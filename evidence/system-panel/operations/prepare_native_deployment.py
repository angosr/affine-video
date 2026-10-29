"""Read-only, reference-only preparation of one native production deployment.

No transfer, credential issuance, writer start, switch, or persisted resume state.
The selected target executes this same file in its explicitly frozen Python root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
from pathlib import Path

PHASES = ("inspect", "stage", "verify", "prepare")


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate deployment configuration field")
        result[key] = value
    return result


def exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"invalid {label} fields")
    return value


def absolute(value):
    if (not isinstance(value, str) or not value.startswith("/")
            or any(part in {".", "..", ""} for part in value[1:].split("/"))
            or any(ord(char) < 32 for char in value)):
        raise ValueError("expected an absolute path without traversal")
    return value


def reference(value):
    exact(value, ("path", "sha256"), "file reference")
    absolute(value["path"])
    if not isinstance(value["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", value["sha256"]):
        raise ValueError("invalid file reference SHA-256")


def configuration(value):
    exact(value, ("source_ssh", "target_ssh", "target_hostname", "runtime", "production",
                  "accepted", "owner_program", "roles"), "deployment")
    for name in ("source_ssh", "target_ssh"):
        row = exact(value[name], ("host", "port", "user", "identity_file", "known_hosts"), name)
        for field in ("host", "user"):
            if (not isinstance(row[field], str)
                    or not re.fullmatch(r"[a-zA-Z0-9][\w.:-]*", row[field])):
                raise ValueError("invalid SSH endpoint")
        if type(row["port"]) is not int or not 1 <= row["port"] <= 65535:
            raise ValueError("invalid SSH port")
        for field in ("identity_file", "known_hosts"):
            absolute(row[field])
    if (not isinstance(value["target_hostname"], str)
            or not re.fullmatch(r"[a-zA-Z0-9][\w.-]*", value["target_hostname"])):
        raise ValueError("invalid target hostname")
    runtime = exact(value["runtime"], ("python", "home", "site_packages"), "runtime paths")
    for path in runtime.values():
        absolute(path)
    for name in ("production", "accepted", "owner_program"):
        reference(value[name])
    roles = exact(value["roles"], ("owner", "producer", "evaluator"), "Unix roles")
    for name, role in roles.items():
        exact(role, ("uid", "gid", "secret_files"), "Unix role")
        for key in ("uid", "gid"):
            if type(role[key]) is not int or role[key] < (0 if name == "owner" else 1):
                raise ValueError("worker Unix roles must be non-root")
        if not isinstance(role["secret_files"], list) or not role["secret_files"]:
            raise ValueError("each role requires explicit secret-file references")
        for path in role["secret_files"]:
            absolute(path)
    if len({role["uid"] for role in roles.values()}) != 3:
        raise ValueError("owner and worker UIDs must be distinct")
    paths = [path for role in roles.values() for path in role["secret_files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("secret files must not be shared across Unix roles")
    return value


def read_reference(row):
    reference(row)
    payload = Path(row["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"referenced file SHA-256 differs: {row['path']}")
    return payload


def check_roles(value):
    """Check metadata only: never read or report a production secret value."""
    owner = value["roles"]["owner"]
    if (owner["uid"], owner["gid"]) != (os.geteuid(), os.getegid()):
        raise ValueError("prepare must run as the configured production owner")
    for role in value["roles"].values():
        for name in role["secret_files"]:
            path = Path(name)
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError("secret reference traverses a symlink")
            metadata = path.lstat()
            if (not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or (metadata.st_uid, metadata.st_gid) != (role["uid"], role["gid"])):
                raise ValueError("secret reference must be its role's 0600 regular file")
    program = Path(value["owner_program"]["path"])
    for path in (program, *program.parents):
        metadata = path.lstat()
        if (path.is_symlink() or metadata.st_uid != owner["uid"]
                or stat.S_IMODE(metadata.st_mode) & 0o022):
            raise ValueError("owner program and ancestors must not be worker-writable")
    compile(read_reference(value["owner_program"]), str(program), "exec")


def inspect_inputs(value):
    from scoring.spec import load_track_spec
    from validator.production.commitment import ArtifactManifest
    from validator.production.config import load_producer_config

    read_reference(value["production"])
    config = load_producer_config(value["production"]["path"])
    track = load_track_spec(config.track_spec)
    payload = config.artifact_manifest.read_bytes()
    manifest = ArtifactManifest.decode(payload)
    # This validates the original data identity without creating/restoring any DB.
    data, _ = config.data_binding(track, payload)
    return config, track, data, payload, manifest


def check_staged(config, manifest):
    """Presence/size only; stage never claims full-SHA or starts another transfer."""
    names = {config.runtime.adapter, config.runtime.predictor, config.runtime.feature_model,
             config.runtime.source_binding, config.runtime.source_bundle,
             config.runtime.dependency_lock, config.runtime.oci_image}
    prefixes = (config.runtime.model_root.rstrip("/") + "/",
                config.runtime.sample_cache.rstrip("/") + "/")
    selected = {item.logical_name for item in manifest.artifacts
                if item.logical_name in names or item.logical_name.startswith(prefixes)}
    if not names <= selected:
        raise ValueError("runtime inputs are absent from the manifest")
    scopes = ((config.artifact_root, selected),
              (config.evaluator_artifact_root, config.evaluator_artifact_names(manifest)))
    for root, required in scopes:
        for item in manifest.artifacts:
            if item.logical_name not in required:
                continue
            path = (root / item.logical_name).resolve()
            if root.resolve() not in path.parents:
                raise ValueError("staged input escapes its configured artifact root")
            if not path.is_file() or path.stat().st_size != item.size_bytes:
                raise ValueError(f"input not staged: {item.logical_name}")


def verify_release(value, config, track, payload, manifest):
    from validator.production.commitment import load_bound_source_pool
    from validator.production.service import _validate_source_capacity
    from validator.release import ReleaseApproval

    image = config.verify_manifest(manifest)
    evaluation_image = config.verify_evaluator_manifest(manifest)
    if image != evaluation_image:
        raise ValueError("producer and evaluator runtime descriptors differ")
    data, _ = config.data_binding(track, payload)
    pool = load_bound_source_pool(
        config.source_pool.read_bytes(),
        config.artifact_path(config.runtime.source_binding).read_bytes(),
        expected_root=data.source_pool_merkle_root,
        expected_commitments=data.source_commitment_documents,
        sample_cache=config.artifact_path(config.runtime.sample_cache),
    )
    _validate_source_capacity(pool, data)
    approval = ReleaseApproval(track.digest, read_reference(value["accepted"]),
                               config.competition_payload(track))
    approval.verify(track, payload, image_id=image["image_id"])
    return approval.public_identity(track, payload)


def run_phase(value, phase):
    configuration(value)
    if phase not in PHASES:
        raise ValueError("switch requires a separately approved final-state handoff")
    if socket.gethostname() != value["target_hostname"]:
        raise ValueError("refusing to prepare a different machine")
    config, track, data, payload, manifest = inspect_inputs(value)
    result = {"phase": phase, "target_hostname": value["target_hostname"],
              "source_host": value["source_ssh"]["host"],
              "track_id": track.track_id, "track_digest": track.digest, "data_digest": data.digest,
              "runtime_manifest_sha256": manifest.digest,
              "producer_gpu_ids": list(config.gpu_ids),
              "evaluation_gpu_ids": list(config.evaluation_gpu_ids),
              "database": str(config.validator_database), "work_dir": str(config.work_dir),
              "business_owner_started": False, "database_opened": False, "transfer_started": False}
    if phase in ("stage", "verify", "prepare"):
        check_staged(config, manifest)
        result["staged_sizes_checked"] = True
    if phase in ("verify", "prepare"):
        result["runtime_release"] = verify_release(value, config, track, payload, manifest)
        result["selected_asset_sha256_checked"] = True
        result["engine_verified"] = False  # An image descriptor is provenance, not a native Engine.
    if phase == "prepare":
        check_roles(value)
        result["role_file_metadata_checked"] = True
        result["managed_owner_wiring_verified"] = False
        # Final DB/S/claim and live owner checks are still required.
        result["ready_to_switch"] = False
    return result


def ssh_command(value):
    target = value["target_ssh"]
    runtime = value["runtime"]
    remote = ["timeout", "--signal=TERM", "--kill-after=10", "900",
              "env", "-i", "PATH=/usr/bin:/bin", "HOME=/tmp", "PYTHONNOUSERSITE=1",
              "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=",
              "PYTHONHOME=" + runtime["home"], "PYTHONPATH=" + runtime["site_packages"],
              runtime["python"], "-S", "-"]
    return ["ssh", "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oForwardAgent=no",
            "-oIdentitiesOnly=yes", "-oIdentityAgent=none", "-oConnectTimeout=15",
            "-oUserKnownHostsFile=" + target["known_hosts"], "-i", target["identity_file"],
            "-p", str(target["port"]), target["user"] + "@" + target["host"], shlex.join(remote)]


def remote_program(value, phase):
    # Only fixed maintained code and non-secret references cross this existing SSH channel.
    return (
        "import site\n"
        f"site.addsitedir({value['runtime']['site_packages']!r})\n"
        "namespace = {'__name__': 'deployment_worker'}\n"
        f"exec(compile({Path(__file__).read_text()!r}, '<deployment-worker>', 'exec'), namespace)\n"
        f"result = namespace['run_phase']({value!r}, {phase!r})\n"
        "print(namespace['json'].dumps(result, sort_keys=True))\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args(argv)
    value = configuration(json.loads(args.config.read_text(), object_pairs_hook=unique))
    completed = subprocess.run(ssh_command(value), input=remote_program(value, args.phase),
                               text=True, capture_output=True, timeout=950, check=False)
    if completed.returncode:
        # No remote tracebacks, environment values or secret-file contents in operator logs.
        parser.exit(1, f"deployment {args.phase} failed; remote exit {completed.returncode}\n")
    print(json.dumps(json.loads(completed.stdout), sort_keys=True))


if __name__ == "__main__":
    main()
