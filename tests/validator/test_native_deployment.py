import hashlib
import importlib.util
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from validator.production.commitment import Artifact, ArtifactManifest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "evidence/system-panel/operations/prepare_native_deployment.py")
spec = importlib.util.spec_from_file_location("native_deployment", SCRIPT)
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


def reference(tmp_path, name, payload=b"fixture"):
    path = tmp_path / name
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def configuration(tmp_path):
    endpoint = {"host": "fixture.invalid", "port": 2222, "user": "root",
                "identity_file": "/operator/ssh-key", "known_hosts": "/operator/known-hosts"}
    return {
        "source_ssh": deepcopy(endpoint), "target_ssh": deepcopy(endpoint),
        "target_hostname": "fixture-target",
        "runtime": {"python": "/frozen/usr/bin/python3.12", "home": "/frozen/usr",
                    "site_packages": "/frozen/site-packages"},
        "production": reference(tmp_path, "production.json"),
        "accepted": reference(tmp_path, "accepted.json"),
        "owner_program": reference(tmp_path, "owner.py", b"raise SystemExit(0)\n"),
        "roles": {
            "owner": {"uid": 0, "gid": 0, "secret_files": ["/owner/manager.env"]},
            "producer": {"uid": 2001, "gid": 2001, "secret_files": ["/producer/role.env"]},
            "evaluator": {"uid": 2002, "gid": 2002, "secret_files": ["/evaluator/role.env"]},
        },
    }


def test_configuration_contains_references_not_business_overrides(configuration):
    assert deployment.configuration(configuration) is configuration
    for field in ("gpu_ids", "inventory_max_bytes", "track_id", "data_digest", "history", "argv"):
        bad = {**configuration, field: "override"}
        with pytest.raises(ValueError, match="deployment fields"):
            deployment.configuration(bad)
    assert "CLOUDFLARE_CONTROL_API_TOKEN" not in json.dumps(configuration)


def test_duplicate_reference_fields_are_not_silently_overridden():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"production": 1, "production": 2}', object_pairs_hook=deployment.unique)


@pytest.mark.parametrize("fault", ["duplicate-uid", "root-worker", "shared-secret", "bad-port",
                                   "shell-host", "relative-path", "bad-sha", "inline-secret"])
def test_configuration_rejects_ambiguous_or_unsafe_values(configuration, fault):
    if fault == "duplicate-uid":
        configuration["roles"]["evaluator"]["uid"] = 2001
    elif fault == "root-worker":
        configuration["roles"]["evaluator"]["uid"] = 0
    elif fault == "shared-secret":
        configuration["roles"]["evaluator"]["secret_files"] = ["/owner/manager.env"]
    elif fault == "bad-port":
        configuration["target_ssh"]["port"] = True
    elif fault == "shell-host":
        configuration["target_ssh"]["host"] = "host; touch /tmp/unwanted"
    elif fault == "relative-path":
        configuration["runtime"]["home"] = "relative"
    elif fault == "bad-sha":
        configuration["production"]["sha256"] = "unknown"
    else:
        configuration["roles"]["owner"]["secret_value"] = "fixture-do-not-send"
    with pytest.raises(ValueError):
        deployment.configuration(configuration)


def test_references_reverify_bytes_without_stat_cache(configuration):
    row = configuration["production"]
    assert deployment.read_reference(row) == b"fixture"
    path = Path(row["path"])
    old = path.stat()
    path.write_bytes(b"changed")  # Same size and restored mtime must not authorize reuse.
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        deployment.read_reference(row)


def test_one_ssh_target_uses_frozen_python_and_no_inherited_management_env(configuration):
    argv = deployment.ssh_command(configuration)
    assert argv[:2] == ["ssh", "-T"]
    assert "-oStrictHostKeyChecking=yes" in argv and "-oForwardAgent=no" in argv
    assert argv[-2] == "root@fixture.invalid"
    assert argv[-1].startswith("timeout --signal=TERM --kill-after=10 900 env -i")
    assert "env -i" in argv[-1] and "/frozen/usr/bin/python3.12 -S -" in argv[-1]
    assert "PYTHONDONTWRITEBYTECODE=1" in argv[-1] and "CUDA_VISIBLE_DEVICES=" in argv[-1]
    program = deployment.remote_program(configuration, "inspect")
    compile(program, "<remote-test>", "exec")
    assert "site.addsitedir('/frozen/site-packages')" in program


@pytest.mark.parametrize("returncode", [0, 1])
def test_operator_entry_is_one_bounded_ssh_call_not_a_copy_or_retry(
    configuration, tmp_path, monkeypatch, capsys, returncode,
):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(configuration))
    calls = []
    monkeypatch.setenv("CLOUDFLARE_CONTROL_API_TOKEN", "fixture-never-serialize")

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv == deployment.ssh_command(configuration)
        assert kwargs["timeout"] == 950 and kwargs["check"] is False
        assert "fixture-never-serialize" not in kwargs["input"]
        return SimpleNamespace(returncode=returncode, stdout='{"business_owner_started":false}',
                               stderr="fixture-secret-must-not-be-logged")

    monkeypatch.setattr(subprocess, "run", run)
    if returncode:
        with pytest.raises(SystemExit) as failure:
            deployment.main(["--config", str(path), "prepare"])
        assert failure.value.code == 1
    else:
        deployment.main(["--config", str(path), "prepare"])
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
    assert len(calls) == 1


@pytest.fixture
def phases(configuration, monkeypatch):
    events = []
    config = SimpleNamespace(gpu_ids=(0, 1), evaluation_gpu_ids=(2,),
                             validator_database=Path("/state/existing.db"),
                             work_dir=Path("/work/original-data"))
    track = SimpleNamespace(track_id="fixed-track", digest="ab" * 32)
    data = SimpleNamespace(digest="cd" * 32)
    manifest = SimpleNamespace(digest="ef" * 32)
    monkeypatch.setattr(deployment.socket, "gethostname", lambda: "fixture-target")
    monkeypatch.setattr(deployment, "inspect_inputs",
                        lambda _value: (config, track, data, b"runtime", manifest))
    monkeypatch.setattr(deployment, "check_staged", lambda *_a: events.append("stage"))
    monkeypatch.setattr(deployment, "verify_release",
                        lambda *_a: events.append("verify") or {"source_sha256": "12" * 32})
    monkeypatch.setattr(deployment, "check_roles", lambda *_a: events.append("roles"))
    return configuration, config, events


@pytest.mark.parametrize("phase,expected", [("inspect", []), ("stage", ["stage"]),
                                           ("verify", ["stage", "verify"]),
                                           ("prepare", ["stage", "verify", "roles"])])
def test_phase_order_never_opens_db_starts_writer_or_transfer(phases, monkeypatch, phase, expected):
    value, config, events = phases
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("no business child"))
    result = deployment.run_phase(value, phase)
    assert events == expected
    assert result["producer_gpu_ids"] == list(config.gpu_ids)
    assert result["evaluation_gpu_ids"] == list(config.evaluation_gpu_ids)
    assert result["database"] == "/state/existing.db"
    assert result["work_dir"] == "/work/original-data"
    assert not any(result[name] for name in ("database_opened", "business_owner_started",
                                             "transfer_started"))
    if phase == "prepare":
        assert result["ready_to_switch"] is result["managed_owner_wiring_verified"] is False


@pytest.mark.parametrize("failed", ["stage", "verify", "roles"])
def test_failure_stops_current_step_without_skipping_or_resume_ledger(phases, monkeypatch, failed):
    value, _config, events = phases

    def failure(*_args):
        events.append("failed")
        raise RuntimeError("fixture failure")

    function = {"stage": "check_staged", "verify": "verify_release", "roles": "check_roles"}[failed]
    monkeypatch.setattr(deployment, function, failure)
    with pytest.raises(RuntimeError, match="fixture failure"):
        deployment.run_phase(value, "prepare")
    assert events == {"stage": ["failed"], "verify": ["stage", "failed"],
                      "roles": ["stage", "verify", "failed"]}[failed]


def test_wrong_host_and_switch_are_rejected_before_any_input_or_actor(phases, monkeypatch):
    value, _config, events = phases
    monkeypatch.setattr(deployment.socket, "gethostname", lambda: "another-machine")
    with pytest.raises(ValueError, match="different machine"):
        deployment.run_phase(value, "prepare")
    with pytest.raises(ValueError, match="switch requires"):
        deployment.run_phase(value, "switch")
    assert events == []


def test_staging_never_accepts_missing_truncated_or_escape_assets(tmp_path):
    names = ("adapter", "predictor", "feature", "binding", "source", "lock", "image")
    artifacts = tuple(Artifact(name, hashlib.sha256(b"x").hexdigest(), 1, "artifacts/" + name)
                      for name in sorted(names))
    manifest = ArtifactManifest(artifacts)
    runtime = SimpleNamespace(**dict(zip(("adapter", "predictor", "feature_model", "source_binding",
                                         "source_bundle", "dependency_lock", "oci_image"), names,
                                        strict=True)),
                              model_root="weights", sample_cache="samples")
    config = SimpleNamespace(runtime=runtime, artifact_root=tmp_path,
                             evaluator_artifact_root=tmp_path,
                             evaluator_artifact_names=lambda _m: names)
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    deployment.check_staged(config, manifest)
    (tmp_path / "adapter").write_bytes(b"")
    with pytest.raises(ValueError, match="not staged"):
        deployment.check_staged(config, manifest)
    (tmp_path / "adapter").unlink()
    (tmp_path / "adapter").symlink_to(tmp_path.parent / "outside")
    with pytest.raises(ValueError, match="escapes"):
        deployment.check_staged(config, manifest)


def test_prepare_rejects_wrong_owner_before_reading_any_secret(configuration, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 9999)
    with pytest.raises(ValueError, match="configured production owner"):
        deployment.check_roles(configuration)


def test_role_check_never_reads_secret_and_rejects_wrong_mode(configuration, tmp_path, monkeypatch):
    uid, gid = os.geteuid(), os.getegid()
    configuration["roles"]["owner"].update(uid=uid, gid=gid)
    secret = tmp_path / "secret"
    secret.write_bytes(b"fixture-must-not-be-read")
    secret.chmod(0o644)
    configuration["roles"]["owner"]["secret_files"] = [str(secret)]
    monkeypatch.setattr(Path, "read_bytes",
                        lambda *_a: pytest.fail("secret value must not be read"))
    with pytest.raises(ValueError, match="0600"):
        deployment.check_roles(configuration)


def test_release_verification_uses_existing_data_source_and_approval_apis(
    configuration, tmp_path, monkeypatch,
):
    from scoring.spec import parse_track_spec
    from tests.scoring.test_spec import valid_spec

    # Actual TrackSpec attributes, not a new deployment-owned copy of source geometry.
    track = parse_track_spec(valid_spec())
    events = []
    source = tmp_path / "source-pool"
    binding = tmp_path / "source-binding"
    source.write_bytes(b"source-pool")
    binding.write_bytes(b"source-binding")
    image = {"image_id": "sha256:" + "ab" * 32}
    config = SimpleNamespace(
        source_pool=source,
        runtime=SimpleNamespace(source_binding="binding", sample_cache="samples"),
        artifact_path=lambda name: binding if name == "binding" else tmp_path,
        verify_manifest=lambda manifest: events.append(("producer", manifest)) or image,
        verify_evaluator_manifest=lambda manifest: events.append(("evaluator", manifest)) or image,
        data_binding=lambda _track, _payload: (track, b"original-data"),
        competition_payload=lambda _track: b"original-competition",
    )

    def pool(pool_bytes, binding_bytes, **kwargs):
        assert pool_bytes == b"source-pool" and binding_bytes == b"source-binding"
        assert kwargs == {"expected_root": track.source_pool_merkle_root,
                          "expected_commitments": track.source_commitment_documents,
                          "sample_cache": tmp_path}
        events.append("source-pool")
        return "original-pool"

    class Approval:
        def __init__(self, digest, accepted, competition):
            assert digest == track.digest and accepted == b"fixture"
            assert competition == b"original-competition"
            events.append("approval")

        def verify(self, rules, payload, **kwargs):
            assert rules is track and payload == b"runtime" and kwargs == image
            events.append("verified")

        def public_identity(self, rules, payload):
            assert rules is track and payload == b"runtime"
            return {"source_sha256": "12" * 32}

    monkeypatch.setattr("validator.production.commitment.load_bound_source_pool", pool)
    monkeypatch.setattr("validator.release.ReleaseApproval", Approval)
    monkeypatch.setattr("validator.production.service._validate_source_capacity",
                        lambda pool, rules: events.append((pool, rules is track)))
    result = deployment.verify_release(configuration, config, track, b"runtime", "manifest")
    assert result == {"source_sha256": "12" * 32}
    assert events == [("producer", "manifest"), ("evaluator", "manifest"), "source-pool",
                      ("original-pool", True), "approval", "verified"]
