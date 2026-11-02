import json
from types import SimpleNamespace

import pytest

import cli as cli_module
from cli import main
from validator.queue import ChainPosition, ChallengeState, submission_id
from validator.store import ChallengeStore, SubmissionRecord


def test_evaluator_cycle_logs_do_not_share_mailbox_stdout(capsys):
    from validator.run import EvaluationCycleResult

    cli_module._print_evaluation_cycle(EvaluationCycleResult(200, None, None, None))
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["finalized_block"] == 200


def test_retired_docker_attach_is_not_a_second_owner_path(monkeypatch):
    monkeypatch.setattr("validator.uploads.attach_from_environment",
                        lambda **_kwargs: pytest.fail("retired owner CLI must not launch"))
    with pytest.raises(SystemExit):
        main(["evaluator", "attach", "--container", "veda-evaluator"])


@pytest.mark.parametrize("overrides,mailbox_url,validator_identity", [
    ([], None, None),
    (["--mailbox-url", "https://public.example/mailbox/v1"],
     "https://public.example/mailbox/v1", None),
    (["--validator-identity", "pinned-validator"], None, "pinned-validator"),
    (["--mailbox-url", "https://public.example/mailbox/v1",
      "--validator-identity", "pinned-validator"],
     "https://public.example/mailbox/v1", "pinned-validator"),
])
def test_private_upload_cli_uses_chain_mailbox_and_returns_unsigned_ready(
    monkeypatch, capsys, overrides, mailbox_url, validator_identity,
):
    from miner import upload as module
    from miner.submit import PrivateModelSubmission

    calls = []
    async def upload(model, **kwargs):
        calls.append((model, kwargs))
        return PrivateModelSubmission("ab" * 32, "cd" * 32, "ef" * 32)
    monkeypatch.setattr(module, "upload_model", upload)
    main(["miner", "upload", "--model", "prepared.safetensors",
          "--wallet-name", "miner", "--hotkey", "original", *overrides])
    result = json.loads(capsys.readouterr().out)
    assert result == {"upload_id": "ab" * 32, "file_sha256": "cd" * 32,
                      "manifest_sha256": "ef" * 32, "uploaded": True, "accepted": False}
    assert calls == [("prepared.safetensors", {
        "mailbox_url": mailbox_url or cli_module.DEFAULT_PUBLIC_STATUS_URL.removesuffix(
            "/status.json") + "/mailbox/v1",
        "validator_identity": validator_identity or (
            "5HfGuaU3VP2JhzABtPH5v1PYyzrk5k3VkFid8guMfKhPe53s"),
        "network": "finney", "wallet_name": "miner", "hotkey_name": "original",
        "wallet_path": None,
    })]


def test_retired_loopback_upload_service_and_http_url_flags_are_not_aliases():
    with pytest.raises(SystemExit):
        main(["evaluator", "uploads", "--database", "original.db", "--track-spec", "track.json"])
    with pytest.raises(SystemExit):
        main(["miner", "upload", "--model", "file", "--upload-url", "https://old.example"])


def test_public_monitor_needs_no_miner_or_validator_setup(monkeypatch, capsys) -> None:
    calls = []

    async def fetch(url, *, client):
        calls.append(url)
        return {
            "netuid": 120,
            "source_block": 12345,
            "track_phase": "ACCEPTING",
            "participants": [],
        }

    def unexpected_role(*_args):
        raise AssertionError("public monitor entered a participant role")

    monkeypatch.setattr(cli_module, "fetch_status", fetch)
    monkeypatch.setattr(cli_module, "_run_miner", unexpected_role)
    monkeypatch.setattr(cli_module, "_run_validator", unexpected_role)
    monkeypatch.setattr(cli_module, "_run_evaluator", unexpected_role)

    main(["monitor"])

    assert calls == [cli_module.DEFAULT_PUBLIC_STATUS_URL]
    assert "ACCEPTING" in capsys.readouterr().out


def test_public_audit_reuses_s3_load_and_http_gets_without_credentials(monkeypatch, capsys):
    import asyncio
    import hashlib
    import time

    import httpx

    from scoring.spec import parse_track_spec
    from tests.scoring.test_spec import valid_spec
    from validator.production.commitment import Artifact, ArtifactManifest, canonical_json
    from validator.publication import S3Repository, build_genesis_artifacts
    from validator.weights import WeightChainState

    manifest = ArtifactManifest((Artifact("fixture/identity", "ab" * 32, 1, "artifacts/fixture"),))
    raw = valid_spec()
    raw["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(raw)
    artifacts = build_genesis_artifacts(
        track, WeightChainState(120, {track.initial_champion.hotkey: 0}, 1, 1.0, 10, {}),
        version_key=0,
    )
    objects = {"track.json": artifacts.track, "summary.json": artifacts.summary,
               "weights.json": artifacts.weights,
               f"runtime/{manifest.digest}/manifest.json": manifest.encode()}
    now = time.time_ns() // 1_000_000
    objects["status.json"] = canonical_json({
        "track_id": track.track_id, "track_spec_digest": track.digest, "source_block": 10,
        "inventory": {"observed_at_ns": now * 1_000_000, "disclosure_status": [
            {"plan_digest": digest, "state": state, "attempt": 1,
             "disclose_at_ms": now - 2, "waived_at_ms": now - 1 if state == "WAIVED" else None,
             "retry_after_ms": None if state == "WAIVED" else now + 60_000}
            for digest, state in [("ab" * 32, "WAIVED"), ("cd" * 32, "FAILED"),
                                  ("ef" * 32, "UPLOADING")]
        ]},
    })
    methods, loads = [], []

    def handler(request):
        methods.append(request.method)
        assert request.method == "GET" and "authorization" not in request.headers
        body = objects.get(request.url.path.removeprefix("/tracks/main/"))
        if body is None:
            return httpx.Response(400, json={"statusCode": "404", "error": "not_found"})
        return httpx.Response(200, content=body, headers={"etag": hashlib.sha256(body).hexdigest()})

    original_client, original_load = httpx.Client, S3Repository.load

    async def load(self, *args, **kwargs):
        loads.append(True)
        return await original_load(self, *args, **kwargs)

    monkeypatch.setattr(S3Repository, "load", load)
    monkeypatch.setattr(S3Repository, "from_environment",
                        lambda **_k: pytest.fail("audit read private credentials"))
    monkeypatch.setattr(cli_module.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    main(["audit", "--public-base-url", "https://public.example/tracks/main"])
    result = json.loads(capsys.readouterr().out)
    assert result["committed_matches"] == 0 and loads == [True]
    assert result["unknown_stage_evidence"] == 0 and set(methods) == {"GET"}
    assert [row["plan_digest"] for row in result["disclosure_overdue"]] == ["cd" * 32]
    assert [row["plan_digest"] for row in result["disclosure_in_progress"]] == ["ef" * 32]
    assert "ab" * 32 not in json.dumps(result)  # Waived, never falsely counted as disclosed.
    assert not asyncio.iscoroutine(result)
    # Corrupt public summary facts remain fatal; audit never falls back to status.json.
    objects["summary.json"] = canonical_json({**json.loads(artifacts.summary), "match_count": 1})
    with pytest.raises(SystemExit) as stopped:
        main(["audit", "--public-base-url", "https://public.example/tracks/main"])
    assert stopped.value.code == 1


def test_public_monitor_watch_and_custom_url(monkeypatch, capsys) -> None:
    calls = []

    async def fetch(url, *, client):
        calls.append(url)
        return {"netuid": 120, "participants": []}

    async def stop_after_refresh(interval):
        assert interval == 30
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "fetch_status", fetch)
    monkeypatch.setattr(cli_module.asyncio, "sleep", stop_after_refresh)

    main(["monitor", "--status-url", "https://example.com/status.json", "--watch", "30"])

    assert calls == ["https://example.com/status.json"]
    assert "No participants yet" in capsys.readouterr().out


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_public_monitor_rejects_nonpositive_watch_before_fetch(monkeypatch, capsys, interval):
    async def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("invalid interval opened a status request")

    monkeypatch.setattr(cli_module, "fetch_status", unexpected_fetch)
    with pytest.raises(SystemExit) as stopped:
        main(["monitor", "--watch", interval])

    assert stopped.value.code == 1
    assert "watch interval must be positive" in capsys.readouterr().err


def test_public_monitor_reports_fetch_errors(monkeypatch, capsys):
    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("public status could not be fetched")

    monkeypatch.setattr(cli_module, "fetch_status", unavailable)
    with pytest.raises(SystemExit) as stopped:
        main(["monitor"])

    assert stopped.value.code == 1
    assert "public status could not be fetched" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command", ["monitor", "evaluate", "produce", "evaluator-policy", "retry",
                "leaderboard", "challenges", "challenge"],
)
def test_validator_does_not_expose_public_or_evaluator_commands(capsys, command):
    with pytest.raises(SystemExit) as stopped:
        main(["validator", command])

    assert stopped.value.code == 2
    assert f"invalid choice: '{command}'" in capsys.readouterr().err


def test_miner_dry_run_does_not_open_chain(monkeypatch, capsys) -> None:
    async def unexpected_submit(*_args, **_kwargs):
        raise AssertionError("dry-run opened the chain")

    monkeypatch.setattr(cli_module, "submit_model", unexpected_submit)
    main(
        [
            "miner",
            "commit",
            "--repo-id",
            "owner/model",
            "--revision",
            "12" * 20,
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "owner/model"
    assert output["revision"] == "12" * 20
    assert output["netuid"] == 120
    assert output["network"] == "finney"
    assert output["hotkey"] == "default"
    assert output["payload_bytes"] <= 128


@pytest.mark.parametrize("netuid", ["-1", "65536"])
def test_miner_preview_rejects_invalid_netuid(netuid, capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "miner",
                "commit",
                "--repo-id",
                "owner/model",
                "--revision",
                "12" * 20,
                "--netuid",
                netuid,
                "--dry-run",
            ]
        )

    assert stopped.value.code == 1
    assert "unsigned 16-bit" in capsys.readouterr().err


def test_miner_preview_rejects_empty_network(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "miner",
                "commit",
                "--repo-id",
                "owner/model",
                "--revision",
                "12" * 20,
                "--netuid",
                "7",
                "--network",
                "",
                "--dry-run",
            ]
        )

    assert stopped.value.code == 1
    assert "network cannot be empty" in capsys.readouterr().err


def test_miner_commit_forwards_chain_arguments(monkeypatch, capsys) -> None:
    called = {}

    async def fake_submit(submission, **kwargs):
        called["submission"] = submission
        called.update(kwargs)
        return SimpleNamespace(
            block_hash="0xabc",
            extrinsic_id="10-2",
            message="finalized",
        )

    monkeypatch.setattr(cli_module, "submit_model", fake_submit)
    main(
        [
            "miner",
            "commit",
            "--repo-id",
            "owner/model",
            "--revision",
            "12" * 20,
            "--netuid",
            "7",
            "--network",
            "test",
            "--wallet-name",
            "alice",
            "--hotkey",
            "miner",
            "--wallet-path",
            "/wallets",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert called["submission"].repo_id == "owner/model"
    assert called["netuid"] == 7
    assert called["network"] == "test"
    assert called["wallet_name"] == "alice"
    assert called["hotkey_name"] == "miner"
    assert called["wallet_path"] == "/wallets"
    assert output["block_hash"] == "0xabc"
    assert output["extrinsic_id"] == "10-2"
    assert output["netuid"] == 7


def test_removed_miner_submit_command_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["miner", "submit"])

    assert stopped.value.code == 2
    assert "invalid choice: 'submit'" in capsys.readouterr().err


def test_private_commit_preview_is_explicit_and_does_not_open_chain(monkeypatch, capsys):
    async def unexpected(*_args, **_kwargs):
        pytest.fail("preparatory private source opened a chain connection")

    monkeypatch.setattr(cli_module, "submit_model", unexpected)
    command = ["miner", "commit", "--upload-id", "ab" * 32, "--file-sha256", "cd" * 32,
               "--manifest-sha256", "ef" * 32]
    main([*command, "--dry-run"])
    document = json.loads(capsys.readouterr().out)
    assert document["source"] == "PRIVATE_S3"
    assert document["file_sha256"] == "cd" * 32
    assert document["payload_bytes"] == 100
    assert "repo_id" not in document and "revision" not in document


def test_private_ready_uses_same_hotkey_finalization_entry(monkeypatch, capsys):
    from miner.submit import PrivateModelSubmission

    calls = []

    async def submit(reference, **arguments):
        calls.append((reference, arguments))
        return SimpleNamespace(block_hash="0xabc", extrinsic_id="10-2", message="finalized")

    monkeypatch.setattr(cli_module, "submit_model", submit)
    main(["miner", "commit", "--upload-id", "ab" * 32, "--file-sha256", "cd" * 32,
          "--manifest-sha256", "ef" * 32, "--wallet-name", "miner", "--hotkey", "original"])
    reference, arguments = calls[0]
    assert reference == PrivateModelSubmission("ab" * 32, "cd" * 32, "ef" * 32)
    assert reference.encode() == b"VDR1" + bytes.fromhex("ab" * 32 + "cd" * 32 + "ef" * 32)
    assert arguments == {"netuid": 120, "network": "finney", "wallet_name": "miner",
                         "hotkey_name": "original", "wallet_path": None}
    document = json.loads(capsys.readouterr().out)
    assert document["source"] == "PRIVATE_S3" and document["block_hash"] == "0xabc"
    assert document["upload_id"] == reference.upload_id
    assert document["file_sha256"] == reference.file_sha256
    assert "repo_id" not in document and "revision" not in document


def test_hf_vda2_is_opt_in_and_reports_its_chain_declaration(capsys):
    main(["miner", "commit", "--repo-id", "owner/model", "--revision", "12" * 20,
          "--file-sha256", "ab" * 32, "--dry-run"])
    body = json.loads(capsys.readouterr().out)
    assert body["payload_hex"].startswith(b"VDA2".hex())
    assert body["declared_file_sha256"] == "ab" * 32
    assert "file_sha256" not in body and body["payload_bytes"] == 68


@pytest.mark.parametrize("arguments", [
    ["--repo-id", "owner/model", "--revision", "12" * 20, "--manifest-sha256", "ab" * 32],
    ["--upload-id", "ab" * 32, "--revision", "12" * 20, "--file-sha256", "cd" * 32],
    ["--upload-id", "ab" * 32],
    ["--upload-id", "ab" * 32, "--file-sha256", "cd" * 32, "--netuid", "7"],
])
def test_commit_rejects_mixed_or_incomplete_sources(arguments, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["miner", "commit", *arguments, "--dry-run"])
    assert stopped.value.code == 1


def test_formal_evaluator_defaults_to_cuda_zero() -> None:
    arguments = cli_module._parser().parse_args(
        [
            "evaluator",
            "run",
            "--track-spec",
            "track.json",
            "--database",
            "validator.db",
            "--model-cache",
            "models",
            "--exact-cache",
            "exact",
            "--production-config",
            "producer.json",
        ]
    )

    assert arguments.scoring_device == "cuda:0"


def test_evaluator_challenges_reads_local_database(tmp_path, capsys) -> None:
    database = tmp_path / "validator.db"
    position = ChainPosition(100, 0, 0)
    block_hash = "0x" + "ab" * 32
    with ChallengeStore(database) as store:
        store.add_submission(
            SubmissionRecord(
                submission_id=submission_id(7, block_hash, position),
                hotkey="hk1",
                netuid=7,
                block_hash=block_hash,
                position=position,
                payload=b"VDA1payload",
                repo_id="owner/model",
                revision="12" * 20,
            ),
            occurred_at_ms=10,
        )

    main(["evaluator", "challenges", "--database", str(database)])

    output = json.loads(capsys.readouterr().out)
    assert output[0]["repo_id"] == "owner/model"
    assert output[0]["state"] == "QUEUED"


def test_evaluator_query_does_not_create_a_missing_database(tmp_path, capsys) -> None:
    database = tmp_path / "missing.db"

    with pytest.raises(SystemExit) as stopped:
        main(["evaluator", "leaderboard", "--database", str(database)])

    assert stopped.value.code == 1
    assert "does not exist" in capsys.readouterr().err
    assert not database.exists()


def test_evaluator_retry_explicitly_resumes_an_infrastructure_block(monkeypatch, capsys) -> None:
    from validator.production import config as config_module
    from validator.production import retries as retries_module
    from validator.production import storage as storage_module

    calls = []

    class Store:
        def __init__(self, database, *, track_spec_digest):
            assert database == "validator.db"
            assert track_spec_digest == "ab" * 32

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def blocked_resume_state(self, challenge_id):
            assert challenge_id == "challenge-1"
            return ChallengeState.CONFIRMING

        def resume_blocked(self, challenge_id):
            assert challenge_id == "challenge-1"
            calls.append("local")
            return ChallengeState.CONFIRMING

    class Objects:
        @classmethod
        def from_environment(cls, **kwargs):
            assert kwargs == {"bucket": "private", "prefix": f"root/tracks/{'ab' * 32}"}
            return cls()

        def verify_conditional_writes(self):
            calls.append("probe")

    class Retries:
        def __init__(self, _objects, *, track_spec_digest):
            assert track_spec_digest == "ab" * 32

        def reset(self, challenge_id, stage):
            assert (challenge_id, stage) == ("challenge-1", "CONFIRMING")
            calls.append("durable")

    monkeypatch.setattr(cli_module, "ChallengeStore", Store)
    monkeypatch.setattr(
        config_module,
        "load_producer_config",
        lambda _path: SimpleNamespace(
            validator_database=cli_module.Path("validator.db"),
            track_spec=cli_module.Path("track.json"),
            artifact_manifest=SimpleNamespace(read_bytes=lambda: b"runtime"),
            data_binding=lambda *_args: (SimpleNamespace(digest="ab" * 32), b"data"),
            storage_bucket="private",
            storage_prefix="root",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_track_spec",
        lambda _path: SimpleNamespace(digest="ab" * 32),
    )
    monkeypatch.setattr(storage_module, "PrivateObjectStore", Objects)
    monkeypatch.setattr(retries_module, "PrivateRetryStore", Retries)

    main(
        [
            "evaluator",
            "retry",
            "challenge-1",
            "--database",
            "validator.db",
            "--production-config",
            "production.json",
        ]
    )

    assert json.loads(capsys.readouterr().out) == {
        "challenge_id": "challenge-1",
        "state": "CONFIRMING",
    }
    assert calls == ["probe", "durable", "local"]


@pytest.mark.parametrize("damage", [None, "data_spec", "data_manifest", "runtime_origin"])
def test_evaluator_retry_uses_authenticated_data_scope_and_original_rule_identity(
    tmp_path, monkeypatch, capsys, damage,
):
    from dataclasses import replace

    from tests.validator.production.test_config import bound_data_configuration
    from tests.validator.production.test_storage import MemoryS3
    from tests.validator.test_store import submission_at
    from validator.production.commitment import canonical_json
    from validator.production.retries import PrivateRetryStore
    from validator.production.storage import RETRY_STATE_NAMESPACE, PrivateObjectStore

    rules, data, config, runtime, original_manifest = bound_data_configuration(tmp_path)
    assert rules.digest != data.digest
    prefix = f"{config.storage_prefix}/tracks/{data.digest}"
    rule_prefix = f"{config.storage_prefix}/tracks/{rules.digest}"
    client = MemoryS3()
    objects = PrivateObjectStore(client, bucket=config.storage_bucket, prefix=prefix)
    retries = PrivateRetryStore(objects, track_spec_digest=rules.digest)
    with ChallengeStore(config.validator_database, track_spec_digest=rules.digest) as store:
        identifier, _ = store.add_submission(submission_at(200), occurred_at_ms=1)
        for state in (ChallengeState.FETCHING, ChallengeState.VALIDATING, ChallengeState.SCORING):
            store.transition(identifier, state, occurred_at_ms=2)
        task = store.claim_next_scoring("original-worker", lease_ms=100, now_ms=3)
        retries.record_failure(task, "SCORING", code="GPU_DOWN", summary="worker unavailable",
                               retry_after_ms=100, max_attempts=1)
        store.block_leased(identifier, task.lease_token, error_code="GPU_DOWN",
                           error_summary="worker unavailable", retry_after_ms=100,
                           occurred_at_ms=4)
        before_row = store.get_challenge(identifier)
        before_events = store.challenge_events(identifier)
        before_public = store.public_challenge(identifier)
    name = f"{RETRY_STATE_NAMESPACE}/{identifier}/scoring.json"
    before = objects.get(name)
    # A wrong namespace must not be probed, read or rewritten as a fallback.
    rule_objects = PrivateObjectStore(client, bucket=config.storage_bucket, prefix=rule_prefix)
    rule_objects.create_plan(name, b"unrelated rule-scope object")
    all_before = dict(client.objects)
    factories = []

    def from_environment(*, bucket, prefix):
        factories.append((bucket, prefix))
        assert (bucket, prefix) == (config.storage_bucket, objects._prefix)
        return objects

    monkeypatch.setattr(PrivateObjectStore, "from_environment", from_environment)
    if damage == "data_spec":
        config.data_track_spec.write_bytes(b"{}")
    elif damage == "data_manifest":
        config.data_competition_manifest.write_bytes(canonical_json({"wrong": True}))
    elif damage == "runtime_origin":
        changed = replace(runtime, artifacts=tuple(
            replace(asset, sha256="ff" * 32) if asset.sha256 == data.digest else asset
            for asset in runtime.artifacts
        ))
        config.artifact_manifest.write_bytes(changed.encode())
    argv = ["evaluator", "retry", identifier, "--database", str(config.validator_database),
            "--production-config", str(tmp_path / "producer.json")]
    if damage is not None:
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 1
        assert factories == [] and client.objects == all_before
        with ChallengeStore(config.validator_database, track_spec_digest=rules.digest) as store:
            assert store.get_challenge(identifier) == before_row
            assert store.challenge_events(identifier) == before_events
        assert objects.get(name) == before
        return

    main(argv)
    assert json.loads(capsys.readouterr().out) == {
        "challenge_id": identifier, "state": "SCORING",
    }
    assert factories == [(config.storage_bucket, prefix)]
    reset = json.loads(objects.get(name).body)
    prior = json.loads(before.body)
    assert reset["track_spec_digest"] == rules.digest and reset["state"] == "ACTIVE"
    assert reset["generation"] == prior["generation"] + 1 and reset["attempts"] == 0
    assert {key: value for key, value in reset.items() if key not in {
        "generation", "attempts", "state", "error_code", "error_summary",
        "last_attempt", "retry_after_ms",
    }} == {key: value for key, value in prior.items() if key not in {
        "generation", "attempts", "state", "error_code", "error_summary",
        "last_attempt", "retry_after_ms",
    }}
    assert rule_objects.get(name).body == b"unrelated rule-scope object"
    assert set(client.objects) == set(all_before)
    with ChallengeStore(config.validator_database, track_spec_digest=rules.digest) as store:
        resumed = store.claim_next_scoring("operator-resumed", lease_ms=100, now_ms=10**9)
        assert resumed.submission_id == task.submission_id and resumed.position == task.position
        assert resumed.challenge_id == task.challenge_id and resumed.attempts == 1
        assert retries.restore(resumed, "SCORING") is None
        assert store.public_challenge(identifier)["block_height"] == before_public["block_height"]
    assert config.data_competition_manifest.read_bytes() == original_manifest.encode()


def test_evaluator_forwards_runtime_and_storage_configuration(monkeypatch, tmp_path) -> None:
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli_module, "run_evaluator", fake_run)
    monkeypatch.setattr(
        cli_module.S3Repository,
        "from_environment",
        lambda **kwargs: called.setdefault("storage", kwargs),
    )
    main(
        [
            "evaluator",
            "run",
            "--track-spec",
            str(tmp_path / "track.json"),
            "--database",
            str(tmp_path / "validator.db"),
            "--model-cache",
            str(tmp_path / "models"),
            "--exact-cache",
            str(tmp_path / "exact.npz"),
            "--production-config",
            str(tmp_path / "production.json"),
            "--initial-model",
            str(tmp_path / "veda.safetensors"),
            "--archive-endpoint",
            "wss://archive-1.example",
            "--archive-endpoint",
            "wss://archive-2.example",
            "--once",
        ]
    )

    assert called["once"] is True
    assert called["network"] == "finney"
    assert called["exact_cache"] == str(tmp_path / "exact.npz")
    assert called["production_config"] == str(tmp_path / "production.json")
    assert called["initial_model"] == str(tmp_path / "veda.safetensors")
    assert called["publisher"] == {
        "bucket": "affine_video",
        "prefix": "tracks/main",
    }
    assert called["archive_endpoints"] == [
        "wss://archive-1.example",
        "wss://archive-2.example",
    ]


def test_validator_run_needs_no_track_or_storage_configuration(monkeypatch) -> None:
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli_module, "run_validator", fake_run)
    main(
        [
            "validator",
            "run",
            "--once",
        ]
    )

    assert called["public_weights_url"] == cli_module.DEFAULT_PUBLIC_WEIGHTS_URL
    assert "netuid" not in called
    assert "track_spec_digest" not in called
    assert called["wallet_name"] == "default"
    assert called["hotkey_name"] == "default"
    assert called["once"] is True


def test_evaluator_produce_forwards_config_without_validator_setup(monkeypatch):
    from validator.production import service

    called = {}

    def produce(config, *, once):
        called.update(config=config, once=once)

    def unexpected_validator(*_args):
        raise AssertionError("producer entered validator weight setting")

    monkeypatch.setattr(service, "run_producer", produce)
    monkeypatch.setattr(cli_module, "_run_validator", unexpected_validator)

    main(["evaluator", "produce", "--config", "production.json", "--once"])

    assert called == {"config": "production.json", "once": True}


def test_evaluator_policy_cli_roundtrip_and_fail_closed(tmp_path):
    from validator.production.storage import evaluator_iam_policy

    path = tmp_path / "evaluator.json"
    args = ["evaluator", "policy", "--bucket", "private", "--prefix", "test/track"]
    main([*args, "--write", str(path)])
    original = path.read_bytes()
    assert json.loads(original) == evaluator_iam_policy(bucket="private", prefix="test/track")
    main([*args, "--verify", str(path)])
    with pytest.raises(SystemExit) as stopped:
        main([*args, "--write", str(path)])
    assert stopped.value.code == 1
    assert path.read_bytes() == original
    policy = json.loads(original)
    policy["Statement"].append({"Effect": "Allow", "Action": "s3:*", "Resource": "*"})
    path.write_text(json.dumps(policy))
    with pytest.raises(SystemExit) as stopped:
        main([*args, "--verify", str(path)])
    assert stopped.value.code == 1
