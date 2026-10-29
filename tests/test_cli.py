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


def test_private_upload_cli_uses_chain_mailbox_and_returns_unsigned_ready(monkeypatch, capsys):
    from miner import upload as module
    from miner.submit import PrivateModelSubmission

    calls = []
    async def upload(model, **kwargs):
        calls.append((model, kwargs))
        return PrivateModelSubmission("ab" * 32, "cd" * 32, "ef" * 32)
    monkeypatch.setattr(module, "upload_model", upload)
    main(["miner", "upload", "--model", "prepared.safetensors",
          "--mailbox-url", "https://public.example/mailbox/v1",
          "--validator-identity", "pinned-validator"])
    result = json.loads(capsys.readouterr().out)
    assert result == {"upload_id": "ab" * 32, "file_sha256": "cd" * 32,
                      "manifest_sha256": "ef" * 32, "uploaded": True, "accepted": False}
    assert calls[0][1]["mailbox_url"] == "https://public.example/mailbox/v1"
    assert calls[0][1]["validator_identity"] == "pinned-validator"


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


@pytest.mark.parametrize("arguments", [
    ["--repo-id", "owner/model", "--revision", "12" * 20, "--file-sha256", "ab" * 32],
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
