import json
from types import SimpleNamespace

import pytest

import cli as cli_module
from cli import main
from validator.queue import ChainPosition, submission_id
from validator.store import ChallengeStore, SubmissionRecord


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


def test_validator_challenges_reads_local_database(tmp_path, capsys) -> None:
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

    main(["validator", "challenges", "--database", str(database)])

    output = json.loads(capsys.readouterr().out)
    assert output[0]["repo_id"] == "owner/model"
    assert output[0]["state"] == "QUEUED"


def test_validator_query_does_not_create_a_missing_database(tmp_path, capsys) -> None:
    database = tmp_path / "missing.db"

    with pytest.raises(SystemExit) as stopped:
        main(["validator", "leaderboard", "--database", str(database)])

    assert stopped.value.code == 1
    assert "does not exist" in capsys.readouterr().err
    assert not database.exists()


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
            "validator",
            "evaluate",
            "--track-spec",
            str(tmp_path / "track.json"),
            "--database",
            str(tmp_path / "validator.db"),
            "--model-cache",
            str(tmp_path / "models"),
            "--exact-cache",
            str(tmp_path / "exact.npz"),
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
