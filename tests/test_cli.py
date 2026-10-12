import json
from types import SimpleNamespace

import pytest

import cli as cli_module
from cli import main
from validator.queue import ChainPosition, submission_id
from validator.store import ChallengeStore, SubmissionRecord


def test_miner_preview_does_not_open_chain(capsys) -> None:
    main(
        [
            "miner",
            "submit",
            "--repo-id",
            "owner/model",
            "--revision",
            "12" * 20,
            "--netuid",
            "7",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "owner/model"
    assert output["revision"] == "12" * 20
    assert output["netuid"] == 7
    assert output["network"] == "finney"
    assert output["hotkey"] == "default"
    assert output["payload_bytes"] <= 128


def test_miner_execute_forwards_chain_arguments(monkeypatch, capsys) -> None:
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
            "submit",
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
            "--execute",
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


def test_validator_run_forwards_explicit_runtime_paths(monkeypatch, tmp_path) -> None:
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli_module, "run_validator", fake_run)
    main(
        [
            "validator",
            "run",
            "--track-spec",
            str(tmp_path / "track.json"),
            "--database",
            str(tmp_path / "validator.db"),
            "--model-cache",
            str(tmp_path / "models"),
            "--screen-cache",
            str(tmp_path / "screen.npz"),
            "--exact-cache",
            str(tmp_path / "exact.npz"),
            "--rollout-runner",
            str(tmp_path / "rollout"),
            "--rollout-panel",
            str(tmp_path / "panel.json"),
            "--archive-endpoint",
            "wss://archive-1.example",
            "--archive-endpoint",
            "wss://archive-2.example",
            "--once",
        ]
    )

    assert called["once"] is True
    assert called["network"] == "finney"
    assert called["rollout_runner"] == str(tmp_path / "rollout")
    assert called["rollout_panel"] == str(tmp_path / "panel.json")
    assert called["archive_endpoints"] == [
        "wss://archive-1.example",
        "wss://archive-2.example",
    ]
