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


@pytest.mark.parametrize("netuid", ["-1", "65536"])
def test_miner_preview_rejects_invalid_netuid(netuid, capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "miner",
                "submit",
                "--repo-id",
                "owner/model",
                "--revision",
                "12" * 20,
                "--netuid",
                netuid,
            ]
        )

    assert stopped.value.code == 1
    assert "unsigned 16-bit" in capsys.readouterr().err


def test_miner_preview_rejects_empty_network(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
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
                "",
            ]
        )

    assert stopped.value.code == 1
    assert "network cannot be empty" in capsys.readouterr().err


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


def test_evaluator_forwards_runtime_and_r2_configuration(monkeypatch, tmp_path) -> None:
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli_module, "run_evaluator", fake_run)
    monkeypatch.setattr(
        cli_module.R2Repository,
        "from_environment",
        lambda **kwargs: called.setdefault("r2", kwargs),
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
            "--r2-account-id",
            "account",
            "--r2-bucket",
            "public-data",
            "--r2-prefix",
            "veda/track",
            "--once",
        ]
    )

    assert called["once"] is True
    assert called["network"] == "finney"
    assert called["rollout_runner"] == str(tmp_path / "rollout")
    assert called["rollout_panel"] == str(tmp_path / "panel.json")
    assert called["publisher"] == {
        "account_id": "account",
        "bucket": "public-data",
        "prefix": "veda/track",
    }
    assert called["archive_endpoints"] == [
        "wss://archive-1.example",
        "wss://archive-2.example",
    ]


def test_validator_run_only_requires_public_weights_and_wallet(monkeypatch) -> None:
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli_module, "run_validator", fake_run)
    main(
        [
            "validator",
            "run",
            "--public-weights-url",
            "https://public.example/veda/weights.json",
            "--netuid",
            "7",
            "--track-spec-digest",
            "ab" * 32,
            "--wallet-name",
            "validator",
            "--hotkey",
            "default",
            "--once",
        ]
    )

    assert called["public_weights_url"] == "https://public.example/veda/weights.json"
    assert called["netuid"] == 7
    assert called["track_spec_digest"] == "ab" * 32
    assert called["wallet_name"] == "validator"
    assert called["hotkey_name"] == "default"
    assert called["once"] is True
