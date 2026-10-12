import json

from validator.export import export_snapshot
from validator.queue import ChainPosition, submission_id
from validator.store import ChallengeStore, SubmissionRecord


def test_export_writes_one_atomic_snapshot(tmp_path) -> None:
    position = ChainPosition(100, 0, 0)
    block_hash = "0x" + "ab" * 32
    submission = SubmissionRecord(
        submission_id=submission_id(7, block_hash, position),
        hotkey="hk1",
        netuid=7,
        block_hash=block_hash,
        position=position,
        payload=b"VDA1payload",
        repo_id="owner/model",
        revision="12" * 20,
    )
    with ChallengeStore(tmp_path / "validator.db") as store:
        store.add_submission(submission, occurred_at_ms=10)
        target = export_snapshot(store, tmp_path / "public")

    snapshot = json.loads(target.read_text())
    assert target.name == "snapshot.json"
    assert snapshot["schema_version"] == 1
    assert snapshot["challenges"][0]["repo_id"] == "owner/model"
    assert not list((tmp_path / "public").glob(".snapshot-*"))
