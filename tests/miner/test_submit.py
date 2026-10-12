import pytest

from miner.submit import (
    MAGIC,
    MAX_PAYLOAD_BYTES,
    MAX_REPO_ID_BYTES,
    ModelSubmission,
    SubmissionFormatError,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_submission_round_trip() -> None:
    submission = ModelSubmission("alice/veda-selector", REVISION.upper())

    payload = submission.encode()

    assert payload.startswith(MAGIC)
    assert ModelSubmission.decode(payload) == ModelSubmission("alice/veda-selector", REVISION)


def test_payload_bound_is_enforced() -> None:
    owner = "a" * 40
    repository = "b" * (MAX_REPO_ID_BYTES - len(owner) - 1)
    payload = ModelSubmission(f"{owner}/{repository}", REVISION).encode()

    assert len(payload) == MAX_PAYLOAD_BYTES == 121


@pytest.mark.parametrize(
    "repo_id",
    ["missing-slash", "too/many/slashes", "/empty", "empty/", "bad owner/repo"],
)
def test_invalid_repo_id_is_rejected(repo_id: str) -> None:
    with pytest.raises(SubmissionFormatError):
        ModelSubmission(repo_id, REVISION)


def test_invalid_revision_is_rejected() -> None:
    with pytest.raises(SubmissionFormatError):
        ModelSubmission("alice/model", "main")


def test_trailing_bytes_are_rejected() -> None:
    payload = ModelSubmission("alice/model", REVISION).encode()

    with pytest.raises(SubmissionFormatError):
        ModelSubmission.decode(payload + b"x")
