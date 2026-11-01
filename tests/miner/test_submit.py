import pytest

from miner.submit import (
    MAGIC,
    MAX_PAYLOAD_BYTES,
    MAX_REPO_ID_BYTES,
    ModelSubmission,
    PrivateModelSubmission,
    SubmissionFormatError,
    declared_model_sha256,
    model_source_fields,
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


def test_vda2_exact_raw128_bound_preserves_legacy_golden_and_source_identity():
    repo = "o/" + "r" * 69
    model = ModelSubmission(repo, REVISION, "ab" * 32)
    assert model.encode() == b"VDA2\x47" + repo.encode() + bytes.fromhex(REVISION + "ab" * 32)
    assert len(model.encode()) == 128 and ModelSubmission.decode(model.encode()) == model
    with pytest.raises(SubmissionFormatError, match="71-byte"):
        ModelSubmission(repo + "x", REVISION, "ab" * 32)
    legacy = ModelSubmission("o/" + "r" * 94, REVISION)
    assert len(legacy.encode()) == 121 and ModelSubmission.decode(legacy.encode()) == legacy
    assert legacy.encode() == b"VDA1\x60" + legacy.repo_id.encode() + bytes.fromhex(REVISION)
    source = model_source_fields(repo, REVISION, declared_file_sha256=model.file_sha256)
    assert declared_model_sha256(source) == model.file_sha256
    assert "declared_file_sha256" not in model_source_fields(repo, REVISION)
    with pytest.raises(SubmissionFormatError):
        declared_model_sha256({**source, "declared_file_sha256": None})
    with pytest.raises(SubmissionFormatError):
        declared_model_sha256({**source, "private_upload": {}})


@pytest.mark.parametrize("mutation", ["truncate", "extra", "length", "magic"])
def test_vda2_rejects_noncanonical_payload(mutation):
    raw = ModelSubmission("owner/model", REVISION, "ab" * 32).encode()
    changed = {"truncate": raw[:-1], "extra": raw + b"x",
               "length": raw[:4] + b"\x01" + raw[5:], "magic": b"VDA3" + raw[4:]}[mutation]
    with pytest.raises(SubmissionFormatError):
        ModelSubmission.decode(changed)


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


def test_private_reference_has_its_own_bounded_type_and_complete_digest():
    reference = PrivateModelSubmission("ab" * 32, "cd" * 32)
    assert len(reference.encode()) == 68
    assert reference.encode() == b"VDS1" + bytes.fromhex("ab" * 32 + "cd" * 32)
    assert PrivateModelSubmission.decode(reference.encode()) == reference
    with pytest.raises(SubmissionFormatError):
        ModelSubmission.decode(reference.encode())
    # Existing public commitments keep their exact byte contract.
    assert ModelSubmission("alice/model", REVISION).encode() == (
        b"VDA1\x0balice/model" + bytes.fromhex(REVISION)
    )


@pytest.mark.parametrize("digest", ["ab" * 20, "AB" * 32, "g" * 64, "../key", ""])
def test_private_reference_rejects_partial_digest_and_paths(digest):
    with pytest.raises(SubmissionFormatError):
        PrivateModelSubmission(digest, "cd" * 32)
    with pytest.raises(SubmissionFormatError):
        PrivateModelSubmission("ab" * 32, digest)


@pytest.mark.parametrize("payload", [b"VDS1", b"VDS1" + b"a" * 63, b"VDS1" + b"a" * 65])
def test_private_payload_rejects_truncation_and_trailing_bytes(payload):
    with pytest.raises(SubmissionFormatError):
        PrivateModelSubmission.decode(payload)
