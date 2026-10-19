from types import SimpleNamespace

import pytest

from validator.s3 import is_missing, is_precondition_failed


@pytest.mark.parametrize("status,code,missing,conflict", [
    (404, None, True, False),
    (None, "404", True, False),
    (None, "NoSuchKey", True, False),
    (None, "NotFound", True, False),
    (412, None, False, True),
    (None, "412", False, True),
    (None, "PreconditionFailed", False, True),
    (403, "AccessDenied", False, False),
    (500, "InternalError", False, False),
    (None, None, False, False),
])
def test_public_and_private_storage_classify_the_same_errors(status, code, missing, conflict):
    error = SimpleNamespace(response={
        "ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code},
    })

    assert is_missing(error) is missing
    assert is_precondition_failed(error) is conflict


@pytest.mark.parametrize("error", [RuntimeError("offline"), SimpleNamespace(response=None)])
def test_non_s3_errors_are_not_missing_objects_or_cas_conflicts(error):
    assert not is_missing(error)
    assert not is_precondition_failed(error)
