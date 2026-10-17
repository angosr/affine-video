import pytest

from scoring.model import ModelValidationError
from scoring.torch_backend import _require_finite_logits


class _FiniteResult:
    def __init__(self, value: bool) -> None:
        self._value = value

    def all(self) -> bool:
        return self._value


class _Torch:
    @staticmethod
    def isfinite(value):
        return _FiniteResult(value == "finite")


def test_non_finite_gpu_logits_are_a_participant_failure() -> None:
    _require_finite_logits(_Torch(), "finite")

    with pytest.raises(ModelValidationError) as invalid:
        _require_finite_logits(_Torch(), "non-finite")

    assert invalid.value.code == "MODEL_RUNTIME_INVALID"
