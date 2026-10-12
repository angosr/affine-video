import pytest

from scoring.metrics import log_cost_auc, relative_squared_error


def test_relative_squared_error() -> None:
    assert relative_squared_error([1.0, 2.0], [1.0, 1.0], epsilon=1e-9) == pytest.approx(0.2)


def test_log_cost_auc_for_constant_quality() -> None:
    assert log_cost_auc([1.0, 2.0, 4.0], [0.75, 0.75, 0.75]) == pytest.approx(0.75)


def test_log_cost_auc_rejects_unsorted_costs() -> None:
    with pytest.raises(ValueError):
        log_cost_auc([1.0, 1.0], [0.5, 0.6])
