from itertools import product
from math import log

import pytest

from scoring.metrics import log_cost_auc, relative_squared_error


def test_relative_squared_error() -> None:
    assert relative_squared_error([1.0, 2.0], [1.0, 1.0], epsilon=1e-9) == pytest.approx(0.2)


def test_log_cost_auc_for_constant_quality() -> None:
    assert log_cost_auc([1.0, 2.0, 4.0], [0.75, 0.75, 0.75]) == pytest.approx(0.75)


def test_log_cost_auc_rejects_unsorted_costs() -> None:
    with pytest.raises(ValueError):
        log_cost_auc([1.0, 1.0], [0.5, 0.6])


@pytest.mark.parametrize("middle_cost", [1.5, 3.0])
def test_three_budget_cost_axis_has_bounded_endpoint_effect(middle_cost: float) -> None:
    original = [1.0, 2.0, 4.0]
    changed = [1.0, middle_cost, 4.0]
    middle_shift = log(middle_cost) / log(4.0) - 0.5
    differences = []
    for quality in (*product((0.0, 1.0), repeat=3), (0.2, 0.4, 0.9)):
        difference = log_cost_auc(changed, quality) - log_cost_auc(original, quality)
        assert difference == pytest.approx(middle_shift * (quality[0] - quality[2]) / 2)
        differences.append(abs(difference))
    assert max(differences) == pytest.approx(abs(middle_shift) / 2)


def test_log_cost_auc_is_invariant_to_common_cost_scale() -> None:
    assert log_cost_auc([0.5, 1.0, 2.0], [0.2, 0.5, 0.9]) == pytest.approx(
        log_cost_auc([4.0, 8.0, 16.0], [0.2, 0.5, 0.9])
    )
