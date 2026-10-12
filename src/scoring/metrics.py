"""Core metric formulas shared by local evaluation and validators."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import fsum, isfinite, log


def relative_squared_error(
    dense: Sequence[float], sparse: Sequence[float], *, epsilon: float
) -> float:
    """Return squared output error normalized by dense output energy."""

    if len(dense) != len(sparse) or len(dense) == 0:
        raise ValueError("dense and sparse outputs must have the same non-zero length")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not all(isfinite(value) for value in (*dense, *sparse)):
        raise ValueError("metric inputs must be finite")

    error = fsum((left - right) ** 2 for left, right in zip(dense, sparse, strict=True))
    energy = fsum(value**2 for value in dense)
    return error / max(energy, epsilon)


def log_cost_auc(costs: Sequence[float], qualities: Sequence[float]) -> float:
    """Compute normalized trapezoidal AUC on the logarithmic cost axis."""

    if len(costs) != len(qualities) or len(costs) < 2:
        raise ValueError("costs and qualities must contain the same two or more points")
    if not all(isfinite(cost) and cost > 0 for cost in costs):
        raise ValueError("costs must be finite and positive")
    if any(right <= left for left, right in pairwise(costs)):
        raise ValueError("costs must be strictly increasing")
    if not all(isfinite(quality) and 0 <= quality <= 1 for quality in qualities):
        raise ValueError("qualities must be finite and within [0, 1]")

    logs = [log(cost) for cost in costs]
    span = logs[-1] - logs[0]
    normalized = [(value - logs[0]) / span for value in logs]
    return fsum(
        (qualities[index] + qualities[index + 1]) * (normalized[index + 1] - normalized[index]) / 2
        for index in range(len(qualities) - 1)
    )
