#!/usr/bin/env python3
"""Small, memory-bounded CPU experiments for the competition design.

The experiments use synthetic tiled attention. They validate evaluator mechanics
and assumptions that do not require a real Video DiT or GPU. They do not claim
to validate video quality or wall-clock GPU speed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import resource
import time
import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-9
SEED = 20261012


@dataclass
class Case:
    q: np.ndarray  # [nq, tile, d]
    k: np.ndarray  # [nk, tile, d]
    v: np.ndarray  # [nk, tile, d]
    dense: np.ndarray  # [nq, tile, d]
    mass: np.ndarray  # [nq, nk], rows sum to one
    teacher: np.ndarray  # [nq, nk], Veda max-pooled distribution; rows sum to one
    avg_q: np.ndarray
    avg_k: np.ndarray
    trip_q: np.ndarray
    trip_k: np.ndarray


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


def descriptors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    avg = x.mean(axis=1)
    trip = np.concatenate([avg, x.max(axis=1), x.min(axis=1)], axis=-1)
    return avg.astype(np.float32), trip.astype(np.float32)


def dense_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nq, tile, d = q.shape
    nk = k.shape[0]
    qf = q.reshape(nq * tile, d).astype(np.float32)
    kf = k.reshape(nk * tile, d).astype(np.float32)
    vf = v.reshape(nk * tile, d).astype(np.float32)
    logits = (qf @ kf.T) / math.sqrt(d)
    attn = softmax(logits, axis=-1)
    out = (attn @ vf).reshape(nq, tile, d)
    a4 = attn.reshape(nq, tile, nk, tile)
    mass = a4.sum(axis=3).mean(axis=1)
    teacher = a4.max(axis=(1, 3))
    teacher /= teacher.sum(axis=1, keepdims=True)
    return out.astype(np.float32), mass.astype(np.float32), teacher.astype(np.float32)


def sparse_row(qrow: np.ndarray, k: np.ndarray, v: np.ndarray, selected: np.ndarray) -> np.ndarray:
    d = qrow.shape[-1]
    ks = k[selected].reshape(-1, d).astype(np.float32)
    vs = v[selected].reshape(-1, d).astype(np.float32)
    logits = (qrow.astype(np.float32) @ ks.T) / math.sqrt(d)
    return (softmax(logits, axis=-1) @ vs).astype(np.float32)


def sparse_attention(case: Case, mask: np.ndarray) -> np.ndarray:
    rows = [
        sparse_row(case.q[i], case.k, case.v, np.flatnonzero(mask[i])) for i in range(mask.shape[0])
    ]
    return np.stack(rows)


def make_case(
    rng: np.random.Generator, nq: int = 8, nk: int = 12, tile: int = 4, d: int = 8
) -> Case:
    # Smooth tile centers make average pooling useful.
    centers = rng.normal(0, 0.65, size=(nk, d)).astype(np.float32)
    q = np.empty((nq, tile, d), np.float32)
    k = centers[:, None, :] + rng.normal(0, 0.18, size=(nk, tile, d)).astype(np.float32)

    # Values differ across tiles so a wrong mask changes the output.
    rotation = rng.normal(0, 1 / math.sqrt(d), size=(d, d)).astype(np.float32)
    v = (k @ rotation + rng.normal(0, 0.08, size=k.shape)).astype(np.float32)

    for i in range(nq):
        local = (3 * i + int(rng.integers(0, 3))) % nk
        q[i] = centers[local] + rng.normal(0, 0.18, size=(tile, d)).astype(np.float32)

        # A zero-mean +/- outlier pair creates a dependency that mean pooling
        # tends to hide but extrema can reveal.
        salient = int(rng.integers(0, nk))
        direction = rng.normal(size=d).astype(np.float32)
        direction /= np.linalg.norm(direction) + EPS
        strength = np.float32(rng.uniform(2.2, 3.2))
        q[i, 0] += strength * direction
        q[i, 1] -= strength * direction
        k[salient, 0] += strength * direction
        k[salient, 1] -= strength * direction
        v[salient, 0] += 0.8 * direction
        v[salient, 1] -= 0.8 * direction

    dense, mass, teacher = dense_attention(q, k, v)
    avg_q, trip_q = descriptors(q)
    avg_k, trip_k = descriptors(k)
    return Case(q, k, v, dense, mass, teacher, avg_q, avg_k, trip_q, trip_k)


def pair_features(qdesc: np.ndarray, kdesc: np.ndarray) -> np.ndarray:
    nq, f = qdesc.shape
    nk = kdesc.shape[0]
    q = np.broadcast_to(qdesc[:, None, :], (nq, nk, f))
    k = np.broadcast_to(kdesc[None, :, :], (nq, nk, f))
    return (
        np.concatenate([q, k, q * k, np.abs(q - k)], axis=-1)
        .reshape(nq * nk, 4 * f)
        .astype(np.float32)
    )


def build_xy(cases: list[Case], kind: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for case in cases:
        qd, kd = (case.avg_q, case.avg_k) if kind == "avg" else (case.trip_q, case.trip_k)
        xs.append(pair_features(qd, kd))
        ys.append(np.log(case.teacher.reshape(-1) + 1e-6))
    return np.concatenate(xs), np.concatenate(ys).astype(np.float32)


def train_model(cases: list[Case], kind: str, hidden: tuple[int, ...], seed: int):
    x, y = build_xy(cases, kind)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=2e-3,
            max_iter=90,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=8,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return model


def model_scores(model, case: Case, kind: str, quantized: bool = False) -> np.ndarray:
    qd, kd = (case.avg_q, case.avg_k) if kind == "avg" else (case.trip_q, case.trip_k)
    if quantized:
        qd = qd.astype(np.float16).astype(np.float32)
        kd = kd.astype(np.float16).astype(np.float32)
    return model.predict(pair_features(qd, kd)).reshape(case.mass.shape).astype(np.float32)


def anchor_mask(nq: int, nk: int) -> np.ndarray:
    mask = np.zeros((nq, nk), dtype=bool)
    mask[np.arange(nq), np.arange(nq) % nk] = True
    return mask


def allocate(
    scores: np.ndarray,
    budget: int,
    adaptive: bool,
    row_cap: int = 6,
    force_anchor: bool = True,
) -> np.ndarray:
    nq, nk = scores.shape
    if not adaptive and budget % nq:
        raise ValueError("fixed allocation requires the same integer k for every query")
    if adaptive and not force_anchor:
        raise ValueError("this experimental global allocator requires one anchor per query")
    minimum = nq if force_anchor or not adaptive else 0
    if budget < minimum or budget > nq * min(nk, row_cap):
        raise ValueError("budget outside feasible range")
    mask = anchor_mask(nq, nk) if force_anchor else np.zeros((nq, nk), dtype=bool)
    counts = mask.sum(axis=1)
    if adaptive:
        # Stable deterministic ordering: score descending, flat index ascending.
        flat_index = np.arange(nq * nk)
        order = np.lexsort((flat_index, -scores.reshape(-1)))
        for idx in order:
            if int(mask.sum()) >= budget:
                break
            i, j = divmod(int(idx), nk)
            if not mask[i, j] and counts[i] < row_cap:
                mask[i, j] = True
                counts[i] += 1
    else:
        base, extra = divmod(budget, nq)
        targets = np.full(nq, base, dtype=int)
        targets[:extra] += 1
        for i in range(nq):
            order = np.lexsort((np.arange(nk), -scores[i]))
            for j in order:
                if counts[i] >= targets[i]:
                    break
                if not mask[i, j]:
                    mask[i, j] = True
                    counts[i] += 1
    if int(mask.sum()) != budget:
        raise AssertionError((mask.sum(), budget))
    return mask


def row_errors(case: Case, mask: np.ndarray) -> np.ndarray:
    sparse = sparse_attention(case, mask)
    num = np.sum((sparse - case.dense) ** 2, axis=(1, 2))
    den = np.sum(case.dense**2, axis=(1, 2)) + EPS
    return num / den


def quality_from_errors(errors: np.ndarray) -> float:
    agg = 0.8 * float(np.mean(errors)) + 0.2 * float(np.percentile(errors, 95))
    return 1.0 / (1.0 + agg)


def log_cost_auc(qualities: list[float] | np.ndarray, costs: list[int] | np.ndarray) -> float:
    q = np.asarray(qualities, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    if q.ndim != 1 or c.ndim != 1 or q.size != c.size or q.size < 2:
        raise ValueError("qualities and costs must be equal-length one-dimensional arrays")
    if np.any(c <= 0) or np.any(np.diff(c) <= 0):
        raise ValueError("costs must be positive and strictly increasing")
    x = (np.log(c) - np.log(c[0])) / (np.log(c[-1]) - np.log(c[0]))
    return float(np.trapezoid(q, x))


def evaluate_scores(
    cases: list[Case], scorer, budgets: list[int], adaptive: bool
) -> tuple[np.ndarray, np.ndarray]:
    per_budget_q = []
    per_case_q = np.zeros((len(cases), len(budgets)), np.float64)
    for cidx, case in enumerate(cases):
        scores = scorer(case)
        for bidx, budget in enumerate(budgets):
            errs = row_errors(
                case,
                allocate(scores, budget, adaptive=adaptive, force_anchor=adaptive),
            )
            per_case_q[cidx, bidx] = quality_from_errors(errs)
    for bidx in range(len(budgets)):
        # Aggregate all case/query errors exactly as the design proposes.
        all_err = []
        for case in cases:
            scores = scorer(case)
            all_err.extend(
                row_errors(
                    case,
                    allocate(
                        scores,
                        budgets[bidx],
                        adaptive=adaptive,
                        force_anchor=adaptive,
                    ),
                )
            )
        per_budget_q.append(quality_from_errors(np.asarray(all_err)))
    return np.asarray(per_budget_q), per_case_q


def prefix_errors(
    case: Case, scores: np.ndarray, row_cap: int = 6
) -> tuple[list[list[int]], np.ndarray]:
    """Return each row's score-ranked prefix and its exact error after every increment."""
    nq, nk = scores.shape
    prefixes: list[list[int]] = []
    errors = np.zeros((nq, row_cap), dtype=np.float32)
    for i in range(nq):
        anchor = i % nk
        order = np.lexsort((np.arange(nk), -scores[i]))
        prefix = [anchor] + [int(j) for j in order if int(j) != anchor][: row_cap - 1]
        prefixes.append(prefix)
        for k in range(1, row_cap + 1):
            out = sparse_row(case.q[i], case.k, case.v, np.asarray(prefix[:k]))
            errors[i, k - 1] = np.sum((out - case.dense[i]) ** 2) / (
                np.sum(case.dense[i] ** 2) + EPS
            )
    return prefixes, errors


def demand_features(case: Case, scores: np.ndarray, row_cap: int = 6) -> np.ndarray:
    probs = softmax(scores, axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1][:, :row_cap]
    entropy = -np.sum(probs * np.log(probs + EPS), axis=1, keepdims=True)
    spread = (scores.max(axis=1) - scores.min(axis=1))[:, None]
    anchor = scores[np.arange(scores.shape[0]), np.arange(scores.shape[0]) % scores.shape[1]][
        :, None
    ]
    return np.concatenate([case.trip_q, sorted_probs, entropy, spread, anchor], axis=1).astype(
        np.float32
    )


def train_demand_model(cases: list[Case], rank_model, seed: int, row_cap: int = 6):
    xs, ys = [], []
    for case in cases:
        scores = model_scores(rank_model, case, "trip")
        _, errors = prefix_errors(case, scores, row_cap)
        gains = errors[:, :-1] - errors[:, 1:]
        xs.append(demand_features(case, scores, row_cap))
        ys.append(np.clip(gains, -3.0, 3.0))
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="relu",
            batch_size=128,
            learning_rate_init=2e-3,
            max_iter=110,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=10,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(np.concatenate(xs), np.concatenate(ys))
    return model


def allocate_by_marginal(
    scores: np.ndarray,
    prefixes: list[list[int]],
    marginal_gains: np.ndarray,
    budget: int,
    row_cap: int = 6,
) -> np.ndarray:
    nq, nk = scores.shape
    mask = anchor_mask(nq, nk)
    counts = np.ones(nq, dtype=int)
    total = nq
    while total < budget:
        next_gains = np.full(nq, -np.inf, dtype=np.float32)
        for i in range(nq):
            if counts[i] < row_cap:
                next_gains[i] = marginal_gains[i, counts[i] - 1]
        row = int(np.argmax(next_gains))
        if not np.isfinite(next_gains[row]):
            raise AssertionError("marginal allocator exhausted candidates")
        mask[row, prefixes[row][counts[row]]] = True
        counts[row] += 1
        total += 1
    return mask


def evaluate_marginal_allocator(
    cases: list[Case], rank_model, demand_model, budgets: list[int], row_cap: int = 6
) -> dict:
    buckets = {
        name: [[] for _ in budgets]
        for name in ("fixed", "naive_global", "learned_marginal", "privileged_marginal")
    }
    gain_pred, gain_true = [], []
    for case in cases:
        scores = model_scores(rank_model, case, "trip")
        prefixes, errors = prefix_errors(case, scores, row_cap)
        true_gains = errors[:, :-1] - errors[:, 1:]
        pred_gains = demand_model.predict(demand_features(case, scores, row_cap)).astype(np.float32)
        gain_pred.extend(pred_gains.reshape(-1))
        gain_true.extend(true_gains.reshape(-1))
        for bidx, budget in enumerate(budgets):
            masks = {
                "fixed": allocate(scores, budget, adaptive=False, row_cap=row_cap),
                "naive_global": allocate(scores, budget, adaptive=True, row_cap=row_cap),
                "learned_marginal": allocate_by_marginal(
                    scores, prefixes, pred_gains, budget, row_cap
                ),
                "privileged_marginal": allocate_by_marginal(
                    scores, prefixes, true_gains, budget, row_cap
                ),
            }
            for name, mask in masks.items():
                buckets[name][bidx].extend(row_errors(case, mask))
    quality = {
        name: [quality_from_errors(np.asarray(v)) for v in per_budget]
        for name, per_budget in buckets.items()
    }
    return {
        "quality_by_budget": quality,
        "learned_minus_fixed": (
            np.asarray(quality["learned_marginal"]) - np.asarray(quality["fixed"])
        ).tolist(),
        "privileged_minus_fixed": (
            np.asarray(quality["privileged_marginal"]) - np.asarray(quality["fixed"])
        ).tolist(),
        "marginal_gain_prediction_spearman": float(spearmanr(gain_pred, gain_true).statistic),
    }


def retained_teacher_probability(case: Case, mask: np.ndarray) -> float:
    return float(np.mean(np.sum(case.teacher * mask, axis=1)))


def proxy_correlation(cases: list[Case], budgets: list[int], seed: int) -> dict:
    qualities, probabilities, recalls = [], [], []
    candidate_case_scores = []
    candidate_params = []
    budget = budgets[1]
    for idx in range(36):
        noise = 0.05 + 0.085 * idx
        bias_scale = 0.04 * (idx % 5)
        q_values, m_values, r_values = [], [], []
        for cidx, case in enumerate(cases):
            local_rng = np.random.default_rng(seed + 1000 * idx + cidx)
            scores = np.log(case.teacher + 1e-6) + noise * local_rng.normal(
                size=case.teacher.shape
            )
            scores += bias_scale * local_rng.normal(size=(case.mass.shape[0], 1))
            mask = allocate(scores, budget, adaptive=False, force_anchor=False)
            errs = row_errors(case, mask)
            q_values.append(quality_from_errors(errs))
            m_values.append(retained_teacher_probability(case, mask))
            oracle = allocate(case.teacher, budget, adaptive=False, force_anchor=False)
            r_values.append(float(np.logical_and(mask, oracle).sum() / budget))
        qualities.append(float(np.mean(q_values)))
        probabilities.append(float(np.mean(m_values)))
        recalls.append(float(np.mean(r_values)))
        candidate_case_scores.append(q_values)
        candidate_params.append(noise)
    return {
        "teacher_probability_vs_exact_quality_spearman": float(
            spearmanr(probabilities, qualities).statistic
        ),
        "recall_vs_exact_quality_spearman": float(spearmanr(recalls, qualities).statistic),
        "candidate_quality_range": [float(min(qualities)), float(max(qualities))],
        "candidate_case_scores": np.asarray(candidate_case_scores),
        "candidate_noise": candidate_params,
    }


def exact_oracle_experiment(seed: int, ncases: int = 120) -> dict:
    rng = np.random.default_rng(seed)
    kkeep = 2
    exact_errors, mass_errors, teacher_errors = [], [], []
    mass_exact, teacher_exact = 0, 0
    rows = 0
    for _ in range(ncases):
        case = make_case(rng, nq=3, nk=6, tile=3, d=6)
        for i in range(case.q.shape[0]):
            choices = list(itertools.combinations(range(case.k.shape[0]), kkeep))
            errors = []
            for choice in choices:
                out = sparse_row(case.q[i], case.k, case.v, np.asarray(choice))
                num = float(np.sum((out - case.dense[i]) ** 2))
                den = float(np.sum(case.dense[i] ** 2)) + EPS
                errors.append(num / den)
            best_idx = int(np.argmin(errors))
            best = set(choices[best_idx])
            mass = set(np.argsort(-case.mass[i], kind="stable")[:kkeep].tolist())
            teacher = set(np.argsort(-case.teacher[i], kind="stable")[:kkeep].tolist())
            exact_errors.append(errors[best_idx])
            mass_errors.append(errors[choices.index(tuple(sorted(mass)))])
            teacher_errors.append(errors[choices.index(tuple(sorted(teacher)))])
            mass_exact += int(mass == best)
            teacher_exact += int(teacher == best)
            rows += 1
    exact_mean = float(np.mean(exact_errors))
    return {
        "rows": rows,
        "exact_mean_relative_squared_error": exact_mean,
        "mass_topk_mean_relative_squared_error": float(np.mean(mass_errors)),
        "teacher_topk_mean_relative_squared_error": float(np.mean(teacher_errors)),
        "mass_exact_mask_rate": mass_exact / rows,
        "teacher_exact_mask_rate": teacher_exact / rows,
        "mass_excess_error_vs_exact": float(
            (np.mean(mass_errors) - exact_mean) / (exact_mean + EPS)
        ),
        "teacher_excess_error_vs_exact": float(
            (np.mean(teacher_errors) - exact_mean) / (exact_mean + EPS)
        ),
    }


def cache_precision_experiment(cases: list[Case], model, kind: str, budget: int) -> dict:
    score_rhos, mask_agree, dense_err, sparse_err = [], [], [], []
    for case in cases:
        s32 = model_scores(model, case, kind, quantized=False)
        s16 = model_scores(model, case, kind, quantized=True)
        score_rhos.append(float(spearmanr(s32.reshape(-1), s16.reshape(-1)).statistic))
        m32 = allocate(s32, budget, adaptive=False, force_anchor=False)
        m16 = allocate(s16, budget, adaptive=False, force_anchor=False)
        mask_agree.append(float(np.mean(m32 == m16)))

        q16 = case.q.astype(np.float16).astype(np.float32)
        k16 = case.k.astype(np.float16).astype(np.float32)
        v16 = case.v.astype(np.float16).astype(np.float32)
        d16, _, _ = dense_attention(q16, k16, v16)
        dense_err.append(float(np.sum((d16 - case.dense) ** 2) / (np.sum(case.dense**2) + EPS)))
        case16 = Case(
            q16,
            k16,
            v16,
            d16,
            case.mass,
            case.teacher,
            case.avg_q,
            case.avg_k,
            case.trip_q,
            case.trip_k,
        )
        sp32 = sparse_attention(case, m32)
        sp16 = sparse_attention(case16, m32)
        sparse_err.append(float(np.sum((sp16 - sp32) ** 2) / (np.sum(sp32**2) + EPS)))
    return {
        "descriptor_score_spearman_mean": float(np.mean(score_rhos)),
        "mask_entry_agreement_mean": float(np.mean(mask_agree)),
        "dense_output_relative_squared_error_mean": float(np.mean(dense_err)),
        "sparse_output_relative_squared_error_mean": float(np.mean(sparse_err)),
    }


def bootstrap_stability(case_scores: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    full = case_scores.mean(axis=1)
    true_top3 = set(np.argsort(-full)[:3].tolist())
    result = {}
    for n in [20, 40, 80, case_scores.shape[1]]:
        hit, rhos = [], []
        for _ in range(200):
            idx = rng.integers(0, case_scores.shape[1], size=n)
            sampled = case_scores[:, idx].mean(axis=1)
            top5 = set(np.argsort(-sampled)[:5].tolist())
            hit.append(len(true_top3 & top5) / 3)
            rhos.append(float(spearmanr(full, sampled).statistic))
        result[str(n)] = {
            "mean_top3_in_top5_recall": float(np.mean(hit)),
            "mean_rank_spearman": float(np.mean(rhos)),
            "p05_rank_spearman": float(np.percentile(rhos, 5)),
        }
    return result


def repeated_submission_lottery(case_scores: np.ndarray, seed: int) -> dict:
    """Quantify best-of-many bias when every submission receives a new panel."""

    rng = np.random.default_rng(seed)
    candidate = case_scores[np.argmax(np.var(case_scores, axis=1))]
    panel_size = 20
    trials = 1000
    result = {"full_score": float(np.mean(candidate)), "panel_size": panel_size}
    for attempts in (1, 5, 20, 100):
        best_scores = []
        for _ in range(trials):
            scores = [
                float(np.mean(candidate[rng.choice(len(candidate), panel_size, replace=False)]))
                for _ in range(attempts)
            ]
            best_scores.append(max(scores))
        result[str(attempts)] = {
            "mean_best_score": float(np.mean(best_scores)),
            "mean_optimism": float(np.mean(best_scores) - np.mean(candidate)),
            "p95_best_score": float(np.percentile(best_scores, 95)),
        }
    return result


def clone_attack_experiment(seed: int) -> dict:
    """Show why exact weight hashes and score-proportional rewards are insufficient."""

    rng = np.random.default_rng(seed)
    descriptor_dim, hidden_dim, projection_dim = 6, 8, 4
    descriptors = rng.normal(size=(12, descriptor_dim)).astype(np.float32)
    hidden_weight = rng.normal(size=(hidden_dim, descriptor_dim)).astype(np.float32)
    hidden_bias = rng.normal(size=hidden_dim).astype(np.float32)
    output_weight = rng.normal(size=(projection_dim, hidden_dim)).astype(np.float32)
    output_bias = rng.normal(size=projection_dim).astype(np.float32)
    permutation = rng.permutation(hidden_dim)

    def project(weight1, bias1, weight2):
        hidden = np.maximum(descriptors @ weight1.T + bias1, 0)
        return hidden @ weight2.T + output_bias

    original = project(hidden_weight, hidden_bias, output_weight)
    permuted = project(
        hidden_weight[permutation],
        hidden_bias[permutation],
        output_weight[:, permutation],
    )
    original_bytes = b"".join(
        array.tobytes() for array in (hidden_weight, hidden_bias, output_weight, output_bias)
    )
    permuted_bytes = b"".join(
        array.tobytes()
        for array in (
            hidden_weight[permutation],
            hidden_bias[permutation],
            output_weight[:, permutation],
            output_bias,
        )
    )
    score = 0.8
    clone_counts = (1, 5, 20)
    return {
        "different_weight_hashes": hashlib.sha256(original_bytes).digest()
        != hashlib.sha256(permuted_bytes).digest(),
        "max_projection_difference": float(np.max(np.abs(original - permuted))),
        "top2_agreement": float(
            np.mean(
                np.argsort(-original, axis=1, kind="stable")[:, :2]
                == np.argsort(-permuted, axis=1, kind="stable")[:, :2]
            )
        ),
        "clone_share_under_score_proportional_rewards": {
            str(count): count * score / ((count + 1) * score) for count in clone_counts
        },
        "clone_frontier_gain_after_original": 0.0,
    }


def teacher_proxy_counterexample() -> dict:
    """Construct a case where retained teacher probability reverses output quality."""

    teacher = np.asarray([0.45, 0.45, 0.10], dtype=np.float64)
    values = np.asarray([-1.0, 1.0, 0.0], dtype=np.float64)
    dense = float(teacher @ values)
    high_mass_output = float(values[0])
    low_mass_output = float(values[2])
    return {
        "dense_output": dense,
        "high_mass_choice": {
            "retained_teacher_probability": float(teacher[0]),
            "squared_output_error": (high_mass_output - dense) ** 2,
        },
        "low_mass_choice": {
            "retained_teacher_probability": float(teacher[2]),
            "squared_output_error": (low_mass_output - dense) ** 2,
        },
        "proxy_order_reversed": bool(
            teacher[0] > teacher[2]
            and (high_mass_output - dense) ** 2 > (low_mass_output - dense) ** 2
        ),
    }


def budget_grid_stability(cases: list[Case], seed: int) -> dict:
    grids = ([8, 24, 40], [8, 16, 24, 32, 40])
    all_scores = []
    for grid in grids:
        candidates = []
        for idx in range(24):
            noise = 0.08 + 0.12 * idx
            values = []
            for cidx, case in enumerate(cases):
                local_rng = np.random.default_rng(seed + 1000 * idx + cidx)
                score = np.log(case.teacher + 1e-6) + noise * local_rng.normal(
                    size=case.teacher.shape
                )
                q = []
                for b in grid:
                    q.append(
                        quality_from_errors(
                            row_errors(
                                case,
                                allocate(score, b, adaptive=False, force_anchor=False),
                            )
                        )
                    )
                values.append(log_cost_auc(q, grid))
            candidates.append(float(np.mean(values)))
        all_scores.append(candidates)
    return {
        "rank_spearman_between_budget_grids": float(
            spearmanr(all_scores[0], all_scores[1]).statistic
        )
    }


def allocator_safety_test() -> dict:
    scores = np.ones((8, 12), dtype=np.float32)
    masks = [allocate(scores, b, adaptive=True) for b in (8, 16, 24, 32)]
    deterministic = all(
        np.array_equal(a, allocate(scores, int(a.sum()), adaptive=True)) for a in masks
    )
    exact_budget = all(int(m.sum()) == b for m, b in zip(masks, (8, 16, 24, 32), strict=True))
    nonempty = all(np.all(m.sum(axis=1) >= 1) for m in masks)
    return {
        "deterministic_ties": deterministic,
        "exact_budget": exact_budget,
        "all_queries_nonempty": nonempty,
    }


def anchor_ablation(cases: list[Case], model, budgets: list[int]) -> dict:
    result = {"forced_anchor": [], "unconstrained_topk": []}
    for force_anchor, key in ((True, "forced_anchor"), (False, "unconstrained_topk")):
        errors = [[] for _ in budgets]
        for case in cases:
            scores = model_scores(model, case, "trip")
            for idx, budget in enumerate(budgets):
                mask = allocate(scores, budget, adaptive=False, force_anchor=force_anchor)
                errors[idx].extend(row_errors(case, mask))
        result[key] = [quality_from_errors(np.asarray(values)) for values in errors]
    result["forced_minus_unconstrained"] = (
        np.asarray(result["forced_anchor"]) - np.asarray(result["unconstrained_topk"])
    ).tolist()
    return result


def seed_sensitivity_experiment() -> dict:
    result = {}
    budgets = [16, 24, 32]
    for seed in (17, 29, 43):
        rng = np.random.default_rng(seed)
        train_cases = [make_case(rng) for _ in range(120)]
        test_cases = [make_case(rng) for _ in range(60)]
        avg = train_model(train_cases, "avg", (24,), seed + 100)
        trip = train_model(train_cases, "trip", (24,), seed + 200)
        avg_scorer = partial(model_scores, avg, kind="avg")
        trip_scorer = partial(model_scores, trip, kind="trip")
        avg_q, _ = evaluate_scores(test_cases, avg_scorer, budgets, adaptive=False)
        trip_fixed, _ = evaluate_scores(test_cases, trip_scorer, budgets, adaptive=False)
        trip_adaptive, _ = evaluate_scores(test_cases, trip_scorer, budgets, adaptive=True)
        exact = exact_oracle_experiment(seed + 300, ncases=40)
        result[str(seed)] = {
            "avg_log_cost_auc": log_cost_auc(avg_q, budgets),
            "trip_log_cost_auc": log_cost_auc(trip_fixed, budgets),
            "trip_minus_avg": log_cost_auc(trip_fixed, budgets) - log_cost_auc(avg_q, budgets),
            "naive_adaptive_minus_fixed": (trip_adaptive - trip_fixed).tolist(),
            "mass_excess_error_vs_exact": exact["mass_excess_error_vs_exact"],
            "teacher_excess_error_vs_exact": exact["teacher_excess_error_vs_exact"],
        }
    return result


def main() -> None:
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    train_cases = [make_case(rng) for _ in range(220)]
    test_cases = [make_case(rng) for _ in range(120)]
    budgets = [16, 24, 32]

    t0 = time.perf_counter()
    avg_model = train_model(train_cases, "avg", (24,), SEED + 1)
    trip_model = train_model(train_cases, "trip", (24,), SEED + 2)
    large_model = train_model(train_cases, "trip", (96, 48), SEED + 3)
    demand_model = train_demand_model(train_cases, trip_model, SEED + 4)
    training_seconds = time.perf_counter() - t0

    models = {
        "avg_small": (avg_model, "avg"),
        "trip_small": (trip_model, "trip"),
        "trip_large_analysis": (large_model, "trip"),
    }
    model_results = {}
    for name, (model, kind) in models.items():
        scorer = partial(model_scores, model, kind=kind)
        fixed_q, _ = evaluate_scores(test_cases, scorer, budgets, adaptive=False)
        adaptive_q, _ = evaluate_scores(test_cases, scorer, budgets, adaptive=True)
        # Prediction correlation is diagnostic only.
        pred, target = [], []
        for case in test_cases:
            pred.extend(scorer(case).reshape(-1))
            target.extend(np.log(case.teacher.reshape(-1) + 1e-6))
        model_results[name] = {
            "teacher_logit_spearman": float(spearmanr(pred, target).statistic),
            "fixed_quality_by_budget": fixed_q.tolist(),
            "adaptive_quality_by_budget": adaptive_q.tolist(),
            "adaptive_minus_fixed": (adaptive_q - fixed_q).tolist(),
            "fixed_log_cost_auc": log_cost_auc(fixed_q, budgets),
            "adaptive_log_cost_auc": log_cost_auc(adaptive_q, budgets),
        }

    def oracle_scorer(case: Case) -> np.ndarray:
        return np.log(case.teacher + 1e-6)

    oracle_fixed, _ = evaluate_scores(test_cases, oracle_scorer, budgets, adaptive=False)
    oracle_adaptive, _ = evaluate_scores(test_cases, oracle_scorer, budgets, adaptive=True)

    proxy = proxy_correlation(test_cases, budgets, SEED + 10)
    candidate_case_scores = proxy.pop("candidate_case_scores")
    stability = bootstrap_stability(candidate_case_scores, SEED + 11)

    results = {
        "scope": {
            "validated": [
                "deterministic budget allocator",
                "multiple-budget scoring mechanics",
                "max-pooled teacher proxy correlation on synthetic attention",
                "TripPool information advantage on synthetic peaked attention",
                "float16 cache fidelity on small attention tensors",
                "exact versus heuristic oracle gap on tiny problems",
                "adaptive versus fixed allocation on equal block budgets",
                "sample-size and ranking stability on synthetic candidates",
                "repeated-submission selection bias",
                "exact-hash clone bypass through hidden-unit permutation",
                "max-pooled teacher screening counterexample",
            ],
            "not_validated": [
                "GPU block count versus latency",
                "selector overhead on a target GPU",
                "proxy score versus real video quality",
                "selector-only quality without adapters in a real Video DiT",
                "public Veda checkpoint compatibility",
            ],
        },
        "dimensions": {
            "train_cases": 220,
            "test_cases": 120,
            "nq": 8,
            "nk": 12,
            "tile": 4,
            "d": 8,
            "budgets": budgets,
        },
        "allocator_safety": allocator_safety_test(),
        "anchor_ablation": anchor_ablation(test_cases, trip_model, budgets),
        "models": model_results,
        "oracle_teacher": {
            "fixed_quality_by_budget": oracle_fixed.tolist(),
            "adaptive_quality_by_budget": oracle_adaptive.tolist(),
            "adaptive_minus_fixed": (oracle_adaptive - oracle_fixed).tolist(),
        },
        "adaptive_marginal": evaluate_marginal_allocator(
            test_cases, trip_model, demand_model, budgets
        ),
        "proxy": proxy,
        "exact_oracle": exact_oracle_experiment(SEED + 20),
        "cache_float16": cache_precision_experiment(test_cases, trip_model, "trip", budgets[1]),
        "bootstrap_stability": stability,
        "repeated_submission_lottery": repeated_submission_lottery(
            candidate_case_scores, SEED + 12
        ),
        "clone_attack": clone_attack_experiment(SEED + 13),
        "teacher_proxy_counterexample": teacher_proxy_counterexample(),
        "budget_grid_stability": budget_grid_stability(test_cases[:60], SEED + 30),
        "seed_sensitivity": seed_sensitivity_experiment(),
        "timing": {
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - start,
        },
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
