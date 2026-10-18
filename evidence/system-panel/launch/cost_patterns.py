"""Deterministic legal allocation probes; production selector still enforces all rules.

The real predictor executes before these precomputed logits are supplied to the
production stable selector. This measures complete dispatch cost under adversarial
legal layouts, not trained models or their quality.
"""

from __future__ import annotations

import hashlib

import numpy as np

PATTERNS = ("clustered", "spread", "boundary_padding", "repeated_keys")


def scores(pattern, valid_count, rows, budget, heads):
    if pattern not in PATTERNS or budget < 1:
        raise ValueError("unknown pattern or invalid budget")
    valid_count = np.asarray(valid_count)
    valid = np.flatnonzero(valid_count > 0)
    if not len(valid):
        raise ValueError("pattern requires nonempty video keys")
    values = np.full((heads, len(rows), len(valid_count)), -1e9, dtype=np.float32)
    ranks = np.arange(len(valid), 0, -1, dtype=np.float32)
    if pattern == "clustered":
        distance = np.abs(valid[None, :] - np.asarray(rows)[:, None])
        order = valid[np.argsort(distance, axis=-1, kind="stable")][None, :, :]
    elif pattern == "spread":
        count = min(budget, len(valid))
        positions = (
            np.arange(count)[None, None, :] * len(valid) // count
            + np.arange(len(rows))[None, :, None] + np.arange(heads)[:, None, None]
        ) % len(valid)
        chosen = np.zeros((heads, len(rows), len(valid)), dtype=np.bool_)
        np.put_along_axis(chosen, positions, True, axis=-1)
        remaining = np.argsort(chosen, axis=-1, kind="stable")[..., :len(valid) - count]
        order = np.concatenate((valid[positions], valid[remaining]), axis=-1)
    elif pattern == "boundary_padding":
        order = np.asarray(sorted(
            valid,
            key=lambda col: (
                int(valid_count[col]), min(int(col), len(valid_count) - 1 - int(col)), int(col),
            ),
        ))[None, None, :]
    else:
        order = valid[None, None, :]
    np.put_along_axis(values, order, ranks, axis=-1)
    return values


def _full_scores(pattern, counts, rows, budget, heads, video_tiles):
    values = scores(pattern, counts[:video_tiles], rows, budget, heads)
    globals_ = np.zeros((heads, len(rows), len(counts) - video_tiles), dtype=np.float32)
    return np.concatenate((values, globals_), axis=-1)


def _expected_mask(values, counts, rows, budget, video_tiles, include_global_rows):
    """Independent stable NumPy allocation, including empty and global tiles."""
    if not 0 < budget <= np.count_nonzero(counts[:video_tiles]):
        raise ValueError("pattern budget cannot be satisfied")
    ranked = values[..., :video_tiles].copy()
    ranked[..., counts[:video_tiles] == 0] = -np.inf
    own = counts[rows] > 0
    ranked[:, np.arange(len(rows)), rows] = np.where(
        own[None, :], np.inf, ranked[:, np.arange(len(rows)), rows],
    )
    order = np.argsort(-ranked, axis=-1, kind="stable")[..., :budget]
    result = np.zeros(values.shape, dtype=np.bool_)
    np.put_along_axis(result[..., :video_tiles], order, True, axis=-1)
    result[..., video_tiles:] = counts[video_tiles:] > 0
    if include_global_rows:
        globals_ = np.broadcast_to(
            counts > 0, (values.shape[0], len(counts) - video_tiles, len(counts)),
        )
        result = np.concatenate((result, globals_), axis=-2)
    return result


def selection_proof(mask, valid_count, rows, budget, video_tiles, include_global_rows):
    mask, counts, rows = np.asarray(mask), np.asarray(valid_count), np.asarray(rows)
    expected_rows = len(rows) + (len(counts) - video_tiles if include_global_rows else 0)
    if (
        mask.ndim != 3 or mask.dtype != np.bool_
        or mask.shape[-2:] != (expected_rows, len(counts))
        or np.any(rows < 0) or np.any(rows >= video_tiles)
    ):
        raise ValueError("selection shape or query rows mismatch")
    if np.any(mask[..., counts == 0]):
        raise ValueError("pattern selects empty keys")
    video_mask = mask[:, :len(rows)]
    if np.any(video_mask[..., :video_tiles].sum(-1) != budget):
        raise ValueError("pattern violates fixed budget")
    own = counts[rows] > 0
    if not video_mask[:, np.arange(len(rows))[own], rows[own]].all():
        raise ValueError("pattern omits compulsory own key")
    if not np.array_equal(
        video_mask[..., video_tiles:],
        np.broadcast_to(counts[video_tiles:] > 0, video_mask[..., video_tiles:].shape),
    ):
        raise ValueError("pattern omits global keys")
    if include_global_rows and not np.array_equal(
        mask[:, len(rows):], np.broadcast_to(counts > 0, mask[:, len(rows):].shape),
    ):
        raise ValueError("global queries must use every valid key")
    return {
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "shape": list(mask.shape), "valid_count": counts.tolist(), "rows": rows.tolist(),
        "budget": budget, "n_video_tiles": video_tiles,
        "include_global_rows": include_global_rows,
    }


def verify_proof(proof):
    counts, rows = np.asarray(proof["valid_count"]), np.asarray(proof["rows"])
    if hashlib.sha256(rows.tobytes()).hexdigest() != proof["rows_sha256"]:
        raise ValueError("pattern query row identity mismatch")
    values = _full_scores(
        proof["pattern"], counts, rows, proof["budget"], proof["shape"][0],
        proof["n_video_tiles"],
    )
    if hashlib.sha256(values.tobytes()).hexdigest() != proof["logits_sha256"]:
        raise ValueError("pattern logits hash mismatch")
    mask = _expected_mask(
        values, counts, rows, proof["budget"], proof["n_video_tiles"],
        proof["include_global_rows"],
    )
    actual = selection_proof(
        mask, counts, rows, proof["budget"], proof["n_video_tiles"],
        proof["include_global_rows"],
    )
    if any(proof[field] != value for field, value in actual.items()):
        raise ValueError("production pattern differs from deterministic legal allocation")


class SelectorPattern:
    """Inject primed logits before the real production mask; no timed host copies."""

    def __init__(self, torch, selector, name):
        self.torch, self.selector, self.name = torch, selector, name
        self.entries, self.proofs = [], []
        self.mode, self.cursor, self.observed = "unprimed", 0, []

    def prime(self, operation):
        if self.mode != "unprimed":
            raise RuntimeError("pattern sequence is already primed")
        self.mode = "prime"
        operation()
        if not self.entries:
            raise RuntimeError("dry operation did not select any masks")
        self.mode = "ready"

    def replay(self, operation):
        def call():
            if self.mode != "ready":
                raise RuntimeError("pattern sequence must be primed and verified")
            self.cursor, self.observed, self.mode = 0, [], "replay"
            result = operation()
            if self.cursor != len(self.entries):
                raise RuntimeError("timed operation omitted primed selector calls")
            self.mode = "pending_verification"
            return result
        return call

    def verify_replay(self):
        """Called after the latency timestamp, never from the timed operation."""
        if self.mode != "pending_verification":
            raise RuntimeError("no complete pattern operation to verify")
        for actual, entry in zip(self.observed, self.entries, strict=True):
            if actual is None and entry["implicit_full_rows"]:
                continue
            if actual is None or not self.torch.equal(actual, entry["rows"]):
                raise RuntimeError("timed operation changed primed query-row sequence")
        self.mode = "ready"

    def __call__(
        self, torch, logits, layout, *, target_blocks, reference_blocks, query_rows=None,
    ):
        if torch is not self.torch:
            raise ValueError("pattern torch module differs from production")
        if reference_blocks != 0 or layout.n_ref_tiles != 0:
            raise ValueError("pattern probe requires production target-only fixed Top-k")
        identity = (id(layout), tuple(logits.shape), target_blocks,
                    None if query_rows is None else tuple(query_rows.shape))
        if self.mode == "prime":
            rows = (
                self.torch.arange(layout.n_video_tiles, device=logits.device)
                if query_rows is None else query_rows
            )
            counts, query_ids = layout.valid_count.cpu().numpy(), rows.cpu().numpy()
            values = _full_scores(
                self.name, counts, query_ids, target_blocks, logits.shape[0], layout.n_video_tiles,
            )
            gpu_values = self.torch.from_numpy(values).to(logits.device)
            selected = self.selector(
                self.torch, gpu_values, layout, target_blocks=target_blocks,
                reference_blocks=reference_blocks, query_rows=query_rows,
            )
            proof = selection_proof(
                selected.cpu().numpy(), counts, query_ids, target_blocks,
                layout.n_video_tiles, query_rows is None,
            )
            proof.update(
                logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                pattern=self.name, rows_sha256=hashlib.sha256(query_ids.tobytes()).hexdigest(),
                sequence_index=len(self.entries),
            )
            self.entries.append({
                "identity": identity, "implicit_full_rows": query_rows is None,
                "logits": gpu_values, "rows": rows.clone(),
            })
            self.proofs.append(proof)
            return selected
        if self.mode != "replay" or self.cursor >= len(self.entries):
            raise RuntimeError("unprimed selector call during timing")
        entry = self.entries[self.cursor]
        if identity != entry["identity"]:
            raise RuntimeError("timed selector signature differs from primed sequence")
        self.cursor += 1
        self.observed.append(query_rows)
        return self.selector(
            self.torch, entry["logits"], layout, target_blocks=target_blocks,
            reference_blocks=reference_blocks, query_rows=query_rows,
        )


def production_self_test(device="cpu"):
    """Actual shared mask, including empty/global tiles and distinct equal-shaped rows."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    from scoring.torch_selector import runtime_block_mask

    counts = torch.tensor([128] * 128 + [4, 8, 0, 2, 0], dtype=torch.long, device=device)
    layout = SimpleNamespace(n_tiles=len(counts), n_video_tiles=131, n_ref_tiles=0,
                             valid_count=counts)
    results = []
    for name in PATTERNS:
        for budget in (4, 16, 32, 64, 130):
            for implicit in (False, True):
                pattern = SelectorPattern(torch, runtime_block_mask, name)
                chunks = (
                    [None] if implicit else [
                        torch.tensor([0, 1, 2], device=device),
                        torch.tensor([64, 65, 66], device=device),
                    ]
                )
                logits = torch.zeros(2, 131 if implicit else 3, len(counts), device=device)

                def operation(pattern=pattern, chunks=chunks, logits=logits, budget=budget):
                    return [
                        pattern(torch, logits, layout, target_blocks=budget, reference_blocks=0,
                                query_rows=rows) for rows in chunks
                    ]

                pattern.prime(operation)
                for proof in pattern.proofs:
                    verify_proof(proof)
                    results.append(proof)
                if not implicit and pattern.proofs[0]["rows_sha256"] == (
                    pattern.proofs[1]["rows_sha256"]
                ):
                    raise AssertionError("equal-shaped different rows shared primed logits")
                replay = pattern.replay(operation)
                with (
                    patch.object(torch.Tensor, "cpu",
                                 side_effect=AssertionError("timed host transfer")),
                    patch.object(torch.Tensor, "numpy",
                                 side_effect=AssertionError("timed host conversion")),
                ):
                    for _ in range(3):
                        replay()
                        pattern.verify_replay()
                    if not implicit:
                        chunks.reverse()
                        replay()
                        try:
                            pattern.verify_replay()
                        except RuntimeError as exc:
                            if "query-row sequence" not in str(exc):
                                raise
                        else:
                            raise AssertionError("different-row replay order was accepted")
    return {
        "format": "veda-cost-production-pattern-regression", "pass": True,
        "torch": torch.__version__, "device": device, "cases": results,
        "host_conversion_disabled_during_replay": True,
        "reversed_equal_shape_sequence_rejected": True,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = production_self_test(args.device)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"pass": result["pass"], "cases": len(result["cases"])}))
