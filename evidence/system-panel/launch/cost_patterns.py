"""Deterministic legal allocation probes; production selector still enforces all rules.

The real predictor executes before these precomputed logits are supplied to the
native stable selector. This measures complete dispatch cost under adversarial
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


def selection_proof(index, keep, valid_count, rows, budget):
    index, keep = np.asarray(index), np.asarray(keep)
    valid_count, rows = np.asarray(valid_count), np.asarray(rows)
    if index.shape != keep.shape or index.shape[1] != len(rows):
        raise ValueError("selection shape mismatch")
    if np.any(index < 0) or np.any(index >= len(valid_count)):
        raise ValueError("out of bounds key")
    expected = min(budget, np.count_nonzero(valid_count))
    counts = keep.sum(-1)
    if np.any(counts != expected) or np.any((valid_count[index] == 0) & keep):
        raise ValueError("pattern violates fixed budget or padding rule")
    for head in range(index.shape[0]):
        for ri, query in enumerate(rows):
            selected = index[head, ri][keep[head, ri]]
            if len(np.unique(selected)) != expected:
                raise ValueError("duplicate selected key")
            if valid_count[query] and query not in selected:
                raise ValueError("pattern omits compulsory own key")
    return {
        "index_sha256": hashlib.sha256(index.tobytes()).hexdigest(),
        "keep_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
        "shape": list(index.shape),
        "valid_count": valid_count.tolist(),
        "rows": rows.tolist(),
        "budget": budget,
        "minimum_kept": int(counts.min()),
        "maximum_kept": int(counts.max()),
        "padding_violations": 0,
        "anchor_violations": 0,
        "duplicate_violations": 0,
    }


def verify_proof(proof):
    counts, rows = np.asarray(proof["valid_count"]), np.asarray(proof["rows"])
    if hashlib.sha256(rows.tobytes()).hexdigest() != proof["rows_sha256"]:
        raise ValueError("pattern query row identity mismatch")
    heads = proof["shape"][0]
    values = scores(proof["pattern"], counts, rows, proof["budget"], heads)
    if hashlib.sha256(values.tobytes()).hexdigest() != proof["logits_sha256"]:
        raise ValueError("pattern logits hash mismatch")
    values[:, :, counts == 0] = -np.inf
    values[:, np.arange(len(rows)), rows] = np.inf
    # Pinned installed split_budget(16, 300) returns (16, 17, 0):
    # index has 17 columns, with exactly 16 kept. Verify against the
    # real installed selector in native_self_test, including saturated budgets.
    width = min(proof["budget"] + 1, len(counts))
    index = np.argsort(-values, axis=-1, kind="stable")[..., :width].astype(np.int64)
    keep = np.broadcast_to(np.arange(width) < proof["budget"], index.shape).copy()
    keep &= np.take_along_axis(values, index, axis=-1) > -np.inf
    actual = selection_proof(index, keep, counts, rows, proof["budget"])
    if any(proof[field] != value for field, value in actual.items()):
        raise ValueError("native pattern differs from deterministic legal allocation")


class NativePattern:
    def __init__(self, torch, native, name):
        self.torch, self.native, self.name = torch, native, name
        self.entries = []
        self.proofs = []
        self.mode = "unprimed"
        self.cursor = 0
        self.observed = []

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
            self.cursor = 0
            self.observed = []
            self.mode = "replay"
            result = operation()
            if self.cursor != len(self.entries):
                raise RuntimeError("timed operation omitted primed selector calls")
            self.mode = "pending_verification"
            return result

        return call

    def verify_replay(self):
        """Called AFTER the latency timestamp, never from the timed operation."""
        if self.mode != "pending_verification":
            raise RuntimeError("no complete pattern operation to verify")
        for actual, entry in zip(self.observed, self.entries, strict=True):
            if actual is None and entry["implicit_full_rows"]:
                continue
            if actual is None or not self.torch.equal(actual, entry["rows"]):
                raise RuntimeError("timed operation changed primed query-row sequence")
        self.mode = "ready"

    def __call__(self, logits, layout, blocks, rows=None, **kwargs):
        native_rows = rows
        implicit_full_rows = rows is None
        if len(blocks) != 1 or blocks[0].start != 0 or kwargs.get("rule", "topk") != "topk":
            raise ValueError("pattern probe requires production target-only fixed Top-k")
        budget = blocks[0].budget.tiles
        if budget is None or budget != int(budget):
            raise ValueError("pattern probe requires integer block budget")
        identity = (
            id(layout),
            tuple(logits.shape),
            int(budget),
            None if implicit_full_rows else tuple(rows.shape),
        )
        if self.mode == "prime":
            if implicit_full_rows:
                rows = self.torch.arange(layout.n_video_tiles, device=logits.device)
            counts = layout.valid_count[: layout.n_video_tiles].cpu().numpy()
            query_rows = rows.detach().cpu().numpy()
            rows_digest = hashlib.sha256(query_rows.tobytes()).hexdigest()
            values = scores(self.name, counts, query_rows, int(budget), logits.shape[0])
            gpu_values = self.torch.from_numpy(values).to(logits.device)
            selection = self.native(gpu_values, layout, blocks, native_rows, **kwargs)
            proof = selection_proof(
                selection.index.cpu().numpy(),
                selection.keep.cpu().numpy(),
                counts,
                query_rows,
                int(budget),
            )
            proof.update(
                logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                pattern=self.name,
                rows_sha256=rows_digest,
                sequence_index=len(self.entries),
                implicit_full_rows=implicit_full_rows,
            )
            self.entries.append(
                {
                    "identity": identity,
                    "implicit_full_rows": implicit_full_rows,
                    "logits": gpu_values,
                    "rows": rows.clone(),
                    "rows_sha256": rows_digest,
                }
            )
            self.proofs.append(proof)
            return selection
        if self.mode != "replay" or self.cursor >= len(self.entries):
            raise RuntimeError("unprimed selector call during timing")
        entry = self.entries[self.cursor]
        if identity != entry["identity"]:
            raise RuntimeError("timed selector signature differs from primed sequence")
        self.cursor += 1
        self.observed.append(native_rows)
        return self.native(entry["logits"], layout, blocks, native_rows, **kwargs)


def native_self_test(device="cpu"):
    """Real installed selector, including equal-shaped distinct query chunks."""
    from types import SimpleNamespace

    import torch
    from miowtion.veda import mask

    counts = torch.tensor([128] * 128 + [4, 8, 0], dtype=torch.int64, device=device)
    layout = SimpleNamespace(n_video_tiles=len(counts), valid_count=counts, kv_ok=counts > 0)
    from unittest.mock import patch

    results = []
    for name in PATTERNS:
        for budget in (4, 16, 32, 64, 131):
            pattern = NativePattern(torch, mask.select_video_blocks, name)
            blocks = [
                mask.ColumnBlock(0, len(counts), mask.Budget(tiles=budget), int(counts.sum()))
            ]
            chunks = [
                torch.tensor([0, 1, 2], device=device),
                torch.tensor([64, 65, 66], device=device),
            ]
            logits = torch.zeros(2, 3, len(counts), device=device)

            def operation(pattern=pattern, logits=logits, blocks=blocks, chunks=chunks):
                return [pattern(logits, layout, blocks, rows) for rows in chunks]

            pattern.prime(operation)
            for proof in pattern.proofs:
                verify_proof(proof)
                results.append(
                    {
                        "pattern": name,
                        "budget": budget,
                        "index_width": proof["shape"][-1],
                        "proof": proof,
                    }
                )
            if (
                len(pattern.entries) != 2
                or pattern.proofs[0]["rows_sha256"] == pattern.proofs[1]["rows_sha256"]
            ):
                raise AssertionError("equal-shaped different rows shared primed logits")
            replay = pattern.replay(operation)
            with (
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("timed host transfer")
                ),
                patch.object(
                    torch.Tensor, "numpy", side_effect=AssertionError("timed host conversion")
                ),
            ):
                for _ in range(3):
                    replay()
                    pattern.verify_replay()
                chunks.reverse()
                replay()
                try:
                    pattern.verify_replay()
                except RuntimeError as exc:
                    if "query-row sequence" not in str(exc):
                        raise
                else:
                    raise AssertionError("different-row replay order was accepted")
            implicit = NativePattern(torch, mask.select_video_blocks, name)
            full_logits = torch.zeros(2, len(counts), len(counts), device=device)
            # Use nonempty rows in this test's full production layout. Empty keys
            # remain covered by the explicit-row cases above.
            full_counts = counts.clamp(min=1)
            full_layout = SimpleNamespace(
                n_video_tiles=len(counts), valid_count=full_counts, kv_ok=full_counts > 0
            )

            def full_operation(
                implicit=implicit, full_logits=full_logits, full_layout=full_layout, blocks=blocks
            ):
                return implicit(full_logits, full_layout, blocks)

            implicit.prime(full_operation)
            verify_proof(implicit.proofs[0])
            replay_full = implicit.replay(full_operation)
            with (
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("timed host transfer")
                ),
                patch.object(
                    torch.Tensor, "numpy", side_effect=AssertionError("timed host conversion")
                ),
            ):
                for _ in range(3):
                    replay_full()
                    implicit.verify_replay()
            results.append(
                {
                    "pattern": name,
                    "budget": budget,
                    "index_width": implicit.proofs[0]["shape"][-1],
                    "implicit_full_rows": True,
                    "proof": implicit.proofs[0],
                }
            )
    return {
        "format": "veda-cost-native-pattern-regression-v2",
        "pass": True,
        "torch": torch.__version__,
        "device": device,
        "cases": results,
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
    result = native_self_test(args.device)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"pass": result["pass"], "cases": len(result["cases"])}))
