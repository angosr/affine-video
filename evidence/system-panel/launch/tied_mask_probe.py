"""Focused E16 mechanism probe; no final-runtime or launch-acceptance claim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def probe():
    import torch
    from miowtion.veda import mask

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    rows = 16
    keys = 32
    layout = SimpleNamespace(
        n_tiles=keys,
        n_video_tiles=keys,
        n_ref_tiles=0,
        target_tokens=128 * keys,
        ref_tokens=0,
        kv_ok=torch.ones(keys, dtype=torch.bool, device=device),
    )
    logits = torch.zeros((1, rows, keys), dtype=torch.float32, device=device)
    row_ids = torch.arange(rows, device=device)
    results = []
    for k in (4, 8, 16):
        blocks = mask.column_blocks(layout, mask.Budget(tiles=k), None)
        selected = mask.select_video_blocks(logits, layout, blocks, rows=row_ids)
        actual = mask.dense_block_mask(selected, layout, global_rows=False)[0]
        expected_logits = logits[0].clone()
        expected_logits[row_ids, row_ids] = float("inf")
        order = torch.argsort(expected_logits, dim=-1, descending=True, stable=True)
        expected = torch.zeros_like(actual).scatter_(1, order[:, :k], True)
        results.append(
            {
                "budget": k,
                "mask_mismatches": int((actual != expected).sum().item()),
                "runtime_first_row": actual[0].nonzero().flatten().cpu().tolist(),
                "canonical_first_row": expected[0].nonzero().flatten().cpu().tolist(),
            }
        )
    return {
        "format": "veda-tied-mask-mechanism-probe-v1",
        "scope": "mechanism_only_not_final_acceptance",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_uuid": str(properties.uuid),
        "pci_bus_id": properties.pci_bus_id,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "results": results,
        "pass": all(row["mask_mismatches"] == 0 for row in results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = probe()
    with arguments.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))
