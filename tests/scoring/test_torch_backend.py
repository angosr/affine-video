from types import SimpleNamespace

import numpy as np
import pytest

from scoring import torch_backend, torch_selector
from scoring.evaluate import segmented_top_k as numpy_top_k
from scoring.evaluate import trip_pool as numpy_trip_pool
from scoring.model import ModelValidationError
from scoring.torch_selector import (
    _require_finite_logits,
    runtime_block_mask,
    segmented_top_k,
    selector_logits,
    trip_pool,
)
from validator.production import runtime


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


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


def test_exact_and_video_use_the_same_production_selector_functions():
    assert torch_backend.trip_pool is torch_selector.trip_pool
    assert runtime.gather_and_pool is torch_selector.gather_and_pool
    assert (
        torch_backend.selector_logits is runtime.selector_logits is torch_selector.selector_logits
    )
    assert torch_backend.segmented_top_k is torch_selector.segmented_top_k
    assert runtime.runtime_block_mask is torch_selector.runtime_block_mask


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_shared_trip_pool_masks_padding_and_handles_head_batches(torch, dtype):
    values = np.random.default_rng(7).normal(size=(3, 5, 4, 2)).astype(dtype)
    counts = np.array([1, 4, 2, 3, 1], dtype=np.int64)
    for tile, count in enumerate(counts):
        values[:, tile, count:] = 100
    actual = trip_pool(torch, torch.from_numpy(values), torch.from_numpy(counts))
    for head in range(3):
        np.testing.assert_allclose(actual[head].numpy(), numpy_trip_pool(values[head], counts))
        assert torch.equal(
            actual[head],
            trip_pool(torch, torch.from_numpy(values[head]), torch.from_numpy(counts)),
        )
    counts[-1] = 0
    empty = trip_pool(torch, torch.from_numpy(values), torch.from_numpy(counts))
    assert torch.equal(empty[:, -1], torch.zeros((3, 6)))
    with pytest.raises(ValueError, match="out of range"):
        trip_pool(torch, torch.from_numpy(values), torch.tensor([1, 5, 2, 3, 0]))


def test_shared_projection_matches_independent_per_head_reference(torch):
    rng = np.random.default_rng(10)
    q, k = (rng.normal(size=shape).astype(np.float32) for shape in ((3, 4, 6), (3, 7, 6)))
    pq, pk = (rng.normal(size=(3, 6, 2)).astype(np.float32) for _ in range(2))
    actual = selector_logits(torch, *(torch.from_numpy(value) for value in (q, k, pq, pk)))
    for head in range(3):
        expected = (q[head] @ pq[head] + q[head, :, :2]) @ (
            k[head] @ pk[head] + k[head, :, :2]
        ).T / np.sqrt(np.float32(2))
        np.testing.assert_allclose(actual[head].numpy(), expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ModelValidationError, match="non-finite"):
        selector_logits(torch, torch.full((2, 6), float("inf")), *(
            torch.from_numpy(value[0]) for value in (k, pq, pk)
        ))


def test_shared_stable_top_k_keeps_required_keys_and_segment_budgets(torch):
    segments = np.array([0, 1, 0, 2, 1, 0, 0, 2], dtype=np.int8)
    counts = np.array([4, 2, 3, 1, 4, 4, 1, 3], dtype=np.int64)
    logits = np.zeros((3, 4, 8), dtype=np.float32)
    logits[1] = np.random.default_rng(11).normal(size=(4, 8))
    required = np.zeros((4, 8), dtype=np.bool_)
    required[:, 6] = True
    actual = segmented_top_k(
        torch, torch.from_numpy(logits), torch.from_numpy(segments), torch.from_numpy(counts),
        target_blocks=2, reference_blocks=1, required_keys=torch.from_numpy(required),
    )
    for head in range(3):
        expected = numpy_top_k(
            logits[head], segments, counts, target_blocks=2, reference_blocks=1,
            required_keys=required,
        )
        np.testing.assert_array_equal(actual[head].numpy(), expected)
    assert actual[:, :, 6].all() and actual[:, :, 3].all() and actual[:, :, 7].all()
    np.testing.assert_array_equal(actual[0, 0].numpy(), [True, True, False, True,
                                                       False, False, True, True])


def test_runtime_mask_uses_shared_semantics_for_full_and_sampled_rows(torch):
    counts = torch.tensor([3, 1, 4, 2, 3, 1, 2, 0])
    layout = SimpleNamespace(n_tiles=8, n_ref_tiles=2, n_video_tiles=6, valid_count=counts)
    logits = torch.zeros((3, 6, 8))
    full = runtime_block_mask(torch, logits, layout, target_blocks=2, reference_blocks=1)
    rows = torch.tensor([1, 4])
    sampled = runtime_block_mask(
        torch, logits[:, rows], layout, target_blocks=2, reference_blocks=1, query_rows=rows,
    )
    assert full.shape == (3, 8, 8)
    assert torch.equal(sampled, full[:, rows])
    assert full[:, :, 6].all() and not full[:, :, 7].any()
    assert torch.equal(full[:, 6:], (counts > 0).expand(3, 2, 8))
    assert full[:, torch.arange(6), torch.arange(6)].all()
    assert torch.equal(full[:, :6, :2].sum(-1), torch.ones((3, 6), dtype=torch.long))
    assert torch.equal(full[:, :6, 2:6].sum(-1), torch.full((3, 6), 2, dtype=torch.long))
    with pytest.raises(ValueError, match="align"):
        runtime_block_mask(
            torch, logits, layout, target_blocks=2, reference_blocks=1, query_rows=rows,
        )


def test_video_attention_adapter_passes_the_shared_mask_to_the_kernel(torch, monkeypatch):
    """Real tensor arithmetic through the adapter; kernel transport is observed."""

    counts = torch.tensor([128, 17, 128, 53, 2])
    layout = SimpleNamespace(n_tiles=5, n_ref_tiles=2, n_video_tiles=4, valid_count=counts)
    budget = SimpleNamespace(target_blocks=1, reference_blocks=1)
    config = SimpleNamespace(
        budget_rule="topk", zero_order_compensation=False,
        fused_compensation=False, triton_compensation=False,
    )
    heads = torch.tensor([1, 0])
    rows = torch.from_numpy(np.random.default_rng(19).normal(size=(640, 2, 2)).astype(np.float16))
    projections = {
        "layers.0.proj_q": torch.zeros((2, 6, 2)),
        "layers.0.proj_k": torch.zeros((2, 6, 2)),
    }
    masks, outputs = [], []

    class Student:
        def __init__(self, clip, plan, predictor):
            assert predictor is None

    def kernel(q, k, v, mask, tile_layout):
        assert tile_layout is layout
        masks.append(mask.clone())
        result = v.clone()
        outputs.append(result)
        return result

    def scatter(out, result, tile_layout, selected_heads):
        assert tile_layout is layout
        assert torch.equal(selected_heads, heads)
        out[:, selected_heads] = result

    module = SimpleNamespace(
        SparseStudent=Student, fa4=SimpleNamespace(block_sparse_attention=kernel),
        _gather=lambda value, _layout, selected_heads: value[:, selected_heads],
        _scatter_=scatter,
    )

    def cpu_gather_and_pool(value, tile_layout, selected_heads):
        gathered = module._gather(value, tile_layout, selected_heads)
        features = trip_pool(
            torch, gathered.view(5, 128, 2, 2).permute(2, 0, 1, 3), tile_layout.valid_count,
        )
        return gathered, features

    monkeypatch.setattr(runtime, "gather_and_pool", cpu_gather_and_pool)
    attention = runtime._selector_attention(
        torch, module, SimpleNamespace(config=config), object(), projections, budget,
    )
    result = torch.zeros_like(rows)
    attention._attend(rows, rows, rows, 0, layout, heads, result)
    expected_features = numpy_trip_pool(rows.numpy().reshape(5, 128, 2, 2)[:, :, 0], counts)
    expected_logits = expected_features[:4, :2] @ expected_features[:, :2].T / np.sqrt(2)
    required = np.eye(5, dtype=np.bool_)[:4]
    expected_mask = numpy_top_k(
        expected_logits, np.array([1, 1, 0, 0, 2]), counts.numpy(),
        target_blocks=1, reference_blocks=1, required_keys=required,
    )
    np.testing.assert_array_equal(masks[0][1, :4].numpy(), expected_mask)
    assert torch.equal(result[:, heads], outputs[0])
    assert masks[0][:, 4].all()
    config.zero_order_compensation = True
    with pytest.raises(ValueError, match="uncompensated"):
        runtime._selector_attention(
            torch, module, SimpleNamespace(config=config), object(), projections, budget,
        )


def test_cuda_tie_order_and_sampled_video_masks_match_independent_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the device tie-order regression")
    torch.backends.cuda.matmul.allow_tf32 = False
    values = np.random.default_rng(29).normal(size=(3, 16, 128, 128)).astype(np.float16)
    counts = np.array([128, 3, 71, 1, 128, 51, 128, 13, 128, 2, 127, 128, 43, 7, 128, 1])
    for tile, count in enumerate(counts):
        values[:, tile, count:] = 100
    gpu_counts = torch.as_tensor(counts, device="cuda:0")
    features = trip_pool(torch, torch.as_tensor(values, device="cuda:0"), gpu_counts)
    for head in range(3):
        np.testing.assert_allclose(
            features[head].cpu().numpy(), numpy_trip_pool(values[head], counts),
            atol=1e-6, rtol=1e-6,
        )
    projection = torch.zeros((3, 384, 128), device="cuda:0")
    # A legal selector cancels the residual mean, forcing every logit to tie.
    projection[:, :128] = -torch.eye(128, device="cuda:0")
    logits = selector_logits(
        torch, features[:, :13], features, projection, torch.zeros_like(projection),
    )
    assert not logits.count_nonzero()
    layout = SimpleNamespace(n_tiles=16, n_ref_tiles=4, n_video_tiles=13, valid_count=gpu_counts)
    query_rows = torch.tensor([0, 3, 5, 12], device="cuda:0")
    segments = np.array([1] * 4 + [0] * 9 + [2] * 3)
    required = np.eye(16, dtype=np.bool_)[:13]
    expected = numpy_top_k(
        np.zeros((13, 16), dtype=np.float32), segments, counts,
        target_blocks=4, reference_blocks=2, required_keys=required,
    )
    for _ in range(10):
        full = runtime_block_mask(torch, logits, layout, target_blocks=4, reference_blocks=2)
        sampled = runtime_block_mask(
            torch, logits[:, query_rows], layout,
            target_blocks=4, reference_blocks=2, query_rows=query_rows,
        )
        assert torch.equal(sampled, full[:, query_rows])
        for head in range(3):
            np.testing.assert_array_equal(full[head, :13].cpu().numpy(), expected)
        assert full[:, 13:].all()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_cuda_cached_pool_matches_fused_runtime_with_permuted_heads(torch, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused runtime pooling")
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(20261018)
    values = torch.randn((7 * 128, 4, 128), generator=generator, device=device).to(
        getattr(torch, dtype),
    )
    counts = torch.tensor([128, 17, 1, 0, 127, 5, 128], device=device)
    permutation = torch.randperm(7 * 128, generator=generator, device=device)
    layout = SimpleNamespace(
        n_tiles=7, num_slots=7 * 128, gather_index=permutation,
        slot_valid=(torch.arange(128, device=device)[None, :] < counts[:, None]).flatten(),
        valid_count=counts,
    )
    heads = torch.tensor([3, 0, 2], device=device)
    gathered, features = torch_selector.gather_and_pool(values, layout, heads)
    expected_rows = values[permutation][:, heads].masked_fill(
        ~layout.slot_valid[:, None, None], 0,
    )
    assert torch.equal(gathered, expected_rows)
    tiles = gathered.view(7, 128, 3, 128).permute(2, 0, 1, 3)
    assert torch.equal(features, trip_pool(torch, tiles, counts))
    assert not features[:, counts == 0].count_nonzero()
    for head in range(3):
        cached = tiles[head].float()
        # Widening cache inputs can change Triton's reduction lowering; the
        # formal descriptor tolerance is 1e-6, not cross-dtype bitwise equality.
        torch.testing.assert_close(
            features[head], trip_pool(torch, cached, counts), rtol=1e-6, atol=1e-6,
        )
        nonempty = counts > 0
        np.testing.assert_allclose(
            features[head, nonempty].cpu().numpy(),
            numpy_trip_pool(cached[nonempty].cpu().numpy(), counts[nonempty].cpu().numpy()),
            rtol=1e-6, atol=1e-6,
        )
