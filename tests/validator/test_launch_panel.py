"""Launch-only gates are tested without importing them into production scoring."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PANEL = Path(__file__).resolve().parents[2] / "evidence/system-panel/launch/runner.py"
spec = importlib.util.spec_from_file_location("launch_panel", PANEL)
assert spec is not None and spec.loader is not None
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def controls():
    return {
        name: {"gain_lower_bound": 0.0 if name == "baseline_self" else -0.01, "promoted": False}
        for name in panel.EXPECTED_IDENTITIES
    }


@pytest.mark.parametrize("replace_selected_inputs", [False, True])
def test_panel_generation_assembles_real_shards_with_the_selected_cache(
    tmp_path, monkeypatch, replace_selected_inputs,
):
    """Stub GPU/domain checks; exercise real assembly, binding, file hashes and resume."""
    from scoring.cache import ExactCacheReader, inspect_exact_cache_batch, write_exact_cache_batch
    from tests.scoring.test_cache import write_batch
    from validator.production import batches, service
    from validator.production.plan import BatchPlan, ReadySampleCache
    from validator.production.storage import INPUTS_NAMESPACE

    original = write_batch(tmp_path / "source")
    with ExactCacheReader(
        original.path, expected_sha256=original.sha256, max_bytes=1_000_000,
    ) as reader:
        cases = tuple(reader)
    results = []
    for rollout in ("rollout-0", "rollout-1"):
        directory = tmp_path / rollout
        shard = write_exact_cache_batch(
            directory, tuple(case for case in cases if case.rollout_id == rollout),
            generated_block=original.generated_block, seed_block_hash=original.seed_block_hash,
            plan_digest=original.plan_digest, track_spec_digest=original.track_spec_digest,
            artifact_manifest_sha256=original.artifact_manifest_sha256,
            source_pool_merkle_root=original.source_pool_merkle_root,
            source_proofs=original.source_proofs, dense_references=(), sample_cache={},
        )
        video, latent = directory / "dense.mp4", directory / "dense.safetensors"
        video.write_bytes(b"GPU video fixture")
        latent.write_bytes(b"GPU latent fixture")
        results.append(service.UnitResult(rollout, str(shard.path), str(video), str(latent), 1, 0))
    plan = SimpleNamespace(
        plan_digest=original.plan_digest, generation_block=original.generated_block,
        block_hash=original.seed_block_hash, batch_seed="ab" * 32,
        track_spec_digest=original.track_spec_digest,
        rollouts=tuple(SimpleNamespace(rollout_id=row.rollout_id) for row in results),
        selected_routes=("route-0", "route-1"),
        source_proofs=(SimpleNamespace(
            leaf=SimpleNamespace(
                source_id="source-0", source_family_id="family-0", digest="12" * 32,
            ),
            document=lambda: original.source_proofs[0],
        ),),
    )
    track = SimpleNamespace(
        digest=original.track_spec_digest,
        artifact_manifest_sha256=original.artifact_manifest_sha256,
        source_pool_merkle_root=original.source_pool_merkle_root,
        model=SimpleNamespace(selector=SimpleNamespace(head_dim=2)),
        scoring=SimpleNamespace(
            case_count=len(cases), stratum_weight_map={"all": 1.0}, sampled_route_count=2,
            rollouts_per_stratum=2, sampled_step_count=2, query_tiles_per_case=2,
        ),
        cache=SimpleNamespace(exact_max_bytes=1_000_000),
    )
    calls = []

    class Workers:
        def __init__(self, config, active_track):
            assert active_track is track
            calls.append("start")

        def run(self, active_plan):
            assert active_plan is plan
            calls.append("generate")
            return tuple(results)

        def close(self):
            calls.append("close")

    selected = []

    def export_inputs(config, active_plan, destination):
        assert active_plan is plan
        destination.mkdir(parents=True)
        index, text = destination / "index.json", destination / "text.safetensors"
        index.write_bytes(b"{}")
        text.write_bytes(b"encoded-input fixture")
        cache = ReadySampleCache(
            service._ready_file(index, INPUTS_NAMESPACE, ".json"),
            service._ready_file(text, INPUTS_NAMESPACE, ".safetensors"),
        )
        selected.append(cache.document())
        return cache, ()

    monkeypatch.setattr(panel, "ROOT", tmp_path)
    monkeypatch.setattr(panel, "labels", lambda: ("production-00",))
    monkeypatch.setattr(
        panel, "load_producer_context",
        lambda: (SimpleNamespace(), track, None, None, {"production-00": plan}),
    )
    monkeypatch.setattr(panel, "record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "WorkerPool", Workers)
    monkeypatch.setattr(service, "_build_selected_sample_cache", export_inputs)
    (tmp_path / "runtime-verification.json").write_text("{}")

    panel.generate()
    report = json.loads((tmp_path / "batches/production-00.json").read_text())
    batch = inspect_exact_cache_batch(report["exact_path"], max_bytes=1_000_000)
    assert len(batch.cases) == len(cases)
    assert batch.sample_cache == report["selected_inputs"] == selected[0]
    assert len(batch.dense_references) == len(results)
    assert report["exact_sha256"] == batch.sha256
    panel.generate()
    assert calls == ["start", "generate", "close"]

    # Continue the generated batch into the actual confirmation loader. Only
    # the heavyweight encoded-input/CUDA/domain validators are stand-ins.
    from scoring import cache as cache_module

    validations = []
    plan.ready = lambda **fields: SimpleNamespace(**vars(plan), **fields)
    monkeypatch.setattr(BatchPlan, "decode", lambda _payload: plan)
    monkeypatch.setattr(panel, "confirmation_labels", lambda: ("production-00",))
    for module, name in (
        (batches, "validate_selected_sample_cache"),
        (batches, "_validate_plan_case_binding"),
        (cache_module, "validate_exact_cache"),
    ):
        monkeypatch.setattr(
            module, name, lambda *_args, _name=name, **_kwargs: validations.append(_name),
        )
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans/production-00.json").write_bytes(b"validated plan fixture")
    if replace_selected_inputs:
        text = Path(report["selected_cache_path"]) / "text.safetensors"
        text.write_bytes(b"replacement encoded inputs")
        report["selected_inputs"]["text"] = service._ready_file(
            text, INPUTS_NAMESPACE, ".safetensors",
        ).document()
        (tmp_path / "batches/production-00.json").write_text(json.dumps(report))
        with pytest.raises(RuntimeError, match="conflicts with its READY plan"):
            panel._load_confirmation_batch("production-00", track)
    else:
        confirmed, _ = panel._load_confirmation_batch("production-00", track)
        assert confirmed.exact.sample_cache == confirmed.plan.sample_cache.document() == selected[0]
        assert len(confirmed.latent_paths) == len(results)
    assert validations == [
        "validate_selected_sample_cache", "validate_exact_cache", "_validate_plan_case_binding",
    ]


def test_panel_positive_and_negative_directions_are_required():
    result = panel._directional_gates(controls(), 0.02, 0.005, "production-00")
    assert result["reverse_veda_vs_shuffled"]["promoted"] is True


@pytest.mark.parametrize("name", sorted(panel.HARD_STABILITY_IDENTITIES))
def test_consistently_wrong_control_direction_fails(name):
    values = controls()
    values[name] = {"gain_lower_bound": 0.01, "promoted": True}
    with pytest.raises(RuntimeError):
        panel._directional_gates(values, 0.02, 0.005, "production-00")


@pytest.mark.parametrize("reverse", [0.005, 0.0, -0.02, float("nan"), float("inf")])
def test_missing_positive_power_fails(reverse):
    with pytest.raises(RuntimeError):
        panel._directional_gates(controls(), reverse, 0.005, "production-00")


def test_full_video_covers_dynamic_incumbent_even_when_baseline_comparison_rejects(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    document = {
        "candidate_track_spec_digest": "track",
        "runtime_contract_sha256": panel.RUNTIME_CONTRACT,
        "model_identities": panel.EXPECTED_IDENTITIES,
        "batches": [
            {"label": f"production-{index:02d}", "partition": "production", "controls": controls()}
            for index in range(panel.PRODUCTION_EXACT)
        ],
    }
    (tmp_path / "exact-summary.json").write_text(json.dumps(document))
    decisions = panel._full_video_eligibility(
        SimpleNamespace(digest="track", scoring=SimpleNamespace(minimum_frontier_gain=0.005))
    )["decisions"]
    assert all(row["eligible"] for row in decisions.values())
    calls = []
    replay = panel._two_stage_fifo_replay(
        order=("trained_near_neighbor", "mean_pool"),
        exact_labels=("production-00", "production-01"),
        confirmation_labels=("production-10", "production-11"),
        exact_bound=lambda challenger, incumbent, _label: (
            0.02 if challenger == "trained_near_neighbor" or incumbent != "baseline_self" else -0.02
        ),
        video_bound=lambda challenger, incumbent, label: (
            calls.append((challenger, incumbent, label)) or 0.02
        ),
        exact_threshold=0.005, video_threshold=0.005,
    )
    assert replay["terminal_champion"] == "mean_pool"
    assert calls[-1][:2] == ("mean_pool", "trained_near_neighbor")
    assert all(decisions[name]["eligible"] for call in calls for name in call[:2])


def test_intermediate_runtime_binding_cannot_launch(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"format": "veda-panel-launch-binding-v1", "frozen": False}))
    with pytest.raises(RuntimeError, match="frozen final runtime"):
        panel.configure(binding)


def test_panel_interval_percentile_and_two_stage_self_checks():
    panel.self_test()


def test_source_log_quarantine_binds_pool_and_durable_inputs(tmp_path, monkeypatch):
    raw = tmp_path / "raw/source-log-scan/smoke.log"
    raw.parent.mkdir(parents=True)
    raw.write_text("source-0001-123456789abc\n")
    document = {
        "format": "veda-source-log-scan-v1",
        "candidate_root": "ab" * 32,
        "candidate_source_pool_sha256": "cd" * 32,
        "source_ids": ["source-0001-123456789abc"],
        "source_id_count": 1,
        "source_families": ["12" * 32],
        "source_family_count": 1,
        "candidate_source_id_intersection": [],
        "candidate_family_intersection": [],
        "files_with_source_ids": [
            {
                "durable_path": "raw/source-log-scan/smoke.log",
                "sha256": panel.sha256(raw),
                "source_ids": ["source-0001-123456789abc"],
            }
        ],
    }
    monkeypatch.setattr(panel, "ROOT", tmp_path)

    panel._verify_source_log_scan(document, SimpleNamespace(root="ab" * 32), "cd" * 32)
    document["candidate_family_intersection"] = ["12" * 32]
    with pytest.raises(RuntimeError, match="quarantine evidence"):
        panel._verify_source_log_scan(document, SimpleNamespace(root="ab" * 32), "cd" * 32)
    document["candidate_family_intersection"] = []
    raw.write_text("changed")
    with pytest.raises(RuntimeError, match="evidence changed"):
        panel._verify_source_log_scan(document, SimpleNamespace(root="ab" * 32), "cd" * 32)


@pytest.fixture
def cost_evidence(tmp_path, monkeypatch):
    """Synthetic reports exercise launch gating; they are never acceptance evidence."""
    from copy import deepcopy

    from scoring.spec import parse_track_spec
    from tests.validator.test_cost_calibration import analysis, cost_registration
    from validator.system_matches import _evidence_contract_digest

    registration = cost_registration()
    registration["runtime_contract_sha256"] = "ab" * 32
    costs = {
        str(b["target_blocks"]): b["calibrated_cost"]
        for b in registration["calibration_track_document"]["scoring"]["budgets"]
    }
    document, digest = analysis.derive_track(registration["calibration_track_document"], costs)
    track = parse_track_spec(document)
    contexts = {
        f"{g}/{t}" for g in registration["geometries"] for t in registration["text_token_counts"]
    }
    report = {
        "format": "veda-h200-cost-fit-v2",
        "source_sha256": "source",
        "environment_sha256": "environment",
        "costs": costs,
        "scales": dict.fromkeys(contexts, 1.0),
        "provisional_cost_track_document": document,
        "provisional_cost_track_spec_digest": digest,
        "cost_independent_track_sha256": registration["cost_independent_track_sha256"],
        "confirmed_evidence_contract_sha256": _evidence_contract_digest(track),
        "model_identities": registration["model_identities"],
        "predictor_bundle_sha256": registration["predictor_bundle_sha256"],
    }
    documents = {"registration": registration}
    binding = {}
    for phase in ("calibration", "confirmation"):
        value = deepcopy(report)
        value.update(
            phase=phase,
            e10_confirmation_pass=phase == "confirmation",
            e11_confirmation_pass=phase == "confirmation",
        )
        value["raw_evidence"] = [
            {"protocol_sha256": f"{seed:064x}", "summary_sha256": f"{seed + 1:064x}"}
            for seed in registration["seeds"][phase]
        ]
        value["gates"] = {}
        for context in contexts:
            for model in [None, *registration["models"]]:
                curves = len(registration["layers"]) * len(registration["seeds"][phase])
                curves *= len(registration["models"]) if model is None else 1
                group = context if model is None else f"{context}/{model}"
                value["gates"][group] = {
                    "row_count": curves * 3,
                    "matched_curve_count": curves,
                    "minimum_matched_curve_spearman": 1.0,
                    "minimum_overhead_fraction": 0.04,
                    "maximum_overhead_fraction": 0.06,
                    "median_relative_error": 0.01,
                    "p95_relative_error": 0.02,
                    "e10_pass": True,
                    "e11_pass": True,
                    "budget_error_gates": {
                        str(b): {"median_relative_error": 0.01, "p95_relative_error": 0.02}
                        for b in registration["budgets"]
                    },
                }
        documents[phase] = value

    def persist():
        for name in ("registration", "calibration", "confirmation"):
            value = documents[name]
            if name != "registration":
                value["registration_sha256"] = binding["registration"]["sha256"]
                value["fit_sha256"] = (
                    None if name == "calibration" else binding["calibration"]["sha256"]
                )
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(value))
            binding[name] = {"path": str(path), "sha256": panel.sha256(path)}

    persist()
    monkeypatch.setattr(panel, "COST_BINDING", binding)
    monkeypatch.setattr(panel, "RUNTIME_CONTRACT", registration["runtime_contract_sha256"])
    return track, documents, persist


def verify_cost(track):
    panel._verify_cost_confirmation(track, source_sha256="source", environment_sha256="environment")


def test_cost_gate_accepts_only_confirmed_costs_and_allows_later_manifest(cost_evidence):
    from dataclasses import replace

    track, _, _ = cost_evidence
    verify_cost(track)
    verify_cost(replace(track, artifact_manifest_sha256="f" * 64))
    with pytest.raises(ValueError, match="registered phase track"):
        verify_cost(replace(track, track_start_block=track.track_start_block + 1))
    budget = replace(track.scoring.budgets[0], calibrated_cost=0.041)
    with pytest.raises(ValueError, match="registered phase track"):
        verify_cost(
            replace(
                track, scoring=replace(track.scoring, budgets=(budget, *track.scoring.budgets[1:]))
            )
        )


@pytest.mark.parametrize("phase", ["calibration", "confirmation"])
@pytest.mark.parametrize(
    "mutation",
    [
        "group",
        "budget",
        "count",
        "error",
        "overhead",
        "nan",
        "source",
        "environment",
        "failed",
        "raw",
    ],
)
def test_cost_gate_rejects_incomplete_or_failed_reports_even_if_rehashed(
    cost_evidence, phase, mutation
):
    track, documents, persist = cost_evidence
    report = documents[phase]
    group = next(iter(report["gates"]))
    gate = report["gates"][group]
    if mutation == "group":
        del report["gates"][group]
    elif mutation == "budget":
        del gate["budget_error_gates"]["16"]
    elif mutation == "count":
        gate["row_count"] -= 1
    elif mutation == "error":
        gate["budget_error_gates"]["16"]["p95_relative_error"] = 0.101
    elif mutation == "overhead":
        gate["maximum_overhead_fraction"] = 0.101
    elif mutation == "nan":
        gate["minimum_matched_curve_spearman"] = float("nan")
    elif mutation in ("source", "environment"):
        report[f"{mutation}_sha256"] = "different"
    elif mutation == "raw":
        report["raw_evidence"] = []
    else:
        gate["e10_pass"] = False
    persist()
    with pytest.raises(RuntimeError, match="cost"):
        verify_cost(track)


def test_cost_gate_checks_content_hash_and_requires_confirmation(cost_evidence, monkeypatch):
    track, _, _ = cost_evidence
    Path(panel.COST_BINDING["confirmation"]["path"]).write_text("{}")
    with pytest.raises(RuntimeError, match="hash changed"):
        verify_cost(track)
    monkeypatch.setattr(panel, "COST_BINDING", {})
    with pytest.raises(RuntimeError, match="requires registered cost"):
        verify_cost(track)


def test_frozen_flag_alone_does_not_authorize_panel_launch(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"format": "veda-panel-launch-binding-v1", "frozen": True}))
    with pytest.raises(RuntimeError, match="requires registered cost"):
        panel.configure(binding)


def test_producer_rejects_missing_cost_evidence_before_loading_private_sources(
    cost_evidence, tmp_path, monkeypatch
):
    import scoring.spec
    import validator.production.commitment as commitment
    import validator.production.config as config
    import validator.production.environment as environment
    import validator.system_matches as matches

    track, _, _ = cost_evidence
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(panel, "MANIFEST", manifest)
    monkeypatch.setattr(panel, "COST_BINDING", {})
    monkeypatch.setattr(panel, "EVIDENCE_CONTRACT", matches._evidence_contract_digest(track))
    monkeypatch.setattr(scoring.spec, "load_track_spec", lambda _: track)
    monkeypatch.setattr(
        config, "load_producer_config", lambda _: SimpleNamespace(verify_manifest=lambda _: None)
    )
    monkeypatch.setattr(
        commitment.ArtifactManifest,
        "decode",
        lambda _: SimpleNamespace(digest=track.artifact_manifest_sha256),
    )
    monkeypatch.setattr(matches, "_evidence_runtime_digest", lambda _: panel.RUNTIME_CONTRACT)
    monkeypatch.setattr(environment, "evaluation_source_digest", lambda: "source")
    monkeypatch.setattr(environment, "evaluation_environment_digest", lambda: "environment")
    monkeypatch.setattr(
        commitment,
        "load_bound_source_pool",
        lambda *a, **k: pytest.fail(
            "private sources must not be consumed before cost confirmation"
        ),
    )
    with pytest.raises(RuntimeError, match="requires registered cost"):
        panel.load_producer_context()
