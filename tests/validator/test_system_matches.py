import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from scoring.spec import load_track_spec
from validator import publication as publication_module
from validator import system_matches as system_matches_module
from validator.production.commitment import Artifact, ArtifactManifest
from validator.release import PRODUCTION_NETUID, PRODUCTION_TRACK_ID
from validator.system_matches import system_match_artifacts, system_match_summaries
from validator.weights import WeightChainState


def _runtime_manifest(
    *,
    asset_sha256: str = "ab" * 32,
    package_sha256: str = "cd" * 32,
    evidence_sha256: str | None = None,
    evidence_name: str = "evidence/system-panel/raw.json",
) -> bytes:
    artifacts = [
        Artifact("runtime/source.zip", package_sha256, 10, "artifacts/" + package_sha256),
        Artifact(
            "weights/predictor.safetensors",
            asset_sha256,
            20,
            "artifacts/" + asset_sha256,
        ),
    ]
    if evidence_sha256 is not None:
        artifacts.append(
            Artifact(
                evidence_name,
                evidence_sha256,
                30,
                "artifacts/" + evidence_sha256,
            )
        )
    return ArtifactManifest(tuple(sorted(artifacts, key=lambda item: item.logical_name))).encode()


def _bind_evidence(monkeypatch, track, runtime_manifest: bytes) -> None:
    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", lambda: "12" * 32)
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        lambda: "34" * 32,
    )
    records = deepcopy(system_matches_module.BENCHMARKS)
    for record in records:
        record["track_contract_sha256"] = system_matches_module._evidence_contract_digest(track)
        record["source_sha256"] = "12" * 32
        record["environment_sha256"] = "34" * 32
        record["runtime_contract_sha256"] = system_matches_module._evidence_runtime_digest(
            runtime_manifest
        )
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", records)


def test_runtime_evidence_contract_is_computed_once(monkeypatch) -> None:
    runtime_manifest = _runtime_manifest()
    calls = {"environment": 0, "source": 0}

    def environment_digest() -> str:
        calls["environment"] += 1
        return "12" * 32

    def source_digest() -> str:
        calls["source"] += 1
        return "34" * 32

    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        environment_digest,
    )
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", source_digest)

    first = system_matches_module._evidence_runtime_digest(runtime_manifest)
    assert system_matches_module._evidence_runtime_digest(runtime_manifest) == first
    assert calls == {"environment": 1, "source": 1}
    system_matches_module._evidence_runtime_digest.cache_clear()


def test_repeated_publication_reuses_one_runtime_environment_scan(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    track = replace(
        track,
        artifact_manifest_sha256=hashlib.sha256(runtime_manifest).hexdigest(),
    )
    calls = {"artifacts": 0, "environment": 0, "source": 0}

    @lru_cache(maxsize=1)
    def environment_digest() -> str:
        calls["environment"] += 1
        return "12" * 32

    @lru_cache(maxsize=1)
    def source_digest() -> str:
        calls["source"] += 1
        return "34" * 32

    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        environment_digest,
    )
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", source_digest)
    runtime_digest = system_matches_module._evidence_runtime_digest(runtime_manifest)
    records = deepcopy(system_matches_module.BENCHMARKS)
    for record in records:
        record["track_contract_sha256"] = system_matches_module._evidence_contract_digest(track)
        record["runtime_contract_sha256"] = runtime_digest
        record["source_sha256"] = "34" * 32
        record["environment_sha256"] = "12" * 32
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", records)
    system_matches_module._evidence_runtime_digest.cache_clear()
    environment_digest.cache_clear()
    source_digest.cache_clear()
    calls["environment"] = calls["source"] = 0
    real_artifacts = system_matches_module.system_match_artifacts

    def counted_artifacts(track, manifest):
        calls["artifacts"] += 1
        return real_artifacts(track, manifest)

    monkeypatch.setattr(publication_module, "system_match_artifacts", counted_artifacts)
    monkeypatch.setattr(
        publication_module,
        "_verify_candidate_release",
        lambda _track, _manifest, _approval: None,
    )
    store = SimpleNamespace(
        chain_cursor=lambda: (track.track_start_block, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": []},
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
    )
    chain = WeightChainState(
        track.track_start_block,
        {track.initial_champion.hotkey: 0},
        1,
        1.0,
        10,
        {track.initial_champion.hotkey: 0},
    )

    try:
        first = publication_module.build_public_artifacts(
            store,
            track,
            chain,
            version_key=0,
            runtime_manifest=runtime_manifest,
        )
        second = publication_module.build_public_artifacts(
            store,
            track,
            chain,
            version_key=0,
            runtime_manifest=runtime_manifest,
        )

        assert first is not None and second is not None
        assert len(first.system_matches) == len(second.system_matches) == 4
        assert len(json.loads(first.status)["system_matches"]) == 4
        assert calls == {"artifacts": 2, "environment": 1, "source": 1}
        monkeypatch.setattr(system_matches_module, "BENCHMARKS", ())
        without_benchmarks = publication_module.build_public_artifacts(
            store, track, chain, version_key=0, runtime_manifest=runtime_manifest
        )
        assert without_benchmarks is not None
        for field in ("weights", "summary", "matches", "abort", "batch_groups"):
            assert getattr(first, field) == getattr(without_benchmarks, field)
    finally:
        system_matches_module._evidence_runtime_digest.cache_clear()


def test_mechanism_formats_bound_stage_records_without_reward_effects(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    assert track.track_id == PRODUCTION_TRACK_ID
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", lambda: "ef" * 32)
    assert system_match_artifacts(track, runtime_manifest) == ()
    _bind_evidence(monkeypatch, track, runtime_manifest)
    artifacts = system_match_artifacts(track, runtime_manifest)

    assert [name for name, _payload in artifacts] == [
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000001.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000002.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000003.json",
        f"system_matches/{PRODUCTION_TRACK_ID}/000000000004.json",
    ]
    documents = [json.loads(payload) for _name, payload in artifacts]
    for document in documents:
        assert document["record_type"] == "SYSTEM_BENCHMARK"
        assert document["affects_tournament"] is False
        assert document["reward_eligible"] is False
        assert (
            document["track_spec_digest"]
            == system_matches_module.BENCHMARKS[0]["track_spec_digest"]
        )
        assert document["evaluation"]["track_contract_sha256"] == (
            system_matches_module._evidence_contract_digest(track)
        )
        assert document["evaluation"]["runtime_contract_sha256"] == (
            system_matches_module._evidence_runtime_digest(runtime_manifest)
        )
        assert document["outcome"] in {"REFERENCE_RETAINED", "REFERENCE_WON"}
        assert document["evaluation"]["batch_count"] == 1
        assert len(document["evaluation"]["batches"]) == 1
        assert "model_digest" in document["reference"]
        assert "model_digest" in document["challenger"]
        assert "runtime_trace" not in document["evaluation"]
        assert all(
            batch["challenger_gain_lower_bound"] < document["evaluation"]["minimum_gain"]
            for batch in document["evaluation"]["batches"]
        )

    assert documents[0]["scores"]["reference"] == 0.9491625628902893
    assert documents[0]["scores"]["challenger"] == 0.9491625628902893
    assert documents[1]["scores"]["challenger"] == 0.7885866430037899
    assert all(
        batch["reference_gain_lower_bound"] > 0.005
        for batch in documents[1]["evaluation"]["batches"]
    )
    assert documents[2]["scores"]["reference"] == 0.7619235479211722
    assert documents[3]["scores"]["challenger"] == 0.6570879920438147
    assert [item["evaluation"]["phase"] for item in documents] == [
        "exact",
        "exact",
        "full_video",
        "full_video",
    ]
    assert [item["evaluation"]["cases_per_batch"] for item in documents] == [384, 384, 48, 48]
    assert len(system_match_summaries(artifacts)) == 4


def test_candidate_track_keeps_the_exact_runtime_schedule() -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")

    assert track.scoring.trajectory_schedule.sigmas == (
        1.0,
        0.9882352941176471,
        0.972972972972973,
        0.9523809523809523,
        0.9230769230769231,
        0.8780487804878049,
        0.8,
        0.631578947368421,
        0.0,
    )


def test_system_evidence_binding_does_not_depend_on_release_artifact_hash(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)
    rebuilt = replace(
        track,
        artifact_manifest_sha256="ef" * 32,
        digest="cd" * 32,
    )

    assert system_match_artifacts(rebuilt, runtime_manifest) == system_match_artifacts(
        track, runtime_manifest
    )


def test_system_evidence_rejects_changed_scoring_assets_but_not_repackaging(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, _runtime_manifest(package_sha256="ef" * 32))
    assert system_match_artifacts(track, _runtime_manifest(evidence_sha256="ef" * 32))
    assert system_match_artifacts(track, _runtime_manifest(asset_sha256="ef" * 32)) == ()
    assert system_match_artifacts(track, None) == ()


def test_only_the_canonical_panel_namespace_is_excluded_from_runtime_semantics(
    monkeypatch,
) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest(evidence_sha256="56" * 32)
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, _runtime_manifest(evidence_sha256="ef" * 32))
    for name in (
        "release/evidence/system-panel/raw.json",
        "evidence/system-panelish/raw.json",
        "other/evidence/system-panel/raw.json",
    ):
        assert (
            system_match_artifacts(
                track,
                _runtime_manifest(evidence_sha256="ef" * 32, evidence_name=name),
            )
            == ()
        )


def test_system_matches_are_not_attached_to_another_track(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, runtime_manifest)
    assert system_match_artifacts(replace(track, track_id="another-track"), runtime_manifest) == ()
    assert (
        system_match_artifacts(replace(track, netuid=PRODUCTION_NETUID + 1), runtime_manifest) == ()
    )
    assert (
        system_match_artifacts(
            replace(
                track,
                scoring=replace(track.scoring, minimum_frontier_gain=0.006),
            ),
            runtime_manifest,
        )
        == ()
    )


def test_system_evidence_is_invalidated_by_source_pool_changes(monkeypatch) -> None:
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    runtime_manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, runtime_manifest)

    assert system_match_artifacts(track, runtime_manifest)
    assert (
        system_match_artifacts(
            replace(track, source_pool_merkle_root="ef" * 32),
            runtime_manifest,
        )
        == ()
    )
    assert (
        system_match_artifacts(
            replace(
                track,
                source_commitments=(
                    replace(track.source_commitments[0], sha256="ef" * 32),
                    *track.source_commitments[1:],
                ),
            ),
            runtime_manifest,
        )
        == ()
    )


def test_historical_records_preserve_real_report_identity_without_authorizing_candidate():
    """CPU formatting checks only: this is not a new GPU or release acceptance."""
    records = system_matches_module.BENCHMARKS
    documents = [system_matches_module._match_document(record) for record in records]
    assert {record["source_sha256"] for record in records} == {
        "3fdf7148a4da81119ee51b6557e2a08066c4ffdc69bdb759fd567e524ab0c025"
    }
    assert len(records) == 4
    assert {row["challenger"]["model_id"] for row in records} == {
        "veda_baseline",
        "shuffled_heads_control",
    }
    assert documents[1]["evaluation"]["evidence_sha256"] == (
        "4a2f9c84bef9067522a83d80b8674d2eba7438439aa8495cc0c550852fa66809"
    )
    full = documents[3]["evaluation"]
    assert full["evidence_sha256"] == (
        "59b6aba46debcb3001848ca667443106b25fc075bc7071acf0487ceb5d37057e"
    )
    assert full["batches"][0]["reference_gain_lower_bound"] == 0.04190771739740623
    assert full["scope"] == "isolated_accelerated_chain_fixture"
    assert {item["evaluation"]["batches"][0]["generation_block"] for item in documents} == {
        9311300,
        9312500,
    }
    for document in (documents[0], documents[2]):
        assert document["scores"]["reference"] == document["scores"]["challenger"]
        assert document["evaluation"]["batches"][0]["challenger_gain_lower_bound"] == 0
        assert document["outcome"] == "REFERENCE_RETAINED"


@pytest.mark.parametrize("changed", ["source", "environment"])
def test_measured_runtime_changes_do_not_reuse_evidence(monkeypatch, changed):
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, manifest)
    assert system_match_artifacts(track, manifest)
    system_matches_module._evidence_runtime_digest.cache_clear()
    monkeypatch.setattr(system_matches_module, f"evaluation_{changed}_digest", lambda: "ef" * 32)
    assert system_match_artifacts(track, manifest) == ()
    system_matches_module._evidence_runtime_digest.cache_clear()


def test_source_mismatch_never_scans_the_environment(monkeypatch):
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, manifest)
    monkeypatch.setattr(system_matches_module, "evaluation_source_digest", lambda: "ef" * 32)
    monkeypatch.setattr(
        system_matches_module,
        "evaluation_environment_digest",
        lambda: pytest.fail("old source must not trigger environment hashing"),
    )
    system_matches_module._evidence_runtime_digest.cache_clear()
    assert system_match_artifacts(track, manifest) == ()


def test_batch_count_and_scores_follow_measured_cohorts_not_an_old_panel_size():
    record = deepcopy(system_matches_module.BENCHMARKS[1])
    second = {**record["batches"][0], "cohort": "second-cohort", "generation_block": 9312500}
    second["reference_score"] = 0.9
    record["batches"] = (*record["batches"], second)
    document = system_matches_module._match_document(record)
    assert document["evaluation"]["batch_count"] == 2
    assert len(document["evaluation"]["batches"]) == 2
    assert document["scores"]["reference"] == (0.9491625628902893 + 0.9) / 2
    assert "evidence_sha256" not in document["evaluation"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("phase", "pending"), ("batches", ()), ("source_sha256", "0" * 64)],
)
def test_unmeasured_or_placeholder_records_are_rejected(field, value):
    record = deepcopy(system_matches_module.BENCHMARKS[0])
    record[field] = value
    with pytest.raises(ValueError, match="system benchmark"):
        system_matches_module._match_document(record)


@pytest.mark.parametrize("role", ["reference", "challenger"])
@pytest.mark.parametrize("field", ["sha256", "model_digest"])
def test_both_model_fingerprints_are_required(role, field):
    record = deepcopy(system_matches_module.BENCHMARKS[0])
    record[role][field] = "missing"
    with pytest.raises(ValueError, match="SHA-256"):
        system_matches_module._match_document(record)


def test_repackaging_and_appending_cannot_rewrite_an_immutable_record(monkeypatch):
    from tests.validator.test_publication import MemoryS3
    from validator.monitor import render_status
    from validator.publication import PublicArtifactError, S3Repository

    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, manifest)
    initial = system_match_artifacts(track, manifest)
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    for name, payload in initial:
        publisher._put_immutable(name, payload)
    original_calls = len(client.calls)
    rebuilt = replace(track, digest="ef" * 32, artifact_manifest_sha256="ab" * 32)
    repackaged = system_match_artifacts(rebuilt, _runtime_manifest(package_sha256="56" * 32))
    assert repackaged == initial
    for name, payload in repackaged:
        publisher._put_immutable(name, payload)
    assert len(client.calls) == original_calls

    records = deepcopy(system_matches_module.BENCHMARKS)
    added = {**records[0], "sequence": 5}
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", (*records, added))
    appended = system_match_artifacts(track, manifest)
    assert appended[: len(initial)] == initial
    assert appended[-1][0].endswith("000000000005.json")
    for name, payload in appended:
        publisher._put_immutable(name, payload)
    assert len(client.calls) == original_calls + 1

    records[0]["batches"][0]["evidence_sha256"] = "ab" * 32
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", records)
    modified_name, modified_payload = system_match_artifacts(track, manifest)[0]
    with pytest.raises(PublicArtifactError, match="immutable"):
        publisher._put_immutable(modified_name, modified_payload)
    assert client.objects[f"tracks/test/{initial[0][0]}"] == initial[0][1]
    rendered = render_status(
        {"participants": [], "system_matches": system_match_summaries(initial)}, color=False
    )
    assert "SYSTEM BENCHMARKS" in rendered
    assert "(exact)" in rendered and "(full_video)" in rendered


def test_duplicate_sequences_are_not_silently_renumbered(monkeypatch):
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    records = deepcopy(system_matches_module.BENCHMARKS)
    records[1]["sequence"] = records[0]["sequence"]
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", records)
    with pytest.raises(ValueError, match="sequences"):
        system_match_artifacts(track, _runtime_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    [("confidence", 0.95), ("minimum_gain", 0.001), ("budgets", (16, 32))],
)
def test_report_context_cannot_disagree_with_the_bound_track(monkeypatch, field, value):
    track = load_track_spec(Path(__file__).parents[2] / "track.json")
    manifest = _runtime_manifest()
    _bind_evidence(monkeypatch, track, manifest)
    records = deepcopy(system_matches_module.BENCHMARKS)
    records[0][field] = value
    monkeypatch.setattr(system_matches_module, "BENCHMARKS", records)
    with pytest.raises(ValueError, match="context differs"):
        system_match_artifacts(track, manifest)


def test_only_public_summary_fields_are_serialized():
    record = deepcopy(system_matches_module.BENCHMARKS[0])
    record["reference"]["local_path"] = "PRIVATE-MODEL-PATH"
    record["batches"][0]["source_ids"] = ["PRIVATE-SOURCE-ID"]
    record["batches"][0]["input_path"] = "PRIVATE-INPUT-PATH"
    payload = system_matches_module._json_bytes(system_matches_module._match_document(record))
    assert b"PRIVATE-" not in payload


@pytest.mark.parametrize("field", ["reference_score", "challenger_gain_lower_bound"])
def test_self_checks_cannot_be_relabelled_as_wins(field):
    record = deepcopy(system_matches_module.BENCHMARKS[0])
    record["batches"][0][field] = 0.1
    with pytest.raises(ValueError, match="self comparison"):
        system_matches_module._match_document(record)
