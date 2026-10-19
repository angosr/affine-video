"""CPU mechanism fixtures only; no recorded GPU scores are shipped with runtime code."""

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from scoring.spec import parse_track_spec, track_spec_dict
from tests.scoring.test_spec import valid_spec
from validator import system_matches as matches
from validator.monitor import render_status
from validator.production.commitment import Artifact, ArtifactManifest
from validator.publication import (
    PublicArtifactError,
    S3Repository,
    build_genesis_artifacts,
    decode_weight_artifact,
)
from validator.weights import WeightChainState

from .test_publication import MemoryS3


@pytest.fixture
def benchmark(monkeypatch):
    """One fixed synthetic record, never a measured model or pending release."""

    calls = {"source": 0, "environment": 0}

    @lru_cache(maxsize=1)
    def source():
        calls["source"] += 1
        return "41" * 32

    @lru_cache(maxsize=1)
    def environment():
        calls["environment"] += 1
        return "42" * 32

    monkeypatch.setattr(matches, "evaluation_source_digest", source)
    monkeypatch.setattr(matches, "evaluation_environment_digest", environment)
    matches._evidence_runtime_digest.cache_clear()
    manifest = ArtifactManifest(
        tuple(
            sorted(
                (
                    Artifact("runtime/source.zip", "51" * 32, 10, "artifacts/source"),
                    Artifact("weights/reference", "11" * 32, 100, "artifacts/reference"),
                    Artifact("weights/control", "21" * 32, 100, "artifacts/control"),
                ),
                key=lambda item: item.logical_name,
            )
        )
    ).encode()
    raw = valid_spec()
    raw["track_id"] = "benchmark-test-track"
    raw["artifact_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    track = parse_track_spec(raw)
    record = {
        "sequence": 1,
        "track_id": track.track_id,
        "netuid": track.netuid,
        "track_spec_digest": track.digest,
        "track_contract_sha256": matches._evidence_contract_digest(track),
        "runtime_contract_sha256": matches._evidence_runtime_digest(manifest),
        "source_sha256": source(),
        "environment_sha256": environment(),
        "image_sha256": "52" * 32,
        "hardware": "CPU synthetic fixture",
        "scope": "unit_test_not_gpu_acceptance",
        "phase": "exact",
        "confidence": track.match_confidence,
        "minimum_gain": track.scoring.minimum_frontier_gain,
        "budgets": tuple(budget.target_blocks for budget in track.scoring.budgets),
        "cluster_unit": "source_video",
        "reference": {
            "model_id": "reference-fixture",
            "label": "Reference fixture",
            "sha256": "11" * 32,
            "model_digest": "12" * 32,
        },
        "challenger": {
            "model_id": "control-fixture",
            "label": "Control fixture",
            "sha256": "21" * 32,
            "model_digest": "22" * 32,
        },
        "batches": (
            {
                "cohort": "synthetic-cohort",
                "partition": "audit",
                "generation_block": 100,
                "plan_digest": "31" * 32,
                "cache_sha256": "32" * 32,
                "ready_sha256": "33" * 32,
                "evidence_sha256": "34" * 32,
                "cases": 16,
                "independent_videos": 4,
                "independent_videos_per_stratum": 4,
                "reference_score": 0.8,
                "challenger_score": 0.7,
                "reference_gain_lower_bound": 0.05,
                "challenger_gain_lower_bound": -0.2,
            },
        ),
    }
    context = SimpleNamespace(track=track, manifest=manifest, record=record, calls=calls)
    context.artifacts = matches.system_match_artifacts(track, manifest, (record,))
    context.chain = WeightChainState(99, {track.initial_champion.hotkey: 0}, 1, 1.0, 0, {})
    yield context
    matches._evidence_runtime_digest.cache_clear()


def genesis(case, *, system_matches=()):
    return build_genesis_artifacts(
        case.track,
        case.chain,
        version_key=0,
        runtime_manifest=case.manifest,
        system_matches=system_matches,
    )


def test_no_builtin_results_and_constructor_preserves_original_schema(benchmark):
    case = benchmark
    assert matches.system_match_artifacts(case.track, case.manifest) == ()
    assert not hasattr(matches, "BENCHMARKS")
    assert not Path(matches.__file__).with_name("system_match_data.py").exists()
    name, payload = case.artifacts[0]
    assert name == f"system_matches/{case.track.track_id}/000000000001.json"
    document = json.loads(payload)
    assert document["record_type"] == "SYSTEM_BENCHMARK"
    assert document["affects_tournament"] is document["reward_eligible"] is False
    assert document["evaluation"]["batch_count"] == 1
    assert document["evaluation"]["cases_per_batch"] == 16
    assert document["scores"] == {"reference": 0.8, "challenger": 0.7}
    assert matches.validate_system_match_artifact(case.track, case.manifest, name, payload) == (
        name,
        payload,
    )


def test_r2_roundtrip_cold_recovery_and_cli_do_not_change_rewards(benchmark):
    case = benchmark
    client = MemoryS3()
    first = S3Repository(client, bucket="public-data", prefix="tracks/test")
    plain = genesis(case)
    populated = genesis(case, system_matches=case.artifacts)
    assert populated.weights == plain.weights
    assert populated.matches == plain.matches == ()
    assert populated.summary == plain.summary
    asyncio.run(first.publish(populated))
    assert client.objects[f"tracks/test/{case.artifacts[0][0]}"] == case.artifacts[0][1]

    # All local result state is gone; the only surviving fact is the S3 object.
    cold = S3Repository(client, bucket="public-data", prefix="tracks/test")
    recovered = asyncio.run(cold.load_system_matches(case.track, case.manifest))
    assert recovered == case.artifacts
    assert matches.system_match_artifacts(case.track, case.manifest) == ()
    puts_before = len(client.calls)
    next_snapshot = genesis(case, system_matches=recovered)
    asyncio.run(cold.publish(next_snapshot))
    assert len(client.calls) == puts_before
    status = json.loads(client.objects["tracks/test/status.json"])
    rendered = render_status(status, color=False)
    assert "SYSTEM BENCHMARKS" in rendered and "(exact)" in rendered
    assert decode_weight_artifact(client.objects["tracks/test/weights.json"]).plan.uids == (0,)


@pytest.mark.parametrize("failure", ["missing", "json", "identity", "network", "changed"])
def test_refresh_failure_retains_last_valid_summary_and_weights(benchmark, caplog, failure):
    case = benchmark

    class Transport(MemoryS3):
        broken = False

        def get_object(self, **kwargs):
            if self.broken and "/system_matches/" in kwargs["Key"]:
                raise OSError("benchmark transport unavailable")
            return super().get_object(**kwargs)

    client = Transport()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    snapshot = genesis(case, system_matches=case.artifacts)
    asyncio.run(repository.publish(snapshot))
    assert asyncio.run(repository.load_system_matches(case.track, case.manifest)) == case.artifacts
    key = f"tracks/test/{case.artifacts[0][0]}"
    if failure == "missing":
        del client.objects[key]
    elif failure == "json":
        client.objects[key] = b"{bad-json"
    elif failure == "identity":
        document = json.loads(case.artifacts[0][1])
        document["netuid"] += 1
        client.objects[key] = matches._json_bytes(document)
    elif failure == "changed":
        record = deepcopy(case.record)
        record["batches"][0]["challenger_score"] = 0.6
        client.objects[key] = matches.system_match_artifacts(case.track, case.manifest, (record,))[
            0
        ][1]
    else:
        client.broken = True
    restored = asyncio.run(repository.load_system_matches(case.track, case.manifest))
    assert restored == case.artifacts
    asyncio.run(repository.publish(genesis(case, system_matches=restored)))
    assert decode_weight_artifact(client.objects["tracks/test/weights.json"]).plan.uids == (0,)
    assert len(json.loads(client.objects["tracks/test/status.json"])["system_matches"]) == 1
    assert "system benchmark" in caplog.text


def test_bad_record_or_failed_information_put_cannot_block_weights(benchmark, caplog):
    case = benchmark

    class Transport(MemoryS3):
        def put_object(self, **kwargs):
            if "/system_matches/" in kwargs["Key"]:
                raise OSError("benchmark PUT unavailable")
            return super().put_object(**kwargs)

    client = Transport()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(genesis(case, system_matches=case.artifacts)))
    assert decode_weight_artifact(client.objects["tracks/test/weights.json"]).plan.uids == (0,)
    assert json.loads(client.objects["tracks/test/status.json"])["system_matches"] == []
    assert "system benchmark publication skipped" in caplog.text
    # Invalid deployment metadata is also non-fatal at the public artifact boundary.
    invalid = ((case.artifacts[0][0], b"not-json"),)
    assert genesis(case, system_matches=invalid).system_matches == ()


def test_information_retry_cannot_rewrite_a_frozen_status_or_weaken_its_guard(benchmark):
    case = benchmark

    class Transport(MemoryS3):
        fail_benchmark = True

        def put_object(self, **kwargs):
            if self.fail_benchmark and "/system_matches/" in kwargs["Key"]:
                raise OSError("temporary benchmark write outage")
            return super().put_object(**kwargs)

    client = Transport()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    snapshot = genesis(case, system_matches=case.artifacts)
    asyncio.run(repository.publish(snapshot))
    original_status = client.objects["tracks/test/status.json"]
    assert json.loads(original_status)["system_matches"] == []
    client.fail_benchmark = False
    asyncio.run(repository.publish(snapshot))
    assert client.objects["tracks/test/status.json"] == original_status
    # The background reader can complete after this head was already published.
    asyncio.run(repository.publish(genesis(case)))
    assert client.objects["tracks/test/status.json"] == original_status
    changed = json.loads(snapshot.status)
    changed["participants"] = [{"hotkey": "changed-real-participant"}]
    with pytest.raises(PublicArtifactError):
        asyncio.run(repository.publish(replace(snapshot, status=matches._json_bytes(changed))))
    assert client.objects["tracks/test/status.json"] == original_status

    case.chain = replace(case.chain, finalized_block=100)
    asyncio.run(repository.publish(genesis(case)))
    assert len(json.loads(client.objects["tracks/test/status.json"])["system_matches"]) == 1
    assert decode_weight_artifact(client.objects["tracks/test/weights.json"]).plan.uids == (0,)


def test_cold_reader_skips_bad_record_then_reads_next_without_an_index(benchmark):
    case = benchmark
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    client.objects[f"tracks/test/{case.artifacts[0][0]}"] = b"not-json"
    second = matches.system_match_artifacts(
        case.track, case.manifest, ({**case.record, "sequence": 2},)
    )
    for name, payload in second:
        repository._put_immutable(name, payload)
    assert asyncio.run(repository.load_system_matches(case.track, case.manifest)) == second
    assert {call["Key"] for call in client.calls} == {f"tracks/test/{second[0][0]}"}


def test_reader_is_bounded_and_stops_at_first_missing_sequence(benchmark):
    case = benchmark

    class Transport(MemoryS3):
        def __init__(self):
            super().__init__()
            self.reads = []

        def get_object(self, **kwargs):
            self.reads.append(kwargs["Key"])
            return super().get_object(**kwargs)

    client = Transport()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    for sequence in range(1, matches.MAX_SYSTEM_MATCHES + 1):
        name, payload = matches.system_match_artifacts(
            case.track, case.manifest, ({**case.record, "sequence": sequence},)
        )[0]
        client.objects[f"tracks/test/{name}"] = payload
    loaded = asyncio.run(repository.load_system_matches(case.track, case.manifest))
    assert len(loaded) == matches.MAX_SYSTEM_MATCHES
    assert len(client.reads) == matches.MAX_SYSTEM_MATCHES
    del client.objects[f"tracks/test/system_matches/{case.track.track_id}/000000000002.json"]
    cold = S3Repository(client, bucket="public-data", prefix="tracks/test")
    client.reads.clear()
    assert len(asyncio.run(cold.load_system_matches(case.track, case.manifest))) == 1
    assert len(client.reads) == 2


@pytest.mark.parametrize(
    "identity",
    [
        "track_id",
        "netuid",
        "track_contract_sha256",
        "source_sha256",
        "environment_sha256",
        "runtime_contract_sha256",
    ],
)
def test_identity_and_every_measured_contract_are_fail_closed(benchmark, identity, caplog):
    case = benchmark
    document = json.loads(case.artifacts[0][1])
    if identity == "track_id":
        document["track_id"] = "another-track"
    elif identity == "netuid":
        document["netuid"] += 1
    else:
        document["evaluation"][identity] = "ef" * 32
    bad = ((case.artifacts[0][0], matches._json_bytes(document)),)
    assert matches.validated_system_matches(case.track, case.manifest, bad) == ()
    assert "system benchmark skipped" in caplog.text


@pytest.mark.parametrize("field", ["sha256", "model_digest"])
def test_both_model_fingerprints_and_exact_canonical_body_are_required(benchmark, field):
    case = benchmark
    document = json.loads(case.artifacts[0][1])
    document["challenger"][field] = "missing"
    with pytest.raises(ValueError):
        matches.validate_system_match_artifact(
            case.track, case.manifest, case.artifacts[0][0], matches._json_bytes(document)
        )
    with pytest.raises(ValueError, match="canonical"):
        matches.validate_system_match_artifact(
            case.track, case.manifest, case.artifacts[0][0], b" " + case.artifacts[0][1]
        )


def test_another_track_has_its_own_sequence_one_and_original_record_is_immutable(benchmark):
    case = benchmark
    raw = track_spec_dict(case.track)
    raw["track_id"] = "other-benchmark-track"
    other = parse_track_spec(raw)
    record = {
        **deepcopy(case.record),
        "track_id": other.track_id,
        "track_spec_digest": other.digest,
        "track_contract_sha256": matches._evidence_contract_digest(other),
    }
    other_artifacts = matches.system_match_artifacts(other, case.manifest, (case.record, record))
    assert len(other_artifacts) == 1 and "/other-benchmark-track/" in other_artifacts[0][0]
    assert matches.system_match_artifacts(case.track, case.manifest, (case.record, record)) == (
        case.artifacts
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    for name, payload in (*case.artifacts, *other_artifacts):
        repository._put_immutable(name, payload)
    assert asyncio.run(repository.load_system_matches(other, case.manifest)) == other_artifacts
    assert asyncio.run(repository.load_system_matches(case.track, case.manifest)) == case.artifacts


def test_repeated_publication_does_not_rescan_the_environment(benchmark):
    case = benchmark
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    snapshot = genesis(case, system_matches=case.artifacts)
    for _ in range(3):
        asyncio.run(repository.publish(snapshot))
        assert (
            asyncio.run(repository.load_system_matches(case.track, case.manifest)) == case.artifacts
        )
    assert case.calls == {"source": 1, "environment": 1}


def test_schema_phase_metrics_and_private_fields_are_not_relaxed(benchmark):
    case = benchmark
    for mutate in (
        lambda doc: doc.update(reward_eligible=True),
        lambda doc: doc["evaluation"].update(phase="pending"),
        lambda doc: doc["evaluation"].update(batch_count=10),
        lambda doc: doc["scores"].update(challenger=0.9),
        lambda doc: doc["evaluation"]["batches"][0].update(source_ids=["private-source"]),
    ):
        document = json.loads(case.artifacts[0][1])
        mutate(document)
        assert (
            matches.validated_system_matches(
                case.track, case.manifest, ((case.artifacts[0][0], matches._json_bytes(document)),)
            )
            == ()
        )
