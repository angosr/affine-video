import asyncio
import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from miner.submit import PrivateModelSubmission, model_source_fields
from scoring.spec import parse_track_spec
from tests.scoring.test_spec import valid_spec
from validator import chunks
from validator import publication as publication_module
from validator.production.batches import BatchDisclosure, DisclosureFile
from validator.production.commitment import (
    Artifact,
    ArtifactManifest,
    SourceLeaf,
    SourcePool,
    canonical_json,
)
from validator.production.plan import (
    DenseReference,
    ReadyObject,
    ReadySampleCache,
    build_plan,
    generation_seed,
    select_source_proofs,
)
from validator.publication import (
    HISTORICAL_MATCHES_PREFIX,
    PublicArtifactError,
    PublicArtifacts,
    PublicFile,
    S3Repository,
    build_genesis_artifacts,
    build_public_artifacts,
    decode_weight_artifact,
    encode_weight_artifact,
    fetch_track_identity,
    fetch_weight_artifact,
    validate_public_weights_url,
)
from validator.queue import ChainPosition, challenge_id
from validator.store import ChallengeStore
from validator.tournament import (
    Tournament,
    challenge_from_match,
    genesis_champion,
    historical_champion,
    resolve_tournament,
)
from validator.weights import WeightArtifact, WeightChainState, WeightPlan

from .test_store import submission_at
from .test_tournament import challenge as scored_challenge


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    def __init__(self) -> None:
        self.objects = {}
        self.metadata = {}
        self.calls = []
        self.before_put = None

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "Body": io.BytesIO(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key], usedforsecurity=False).hexdigest()}"',
        }

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.before_put is not None:
            self.before_put(kwargs)
        existing = self.objects.get(kwargs["Key"])
        etag = (
            None if existing is None else hashlib.md5(existing, usedforsecurity=False).hexdigest()
        )
        if kwargs.get("IfNoneMatch") == "*" and existing is not None:
            raise precondition_failed()
        if "IfMatch" in kwargs and kwargs["IfMatch"] != etag:
            raise precondition_failed()
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.metadata[kwargs["Key"]] = kwargs.get("Metadata", {})

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject
        return {
            "ContentLength": len(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key], usedforsecurity=False).hexdigest()}"',
            "Metadata": self.metadata.get(Key, {}),
        }

    def delete_object(self, *, Bucket, Key):
        del Bucket
        self.objects.pop(Key, None)
        self.metadata.pop(Key, None)


def precondition_failed():
    error = RuntimeError("precondition failed")
    error.response = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }
    return error


def weight_artifact() -> WeightArtifact:
    return WeightArtifact(
        track_id="track",
        track_spec_digest="ab" * 32,
        netuid=7,
        mechid=2,
        version_key=9,
        source_block=120,
        plan=WeightPlan(120, ("alice", "bob"), (1, 4), (0.6, 0.4)),
        match_sequence=2,
        terminal_match_sha256="cd" * 32,
    )


def test_only_pinned_legacy_weights_may_omit_abort_fence() -> None:
    legacy = {
        "track_id": "veda-sn120-v1",
        "track_spec_digest": (
            "0fcae991aef29f87b97fbb07cf79c49e5eb9cc2a5d9aecdd2a566443e2b81dd5"
        ),
        "netuid": 120,
        "mechid": 1,
        "version_key": 0,
        "source_block": 9305106,
        "weights": [{"hotkey": "hotkey", "uid": 0, "weight": 1.0}],
    }
    decoded = decode_weight_artifact(json.dumps(legacy).encode())
    assert decoded.track_aborted is False
    assert decoded.match_sequence == 0
    assert decoded.terminal_match_sha256 is None

    legacy["track_id"] = "another-track"
    with pytest.raises(PublicArtifactError, match="fields"):
        decode_weight_artifact(json.dumps(legacy).encode())


def pointer_track(track_id: str = "track") -> bytes:
    return json.dumps(
        {"spec": {"track_id": track_id}, "track_spec_digest": "ab" * 32},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def pointer_weights(
    block: int,
    track_id: str = "track",
    *,
    match_sequence: int = 0,
    terminal_match_sha256: str | None = None,
) -> bytes:
    return encode_weight_artifact(
        WeightArtifact(
            track_id,
            "ab" * 32,
            7,
            0,
            0,
            block,
            WeightPlan(block, ("hk",), (1,), (1.0,)),
            match_sequence=match_sequence,
            terminal_match_sha256=terminal_match_sha256,
        )
    )


def two_stage_challenge(value: dict) -> dict:
    result = dict(value)
    if result.get("state") != "SUCCEEDED":
        return result
    final_batch = dict(result["evaluation_batch"])
    final_batch["generated_block"] = max(final_batch["generated_block"], result["block_height"] + 5)
    exact_batch = {
        "sha256": hashlib.sha256((result["challenge_id"] + ":exact").encode()).hexdigest(),
        "generated_block": result["block_height"] + 4,
        "plan_digest": hashlib.sha256(
            (result["challenge_id"] + ":exact-plan").encode()
        ).hexdigest(),
        "seed_block_hash": "0x" + "cd" * 32,
    }
    exact_decision = result["block_height"] + 4
    result["evaluation_batch"] = final_batch
    result["score_block"] = max(result["score_block"], final_batch["generated_block"])
    result["stages"] = [
        {
            "stage": "SCORING",
            "score": result["final_score"],
            "score_block": exact_decision,
            "incumbent_score": result["incumbent_score"],
            "paired_gain_lower_bound": result["paired_gain_lower_bound"],
            "evaluation_batch": exact_batch,
        },
        {
            "stage": "CONFIRMING",
            "score": result["final_score"],
            "score_block": result["score_block"],
            "incumbent_score": result["incumbent_score"],
            "paired_gain_lower_bound": result["paired_gain_lower_bound"],
            "evaluation_batch": final_batch,
        },
    ]
    return result


@pytest.fixture(params=[False, True], ids=["hf", "private-s3"])
def maintenance_history(request):
    """Canonical outcome facts, not GPU measurements or a second champion source."""
    runtime = Artifact("runtime/source.zip", "ab" * 32, 1, "source")
    previous_manifest = ArtifactManifest((runtime,))
    raw = valid_spec()
    raw["artifact_manifest_sha256"] = previous_manifest.digest
    previous = parse_track_spec(raw)
    winner = scored_challenge(1, "alice", score=0.8, incumbent_score=0.7)
    if request.param:
        winner.update(model_source_fields(None, None,
                                          PrivateModelSubmission("ef" * 32, "ab" * 32)))
    history = resolve_tournament(
        [winner], track_id=previous.track_id, track_spec_digest=previous.digest, netuid=7,
        minimum_gain=previous.full_video.minimum_gain,
        initial_champion=genesis_champion(previous),
    ).matches
    body = canonical_json(history[0].document)
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + history[0].object_name,
        hashlib.sha256(body).hexdigest(), len(body), history[0].object_name,
    ), runtime))
    raw.update(track_id="maintenance", track_start_block=300,
               artifact_manifest_sha256=manifest.digest)
    raw["full_video"]["minimum_gain"] = 0.5  # Old win must not be rejudged.
    current = parse_track_spec(raw)
    chain = WeightChainState(300, {"genesis-hotkey": 0, "alice": 3}, 1, 1.0, 1, {})
    client = MemoryS3()
    repository = S3Repository(client, bucket="test", prefix="isolated/maintenance")
    original = build_genesis_artifacts(previous, chain, version_key=0,
                                       runtime_manifest=previous_manifest.encode())
    tournament = resolve_tournament(
        [winner], track_id=previous.track_id, track_spec_digest=previous.digest, netuid=7,
        minimum_gain=previous.full_video.minimum_gain,
        initial_champion=genesis_champion(previous),
    )
    original = replace(
        original,
        matches=((history[0].object_name, body),),
        summary=canonical_json(publication_module._summary_document(previous, 300, tournament)),
        weights=encode_weight_artifact(WeightArtifact(
            previous.track_id, previous.digest, 7, 0, 0, 300,
            WeightPlan(300, ("alice",), (3,), (1.0,)),
            match_sequence=1, terminal_match_sha256=hashlib.sha256(body).hexdigest(),
        )),
    )
    asyncio.run(repository.publish(original))
    return SimpleNamespace(
        previous=previous, track=current, manifest=manifest,
        chain=replace(chain, finalized_block=301),
        history=history, body=body, client=client, repository=repository,
    )


def test_maintenance_cold_load_and_publication_carry_original_champion(maintenance_history):
    case = maintenance_history
    # Startup may precede the new mutable pointer. Original match bytes alone are
    # sufficient, pinned in the existing manifest and read through the repository.
    ledger = asyncio.run(case.repository.load(case.track, runtime_manifest=case.manifest.encode()))
    assert ledger.matches == () and ledger.historical_matches == case.history
    artifacts = build_genesis_artifacts(
        case.track, case.chain, version_key=0, runtime_manifest=case.manifest.encode(),
        historical_matches=ledger.historical_matches,
    )
    assert decode_weight_artifact(artifacts.weights).plan.uids == (3,)
    summary = json.loads(artifacts.summary)
    assert summary["champion"] == case.history[-1].champion_after.public()
    assert summary["match_count"] == 0  # No cloned/relabelled historical matches.
    assert json.loads(artifacts.status)["participants"][-1]["champion"] is True
    asyncio.run(case.repository.publish(artifacts))
    cold = S3Repository(case.client, bucket="test", prefix="isolated/maintenance")
    restored = asyncio.run(cold.load(case.track))
    assert restored.historical_matches == case.history and restored.matches == ()
    assert case.client.objects[
        "isolated/maintenance/" + case.history[0].object_name
    ] == case.body
    assert not any("matches/maintenance/" in key for key in case.client.objects)


@pytest.mark.parametrize("private", [False, True], ids=["hf", "private"])
@pytest.mark.parametrize("kind", ["exact_loss", "full_loss", "win", "failure"])
def test_status_projects_original_historical_results_not_a_history_label(private, kind):
    previous = parse_track_spec(valid_spec())
    first = scored_challenge(1, "alice", score=0.8, incumbent_score=0.7)
    if kind == "failure":
        second = scored_challenge(2, "bob", state="PARTICIPANT_FAILED",
                                  error_code="INVALID_MODEL")
    else:
        second = scored_challenge(2, "bob", score=0.9, incumbent_score=0.8,
                                  paired_gain=0.0 if kind == "exact_loss" else 0.1)
        if kind == "full_loss":
            second.update(final_score=0.7, paired_gain_lower_bound=-0.1)
            second["stages"][-1].update(score=0.7, paired_gain_lower_bound=-0.1)
    if private:
        for index, item in enumerate((first, second), 1):
            item.update(model_source_fields(
                None, None, PrivateModelSubmission(f"{index:064x}", "ab" * 32),
            ))
    history = resolve_tournament(
        [first, second], track_id=previous.track_id, track_spec_digest=previous.digest,
        netuid=previous.netuid, minimum_gain=previous.full_video.minimum_gain,
        initial_champion=genesis_champion(previous),
    ).matches
    original = tuple(canonical_json(match.document) for match in history)
    raw = valid_spec()
    raw.update(track_id="maintenance", track_start_block=300)
    current = parse_track_spec(raw)
    champion = historical_champion(history, genesis_champion(current))
    challenges = [
        {**challenge_from_match(match.document), "track_spec_digest": previous.digest,
         "attempts": 0, "created_at_ms": 0}
        for match in history
    ]
    competitors = [{**item, "uid": index + 3} for index, item in enumerate(challenges)]
    status = publication_module._status_document(
        current, 301, Tournament((), champion, ()), competitors, challenges,
        system_matches=[], historical_matches=history,
    )
    rows = {row["challenge_id"]: row for row in status["participants"]}
    for match in history:
        row = rows[match.challenge_id]
        assert row["outcome"] == match.outcome
        assert row["match_sequence"] == match.sequence
        assert row["score"] == match.final_score
        assert row["score_gain"] == match.score_gain
        assert row["incumbent_score"] == match.document["evaluation"]["incumbent_score"]
        assert row["track_spec_digest"] == previous.digest
        assert row["attempts"] is None and row["queue_position"] is None
        assert row["champion"] == (champion.challenge_id == match.challenge_id)
    assert "HISTORICAL" not in json.dumps(status)
    assert tuple(canonical_json(match.document) for match in history) == original


def test_status_historical_champion_without_active_commitment_keeps_original_result(
    maintenance_history,
):
    case = maintenance_history
    status = json.loads(build_genesis_artifacts(
        case.track, case.chain, version_key=0, runtime_manifest=case.manifest.encode(),
        historical_matches=case.history,
    ).status)
    match = case.history[-1]
    row = next(row for row in status["participants"] if row["challenge_id"] == match.challenge_id)
    assert row["outcome"] == match.outcome
    assert row["match_sequence"] == match.sequence
    assert row["state"] == match.state
    assert row["score"] == match.final_score
    assert row["score_gain"] == match.score_gain
    assert row["incumbent_score"] == match.document["evaluation"]["incumbent_score"]
    assert row["champion"] is True


@pytest.mark.parametrize("legacy_public", [False, True])
def test_cold_history_preserves_private_loser_archive_policy_without_reading_its_model(
    legacy_public, tmp_path,
):
    runtime = Artifact("runtime/source.zip", "ab" * 32, 1, "source")
    previous = parse_track_spec(valid_spec())
    item = scored_challenge(1, "alice", score=0.7, incumbent_score=0.7)
    private = PrivateModelSubmission("ef" * 32, "ab" * 32)
    item.update(model_source_fields(None, None, private))
    record = replace(submission_at(101, hotkey="alice"),
                     payload=private.encode(), repo_id=None, revision=None)
    item.update(challenge_id=challenge_id(record.submission_id), submission_id=record.submission_id)
    match = resolve_tournament(
        [item], track_id=previous.track_id, track_spec_digest=previous.digest,
        netuid=previous.netuid, minimum_gain=previous.full_video.minimum_gain,
        initial_champion=genesis_champion(previous),
    ).matches[0]
    document = json.loads(canonical_json(match.document))
    if legacy_public:
        document["evaluation"]["model_object"] = (
            f"models/{previous.track_id}/{'ab' * 32}/manifest.json"
        )
    original = canonical_json(document)
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + match.object_name, hashlib.sha256(original).hexdigest(),
        len(original), match.object_name,
    ), runtime))
    raw = valid_spec()
    raw.update(track_id="maintenance", track_start_block=300,
               artifact_manifest_sha256=manifest.digest)
    track = parse_track_spec(raw)
    client = MemoryS3()
    client.objects[f"isolated/history/{match.object_name}"] = original
    client.objects[f"isolated/history/runtime/{manifest.digest}/manifest.json"] = manifest.encode()
    get = client.get_object
    def read(**kwargs):
        assert "/models/" not in kwargs["Key"], "historical losing payload must not be required"
        return get(**kwargs)
    client.get_object = read
    cold = S3Repository(client, bucket="test", prefix="isolated/history")
    ledger = asyncio.run(cold.load(track))
    restored = ledger.historical_matches[0]
    assert canonical_json(restored.document) == original
    assert restored.outcome == "CHAMPION_RETAINED"
    assert restored.champion_after == match.champion_after
    assert ledger.matches == () and client.calls == []
    # Synthetic finalized input exercises the original DB restore and opportunity
    # gate; this is not a real chain transaction or cloud validation.
    with ChallengeStore(tmp_path / "cold.db", track_spec_digest=track.digest) as store:
        assert store.restore_public_matches(
            ledger.historical_matches, submissions={record.submission_id: record},
        ) == 1
        store.advance_chain_cursor(300, "0x" + "12" * 32)
        row = store.public_snapshot()["challenges"][0]
        assert row["state"] == "SUCCEEDED" and row["block_height"] == record.position.block
        assert row["track_spec_digest"] == previous.digest
        assert row["private_upload"]["upload_id"] == private.upload_id
        with pytest.raises(ValueError, match="formal submission"):
            store.require_upload_eligible("alice", uid=3, finalized_block=300)
    assert canonical_json(restored.document) == original


@pytest.mark.parametrize("state", ["current", "inherited", "combined", "staged"])
def test_maintenance_collects_original_history_from_recover_not_a_manual_match_list(
    maintenance_history, state,
):
    case = maintenance_history
    expected = tuple(artifact for artifact in case.manifest.artifacts
                     if artifact.logical_name.startswith(HISTORICAL_MATCHES_PREFIX))
    if state in {"inherited", "combined"}:
        artifacts = build_genesis_artifacts(
            case.track, case.chain, version_key=0, runtime_manifest=case.manifest.encode(),
            historical_matches=case.history,
        )
        if state == "combined":
            failed = scored_challenge(2, "bob", state="PARTICIPANT_FAILED",
                                      error_code="INVALID_MODEL")
            failed["block_height"] = 301
            tournament = resolve_tournament(
                [failed], track_id=case.track.track_id, track_spec_digest=case.track.digest,
                netuid=case.track.netuid, minimum_gain=case.track.full_video.minimum_gain,
                initial_champion=genesis_champion(case.track), historical_matches=case.history,
            )
            match = tournament.matches[0]
            body = canonical_json(match.document)
            artifacts = replace(
                artifacts, matches=((match.object_name, body),),
                summary=canonical_json(publication_module._summary_document(
                    case.track, 301, tournament,
                )),
                weights=encode_weight_artifact(replace(
                    decode_weight_artifact(artifacts.weights), match_sequence=1,
                    terminal_match_sha256=hashlib.sha256(body).hexdigest(),
                )),
            )
            expected = tuple(sorted((*expected, Artifact(
                HISTORICAL_MATCHES_PREFIX + match.object_name, hashlib.sha256(body).hexdigest(),
                len(body), match.object_name,
            )), key=lambda artifact: artifact.logical_name))
        asyncio.run(case.repository.publish(artifacts))
    elif state == "staged":
        # A valid immutable tail may exist before its mutable weight commit.
        empty = build_genesis_artifacts(case.previous, case.chain, version_key=0)
        case.client.objects["isolated/maintenance/summary.json"] = empty.summary
        case.client.objects["isolated/maintenance/weights.json"] = empty.weights
    client = case.client
    original = dict(client.objects)
    puts = list(client.calls)
    cold = S3Repository(client, bucket="test", prefix="isolated/maintenance")
    assert asyncio.run(cold.collect_history_artifacts()) == expected
    assert client.objects == original and client.calls == puts
    assert client.objects["isolated/maintenance/" + case.history[0].object_name] == case.body


@pytest.mark.parametrize("damage", ["missing_match", "bad_fence", "moving_fence"])
def test_automatic_history_collection_rejects_missing_or_unfenced_facts(
    maintenance_history, damage,
):
    case = maintenance_history
    if damage == "missing_match":
        case.client.objects.pop("isolated/maintenance/" + case.history[0].object_name)
    elif damage == "bad_fence":
        key = "isolated/maintenance/weights.json"
        weight = decode_weight_artifact(case.client.objects[key])
        case.client.objects[key] = encode_weight_artifact(
            replace(weight, terminal_match_sha256="00" * 32),
        )
    else:
        get = case.client.get_object
        reads = 0
        def changing(**kwargs):
            nonlocal reads
            if kwargs["Key"].endswith("/weights.json"):
                reads += 1
                if reads == 3:
                    weight = decode_weight_artifact(case.client.objects[kwargs["Key"]])
                    case.client.objects[kwargs["Key"]] = encode_weight_artifact(
                        replace(weight, version_key=weight.version_key + 1),
                    )
            return get(**kwargs)
        case.client.get_object = changing
    writes = len(case.client.calls)
    cold = S3Repository(case.client, bucket="test", prefix="isolated/maintenance")
    with pytest.raises(PublicArtifactError):
        asyncio.run(cold.collect_history_artifacts())
    assert len(case.client.calls) == writes


def test_maintenance_rejects_accidental_genesis_reset_before_pointers(maintenance_history):
    case = maintenance_history
    before = {key: value for key, value in case.client.objects.items()
              if key.endswith(("track.json", "weights.json", "summary.json", "status.json"))}
    forgotten = build_genesis_artifacts(
        case.track, case.chain, version_key=0, runtime_manifest=case.manifest.encode(),
    )
    with pytest.raises(PublicArtifactError, match="reset its historical champion"):
        asyncio.run(case.repository.publish(forgotten))
    assert all(case.client.objects[key] == body for key, body in before.items())


@pytest.mark.parametrize("excluded", [False, True])
def test_maintenance_prestart_owner_loop_keeps_champion_and_existing_eligibility(
    maintenance_history, tmp_path, monkeypatch, excluded,
):
    from scoring.spec import track_spec_dict
    from validator import run as run_module

    case = maintenance_history
    raw = track_spec_dict(case.track)
    raw["track_start_block"] = 400
    track = parse_track_spec(raw)
    case.client.objects[
        f"isolated/maintenance/runtime/{case.manifest.digest}/manifest.json"
    ] = case.manifest.encode()
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            return None
        async def finalized_block(self):
            return 301
    async def seed(_client, store, **kwargs):
        assert kwargs == {"netuid": 7, "before_block": 301}
        if excluded:
            store.seed_legacy_exclusions({"alice": 300})
    async def chain_state(_client, **kwargs):
        assert kwargs == {"netuid": 7, "mechid": 0, "finalized_block": 301}
        return case.chain
    monkeypatch.setattr(run_module, "load_track_spec", lambda _path: track)
    monkeypatch.setattr(run_module, "verify_initial_champion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_module.bt, "Client", lambda *_args, **_kwargs: Client())
    monkeypatch.setattr(run_module, "seed_legacy_exclusions", seed)
    monkeypatch.setattr(run_module, "read_weight_chain_state", chain_state)
    (tmp_path / "exact").mkdir()
    asyncio.run(run_module.run_evaluator(
        track_spec="unused", database=tmp_path / "fresh.db", model_cache=tmp_path / "models",
        exact_cache=tmp_path / "exact", initial_model=tmp_path / "unused-baseline",
        network="test", archive_endpoints=None, worker_id="fixture", publisher=case.repository,
        once=True,
    ))
    weights = decode_weight_artifact(case.client.objects["isolated/maintenance/weights.json"])
    assert weights.plan.uids == ((0,) if excluded else (3,))
    summary = json.loads(case.client.objects["isolated/maintenance/summary.json"])
    assert summary["champion"] == case.history[-1].champion_after.public()
    assert summary["match_count"] == 0


@pytest.mark.parametrize("uncommitted_tail", [False, True])
def test_maintenance_cannot_drop_history_from_manifest(maintenance_history, uncommitted_tail):
    case = maintenance_history
    if uncommitted_tail:
        pending = build_genesis_artifacts(case.previous, case.chain, version_key=0)
        case.client.objects["isolated/maintenance/summary.json"] = pending.summary
        case.client.objects["isolated/maintenance/weights.json"] = pending.weights
    manifest = ArtifactManifest((Artifact("runtime/source.zip", "ab" * 32, 1, "source"),))
    raw = valid_spec()
    raw.update(track_id="maintenance", track_start_block=300,
               artifact_manifest_sha256=manifest.digest)
    track = parse_track_spec(raw)
    forgotten = build_genesis_artifacts(track, case.chain, version_key=0,
                                       runtime_manifest=manifest.encode())
    with pytest.raises(PublicArtifactError, match="drop, replace or invent"):
        asyncio.run(case.repository.publish(forgotten))
    assert json.loads(case.client.objects["isolated/maintenance/track.json"])[
        "track_spec_digest"
    ] == case.previous.digest


@pytest.mark.parametrize("damage", ["missing", "sha", "identity", "subnet", "sequence", "link"])
@pytest.mark.parametrize("operation", ["recover", "load"])
def test_maintenance_history_fails_closed(maintenance_history, damage, operation):
    case = maintenance_history
    name = case.history[0].object_name
    body = case.body
    if damage in {"identity", "subnet", "sequence", "link"}:
        document = json.loads(body)
        if damage == "identity":
            document["track_id"] = case.track.track_id
        elif damage == "subnet":
            document["netuid"] = 8
        elif damage == "sequence":
            document["sequence"] = 2
            name = name.replace("000000000001", "000000000002")
        else:
            document["incumbent_before"]["hotkey"] = "other-baseline"
        body = canonical_json(document)
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + name,
        hashlib.sha256(body).hexdigest(), len(body), name,
    ),))
    raw = valid_spec()
    raw.update(track_id="maintenance", track_start_block=300,
               artifact_manifest_sha256=manifest.digest)
    track = parse_track_spec(raw)
    key = "isolated/maintenance/" + name
    if damage == "missing":
        case.client.objects.pop(key)
    else:
        case.client.objects[key] = body if damage != "sha" else body + b" "
    cold = S3Repository(case.client, bucket="test", prefix="isolated/maintenance")
    with pytest.raises(PublicArtifactError, match="historical"):
        asyncio.run(getattr(cold, operation)(track, runtime_manifest=manifest.encode()))


def test_weight_artifact_has_one_strict_canonical_representation() -> None:
    encoded = encode_weight_artifact(weight_artifact())
    decoded = decode_weight_artifact(encoded)

    assert encoded == (
        b'{"match_sequence":2,"mechid":2,"netuid":7,"source_block":120,'
        b'"terminal_match_sha256":"' + b"cd" * 32
        + b'","track_aborted":false,"track_id":"track","track_spec_digest":"'
        + b"ab" * 32
        + b'","version_key":9,"weights":[{"hotkey":"alice","uid":1,"weight":0.6},'
        b'{"hotkey":"bob","uid":4,"weight":0.4}]}'
    )
    assert decoded == weight_artifact()
    assert encoded == encode_weight_artifact(decoded)
    with pytest.raises(PublicArtifactError, match="duplicate"):
        decode_weight_artifact(
            b'{"track_id":"a","track_id":"b","track_spec_digest":"'
            + b"ab" * 32
            + b'","netuid":7,"mechid":0,"version_key":0,"source_block":1,"weights":null}'
        )


@pytest.mark.parametrize("positions", [
    ((200, 1, 9), (200, 2, 1)),
    ((200, 1, 1), (200, 1, 2)),
])
@pytest.mark.parametrize("reverse_input", [False, True])
def test_status_queue_positions_follow_chain_order_not_challenge_id(positions, reverse_input):
    track = parse_track_spec(valid_spec())
    baseline = genesis_champion(track)
    challenges, competitors = [], []
    for index, (position, identifier) in enumerate(
        zip(positions, ("ff" * 32, "00" * 32), strict=True), 1,
    ):
        item = scored_challenge(index, f"miner-{index}", state="QUEUED")
        item.update(challenge_id=identifier, block_height=position[0],
                    extrinsic_index=position[1], event_index=position[2],
                    created_at_ms=1, attempts=0)
        challenges.append(item)
        competitors.append({**item, "uid": 3 - index})
    if reverse_input:
        challenges.reverse()
        competitors.reverse()
    store = SimpleNamespace(
        chain_cursor=lambda: (211, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": challenges},
        active_competitors=lambda _mapping: competitors,
        legacy_excluded_hotkeys=lambda: frozenset(),
    )
    chain = WeightChainState(211, {baseline.hotkey: 0}, 1, 1.0, 10, {baseline.hotkey: 0})

    artifacts = build_public_artifacts(store, track, chain, version_key=0)

    status = json.loads(artifacts.status)
    rows = {row["challenge_id"]: row for row in status["participants"]}
    assert rows["ff" * 32]["queue_position"] == 1
    assert rows["00" * 32]["queue_position"] == 2
    assert all("extrinsic_index" not in row and "event_index" not in row for row in rows.values())
    assert status["champion"] == baseline.public()
    assert decode_weight_artifact(artifacts.weights).plan == WeightPlan(
        211, (baseline.hotkey,), (0,), (1.0,),
    )
    assert all(item["state"] == "QUEUED" and item["attempts"] == 0 for item in challenges)


def test_confirmation_status_exposes_exact_aggregate_without_raw_inputs():
    from validator.tournament import Tournament

    track = parse_track_spec(valid_spec())
    baseline = genesis_champion(track)
    allocation = {"plan_digest": "ab" * 32, "ready_sha256": "cd" * 32}
    stage = {
        "stage": "SCORING", "score": 0.8, "score_block": 210,
        "incumbent_score": 0.7, "paired_gain_lower_bound": 0.06,
        "evaluation_batch": allocation, "prompt": "must stay private",
        "secret": "must stay private", "cases": ["not a safe aggregate"],
    }
    challenge = {
        "submission_id": "submission", "challenge_id": "model", "state": "CONFIRMING",
        "final_score": None, "incumbent_score": 0.7, "paired_gain_lower_bound": 0.06,
        "evaluation_batch": {"generated_block": 100}, "created_at_ms": 1,
        "attempts": 1, "error_code": None, "file_sha256": "e5" * 32,
        "model_digest": "f6" * 32, "stages": [stage],
    }
    status = publication_module._status_document(
        track, 211, Tournament((), baseline, ()),
        [{"submission_id": "submission", "uid": 7, "hotkey": "miner", "block_height": 200,
          "extrinsic_index": 0, "event_index": 0}],
        [challenge], system_matches=[],
    )
    miner = next(row for row in status["participants"] if row["challenge_id"] == "model")
    assert miner["score"] == 0.8 and miner["score_gain"] == 0.06
    assert miner["file_sha256"] == "e5" * 32 and miner["model_digest"] == "f6" * 32
    assert miner["stages"] == [{k: v for k, v in stage.items()
                                if k not in {"prompt", "secret", "cases"}}]
    assert miner["match_sequence"] is miner["outcome"] is None
    assert miner["champion"] is False
    assert status["champion"] == baseline.public()
    assert "must stay private" not in json.dumps(status)


def test_public_repository_rejects_active_track_pointer_rollback() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})
    current = PublicArtifacts(
        source_block=10,
        track=track,
        summary=encoded(
            {"track_id": "track", "source_block": 10, "match_count": 1, "marker": "current"}
        ),
        weights=pointer_weights(10),
        matches=(),
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(current))

    with pytest.raises(PublicArtifactError, match="rollback"):
        asyncio.run(
            repository.publish(
                PublicArtifacts(
                    source_block=9,
                    track=track,
                    summary=encoded(
                        {
                            "track_id": "track",
                            "source_block": 9,
                            "match_count": 1,
                            "marker": "older",
                        }
                    ),
                    weights=pointer_weights(9),
                    matches=(),
                )
            )
        )
    with pytest.raises(PublicArtifactError, match="two public snapshots"):
        asyncio.run(
            repository.publish(
                PublicArtifacts(
                    source_block=10,
                    track=track,
                    summary=encoded(
                        {
                            "track_id": "track",
                            "source_block": 10,
                            "match_count": 1,
                            "marker": "conflict",
                        }
                    ),
                    weights=pointer_weights(10),
                    matches=(),
                )
            )
        )


def test_weights_reject_a_higher_block_with_a_stale_match_prefix_before_mutation() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})
    current = PublicArtifacts(
        source_block=10,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 10, "match_count": 2}),
        weights=pointer_weights(
            10,
            match_sequence=2,
            terminal_match_sha256="22" * 32,
        ),
        matches=(),
    )
    stale = PublicArtifacts(
        source_block=11,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 11, "match_count": 1}),
        weights=pointer_weights(
            11,
            match_sequence=1,
            terminal_match_sha256="11" * 32,
        ),
        matches=(),
    )
    conflicting = PublicArtifacts(
        source_block=12,
        track=track,
        summary=encoded({"track_id": "track", "source_block": 12, "match_count": 2}),
        weights=pointer_weights(
            12,
            match_sequence=2,
            terminal_match_sha256="33" * 32,
        ),
        matches=(),
    )
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(current))
    weights_before = client.objects["tracks/test/weights.json"]
    summary_before = client.objects["tracks/test/summary.json"]

    with pytest.raises(PublicArtifactError, match="rollback"):
        asyncio.run(repository.publish(stale))
    assert client.objects["tracks/test/weights.json"] == weights_before
    assert client.objects["tracks/test/summary.json"] == summary_before

    with pytest.raises(PublicArtifactError, match="two histories"):
        asyncio.run(repository.publish(conflicting))
    assert client.objects["tracks/test/weights.json"] == weights_before
    assert client.objects["tracks/test/summary.json"] == summary_before


def test_public_repository_write_probe_checks_visibility_and_cleans_up() -> None:
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")

    repository.verify_writes()

    assert client.objects == {}


def test_single_public_publisher_supports_backend_without_condition_enforcement() -> None:
    class UnsafeS3(MemoryS3):
        def put_object(self, **kwargs):
            kwargs.pop("IfMatch", None)
            kwargs.pop("IfNoneMatch", None)
            return super().put_object(**kwargs)

    repository = S3Repository(UnsafeS3(), bucket="public-data", prefix="tracks/test")

    repository.verify_writes()


def test_public_repository_rejects_writes_that_cannot_be_read_back() -> None:
    class DropsWrites(MemoryS3):
        def put_object(self, **kwargs):
            return {}

    client = DropsWrites()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    with pytest.raises(PublicArtifactError, match="creation was not durable"):
        repository.verify_writes()
    assert client.objects == {}


def test_public_pointer_write_is_checked_after_upload() -> None:
    class CorruptsWrites(MemoryS3):
        def put_object(self, **kwargs):
            result = super().put_object(**kwargs)
            self.objects[kwargs["Key"]] = b'{"incorrect":true}'
            return result

    repository = S3Repository(CorruptsWrites(), bucket="public-data", prefix="tracks/test")
    with pytest.raises(PublicArtifactError, match="write verification"):
        repository._put_mutable("status.json", b'{"source_block":1}')


def test_one_writer_lock_covers_all_database_names_in_the_deployment(tmp_path) -> None:
    first = S3Repository(MemoryS3(), bucket="public-data", prefix="tracks/test")
    second = S3Repository(MemoryS3(), bucket="public-data", prefix="tracks/test")
    first.acquire_writer_lock(tmp_path / "first.db")
    try:
        first.acquire_writer_lock(tmp_path / "first.db")
        with pytest.raises(PublicArtifactError, match="another evaluator"):
            second.acquire_writer_lock(tmp_path / "second.db")
        first.close()
        second.acquire_writer_lock(tmp_path / "second.db")
    finally:
        first.close()
        second.close()


def test_concurrent_stale_summary_write_is_fenced() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    track = encoded({"spec": {"track_id": "track"}, "track_spec_digest": "ab" * 32})

    def artifacts(block):
        return PublicArtifacts(
            source_block=block,
            track=track,
            summary=encoded(
                {
                    "track_id": "track",
                    "source_block": block,
                    "match_count": 1,
                }
            ),
            weights=pointer_weights(block),
            matches=(),
        )

    client = MemoryS3()
    stale = S3Repository(client, bucket="public-data", prefix="tracks/test")
    current = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(current.publish(artifacts(10)))

    def interleave(arguments):
        if arguments["Key"] != "tracks/test/summary.json" or "IfMatch" not in arguments:
            return
        client.before_put = None
        current._put_mutable("summary.json", artifacts(12).summary)

    client.before_put = interleave
    with pytest.raises(PublicArtifactError, match="fenced"):
        asyncio.run(stale.publish(artifacts(11)))

    assert json.loads(client.objects["tracks/test/summary.json"])["source_block"] == 12


def test_pointer_from_previous_track_cannot_overwrite_active_track() -> None:
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    old_track = encoded(
        {
            "spec": {"track_id": "old", "track_start_block": 10},
            "track_spec_digest": "ab" * 32,
        }
    )
    new_track = encoded(
        {
            "spec": {"track_id": "new", "track_start_block": 20},
            "track_spec_digest": "cd" * 32,
        }
    )
    old_summary = encoded({"track_id": "old", "source_block": 11, "match_count": 0})
    new_summary = encoded({"track_id": "new", "source_block": 21, "match_count": 0})
    stale_summary = encoded({"track_id": "old", "source_block": 22, "match_count": 0})

    repository._put_mutable("track.json", old_track)
    repository._put_mutable("summary.json", old_summary)
    repository._put_mutable("track.json", new_track)
    repository._put_mutable("summary.json", new_summary)

    with pytest.raises(PublicArtifactError, match="inactive track"):
        repository._put_mutable("summary.json", stale_summary)


def test_fetch_weight_artifact_is_https_only_and_size_bounded() -> None:
    payload = encode_weight_artifact(weight_artifact())

    async def run() -> WeightArtifact:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_weight_artifact("https://public.example/weights.json", client=client)

    assert asyncio.run(run()) == weight_artifact()

    async def oversized() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"x" * 20))
        async with httpx.AsyncClient(transport=transport) as client:
            await fetch_weight_artifact(
                "https://public.example/weights.json", client=client, max_bytes=10
            )

    with pytest.raises(PublicArtifactError, match="byte limit"):
        asyncio.run(oversized())
    with pytest.raises(ValueError, match="HTTPS"):
        validate_public_weights_url("http://public.example/weights.json")


def test_weight_setter_accepts_a_digest_bound_legacy_track_identity() -> None:
    spec = {
        "track_id": "veda-sn120-v1",
        "netuid": 120,
        "mechid": 1,
        "legacy_scoring": {"format": "retired-v1"},
    }
    digest = hashlib.sha256(canonical_json(spec)).hexdigest()
    payload = canonical_json({"spec": spec, "track_spec_digest": digest})

    async def run():
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_track_identity(
                "https://public.example/track.json",
                client=client,
            )

    identity = asyncio.run(run())
    assert identity.track_id == "veda-sn120-v1"
    assert identity.track_spec_digest == digest
    assert identity.netuid == 120
    assert identity.mechid == 1


@pytest.mark.parametrize("prefix", ["../track", "track//main", "track name", "track\nmain"])
def test_storage_rejects_unsafe_object_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        S3Repository(object(), bucket="public-data", prefix=prefix)


def test_storage_factory_requires_complete_safe_environment(monkeypatch) -> None:
    for name in ("STORAGE_URL", "ACCESS_KEY", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="required"):
        S3Repository.from_environment(bucket="public-data", prefix="veda/track")

    monkeypatch.setenv("STORAGE_URL", "http://storage.example/s3")
    monkeypatch.setenv("ACCESS_KEY", "access")
    monkeypatch.setenv("SECRET_ACCESS_KEY", "secret")
    with pytest.raises(ValueError, match="HTTPS"):
        S3Repository.from_environment(bucket="public-data", prefix="veda/track")


def test_staging_uploads_only_immutable_facts_without_any_pointer() -> None:
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="veda/track")
    artifacts = PublicArtifacts(
        120, pointer_track(), b"summary", pointer_weights(120),
        (("matches/track/000000000001.json", b"match"),),
    )
    asyncio.run(publisher.stage(artifacts))
    assert client.objects == {"veda/track/matches/track/000000000001.json": b"match"}
    assert client.calls[0]["IfNoneMatch"] == "*"
    asyncio.run(publisher.publish(artifacts))
    assert client.objects["veda/track/weights.json"] == artifacts.weights
    assert len([call for call in client.calls if call["Key"].endswith("000000000001.json")]) == 1


def test_storage_publishes_immutable_facts_before_summary_and_weights() -> None:
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="veda/track")
    artifacts = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (("matches/track/000000000001.json", b"match"),),
    )
    asyncio.run(publisher.publish(artifacts))

    writes = [call for call in client.calls if "Key" in call and "Body" in call]
    assert [call["Key"] for call in writes] == [
        "veda/track/matches/track/000000000001.json",
        "veda/track/track.json",
        "veda/track/weights.json",
        "veda/track/summary.json",
    ]
    assert writes[0]["CacheControl"].endswith("immutable")
    assert writes[1]["CacheControl"] == "no-store, max-age=0"
    assert writes[2]["CacheControl"] == writes[3]["CacheControl"] == "no-store, max-age=0"

    client.calls.clear()
    asyncio.run(publisher.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == []

    conflicting = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (("matches/track/000000000001.json", b"different"),),
    )
    with pytest.raises(PublicArtifactError, match="conflicts"):
        asyncio.run(publisher.publish(conflicting))


def test_runtime_manifest_is_content_addressed_and_published_before_track() -> None:
    payload = ArtifactManifest((Artifact("runtime/source.zip", "ab" * 32, 1, "source"),)).encode()
    digest = hashlib.sha256(payload).hexdigest()
    raw = valid_spec()
    raw["artifact_manifest_sha256"] = digest
    track = parse_track_spec(raw)
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    artifacts = build_genesis_artifacts(
        track, WeightChainState(120, {track.initial_champion.hotkey: 0}, 1, 1.0, 1, {}),
        version_key=0, runtime_manifest=payload,
    )

    asyncio.run(publisher.publish(artifacts))

    writes = [call["Key"] for call in client.calls if "Body" in call]
    assert writes[:2] == [
        f"tracks/test/runtime/{digest}/manifest.json",
        "tracks/test/track.json",
    ]


def test_storage_publishes_large_files_as_bounded_verified_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    source = tmp_path / "batch.npz"
    source.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    artifacts = PublicArtifacts(
        120,
        pointer_track(),
        b"summary",
        pointer_weights(120),
        (),
        files=(PublicFile(f"batches/{digest}/manifest.json", source, digest),),
    )

    asyncio.run(publisher.publish(artifacts))

    prefix = f"tracks/test/batches/{digest}/"
    assert sorted(key for key in client.objects if key.startswith(prefix)) == [
        prefix + "manifest.json",
        prefix + "part-00000",
        prefix + "part-00001",
        prefix + "part-00002",
    ]

    source.unlink()
    client.calls.clear()
    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(restarted.publish(artifacts))
    assert [call["Key"] for call in client.calls if "Body" in call] == []

    del client.objects[prefix + "part-00001"]
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        asyncio.run(
            S3Repository(client, bucket="public-data", prefix="tracks/test").publish(artifacts)
        )


@pytest.mark.parametrize("operation", ["load", "existing_upload", "resumed_upload"])
@pytest.mark.parametrize("corruption", ["same_size", "truncated", "oversized", "complete_digest"])
def test_public_file_verification_reads_actual_bytes(tmp_path, monkeypatch, operation, corruption):
    monkeypatch.setattr(chunks, "CHUNK_BYTES", 4)
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"abcdefghij")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    name = f"models/{digest}/manifest.json"
    artifact = PublicFile(name, source, digest)
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    repository._put_file_immutable(artifact)
    manifest_key = f"tracks/test/{name}"
    part_key = f"tracks/test/{chunks.chunk_object_name(name, 'part-00001')}"
    if corruption == "complete_digest":
        manifest = chunks.ChunkManifest.decode(client.objects[manifest_key])
        digest = "ff" * 32
        client.objects[manifest_key] = chunks.ChunkManifest(
            digest, manifest.size, manifest.parts,
        ).encode()
        artifact = PublicFile(name, source, digest)
    else:
        client.objects[part_key] = {
            "same_size": b"XXXX", "truncated": b"X", "oversized": b"XXXXX",
        }[corruption]
        # Length and SHA metadata can describe the original bytes even when
        # object content is corrupted; none of it is proof of integrity.
        assert client.metadata[part_key]["sha256"] == hashlib.sha256(b"efgh").hexdigest()

    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    if operation == "resumed_upload":
        if corruption == "complete_digest":
            # Preserve individually valid chunks under a false complete digest.
            monkeypatch.setattr(
                publication_module,
                "build_chunk_manifest",
                lambda _path, **_kwargs: chunks.ChunkManifest(
                    digest, manifest.size, manifest.parts
                ),
            )
        del client.objects[manifest_key]
    with pytest.raises(PublicArtifactError):
        if operation == "load":
            restarted._verify_public_file(name, digest, 10)
        else:
            restarted._put_file_immutable(artifact)
    assert name not in restarted._confirmed_immutable
    if operation == "resumed_upload":
        assert manifest_key not in client.objects


def test_public_upload_verifies_bytes_before_committing_manifest(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"model")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"models/{digest}/manifest.json"
    client = MemoryS3()

    def corrupt_upload(arguments):
        if arguments["Key"].endswith("part-00000"):
            arguments["Body"] = b"other"

    client.before_put = corrupt_upload
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    with pytest.raises(PublicArtifactError, match="conflicts with manifest"):
        repository._put_file_immutable(PublicFile(name, path, digest))
    assert f"tracks/test/{name}" not in client.objects
    assert name not in repository._confirmed_immutable


def test_public_chunk_verification_streams_and_closes_bodies(tmp_path):
    path = tmp_path / "batch.npz"
    path.write_bytes(b"x" * 150_000)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"batches/{digest}/manifest.json"
    bodies = []

    class BoundedBody(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64 * 1024
            return super().read(size)

    class StreamingS3(MemoryS3):
        def get_object(self, *, Bucket, Key):
            response = super().get_object(Bucket=Bucket, Key=Key)
            if Key.endswith("part-00000"):
                response["Body"] = BoundedBody(self.objects[Key])
                bodies.append(response["Body"])
            return response

    client = StreamingS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    repository._put_file_immutable(PublicFile(name, path, digest))
    S3Repository(client, bucket="public-data", prefix="tracks/test")._verify_public_file(
        name, digest, path.stat().st_size,
    )
    assert len(bodies) == 2
    assert all(body.closed for body in bodies)


def test_invalid_model_restore_cannot_publish_an_archive(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"broken")
    digest = hashlib.sha256(b"correct").hexdigest()
    name = f"models/{digest}/manifest.json"
    calls = []

    def restore():
        calls.append(True)
        path.write_bytes(b"also broken")
        return path

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    for _ in range(2):
        with pytest.raises(PublicArtifactError, match="restored public file digest changed"):
            repository._put_file_immutable(PublicFile(name, path, digest, restore))
    assert len(calls) == 2
    assert not client.objects
    assert name not in repository._confirmed_immutable


@pytest.mark.parametrize("private", [False, True])
def test_storage_ledger_rebuilds_and_verifies_every_committed_match(tmp_path, private) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    snapshot = {
        "challenges": [
            two_stage_challenge(
                {
                    "challenge_id": "challenge",
                    "submission_id": "submission",
                    "hotkey": "alice",
                    "repo_id": "owner/model",
                    "revision": "12" * 20,
                    "block_height": 100,
                    "extrinsic_index": 0,
                    "event_index": 0,
                    "state": "SUCCEEDED",
                    "error_code": None,
                    "file_sha256": model_sha256,
                    "model_digest": "cd" * 32,
                    "model_size_bytes": model.stat().st_size,
                    "final_score": 0.8,
                    "incumbent_score": 0.5,
                    "paired_gain_lower_bound": 0.3,
                    "evaluation_batch": {
                        "sha256": "de" * 32,
                        "generated_block": 104,
                        "plan_digest": "ef" * 32,
                        "seed_block_hash": "0x" + "ab" * 32,
                    },
                    "score_block": 110,
                    "stages": [],
                }
            )
        ]
    }
    reference = PrivateModelSubmission("ef" * 32, model_sha256, "cd" * 32) if private else None
    if reference is not None:
        snapshot["challenges"][0].update(model_source_fields(None, None, reference))
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _challenge_id: str(model),
    )
    chain = WeightChainState(120, {"alice": 4}, 1, 1.0, 10, {"alice": 0})
    artifacts = build_public_artifacts(store, track, chain, version_key=0)
    assert artifacts is not None

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="veda/track")
    asyncio.run(repository.publish(artifacts))
    ledger = asyncio.run(repository.load(track))

    assert ledger.source_block == 120
    assert ledger.required_finalized_block == 120
    assert len(ledger.matches) == 1
    assert ledger.matches[0].challenge_id == "challenge"
    assert ledger.matches[0].champion_after.private_submission == reference

    # Startup has the same immutable facts/fence, but must not read the model
    # or disclosed payloads before scanning and refreshing current weights.
    cold = S3Repository(client, bucket="public-data", prefix="veda/track")
    def blocked_bulk(*_args):
        pytest.fail("authoritative fact recovery attempted a bulk audit")
    cold._verify_model_archives = blocked_bulk
    cold._load_batch_disclosures = blocked_bulk
    facts = asyncio.run(cold.recover(track))
    assert facts.matches == ledger.matches
    assert facts.source_block == ledger.source_block
    assert facts.required_finalized_block == ledger.required_finalized_block
    assert facts.disclosed_plan_digests == frozenset()

    weights_key = "veda/track/weights.json"
    original_weights = client.objects[weights_key]
    wrong_fence = json.loads(original_weights)
    wrong_fence["terminal_match_sha256"] = "fa" * 32
    client.objects[weights_key] = canonical_json(wrong_fence)
    for operation in ("recover", "load"):
        with pytest.raises(PublicArtifactError, match="does not bind"):
            asyncio.run(getattr(
                S3Repository(client, bucket="public-data", prefix="veda/track"), operation,
            )(track))
    client.objects[weights_key] = original_weights

    summary_key = "veda/track/summary.json"
    original_summary = client.objects[summary_key]
    wrong_summary = json.loads(original_summary)
    wrong_summary["leaderboard_total"] += 1
    client.objects[summary_key] = canonical_json(wrong_summary)
    for operation in ("recover", "load"):
        with pytest.raises(PublicArtifactError, match="summary does not match"):
            asyncio.run(getattr(S3Repository(
                client, bucket="public-data", prefix="veda/track",
            ), operation)(track))
    client.objects[summary_key] = original_summary

    model_prefix = f"veda/track/models/{track.track_id}/{model_sha256}/"
    model_part = next(key for key in client.objects if key.startswith(model_prefix + "part-"))
    model_bytes = client.objects.pop(model_part)
    assert asyncio.run(S3Repository(
        client, bucket="public-data", prefix="veda/track",
    ).recover(track)).matches == ledger.matches
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").load(track))
    client.objects[model_part] = model_bytes

    original_match = client.objects.pop("veda/track/matches/veda-test-v1/000000000001.json")
    for operation in ("recover", "load"):
        with pytest.raises(PublicArtifactError, match="committed match is missing"):
            asyncio.run(getattr(S3Repository(
                client, bucket="public-data", prefix="veda/track",
            ), operation)(track))
    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = original_match

    client.objects["veda/track/matches/veda-test-v1/000000000001.json"] = b"{}"
    for operation in ("recover", "load"):
        with pytest.raises(PublicArtifactError, match="match is invalid"):
            asyncio.run(getattr(S3Repository(
                client, bucket="public-data", prefix="veda/track",
            ), operation)(track))


@pytest.mark.parametrize(
    "cache_state", ["valid", "missing", "forgotten", "same_size_corruption", "truncated"]
)
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("full_loss", [False, True])
def test_public_artifacts_archive_a_scored_challenger_that_does_not_win(
    tmp_path, cache_state, private, full_loss, monkeypatch,
) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "retained-challenger.safetensors"
    archived_bytes = b"retained challenger"
    model.write_bytes(archived_bytes)
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    batch = {
        "sha256": "de" * 32,
        "generated_block": 104,
        "plan_digest": "ef" * 32,
        "seed_block_hash": "0x" + "ab" * 32,
    }
    challenge = {
        "challenge_id": "challenge",
        "submission_id": "submission",
        "hotkey": "alice",
        "repo_id": "owner/model",
        "revision": "12" * 20,
        "block_height": 100,
        "extrinsic_index": 0,
        "event_index": 0,
        "state": "SUCCEEDED",
        "error_code": None,
        "file_sha256": model_sha256,
        "model_digest": "cd" * 32,
        "model_size_bytes": model.stat().st_size,
        "final_score": 0.49,
        "incumbent_score": 0.5,
        "paired_gain_lower_bound": -0.01,
        "evaluation_batch": batch,
        "score_block": 110,
        "stages": [
            {
                "stage": "SCORING",
                "score": 0.49,
                "score_block": 110,
                "incumbent_score": 0.5,
                "paired_gain_lower_bound": -0.01,
                "evaluation_batch": batch,
            }
        ],
    }
    if full_loss:
        challenge = two_stage_challenge(challenge)
        challenge["stages"][0].update(score=0.8, paired_gain_lower_bound=0.3)
    if private:
        challenge.update(model_source_fields(
            None, None, PrivateModelSubmission("ef" * 32, model_sha256, "cd" * 32),
        ))
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": [challenge]},
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _challenge_id: None if cache_state == "forgotten" else str(model),
    )
    chain = WeightChainState(
        120,
        {track.initial_champion.hotkey: 0, "alice": 4},
        1,
        1.0,
        10,
        {},
    )

    restored = []

    def restore(sha256, size_bytes):
        assert (sha256, size_bytes) == (model_sha256, len(archived_bytes))
        restored.append(sha256)
        model.write_bytes(archived_bytes)
        return model

    if cache_state in {"missing", "forgotten"}:
        model.unlink()
    elif cache_state == "same_size_corruption":
        model.write_bytes(b"X" * len(archived_bytes))
    elif cache_state == "truncated":
        model.write_bytes(b"short")
    artifacts = build_public_artifacts(
        store, track, chain, version_key=0, model_resolver=restore,
    )

    assert artifacts is not None
    assert json.loads(artifacts.matches[0][1])["outcome"] == "CHAMPION_RETAINED"
    if private:
        assert artifacts.files == () and restored == []
        document = json.loads(artifacts.matches[0][1])
        assert document["evaluation"]["model_object"] is None
        assert document["evaluation"]["file_sha256"] == model_sha256
        client = MemoryS3()
        repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
        asyncio.run(repository.publish(artifacts))
        assert not any("/models/" in key for key in client.objects)
        cold = S3Repository(client, bucket="public-data", prefix="tracks/test")
        monkeypatch.setattr(cold, "_verify_public_file", lambda *_args: pytest.fail(
            "private losing payload was requested from public storage",
        ))
        ledger = asyncio.run(cold.load(track))
        assert ledger.matches[0].document == document
        assert asyncio.run(cold.recover(track)).matches == ledger.matches
        before = dict(client.objects)
        with pytest.raises(PublicArtifactError, match="public model"):
            asyncio.run(repository.stage(replace(artifacts, files=(PublicFile(
                f"models/{track.track_id}/{model_sha256}/manifest.json", model, model_sha256,
            ),))))
        assert client.objects == before
        document["evaluation"]["model_object"] = (
            f"models/{track.track_id}/{model_sha256}/manifest.json"
        )
        altered = replace(artifacts, matches=((artifacts.matches[0][0], canonical_json(document)),))
        with pytest.raises(PublicArtifactError, match="public model"):
            asyncio.run(repository.stage(altered))
        assert client.objects == before
        return
    assert len(artifacts.files) == 1
    assert artifacts.files[0].object_name == (
        f"models/{track.track_id}/{model_sha256}/manifest.json"
    )
    assert artifacts.files[0].path == (None if cache_state == "forgotten" else model)

    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(repository.publish(artifacts))
    assert model.read_bytes() == archived_bytes
    assert len(restored) == (cache_state != "valid")
    assert asyncio.run(repository.load(track)).matches[0].document["outcome"] == "CHAMPION_RETAINED"

    # A fresh publisher verifies the remote archive without restoring or hashing
    # the same historical local model on every weight refresh.
    model.write_bytes(b"corrupt after publication")
    repeated = build_public_artifacts(
        store, track, chain, version_key=0, model_resolver=restore,
    )
    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    asyncio.run(restarted.publish(repeated))
    assert len(restored) == (cache_state != "valid")
    asyncio.run(restarted.publish(repeated))
    assert len(restored) == (cache_state != "valid")


@pytest.mark.parametrize("gain,won,archive_failure", [
    (0.0025, False, False), (0.00250001, True, False), (0.00250001, True, True),
])
def test_exact_tenure_private_result_publication_and_cold_reader(
    tmp_path, gain, won, archive_failure,
):
    from tests.validator.test_tournament import tenure_challenge
    from validator.tournament import genesis_champion

    raw = valid_spec()
    raw["promotion"] = {"format": "veda-exact-tenure-v1", "minimum_gain": 0.002,
                        "decay_seconds": 604800}
    raw["scoring"].update(minimum_frontier_gain=0.003, match_confidence=0.9975,
                           bootstrap_resamples=4000)
    raw["full_video"]["minimum_gain"] = 0.003
    track = parse_track_spec(raw)
    # One authenticated old baseline fact; the new track start is not its tenure.
    legacy = replace(track, track_id="legacy", track_start_block=80, promotion=None)
    old = scored_challenge(1, "old-loser", score=0.7, incumbent_score=0.7)
    old["block_height"] = 90
    old["score_block"] = old["stages"][0]["score_block"] = 99
    old["evaluation_batch"]["generated_block"] = 95
    history = resolve_tournament(
        [old], track_id=legacy.track_id, track_spec_digest=legacy.digest, netuid=legacy.netuid,
        minimum_gain=0.003, initial_champion=genesis_champion(legacy),
    ).matches
    historical_body = canonical_json(history[0].document)
    manifest = ArtifactManifest((Artifact(
        HISTORICAL_MATCHES_PREFIX + history[0].object_name,
        hashlib.sha256(historical_body).hexdigest(), len(historical_body), history[0].object_name,
    ),))
    raw["artifact_manifest_sha256"] = manifest.digest
    track = parse_track_spec(raw)
    model = tmp_path / "synthetic-model"
    model.write_bytes(b"synthetic model bytes; not a GPU or real miner test")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    from validator.tournament import historical_champion
    item = tenure_challenge(1, historical_champion(history, genesis_champion(track)),
                            gain=gain, submitted_ms=302401000)
    item.update(file_sha256=digest, model_size_bytes=model.stat().st_size)
    item.update(model_source_fields(None, None, PrivateModelSubmission("ef" * 32, digest)))
    store = SimpleNamespace(
        chain_cursor=lambda: (1000, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": [item]},
        active_competitors=lambda _mapping: [], legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _identifier: str(model),
    )
    chain = WeightChainState(1000, {track.initial_champion.hotkey: 0, item["hotkey"]: 4},
                             1, 1.0, 10, {})
    champion_archives = []

    def archive_champion(sha256, size_bytes):
        assert sha256 == digest and size_bytes == model.stat().st_size
        assert not any("/models/" in call["Key"] or "/matches/" in call["Key"]
                       for call in client.calls)
        champion_archives.append(sha256)
        if archive_failure:
            raise OSError("R2 champion copy temporarily unavailable")

    artifacts = build_public_artifacts(
        store, track, chain, version_key=0,
        historical_matches=history, runtime_manifest=manifest.encode(),
        private_winner_archive=archive_champion,
    )
    document = json.loads(artifacts.matches[0][1])
    assert document["minimum_gain"] == 0.0025
    assert document["outcome"] == ("CHALLENGER_WON" if won else "CHAMPION_RETAINED")
    assert len(document["evaluation"]["stages"]) == 1
    client = MemoryS3()
    client.objects["isolated/tenure/" + history[0].object_name] = historical_body
    publisher = S3Repository(client, bucket="test", prefix="isolated/tenure")
    if archive_failure:
        with pytest.raises(OSError, match="R2 champion"):
            asyncio.run(publisher.publish(artifacts))
        assert champion_archives == [digest]
        assert not any("/models/" in call["Key"] or "/matches/" in call["Key"]
                       or call["Key"].endswith("/weights.json") for call in client.calls)
        return
    asyncio.run(publisher.publish(artifacts))
    assert champion_archives == ([digest] if won else [])
    order = [call["Key"] for call in client.calls]
    model_keys = [key for key in order if "/models/" in key]
    if won:
        assert model_keys
        assert max(order.index(key) for key in model_keys) < order.index(
            "isolated/tenure/" + artifacts.matches[0][0],
        )
    else:
        assert not model_keys and document["evaluation"]["model_object"] is None
    cold = S3Repository(client, bucket="test", prefix="isolated/tenure")
    recovered = asyncio.run(cold.recover(track))
    assert recovered.matches[0].document == document
    assert recovered.matches[0].champion_after.hotkey == (
        item["hotkey"] if won else track.initial_champion.hotkey
    )


def test_failed_private_model_has_public_result_without_payload_and_cold_recovers(tmp_path):
    from validator.queue import ChallengeState

    track = parse_track_spec(valid_spec())
    item = scored_challenge(1, "alice", state=ChallengeState.PARTICIPANT_FAILED,
                            error_code="INVALID_MODEL")
    item.update(model_source_fields(None, None, PrivateModelSubmission("ef" * 32, "ab" * 32)))
    store = SimpleNamespace(
        chain_cursor=lambda: (300, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": [item]},
        active_competitors=lambda _mapping: [], legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _identifier: pytest.fail("failed private model path was requested"),
    )
    chain = WeightChainState(300, {track.initial_champion.hotkey: 0, "alice": 4}, 1, 1.0, 1, {})
    artifacts = build_public_artifacts(store, track, chain, version_key=0)
    assert artifacts.files == ()
    client = MemoryS3()
    asyncio.run(S3Repository(client, bucket="test", prefix="tracks/test").publish(artifacts))
    ledger = asyncio.run(S3Repository(client, bucket="test", prefix="tracks/test").load(track))
    match = ledger.matches[0]
    assert match.document["outcome"] == "CHALLENGE_FAILED"
    assert match.document["evaluation"]["model_object"] is None
    assert not any("/models/" in key for key in client.objects)
    assert decode_weight_artifact(artifacts.weights).plan.uids == (0,)


@pytest.mark.parametrize("switch_block", [252, 1252])
def test_legacy_switch_stops_rewards_without_rewriting_or_losing_match_history(
    tmp_path, switch_block,
) -> None:
    track = parse_track_spec(valid_spec())
    record = submission_at(101, hotkey="alice")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    client = MemoryS3()
    repository = S3Repository(client, bucket="public-data", prefix="tracks/test")

    def observe(store, *, legacy=False):
        store.observe_commitment(
            hotkey=record.hotkey, netuid=track.netuid,
            block_hash="0x" + f"{switch_block:064x}" if legacy else record.block_hash,
            position=ChainPosition(switch_block, 0, 0) if legacy else record.position,
            payload=b"signed-old-activation" if legacy else record.payload,
            submission_id=None if legacy else record.submission_id,
            legacy_excluded=legacy,
        )

    def publishable(store, block):
        return build_public_artifacts(
            store, track,
            WeightChainState(block, {track.initial_champion.hotkey: 0, "alice": 4}, 1, 1.0, 10, {}),
            version_key=0, model_resolver=lambda *_args: model,
        )

    with ChallengeStore(tmp_path / "original.db", track_spec_digest=track.digest) as store:
        identifier, _ = store.add_submission(record)
        observe(store)
        challenge = {
            **scored_challenge(1, "alice", score=0.8, incumbent_score=0.7, paired_gain=0.1),
            "challenge_id": identifier, "submission_id": record.submission_id,
            "repo_id": record.repo_id, "revision": record.revision,
            "file_sha256": model_sha, "model_size_bytes": model.stat().st_size,
        }
        tournament = resolve_tournament(
            [challenge], track_id=track.track_id, track_spec_digest=track.digest,
            netuid=track.netuid, minimum_gain=track.full_video.minimum_gain,
            minimum_exact_gain=track.scoring.minimum_frontier_gain,
            initial_champion=genesis_champion(track),
        )
        store.restore_public_matches(tournament.matches)
        store.advance_chain_cursor(250, "0x" + f"{250:064x}")
        before = publishable(store, 250)
        assert decode_weight_artifact(before.weights).plan.hotkeys == ("alice",)
        asyncio.run(repository.publish(before))

        observe(store, legacy=True)
        for height in range(251, switch_block + 1):
            store.advance_chain_cursor(height, "0x" + f"{height:064x}")
        after = publishable(store, switch_block)
        weights = decode_weight_artifact(after.weights)
        assert weights.plan.hotkeys == (track.initial_champion.hotkey,)
        assert weights.plan.uids == (0,) and weights.plan.weights == (1.0,)
        assert after.matches == before.matches
        assert weights.terminal_match_sha256 == decode_weight_artifact(
            before.weights
        ).terminal_match_sha256
        assert json.loads(after.summary)["champion"]["hotkey"] == "alice"
        asyncio.run(repository.publish(after))

    ledger = asyncio.run(
        S3Repository(client, bucket="public-data", prefix="tracks/test").load(track)
    )
    with ChallengeStore(tmp_path / "fresh.db", track_spec_digest=track.digest) as store:
        store.add_submission(record)
        observe(store)
        observe(store, legacy=True)
        # Chain recovery precedes importing historical terminal matches.
        store.restore_public_matches(ledger.matches)
        store.advance_chain_cursor(switch_block, "0x" + f"{switch_block:064x}")
        restored = publishable(store, switch_block)
        assert restored.weights == after.weights
        assert restored.matches == after.matches
        assert restored.summary == after.summary
        assert store.active_competitors({"alice": 4}) == []


@pytest.mark.parametrize("block", [120, 100_000_000])
def test_empty_track_publishes_uid_zero_veda_as_genesis_champion(block) -> None:
    track = parse_track_spec(valid_spec())
    store = SimpleNamespace(
        chain_cursor=lambda: (block, "0x" + "12" * 32),
        public_snapshot=lambda: {"challenges": []},
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
    )
    chain = WeightChainState(
        block,
        {track.initial_champion.hotkey: 0},
        1,
        1.0,
        10,
        {track.initial_champion.hotkey: 0},
    )

    artifacts = build_public_artifacts(store, track, chain, version_key=0)

    assert artifacts is not None
    weights = decode_weight_artifact(artifacts.weights)
    summary = json.loads(artifacts.summary)
    status = json.loads(artifacts.status)
    assert weights.plan == WeightPlan(
        block,
        (track.initial_champion.hotkey,),
        (0,),
        (1.0,),
    )
    assert summary["match_count"] == 0
    assert summary["champion"]["uid"] == 0
    assert summary["champion"]["source"] == "VEDA_BASELINE"
    assert status["participants"][0]["challenge_id"] == "genesis"
    assert status["participants"][0]["champion"] is True
    assert status["track_phase"] == "ACCEPTING"


def test_storage_loader_recovers_match_uploaded_before_mutable_summary(tmp_path) -> None:
    track = parse_track_spec(valid_spec())
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    first = two_stage_challenge(
        {
            "challenge_id": "challenge-1",
            "submission_id": "submission-1",
            "hotkey": "alice",
            "repo_id": "owner/model-1",
            "revision": "12" * 20,
            "block_height": 100,
            "extrinsic_index": 0,
            "event_index": 0,
            "state": "SUCCEEDED",
            "error_code": None,
            "file_sha256": model_sha256,
            "model_digest": "cd" * 32,
            "model_size_bytes": model.stat().st_size,
            "final_score": 0.8,
            "incumbent_score": 0.5,
            "paired_gain_lower_bound": 0.3,
            "evaluation_batch": {
                "sha256": "de" * 32,
                "generated_block": 104,
                "plan_digest": "ef" * 32,
                "seed_block_hash": "0x" + "ab" * 32,
            },
            "score_block": 110,
            "stages": [],
        }
    )
    snapshot = {"challenges": [first]}
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _challenge_id: str(model),
    )
    chain = WeightChainState(120, {"alice": 4, "bob": 5}, 1, 1.0, 10, {})
    committed = build_public_artifacts(store, track, chain, version_key=0)
    assert committed is not None
    client = MemoryS3()
    asyncio.run(S3Repository(client, bucket="public-data", prefix="veda/track").publish(committed))

    second = two_stage_challenge(
        {
            **first,
            "challenge_id": "challenge-2",
            "submission_id": "submission-2",
            "hotkey": "bob",
            "repo_id": "owner/model-2",
            "revision": "34" * 20,
            "block_height": 121,
            "model_digest": "ef" * 32,
            "final_score": 0.9,
            "incumbent_score": 0.8,
            "paired_gain_lower_bound": 0.1,
            "evaluation_batch": {
                "sha256": "fa" * 32,
                "generated_block": 125,
                "plan_digest": "fb" * 32,
                "seed_block_hash": "0x" + "ab" * 32,
            },
            "score_block": 125,
        }
    )
    snapshot["challenges"] = [first, second]
    store.chain_cursor = lambda: (130, "0x" + "13" * 32)
    newer = build_public_artifacts(
        store,
        track,
        WeightChainState(130, {"alice": 4, "bob": 5}, 1, 1.0, 10, {}),
        version_key=0,
    )
    assert newer is not None
    orphan_name, orphan_payload = newer.matches[1]
    client.objects[f"veda/track/{orphan_name}"] = orphan_payload

    recovered = asyncio.run(
        S3Repository(client, bucket="public-data", prefix="veda/track").load(track)
    )
    assert recovered.source_block == 120
    assert recovered.committed_match_count == 1
    assert recovered.required_finalized_block == 126
    assert [match.challenge_id for match in recovered.matches] == [
        "challenge-1",
        "challenge-2",
    ]


@pytest.mark.parametrize("phase", ["previous", "track", "weights", "summary"])
def test_loader_reconciles_fixed_path_cutover_without_parsing_previous_scoring_schema(phase):
    payload = ArtifactManifest((Artifact("runtime/source.zip", "ab" * 32, 1, "source"),)).encode()
    raw = valid_spec()
    raw["artifact_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    track = parse_track_spec(raw)
    previous_spec = {
        "track_id": "previous-track", "netuid": track.netuid, "mechid": track.mechid,
        "track_start_block": 1, "retired_scoring_schema": {"version": "not-v2"},
    }
    previous_document = {
        "spec": previous_spec,
        "track_spec_digest": hashlib.sha256(canonical_json(previous_spec)).hexdigest(),
    }
    client = MemoryS3()
    client.objects = {
        f"tracks/main/runtime/{track.artifact_manifest_sha256}/manifest.json": payload,
        "tracks/main/track.json": canonical_json(previous_document),
        "tracks/main/weights.json": canonical_json({"track_id": "previous-track", "old_format": 1}),
        "tracks/main/summary.json": canonical_json({"track_id": "previous-track", "old_format": 1}),
    }
    current = build_genesis_artifacts(
        track, WeightChainState(120, {track.initial_champion.hotkey: 0}, 1, 1.0, 1, {}),
        version_key=0,
    )
    if phase != "previous":
        client.objects["tracks/main/track.json"] = current.track
    if phase in {"weights", "summary"}:
        client.objects["tracks/main/weights.json"] = current.weights
    if phase == "summary":
        client.objects["tracks/main/summary.json"] = current.summary
    repository = S3Repository(client, bucket="public-data", prefix="tracks/main")
    recovered = asyncio.run(repository.load(track))
    assert recovered.matches == () and recovered.committed_match_count == 0
    assert recovered.source_block == (120 if phase in {"weights", "summary"} else None)
    assert client.calls == []
    if phase == "previous":
        previous_document["spec"]["track_start_block"] = 2
        client.objects["tracks/main/track.json"] = canonical_json(previous_document)
        with pytest.raises(PublicArtifactError, match="digest"):
            asyncio.run(repository.load(track))


@pytest.mark.parametrize("reuse_data_identity", [False, True])
def test_restart_verifies_disclosed_batches_without_restoring_private_files(
    tmp_path, reuse_data_identity,
) -> None:
    from scoring.spec import data_spec_logical_name, track_spec_dict
    from tests.scoring.test_spec import rules_for_data
    from validator.production.commitment import Artifact
    leaves = tuple(
        SourceLeaf(
            "production",
            f"family-{index}",
            f"source-{index}",
            f"prompt {index}",
            (),
            "all",
            "internal-cleared-v1",
            hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
        )
        for index in range(20)
    )
    audit = SourceLeaf(
        "audit",
        "audit-family",
        "audit-source",
        "audit prompt",
        (),
        "all",
        "internal-cleared-v1",
        hashlib.sha256(b"audit-nonce").hexdigest(),
    )
    pool = SourcePool((*leaves, audit))
    raw_track = valid_spec()
    data_manifest = ArtifactManifest((
        Artifact("weights/teacher", "ab" * 32, 1, "artifacts/model"),
    ))
    raw_track["artifact_manifest_sha256"] = data_manifest.digest
    raw_track["source_pool_merkle_root"] = pool.root
    raw_track["source_commitments"] = {
        "production": {
            "all": [
                leaf.digest for leaf in pool.leaves if leaf.partition == "production"
            ]
        },
        "audit": {"all": [leaf.digest for leaf in pool.leaves if leaf.partition == "audit"]},
    }
    track = parse_track_spec(raw_track)
    block_hash = "0x" + "12" * 32
    proofs = select_source_proofs(
        pool,
        commitments=track.source_commitments,
        partition="production",
        strata=tuple(track.scoring.stratum_weight_map),
        sources_per_stratum=track.scoring.rollouts_per_stratum,
        batch_seed=generation_seed(track.digest, block_hash, "a5" * 32),
        excluded_source_digests=frozenset(), inventory_slot=0,
    )
    plan = build_plan(secret="a5" * 32, inventory_slot=0,
        track_spec_digest=track.digest,
        generation_block=200,
        block_hash=block_hash,
        source_proofs=proofs,
        scoring=track.scoring,
        available_routes=track.model.selector.routes,
    )

    def ready_file(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return path, ReadyObject(f"private/{name}", digest, len(content))

    exact_path, exact = ready_file("batch.npz", b"exact-batch")
    index_path, input_index = ready_file("index.json", b'{"samples":[]}')
    text_path, text = ready_file("text.safetensors", b"encoded-inputs")
    sample_cache = ReadySampleCache(input_index, text)
    references = []
    local_files = [
        (f"batches/{exact.sha256}/manifest.json", exact_path, exact),
        (
            f"batch_inputs/{input_index.sha256}/manifest.json",
            index_path,
            input_index,
        ),
        (f"batch_inputs/{text.sha256}/manifest.json", text_path, text),
    ]
    for index, rollout in enumerate(plan.rollouts):
        video_path, video = ready_file(f"video-{index}.mp4", f"video-{index}".encode())
        latent_path, latent = ready_file(
            f"latent-{index}.safetensors", f"latent-{index}".encode()
        )
        references.append(DenseReference(rollout.rollout_id, video, latent))
        local_files.extend(
            (
                (f"batch_assets/{video.sha256}/manifest.json", video_path, video),
                (f"batch_assets/{latent.sha256}/manifest.json", latent_path, latent),
            )
        )
    ready = plan.ready(
        completed_at_ms=(plan.generation_block + 1) * 1000,
        completed_block=plan.generation_block + 1,
        exact=exact,
        dense_references=tuple(references),
        sample_cache=sample_cache,
    )
    allocation = {
        "ordinal": 1, "challenge_id": "a1" * 32, "stage": "SCORING",
        "plan_digest": ready.plan_digest,
        "ready_sha256": hashlib.sha256(ready.encode()).hexdigest(),
        "file_sha256": "b2" * 32, "model_digest": "c3" * 32, "submission_block": 300,
        "submission_timestamp_ms": 300_000,
    }
    public_references = []
    for reference in ready.dense_references:
        public = {"rollout_id": reference.rollout_id}
        for kind, item in (("video", reference.video), ("latent", reference.latent)):
            public[kind] = {
                "object": f"batch_assets/{item.sha256}/manifest.json",
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
        public_references.append(public)
    record = canonical_json(
        {
            "format": "veda-batch-disclosure-v3",
            "plan": ready.document(),
            "public_exact_object": f"batches/{exact.sha256}/manifest.json",
            "public_dense_references": public_references,
            "public_selected_inputs": {
                "condition": None,
                "index": {
                    "object": f"batch_inputs/{input_index.sha256}/manifest.json",
                    "sha256": input_index.sha256,
                    "size_bytes": input_index.size_bytes,
                },
                "text": {
                    "object": f"batch_inputs/{text.sha256}/manifest.json",
                    "sha256": text.sha256,
                    "size_bytes": text.size_bytes,
                },
            },
        }
    )
    batch = {
        "sha256": exact.sha256,
        "generated_block": ready.generation_block,
        "plan_digest": ready.plan_digest,
        "seed_block_hash": ready.block_hash,
        "object": f"batches/{exact.sha256}/manifest.json",
        "ready_sha256": hashlib.sha256(ready.encode()).hexdigest(),
        "completed_block": ready.completed_block,
        "completed_at_ms": ready.completed_at_ms,
        "allocation": allocation,
    }
    match = SimpleNamespace(
        challenge_id="a1" * 32,
        document={
            "challenge_id": "a1" * 32,
            "submission": {"block_height": 300},
            "evaluation": {
                "file_sha256": "b2" * 32, "model_digest": "c3" * 32,
                "stages": [
                    {
                        "stage": "SCORING",
                        "score_block": 301,
                        "evaluation_batch": batch,
                    }
                ],
                "evaluation_batch": batch,
            }
        },
    )

    publication_module._require_inventory_bindings(track, match.document)
    for field in ("ready_sha256", "completed_block", "completed_at_ms", "allocation"):
        incomplete = json.loads(json.dumps(match.document))
        del incomplete["evaluation"]["stages"][0]["evaluation_batch"][field]
        with pytest.raises(PublicArtifactError):
            publication_module._require_inventory_bindings(track, incomplete)
    for field, value in (("completed_block", 300), ("completed_at_ms", 300_000)):
        post_submission = json.loads(json.dumps(match.document))
        post_submission["evaluation"]["stages"][0]["evaluation_batch"][field] = value
        with pytest.raises(PublicArtifactError):
            publication_module._require_inventory_bindings(track, post_submission)

    client = MemoryS3()
    publisher = S3Repository(client, bucket="public-data", prefix="tracks/test")
    disclosure = BatchDisclosure(f"batch_groups/{ready.plan_digest}.json", record, tuple(
        DisclosureFile(name, path, item.sha256) for name, path, item in local_files
    ))
    reveal_at = ready.completed_at_ms + track.cache.disclosure_delay_seconds * 1000
    with pytest.raises(PublicArtifactError, match="premature"):
        asyncio.run(publisher.publish_disclosure(disclosure, track=track, now_ms=reveal_at - 1))
    assert client.objects == {}
    asyncio.run(publisher.publish_disclosure(disclosure, track=track, now_ms=reveal_at))
    assert not any(key.endswith(("weights.json", "track.json", "summary.json", "status.json"))
                   for key in client.objects)

    restarted = S3Repository(client, bucket="public-data", prefix="tracks/test")
    if reuse_data_identity:
        rule_document = rules_for_data(track)
        provisional = parse_track_spec(rule_document)
        origin = canonical_json(track_spec_dict(track))
        runtime = ArtifactManifest(tuple(sorted((*data_manifest.artifacts, Artifact(
            data_spec_logical_name(provisional), track.digest, len(origin),
            "artifacts/" + track.digest,
        )), key=lambda a: a.logical_name)))
        rule_document["artifact_manifest_sha256"] = runtime.digest
        rules = parse_track_spec(rule_document)
        client.objects["tracks/test/artifacts/" + track.digest] = origin
        client.objects[f"tracks/test/runtime/{runtime.digest}/manifest.json"] = runtime.encode()
        client.objects[f"tracks/test/runtime/{data_manifest.digest}/manifest.json"] = (
            data_manifest.encode()
        )
        resolved = restarted._load_data_track(rules)
        assert resolved == track and rules.digest != resolved.digest
        assert restarted._load_batch_disclosures(resolved, (match,)) == frozenset({
            ready.plan_digest,
        })
        original_body = client.objects["tracks/test/artifacts/" + track.digest]
        client.objects["tracks/test/artifacts/" + track.digest] = origin + b" "
        with pytest.raises(PublicArtifactError, match="unavailable or changed"):
            restarted._load_data_track(rules)
        client.objects["tracks/test/artifacts/" + track.digest] = original_body
    assert restarted._load_batch_disclosures(track, (match,)) == frozenset(
        {ready.plan_digest}
    )
    another_allocation = {
        **allocation, "ordinal": 2, "challenge_id": "d4" * 32,
        "file_sha256": "e5" * 32, "model_digest": "f6" * 32, "submission_block": 301,
        "submission_timestamp_ms": 301_000,
    }
    another_batch = {**batch, "allocation": another_allocation}
    another_match = SimpleNamespace(challenge_id="d4" * 32, document={
        "submission": {"block_height": 301}, "evaluation": {
            "file_sha256": "e5" * 32, "model_digest": "f6" * 32,
            "evaluation_batch": another_batch,
            "stages": [{"stage": "SCORING", "score_block": 302,
                        "evaluation_batch": another_batch}],
        },
    })
    before_record = client.objects[f"tracks/test/batch_groups/{ready.plan_digest}.json"]
    assert restarted._load_batch_disclosures(track, (match, another_match)) == frozenset({
        ready.plan_digest,
    })
    assert client.objects[f"tracks/test/batch_groups/{ready.plan_digest}.json"] == before_record

    tampered = json.loads(record)
    tampered["public_selected_inputs"]["index"]["sha256"] = "ff" * 32
    tampered_client = MemoryS3()
    tampered_client.objects = dict(client.objects)
    tampered_client.metadata = dict(client.metadata)
    disclosure_key = f"tracks/test/batch_groups/{ready.plan_digest}.json"
    tampered_client.objects[disclosure_key] = canonical_json(tampered)
    with pytest.raises(PublicArtifactError, match="not canonical"):
        S3Repository(
            tampered_client,
            bucket="public-data",
            prefix="tracks/test",
        )._load_batch_disclosures(track, (match,))

    # The original submission boundary is part of the immutable allocation.
    later_submission = SimpleNamespace(
        challenge_id="a1" * 32,
        document={**match.document, "submission": {"block_height": 301}},
    )
    with pytest.raises(PublicArtifactError, match="claimant"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (later_submission,),
        )

    wrong_assignment = SimpleNamespace(
        challenge_id="a1" * 32,
        document={
            **match.document,
            "evaluation": {
                **match.document["evaluation"],
                "stages": [
                    {
                        **match.document["evaluation"]["stages"][0],
                        "stage": "CONFIRMING",
                    }
                ],
            },
        },
    )
    with pytest.raises(PublicArtifactError, match="claimant"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (wrong_assignment,)
        )

    missing_batch = {
        **batch, "generated_block": ready.generation_block - 1, "plan_digest": "00" * 32,
    }
    earlier_missing = SimpleNamespace(
        challenge_id="dd" * 32,
        document={"evaluation": {
            "stages": [{"stage": "SCORING", "evaluation_batch": missing_batch}],
            "evaluation_batch": missing_batch,
        }},
    )
    assert S3Repository(
        client, bucket="public-data", prefix="tracks/test",
    )._load_batch_disclosures(track, (earlier_missing, match)) == frozenset({ready.plan_digest})

    client.objects.pop(f"tracks/test/batches/{exact.sha256}/part-00000")
    with pytest.raises(PublicArtifactError, match="chunk is missing"):
        S3Repository(client, bucket="public-data", prefix="tracks/test")._load_batch_disclosures(
            track, (match,)
        )


def test_public_artifacts_bind_summary_matches_and_weights_to_one_snapshot(
    monkeypatch, tmp_path
) -> None:
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"model")
    snapshot = {
        "challenges": [
            two_stage_challenge(
                {
                    "challenge_id": "challenge",
                    "submission_id": "submission",
                    "hotkey": "alice",
                    "repo_id": "owner/model",
                    "revision": "12" * 20,
                    "block_height": 100,
                    "extrinsic_index": 0,
                    "event_index": 0,
                    "state": "SUCCEEDED",
                    "error_code": None,
                    "file_sha256": "ab" * 32,
                    "model_digest": "cd" * 32,
                    "model_size_bytes": 123,
                    "final_score": 0.8,
                    "incumbent_score": 0.5,
                    "paired_gain_lower_bound": 0.3,
                    "evaluation_batch": {
                        "sha256": "de" * 32,
                        "generated_block": 104,
                        "plan_digest": "ef" * 32,
                        "seed_block_hash": "0x" + "ab" * 32,
                    },
                    "score_block": 110,
                    "stages": [],
                }
            )
        ],
    }
    store = SimpleNamespace(
        chain_cursor=lambda: (120, "0x" + "12" * 32),
        public_snapshot=lambda: snapshot,
        active_competitors=lambda _mapping: [],
        legacy_excluded_hotkeys=lambda: frozenset(),
        model_path=lambda _challenge_id: str(model_path),
    )
    track = SimpleNamespace(
        track_id="track",
        digest="ab" * 32,
        netuid=7,
        mechid=2,
        track_start_block=100,
        initial_champion=SimpleNamespace(
            uid=0,
            hotkey="genesis-hotkey",
            repo_id="Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
            revision="76f202874608115408d73280be9531c4ab888242",
            score=0.5,
        ),
        scoring=SimpleNamespace(minimum_frontier_gain=0.001),
        full_video=SimpleNamespace(minimum_gain=0.001),
    )
    chain = WeightChainState(
        120,
        {"genesis-hotkey": 0, "alice": 4},
        1,
        1.0,
        10,
        {"alice": 0},
    )
    monkeypatch.setattr(publication_module, "track_spec_dict", lambda _track: {"track_id": "track"})

    artifacts = build_public_artifacts(store, track, chain, version_key=9)
    assert artifacts is not None
    public_weights = decode_weight_artifact(artifacts.weights)
    public_summary = json.loads(artifacts.summary)
    public_match = json.loads(artifacts.matches[0][1])
    assert public_weights.source_block == public_summary["source_block"] == 120
    assert public_weights.match_sequence == 1
    assert public_weights.terminal_match_sha256 == hashlib.sha256(
        artifacts.matches[0][1]
    ).hexdigest()
    assert public_weights.plan is not None
    assert public_weights.plan.hotkeys == ("alice",)
    assert public_summary["champion"]["hotkey"] == "alice"
    assert public_summary["match_count"] == 1
    assert public_summary["leaderboard_total"] == 2
    assert public_match["outcome"] == "CHALLENGER_WON"

    previous_prefix = build_public_artifacts(store, track, chain, version_key=9, match_limit=0)
    assert previous_prefix.matches == previous_prefix.files == ()
    assert decode_weight_artifact(previous_prefix.weights).plan.hotkeys == ("genesis-hotkey",)
    assert json.loads(previous_prefix.summary)["match_count"] == 0
    for invalid in (-1, True):
        with pytest.raises(ValueError, match="match limit"):
            build_public_artifacts(store, track, chain, version_key=9, match_limit=invalid)
    with pytest.raises(PublicArtifactError, match="prefix is missing"):
        build_public_artifacts(store, track, chain, version_key=9, match_limit=2)

    missing_champion = WeightChainState(
        121,
        {"genesis-hotkey": 0},
        1,
        1.0,
        10,
        {"genesis-hotkey": 0},
    )
    fallback_store = SimpleNamespace(
        chain_cursor=lambda: (121, "0x" + "34" * 32),
        public_snapshot=store.public_snapshot,
        active_competitors=store.active_competitors,
        legacy_excluded_hotkeys=store.legacy_excluded_hotkeys,
        model_path=store.model_path,
    )
    fallback = build_public_artifacts(fallback_store, track, missing_champion, version_key=9)
    assert fallback is not None
    fallback_weights = decode_weight_artifact(fallback.weights)
    assert fallback_weights.plan == WeightPlan(
        121,
        ("genesis-hotkey",),
        (0,),
        (1.0,),
    )
    assert json.loads(fallback.summary)["champion"]["hotkey"] == "alice"

    metadata = build_public_artifacts(store, track, chain, version_key=9)
    assert metadata is not None
    # A decision publishes its commitments immediately, without restoring any
    # raw evaluation inputs. Production-age disclosure is a separate task.
    assert not any(item.object_name.startswith(("batches/", "batch_assets/"))
                   for item in metadata.files)
    assert metadata.batch_groups == ()

    store.chain_cursor = lambda: (119, "0x" + "11" * 32)
    assert build_public_artifacts(store, track, chain, version_key=9) is None

    store.chain_cursor = lambda: (120, "0x" + "12" * 32)
    snapshot["challenges"][0]["score_block"] = 121
    with pytest.raises(PublicArtifactError, match="newer"):
        build_public_artifacts(store, track, chain, version_key=9)
