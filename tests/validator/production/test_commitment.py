import hashlib
import json
from dataclasses import replace

import pytest

from validator.production.commitment import (
    Artifact,
    ArtifactManifest,
    SourceCacheBinding,
    SourceLeaf,
    SourcePool,
    canonical_json,
    load_bound_source_pool,
    verify_source_proof,
)


def source(index: int, *, family: str | None = None, partition: str = "production") -> SourceLeaf:
    return SourceLeaf(
        partition=partition,
        source_family_id=family or f"family-{index}",
        source_id=f"source-{index}",
        prompt=f"prompt {index}",
        condition_asset_sha256=(hashlib.sha256(f"asset-{index}".encode()).hexdigest(),),
        generation_geometry="16:9@37",
        redistribution_license_id="internal-cleared-v1",
        commitment_nonce=hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
    )


def test_source_pool_proves_each_sorted_leaf_without_revealing_others() -> None:
    pool = SourcePool((source(2), source(0), source(1)))
    commitments = tuple(leaf.digest for leaf in pool.leaves)

    assert SourcePool.decode(pool.encode()).root == pool.root
    assert [leaf.source_id for leaf in pool.leaves] == ["source-0", "source-1", "source-2"]
    assert len(pool.root) == 64
    for leaf in pool.leaves:
        proof = pool.proof(leaf.source_id)
        assert verify_source_proof(
            proof,
            pool.root,
            commitment_digests=commitments,
        )
        assert proof.leaf == leaf

    proof = pool.proof("source-1")
    assert not verify_source_proof(proof, "00" * 32)


def test_source_proof_is_bound_to_its_public_index_and_canonical_path() -> None:
    pool = SourcePool((source(0), source(1), source(2)))
    commitments = tuple(leaf.digest for leaf in pool.leaves)
    proof = pool.proof("source-2")

    assert not verify_source_proof(
        replace(proof, index=3),
        pool.root,
        commitment_digests=commitments,
    )
    assert not verify_source_proof(
        replace(proof, siblings=proof.siblings[:-1]),
        pool.root,
        commitment_digests=commitments,
    )
    wrong_side = replace(
        proof.siblings[0],
        side="left" if proof.siblings[0].side == "right" else "right",
    )
    assert not verify_source_proof(
        replace(proof, siblings=(wrong_side, *proof.siblings[1:])),
        pool.root,
        commitment_digests=commitments,
    )


def test_public_source_commitments_hide_inputs_but_reconstruct_the_root() -> None:
    first = source(0)
    same_input_new_nonce = SourceLeaf(
        partition=first.partition,
        source_family_id=first.source_family_id,
        source_id=first.source_id,
        prompt=first.prompt,
        condition_asset_sha256=first.condition_asset_sha256,
        generation_geometry=first.generation_geometry,
        redistribution_license_id=first.redistribution_license_id,
        commitment_nonce="ff" * 32,
    )
    pool = SourcePool((first, source(1, partition="audit")))
    public = pool.public_commitments()

    assert first.digest != same_input_new_nonce.digest
    assert tuple(item["sha256"] for item in public) == tuple(
        leaf.digest for leaf in pool.leaves
    )
    assert all(set(item) == {"generation_geometry", "partition", "sha256"} for item in public)
    assert first.prompt not in json.dumps(public)
    assert first.source_id not in json.dumps(public)
    assert first.commitment_nonce not in json.dumps(public)


def test_source_family_must_be_unique_across_private_pool() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        SourcePool(
            (
                source(0, family="related", partition="production"),
                source(1, family="related", partition="audit"),
            )
        )


def test_source_cache_binding_rejects_id_preserving_encoded_input_change(tmp_path) -> None:
    leaves = tuple(
        SourceLeaf(
            partition="production",
            source_family_id=f"family-{index}",
            source_id=f"source-{index}",
            prompt=f"prompt {index}",
            condition_asset_sha256=(),
            generation_geometry="16:9@37",
            redistribution_license_id="generated-original-v1",
            commitment_nonce=hashlib.sha256(f"nonce-{index}".encode()).hexdigest(),
        )
        for index in range(2)
    )
    pool = SourcePool(leaves)
    cache = tmp_path / "cache"
    cache.mkdir()
    index = cache / "index.json"
    text = cache / "text.safetensors"
    index.write_text(
        json.dumps({"samples": [{"id": leaf.source_id} for leaf in leaves]}),
        encoding="utf-8",
    )
    text.write_bytes(b"encoded prompt tensors")
    binding = SourceCacheBinding(
        source_pool_merkle_root=pool.root,
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        text_sha256=hashlib.sha256(text.read_bytes()).hexdigest(),
        condition_sha256=None,
    )

    decoded = SourceCacheBinding.decode(binding.encode())
    decoded.verify_local(pool, cache)

    text.write_bytes(b"different tensors under the same source IDs")
    with pytest.raises(ValueError, match="sample text differs"):
        decoded.verify_local(pool, cache)


def test_source_cache_binding_requires_exact_leaf_and_sample_coverage(tmp_path) -> None:
    pool = SourcePool((source(0), source(1)))
    cache = tmp_path / "cache"
    cache.mkdir()
    index = cache / "index.json"
    text = cache / "text.safetensors"
    index.write_text(json.dumps({"samples": [{"id": "source-0"}]}), encoding="utf-8")
    text.write_bytes(b"encoded")
    binding = SourceCacheBinding(
        pool.root,
        hashlib.sha256(index.read_bytes()).hexdigest(),
        hashlib.sha256(text.read_bytes()).hexdigest(),
        None,
    )

    with pytest.raises(ValueError, match="sample IDs differ"):
        binding.verify_local(pool, cache)

    with pytest.raises(ValueError, match="track commitment"):
        load_bound_source_pool(
            pool.encode(),
            binding.encode(),
            expected_root="00" * 32,
            sample_cache=cache,
        )


def test_artifact_manifest_is_canonical_and_verifies_local_bytes(tmp_path, monkeypatch) -> None:
    from validator.production import commitment

    content = b"pinned runtime bytes"
    (tmp_path / "runtime.tar").write_bytes(content)
    artifact = Artifact(
        "runtime.tar",
        hashlib.sha256(content).hexdigest(),
        len(content),
        f"artifacts/{hashlib.sha256(content).hexdigest()}",
    )
    manifest = ArtifactManifest((artifact,))

    original_sha = commitment._file_sha256
    hashed = []

    def count_hash(path):
        hashed.append(path)
        return original_sha(path)

    monkeypatch.setattr(commitment, "_file_sha256", count_hash)
    manifest.verify_local(tmp_path)
    assert len(manifest.digest) == 64
    assert ArtifactManifest.decode(canonical_json(manifest.document())) == manifest
    ArtifactManifest.decode(manifest.encode()).verify_local(tmp_path)
    assert hashed == [tmp_path / "runtime.tar"] * 2

    (tmp_path / "runtime.tar").write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="digest"):
        manifest.verify_local(tmp_path)
    assert len(hashed) == 3

    (tmp_path / "runtime.tar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="wrong size"):
        manifest.verify_local(tmp_path)


def test_artifact_manifest_must_cover_every_runtime_file(tmp_path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "__pycache__").mkdir()
    model = tmp_path / "model" / "weights.safetensors"
    selector = tmp_path / "selector.safetensors"
    model.write_bytes(b"model")
    selector.write_bytes(b"selector")
    (tmp_path / "model" / "__pycache__" / "runtime.pyc").write_bytes(b"disposable")

    def artifact(path):
        content = path.read_bytes()
        name = path.relative_to(tmp_path).as_posix()
        return Artifact(name, hashlib.sha256(content).hexdigest(), len(content), f"objects/{name}")

    incomplete = ArtifactManifest((artifact(selector),))
    with pytest.raises(ValueError, match="omits runtime bytes"):
        incomplete.verify_runtime_coverage(
            tmp_path,
            files=("selector.safetensors",),
            directories=("model",),
        )

    complete = ArtifactManifest(
        tuple(sorted((artifact(model), artifact(selector)), key=lambda item: item.logical_name))
    )
    complete.verify_runtime_coverage(
        tmp_path,
        files=("selector.safetensors",),
        directories=("model",),
    )


def test_artifact_manifest_allows_identical_content_to_share_an_object() -> None:
    manifest = ArtifactManifest(
        (
            Artifact("one", "a" * 64, 1, "artifacts/shared"),
            Artifact("two", "a" * 64, 1, "artifacts/shared"),
        )
    )

    assert len(manifest.artifacts) == 2


def test_artifact_manifest_rejects_an_object_key_collision() -> None:
    with pytest.raises(ValueError, match="different content"):
        ArtifactManifest(
            (
                Artifact("one", "a" * 64, 1, "artifacts/shared"),
                Artifact("two", "b" * 64, 1, "artifacts/shared"),
            )
        )


@pytest.mark.parametrize("logical_name", ("/absolute", "../escape", "dir//file"))
def test_artifact_manifest_rejects_unsafe_logical_paths(logical_name) -> None:
    with pytest.raises(ValueError, match="logical name"):
        Artifact(logical_name, "a" * 64, 1, "artifacts/safe")
