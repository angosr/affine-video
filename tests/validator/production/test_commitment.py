import hashlib

import pytest

from validator.production.commitment import (
    Artifact,
    ArtifactManifest,
    SourceLeaf,
    SourcePool,
    canonical_json,
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
    )


def test_source_pool_proves_each_sorted_leaf_without_revealing_others() -> None:
    pool = SourcePool((source(2), source(0), source(1)))

    assert SourcePool.decode(pool.encode()).root == pool.root
    assert [leaf.source_id for leaf in pool.leaves] == ["source-0", "source-1", "source-2"]
    assert len(pool.root) == 64
    for leaf in pool.leaves:
        proof = pool.proof(leaf.source_id)
        assert verify_source_proof(proof, pool.root)
        assert proof.leaf == leaf

    proof = pool.proof("source-1")
    assert not verify_source_proof(proof, "00" * 32)


def test_source_family_cannot_cross_private_partitions() -> None:
    with pytest.raises(ValueError, match="cannot cross"):
        SourcePool(
            (
                source(0, family="related", partition="production"),
                source(1, family="related", partition="audit"),
            )
        )


def test_artifact_manifest_is_canonical_and_verifies_local_bytes(tmp_path) -> None:
    content = b"pinned runtime bytes"
    (tmp_path / "runtime.tar").write_bytes(content)
    artifact = Artifact(
        "runtime.tar",
        hashlib.sha256(content).hexdigest(),
        len(content),
        f"artifacts/{hashlib.sha256(content).hexdigest()}",
    )
    manifest = ArtifactManifest((artifact,))

    manifest.verify_local(tmp_path)
    assert len(manifest.digest) == 64
    assert ArtifactManifest.decode(canonical_json(manifest.document())) == manifest

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
