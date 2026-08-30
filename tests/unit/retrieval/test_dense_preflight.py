from __future__ import annotations

from legal_rag.retrieval.dense_preflight import (
    DenseCheckpointIdentity,
    DensePreflightInput,
    evaluate_dense_preflight,
)


def _input(**overrides: object) -> DensePreflightInput:
    values: dict[str, object] = {
        "arm_id": "R-DISC-1",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "corpus_checksum": "sha256:" + "a" * 64,
        "model_artifact_checksum": "sha256:" + "b" * 64,
        "tokenizer_checksum": "sha256:" + "c" * 64,
        "indexing_config_checksum": "sha256:" + "d" * 64,
        "chunk_count": 641_118,
        "dimension": 1024,
        "bytes_per_component": 2,
        "observed_documents": 512,
        "observed_seconds_millis": 10_000,
        "maximum_runtime_seconds": 21_600,
        "available_storage_bytes": 2_000_000_000,
        "required_free_storage_bytes": 100_000_000,
    }
    values.update(overrides)
    return DensePreflightInput(**values)  # type: ignore[arg-type]


def test_dense_preflight_passes_only_when_runtime_and_storage_fit() -> None:
    report = evaluate_dense_preflight(_input())

    assert report.status == "PASS"
    assert report.projected_runtime_seconds == 12_522
    assert report.projected_vector_bytes == 1_313_009_664
    assert report.resumable_namespace.startswith("d066-r-disc-1-")

    failed = evaluate_dense_preflight(_input(observed_seconds_millis=30_000))
    assert failed.status == "BLOCKED"
    assert "OPS002_RUNTIME_LIMIT" in failed.blocker_codes


def test_dense_preflight_rejects_stale_partial_instead_of_resuming_it() -> None:
    stale = DenseCheckpointIdentity(
        namespace="historical-partial",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        corpus_checksum="sha256:" + "b" * 64,
        model_artifact_checksum="sha256:" + "b" * 64,
        tokenizer_checksum="sha256:" + "c" * 64,
        indexing_config_checksum="sha256:" + "d" * 64,
        chunk_count=12_147,
        dimension=1024,
    )

    report = evaluate_dense_preflight(_input(), checkpoint=stale)

    assert report.status == "BLOCKED"
    assert report.checkpoint_disposition == "REJECT_STALE_PARTIAL"
    assert "DENSE_CHECKPOINT_FINGERPRINT_MISMATCH" in report.blocker_codes


def test_dense_preflight_allows_only_exact_fingerprint_resume() -> None:
    preflight = _input(chunk_count=10, observed_documents=10, observed_seconds_millis=100)
    first = evaluate_dense_preflight(preflight)
    exact = DenseCheckpointIdentity(
        namespace=first.resumable_namespace,
        model_id=preflight.model_id,
        model_revision=preflight.model_revision,
        corpus_checksum=preflight.corpus_checksum,
        model_artifact_checksum=preflight.model_artifact_checksum,
        tokenizer_checksum=preflight.tokenizer_checksum,
        indexing_config_checksum=preflight.indexing_config_checksum,
        chunk_count=preflight.chunk_count,
        dimension=preflight.dimension,
    )

    resumed = evaluate_dense_preflight(preflight, checkpoint=exact)

    assert resumed.status == "PASS"
    assert resumed.checkpoint_disposition == "RESUME_EXACT"
    assert resumed.resumable_namespace == first.resumable_namespace


def test_dense_preflight_namespace_binds_model_tokenizer_and_indexing_config() -> None:
    baseline = evaluate_dense_preflight(_input())

    assert (
        evaluate_dense_preflight(
            _input(tokenizer_checksum="sha256:" + "e" * 64)
        ).resumable_namespace
        != baseline.resumable_namespace
    )
