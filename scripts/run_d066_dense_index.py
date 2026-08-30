"""Build or resume the exact D-066 R-DISC-1 dense index locally."""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.models.huggingface_local import Qwen3EmbeddingBackend
from legal_rag.models.torch_audit import audit_safetensors_directory
from legal_rag.retrieval.dense_store import audit_dense_store
from legal_rag.retrieval.resumable_dense_store import (
    DenseBuildIdentity,
    DenseSourceRow,
    build_resumable_dense_store,
)

_PREFLIGHT = Path(
    "artifacts/evaluations/post-d062/D066-candidate-discovery-v1/preflight/"
    "R-DISC-1.dense-preflight.v1.json"
)
_PREFLIGHT_CHECKSUM = "sha256:70fd25ab9f486057a1b33cb083aa65033dc39f56a1577b28adc46be972f2838b"
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_CHUNKS_CHECKSUM = "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143"
_DATABASE = Path("artifacts/indices/bm25.v1.active.sqlite3")
_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_CHECKPOINT = Path(f".local/models/qwen3-embedding-0.6b/{_MODEL_REVISION}")
_HISTORICAL_PARTIAL = Path("artifacts/indices/dense/qwen3-embedding-0.6b-v1")
_CHUNK_COUNT = 641_118
_PARAMETER_COUNT = 595_776_512
_BATCH_SIZE = 8
_MAXIMUM_LENGTH = 2_048


def _streaming_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _tokenizer_checksum() -> str:
    members = {
        name.replace(".", "_"): _streaming_checksum(_CHECKPOINT / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    }
    return checksum_bytes(content_json_bytes(members))


def _source_rows() -> tuple[DenseSourceRow, ...]:
    connection = sqlite3.connect(_DATABASE)
    connection.execute("PRAGMA query_only=ON")
    try:
        values = connection.execute(
            """
            SELECT chunk_id, document_length, source_offset, source_length
            FROM documents
            ORDER BY document_length, chunk_id
            """
        ).fetchall()
    finally:
        connection.close()
    rows = tuple(
        DenseSourceRow(str(chunk_id), int(length), int(offset), int(size))
        for chunk_id, length, offset, size in values
    )
    if len(rows) != _CHUNK_COUNT:
        raise SystemExit("D-066 dense source count drift")
    chunk_ids = tuple(row.chunk_id for row in rows)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise SystemExit("D-066 dense source contains duplicate chunk IDs")
    return rows


def main() -> int:
    preflight_data = _PREFLIGHT.read_bytes()
    if checksum_bytes(preflight_data) != _PREFLIGHT_CHECKSUM:
        raise SystemExit("D-066 dense preflight checksum drift")
    preflight = json.loads(preflight_data)
    if preflight.get("status") != "PASS":
        raise SystemExit("D-066 dense preflight did not pass")
    measured = preflight["preflight"]
    config = preflight["indexing_config"]
    if (
        config.get("model_id") != _MODEL_ID
        or config.get("model_revision") != _MODEL_REVISION
        or config.get("batch_size") != _BATCH_SIZE
        or config.get("maximum_length") != _MAXIMUM_LENGTH
        or config.get("dimension") != 1_024
        or config.get("storage_dtype") != "float16"
        or config.get("ordering") != "document-length-then-chunk-id.v1"
        or measured.get("resumable_namespace") != "d066-r-disc-1-116f6cf195224b12"
        or preflight.get("historical_partial_disposition") != "REJECT_STALE_PARTIAL"
        or preflight.get("historical_partial_rows") != 12_147
    ):
        raise SystemExit("D-066 dense indexing config drift")
    if _streaming_checksum(_CHUNKS) != _CHUNKS_CHECKSUM:
        raise SystemExit("D-066 canonical chunk checksum drift")
    if _streaming_checksum(_CHECKPOINT / "model.safetensors") != preflight.get(
        "model_artifact_checksum"
    ):
        raise SystemExit("D-066 dense model artifact checksum drift")
    if _tokenizer_checksum() != preflight.get("tokenizer_checksum"):
        raise SystemExit("D-066 dense tokenizer checksum drift")
    if audit_safetensors_directory(_CHECKPOINT).exact_parameter_count != _PARAMETER_COUNT:
        raise SystemExit("D-066 dense model parameter count drift")
    identity = DenseBuildIdentity(
        namespace=str(measured["resumable_namespace"]),
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        corpus_checksum=_CHUNKS_CHECKSUM,
        model_artifact_checksum=str(preflight["model_artifact_checksum"]),
        tokenizer_checksum=str(preflight["tokenizer_checksum"]),
        indexing_config_checksum=str(preflight["indexing_config_checksum"]),
        dimension=int(config["dimension"]),
        storage_dtype="float16",
        ordering="document-length-then-chunk-id.v1",
    )
    rows = _source_rows()
    backend = Qwen3EmbeddingBackend(
        _CHECKPOINT,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        device="cuda",
        batch_size=_BATCH_SIZE,
        maximum_length=_MAXIMUM_LENGTH,
        query_instruction=str(config["query_instruction"]),
    )
    output_directory = Path("artifacts/indices/dense") / identity.namespace
    if output_directory.resolve() == _HISTORICAL_PARTIAL.resolve():
        raise SystemExit("D-066 rejected historical partial cannot be reused")
    started = time.perf_counter()

    def progress(completed: int, total: int) -> None:
        if completed % 4_096 < _BATCH_SIZE or completed == total:
            print(f"R-DISC-1-INDEX: {completed}/{total}", flush=True)

    result = build_resumable_dense_store(
        chunks_path=_CHUNKS,
        output_directory=output_directory,
        backend=backend,
        identity=identity,
        source_rows=rows,
        batch_size=_BATCH_SIZE,
        checkpoint_interval_rows=1_024,
        progress=progress,
    )
    elapsed = time.perf_counter() - started
    if result.status != "COMPLETE":
        raise SystemExit("D-066 dense index did not complete")
    del backend
    gc.collect()
    expected_chunk_ids = tuple(row.chunk_id for row in rows)
    audit = audit_dense_store(
        output_directory,
        expected_chunk_ids=expected_chunk_ids,
    )
    if (
        audit.chunk_count != _CHUNK_COUNT
        or audit.dimension != 1_024
        or audit.storage_dtype != "float16"
        or audit.missing_chunk_count != 0
        or audit.duplicate_chunk_count != 0
        or audit.nonfinite_vector_count != 0
        or audit.zero_vector_count != 0
        or audit.nonunit_vector_count != 0
        or not audit.deterministic_mapping
    ):
        raise SystemExit("D-066 dense index post-build audit failed")
    report_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r-disc-1-index-build.v1",
            "status": result.status,
            "namespace": identity.namespace,
            "completed_count": result.completed_count,
            "total_count": result.total_count,
            "checkpoint_checksum": result.checkpoint_checksum,
            "manifest_checksum": result.manifest_checksum,
            "preflight_checksum": _PREFLIGHT_CHECKSUM,
            "wall_seconds": elapsed,
            "documents_per_second": (
                float(result.completed_count) / elapsed if elapsed > 0.0 else None
            ),
            "vector_bytes": (output_directory / "vectors.f16.npy").stat().st_size,
            "ids_bytes": (output_directory / "chunk-ids.jsonl").stat().st_size,
            "store_audit": asdict(audit),
            "historical_partial_path_used": False,
            "historical_partial_rows": 12_147,
            "execution_mode": "local-gpu-offline",
            "paid_service_used": False,
            "cost_usd": 0,
            "fit_performed": False,
        }
    )
    report_checksum = write_immutable_bytes(
        Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1/")
        / "R-DISC-1.index-build.v1.json",
        report_data,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "completed": result.completed_count,
                "report_checksum": report_checksum,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
