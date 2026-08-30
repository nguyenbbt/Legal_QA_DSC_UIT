"""Run the fresh local GPU preflight for the exact D-066 Qwen dense arm."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.models.huggingface_local import Qwen3EmbeddingBackend
from legal_rag.models.torch_audit import audit_safetensors_directory
from legal_rag.retrieval.dense_preflight import (
    DenseCheckpointIdentity,
    DensePreflightInput,
    evaluate_dense_preflight,
)
from legal_rag.retrieval.dense_sampling import stratified_positions

_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_CHECKPOINT = Path(f".local/models/qwen3-embedding-0.6b/{_MODEL_REVISION}")
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_CHUNKS_CHECKSUM = "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143"
_DATABASE = Path("artifacts/indices/bm25.v1.active.sqlite3")
_DATABASE_CHECKSUM = "sha256:2b977231a0be77fa2409988ecb1f0955bd22d7175130b08affc49fd04771fdc1"
_HISTORICAL = Path("artifacts/indices/dense/qwen3-embedding-0.6b-v1")
_CHUNK_COUNT = 641_118
_DIMENSION = 1024
_SAMPLE_COUNT = 512
_BATCH_SIZE = 8
_MAXIMUM_LENGTH = 2_048
_MAXIMUM_RUNTIME_SECONDS = 21_600
_STORAGE_RESERVE_BYTES = 2_000_000_000
_PUBLISHED_PARAMETER_COUNT = 595_776_512
_WHOLE_SYSTEM_PARAMETER_COUNT = 3_223_292_928


def _streaming_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _tokenizer_checksum() -> str:
    members = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        members[name.replace(".", "_")] = _streaming_checksum(_CHECKPOINT / name)
    return checksum_bytes(content_json_bytes(members))


def _sample() -> tuple[tuple[str, ...], tuple[str, ...], str]:
    connection = sqlite3.connect(_DATABASE)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            """
            SELECT chunk_id, document_length, source_offset, source_length
            FROM documents
            ORDER BY document_length, chunk_id
            """
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != _CHUNK_COUNT:
        raise SystemExit("D-066 dense sample source count drift")
    positions = stratified_positions(total_count=len(rows), sample_count=_SAMPLE_COUNT)
    selected = tuple(rows[position] for position in positions)
    ids: list[str] = []
    texts: list[str] = []
    with _CHUNKS.open("rb") as stream:
        for raw_chunk_id, _length, raw_offset, raw_size in selected:
            stream.seek(int(raw_offset))
            value = json.loads(stream.read(int(raw_size)))
            chunk_id = value.get("chunk_id")
            retrieval_text = value.get("retrieval_text")
            if (
                chunk_id != raw_chunk_id
                or not isinstance(retrieval_text, str)
                or not retrieval_text
            ):
                raise SystemExit("D-066 dense sample row drift")
            ids.append(str(chunk_id))
            texts.append(retrieval_text)
    sample_data = content_json_bytes(
        {
            "schema_version": "retrieval.dense-preflight-sample.v1",
            "strategy": "document-length-stratified-then-sorted.v1",
            "source_chunk_checksum": _CHUNKS_CHECKSUM,
            "chunk_ids": ids,
        }
    )
    return tuple(ids), tuple(texts), checksum_bytes(sample_data)


def _historical_checkpoint(
    *, model_artifact_checksum: str, tokenizer_checksum: str, indexing_config_checksum: str
) -> DenseCheckpointIdentity:
    report = json.loads((_HISTORICAL / "rejected-run-report.json").read_bytes())
    return DenseCheckpointIdentity(
        namespace=str(report["run_id"]),
        model_id=str(report["model_id"]),
        model_revision=str(report["model_revision"]),
        corpus_checksum=str(report["source_chunk_checksum"]),
        model_artifact_checksum=model_artifact_checksum,
        tokenizer_checksum=tokenizer_checksum,
        indexing_config_checksum=indexing_config_checksum,
        chunk_count=int(report["partial_rows"]),
        dimension=int(report["dimension"]),
    )


def main() -> int:
    import torch

    if _streaming_checksum(_CHUNKS) != _CHUNKS_CHECKSUM:
        raise SystemExit("D-066 chunk checksum drift")
    if _streaming_checksum(_DATABASE) != _DATABASE_CHECKSUM:
        raise SystemExit("D-066 BM25 database checksum drift")
    model_artifact_checksum = _streaming_checksum(_CHECKPOINT / "model.safetensors")
    tokenizer_checksum = _tokenizer_checksum()
    parameter_audit = audit_safetensors_directory(_CHECKPOINT)
    if parameter_audit.exact_parameter_count != _PUBLISHED_PARAMETER_COUNT:
        raise SystemExit("D-066 Qwen embedding parameter count drift")
    if not torch.cuda.is_available():
        raise SystemExit("D-066 fresh dense preflight requires the local CUDA device")

    indexing_config = {
        "schema_version": "retrieval.dense-indexing-config.v1",
        "model_id": _MODEL_ID,
        "model_revision": _MODEL_REVISION,
        "dimension": _DIMENSION,
        "storage_dtype": "float16",
        "batch_size": _BATCH_SIZE,
        "maximum_length": _MAXIMUM_LENGTH,
        "ordering": "document-length-then-chunk-id.v1",
        "sample_strategy": "document-length-stratified-then-sorted.v1",
        "query_instruction": "Retrieve Vietnamese legal passages that answer the question.",
    }
    indexing_config_checksum = checksum_bytes(content_json_bytes(indexing_config))
    sample_ids, sample_texts, sample_checksum = _sample()
    backend = Qwen3EmbeddingBackend(
        _CHECKPOINT,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        device="cuda",
        batch_size=_BATCH_SIZE,
        maximum_length=_MAXIMUM_LENGTH,
        query_instruction=str(indexing_config["query_instruction"]),
    )
    backend.encode_documents(sample_texts[:_BATCH_SIZE])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    vectors = backend.encode_documents(sample_texts)
    torch.cuda.synchronize()
    observed_seconds = time.perf_counter() - started
    if len(vectors) != _SAMPLE_COUNT or any(len(vector) != _DIMENSION for vector in vectors):
        raise SystemExit("D-066 dense preflight output shape drift")
    peak_vram = int(torch.cuda.max_memory_allocated())
    available_storage = shutil.disk_usage(Path.cwd()).free
    value = DensePreflightInput(
        arm_id="R-DISC-1",
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        corpus_checksum=_CHUNKS_CHECKSUM,
        model_artifact_checksum=model_artifact_checksum,
        tokenizer_checksum=tokenizer_checksum,
        indexing_config_checksum=indexing_config_checksum,
        chunk_count=_CHUNK_COUNT,
        dimension=_DIMENSION,
        bytes_per_component=2,
        observed_documents=_SAMPLE_COUNT,
        observed_seconds_millis=max(1, round(observed_seconds * 1000.0)),
        maximum_runtime_seconds=_MAXIMUM_RUNTIME_SECONDS,
        available_storage_bytes=available_storage,
        required_free_storage_bytes=_STORAGE_RESERVE_BYTES,
    )
    report = evaluate_dense_preflight(value)
    historical_report = evaluate_dense_preflight(
        value,
        checkpoint=_historical_checkpoint(
            model_artifact_checksum=model_artifact_checksum,
            tokenizer_checksum=tokenizer_checksum,
            indexing_config_checksum=indexing_config_checksum,
        ),
    )
    if historical_report.checkpoint_disposition != "REJECT_STALE_PARTIAL":
        raise SystemExit("D-066 historical dense partial was not rejected")
    operational_status = report.status
    output: dict[str, Any] = {
        "schema_version": "evaluation.d066-r-disc-1-preflight.v1",
        "status": operational_status,
        "operational_blocker_codes": report.blocker_codes,
        "preflight": asdict(report),
        "historical_checkpoint_preflight": asdict(historical_report),
        "historical_partial_disposition": "REJECT_STALE_PARTIAL",
        "historical_partial_rows": 12_147,
        "sample_checksum": sample_checksum,
        "sample_count": len(sample_ids),
        "sample_text_published": False,
        "model_artifact_checksum": model_artifact_checksum,
        "tokenizer_checksum": tokenizer_checksum,
        "indexing_config": indexing_config,
        "indexing_config_checksum": indexing_config_checksum,
        "parameter_audit_checksum": parameter_audit.parameter_audit_checksum,
        "exact_model_parameter_count": parameter_audit.exact_parameter_count,
        "whole_system_parameter_count": _WHOLE_SYSTEM_PARAMETER_COUNT,
        "competition_limit_exclusive": 4_000_000_000,
        "whole_system_passes_limit": _WHOLE_SYSTEM_PARAMETER_COUNT < 4_000_000_000,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_vram_bytes": peak_vram,
        "available_storage_bytes": available_storage,
        "execution_mode": "local-gpu-offline",
        "paid_service_used": False,
        "cost_usd": 0,
        "fit_performed": False,
    }
    output_data = content_json_bytes(output)
    output_path = Path(
        "artifacts/evaluations/post-d062/D066-candidate-discovery-v1/preflight/"
        "R-DISC-1.dense-preflight.v1.json"
    )
    output_checksum = write_immutable_bytes(output_path, output_data)
    print(json.dumps({"status": operational_status, "checksum": output_checksum}), flush=True)
    return 0 if operational_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
