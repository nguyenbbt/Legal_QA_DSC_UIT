"""Checksum-bound resumable float16 dense-store construction."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import numpy as np

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.dense import DenseRetrievalError, EmbeddingBackend


@dataclass(frozen=True, slots=True)
class DenseSourceRow:
    chunk_id: str
    document_length: int
    source_offset: int
    source_length: int


@dataclass(frozen=True, slots=True)
class DenseBuildIdentity:
    namespace: str
    model_id: str
    model_revision: str
    corpus_checksum: str
    model_artifact_checksum: str
    tokenizer_checksum: str
    indexing_config_checksum: str
    dimension: int
    storage_dtype: Literal["float16"]
    ordering: Literal["document-length-then-chunk-id.v1"]

    def __post_init__(self) -> None:
        checksums = (
            self.corpus_checksum,
            self.model_artifact_checksum,
            self.tokenizer_checksum,
            self.indexing_config_checksum,
        )
        if (
            not self.namespace
            or not self.model_id
            or not self.model_revision
            or self.dimension < 1
            or any(not value.startswith("sha256:") or len(value) != 71 for value in checksums)
        ):
            raise ValueError("dense build identity is incomplete")


@dataclass(frozen=True, slots=True)
class DenseBuildResult:
    status: Literal["INCOMPLETE", "COMPLETE"]
    completed_count: int
    total_count: int
    checkpoint_checksum: str
    manifest_checksum: str | None


def _fail(code: str, message: str) -> NoReturn:
    raise DenseRetrievalError(code, message)


def _streaming_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _write_checkpoint(path: Path, value: dict[str, Any]) -> str:
    data = content_json_bytes(value)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return checksum_bytes(data)


def _source_rows(rows: Sequence[DenseSourceRow]) -> tuple[DenseSourceRow, ...]:
    ordered = tuple(rows)
    expected = tuple(
        sorted(ordered, key=lambda item: (item.document_length, item.chunk_id.encode("utf-8")))
    )
    ids = tuple(item.chunk_id for item in ordered)
    if (
        not ordered
        or ordered != expected
        or any(
            not item.chunk_id
            or item.document_length < 0
            or item.source_offset < 0
            or item.source_length < 1
            for item in ordered
        )
        or len(ids) != len(set(ids))
    ):
        _fail("DENSE_SOURCE_ORDER_INVALID", "dense source rows are not canonical and unique")
    return ordered


def _read_texts(
    chunks_path: Path, rows: Sequence[DenseSourceRow]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ids: list[str] = []
    texts: list[str] = []
    with chunks_path.open("rb") as stream:
        for row in rows:
            stream.seek(row.source_offset)
            try:
                value = json.loads(stream.read(row.source_length))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise DenseRetrievalError(
                    "DENSE_SOURCE_INVALID", "dense source row is invalid JSON"
                ) from error
            chunk_id = value.get("chunk_id") if isinstance(value, dict) else None
            retrieval_text = value.get("retrieval_text") if isinstance(value, dict) else None
            if (
                chunk_id != row.chunk_id
                or not isinstance(retrieval_text, str)
                or not retrieval_text
            ):
                _fail("DENSE_SOURCE_INVALID", "dense source identity or text drifted")
            ids.append(row.chunk_id)
            texts.append(retrieval_text)
    return tuple(ids), tuple(texts)


def _checkpoint_value(
    identity: DenseBuildIdentity,
    *,
    total_count: int,
    completed_count: int,
    ids_prefix_checksum: str,
    ids_prefix_bytes: int,
    vector_prefix_checksum: str,
    status: Literal["INCOMPLETE", "COMPLETE"],
) -> dict[str, Any]:
    return {
        "schema_version": "dense.store.checkpoint.v2",
        "identity": asdict(identity),
        "total_count": total_count,
        "completed_count": completed_count,
        "ids_prefix_checksum": ids_prefix_checksum,
        "ids_prefix_bytes": ids_prefix_bytes,
        "vector_prefix_checksum": vector_prefix_checksum,
        "status": status,
    }


def _vector_prefix_digest(matrix: Any, completed: int) -> Any:
    digest = hashlib.sha256()
    for start in range(0, completed, 4_096):
        stop = min(start + 4_096, completed)
        digest.update(np.asarray(matrix[start:stop], dtype=np.float16).tobytes(order="C"))
    return digest


def _load_checkpoint(
    path: Path,
    identity: DenseBuildIdentity,
    rows: tuple[DenseSourceRow, ...],
    ids_path: Path,
    matrix: Any,
) -> tuple[int, Any, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DenseRetrievalError(
            "DENSE_CHECKPOINT_INVALID", "dense checkpoint is unreadable"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "dense.store.checkpoint.v2"
        or value.get("identity") != asdict(identity)
        or value.get("total_count") != len(rows)
        or value.get("status") != "INCOMPLETE"
        or not isinstance(value.get("completed_count"), int)
    ):
        _fail(
            "DENSE_CHECKPOINT_FINGERPRINT_MISMATCH",
            "dense checkpoint identity differs from the requested build",
        )
    completed = int(value["completed_count"])
    if completed < 0 or completed > len(rows) or not ids_path.is_file():
        _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint progress is invalid")
    prefix_bytes = value.get("ids_prefix_bytes")
    if not isinstance(prefix_bytes, int) or prefix_bytes < 0:
        _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint ID prefix size is invalid")
    with ids_path.open("r+b") as ids_stream:
        ids_data = ids_stream.read(prefix_bytes)
        if len(ids_data) != prefix_bytes:
            _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint ID prefix is truncated")
        ids_stream.truncate(prefix_bytes)
    if checksum_bytes(ids_data) != value.get("ids_prefix_checksum"):
        _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint ID prefix checksum drifted")
    try:
        ids = tuple(json.loads(line)["chunk_id"] for line in ids_data.splitlines())
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DenseRetrievalError(
            "DENSE_CHECKPOINT_INVALID", "dense checkpoint ID prefix is invalid"
        ) from error
    if ids != tuple(row.chunk_id for row in rows[:completed]):
        _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint ID prefix ordering drifted")
    ids_digest = hashlib.sha256()
    ids_digest.update(ids_data)
    vector_digest = _vector_prefix_digest(matrix, completed)
    if f"sha256:{vector_digest.hexdigest()}" != value.get("vector_prefix_checksum"):
        _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint vector prefix drifted")
    return completed, ids_digest, vector_digest


def _complete_result(
    output_directory: Path,
    checkpoint_path: Path,
    total_count: int,
    identity: DenseBuildIdentity,
) -> DenseBuildResult:
    manifest_path = output_directory / "manifest.json"
    vector_path = output_directory / "vectors.f16.npy"
    ids_path = output_directory / "chunk-ids.jsonl"
    if not manifest_path.is_file() or not vector_path.is_file() or not ids_path.is_file():
        _fail("DENSE_CHECKPOINT_INVALID", "complete checkpoint has no final manifest")
    try:
        manifest = json.loads(manifest_path.read_bytes())
        checkpoint = json.loads(checkpoint_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DenseRetrievalError(
            "DENSE_CHECKPOINT_INVALID", "complete dense metadata is unreadable"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "dense.store.manifest.v2"
        or manifest.get("identity") != asdict(identity)
        or manifest.get("chunk_count") != total_count
        or not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != "dense.store.checkpoint.v2"
        or checkpoint.get("completed_count") != total_count
        or checkpoint.get("total_count") != total_count
    ):
        _fail("DENSE_CHECKPOINT_INVALID", "complete dense metadata identity drifted")
    if _streaming_checksum(vector_path) != manifest.get("vector_checksum") or _streaming_checksum(
        ids_path
    ) != manifest.get("ids_checksum"):
        _fail("DENSE_INDEX_CHECKSUM_MISMATCH", "complete dense files differ from manifest")
    return DenseBuildResult(
        status="COMPLETE",
        completed_count=total_count,
        total_count=total_count,
        checkpoint_checksum=_streaming_checksum(checkpoint_path),
        manifest_checksum=_streaming_checksum(manifest_path),
    )


def build_resumable_dense_store(
    *,
    chunks_path: Path,
    output_directory: Path,
    backend: EmbeddingBackend,
    identity: DenseBuildIdentity,
    source_rows: Sequence[DenseSourceRow],
    batch_size: int,
    checkpoint_interval_rows: int = 1_024,
    maximum_batches: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DenseBuildResult:
    """Build or resume one exact fingerprint; never reinterpret another partial."""

    if (
        batch_size < 1
        or checkpoint_interval_rows < 1
        or (maximum_batches is not None and maximum_batches < 1)
    ):
        _fail("DENSE_BATCH_INVALID", "dense batch and batch budget must be positive")
    if (
        backend.model_id != identity.model_id
        or backend.model_revision != identity.model_revision
        or backend.dimension != identity.dimension
    ):
        _fail("DENSE_MODEL_IDENTITY_MISMATCH", "dense backend differs from build identity")
    rows = _source_rows(source_rows)
    if _streaming_checksum(chunks_path) != identity.corpus_checksum:
        _fail("DENSE_SOURCE_CHECKSUM_MISMATCH", "dense chunk source checksum drifted")

    output_directory.mkdir(parents=True, exist_ok=True)
    vector_partial = output_directory / "vectors.f16.npy.partial"
    ids_partial = output_directory / "chunk-ids.jsonl.partial"
    checkpoint_path = output_directory / "checkpoint.json"
    final_vector = output_directory / "vectors.f16.npy"
    final_ids = output_directory / "chunk-ids.jsonl"
    manifest_path = output_directory / "manifest.json"
    if checkpoint_path.is_file():
        raw_checkpoint = cast(dict[str, Any], json.loads(checkpoint_path.read_bytes()))
        if raw_checkpoint.get("identity") != asdict(identity):
            _fail(
                "DENSE_CHECKPOINT_FINGERPRINT_MISMATCH",
                "dense checkpoint identity differs from the requested build",
            )
        if raw_checkpoint.get("status") == "COMPLETE":
            return _complete_result(output_directory, checkpoint_path, len(rows), identity)
        if not vector_partial.is_file():
            _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint vector partial is absent")
        matrix = np.load(vector_partial, mmap_mode="r+")
        if matrix.shape != (len(rows), identity.dimension) or matrix.dtype != np.float16:
            _fail("DENSE_CHECKPOINT_INVALID", "dense checkpoint vector shape drifted")
        completed, ids_digest, vector_digest = _load_checkpoint(
            checkpoint_path, identity, rows, ids_partial, matrix
        )
        checkpoint_checksum = _streaming_checksum(checkpoint_path)
    else:
        if any(
            path.exists()
            for path in (vector_partial, ids_partial, final_vector, final_ids, manifest_path)
        ):
            _fail("DENSE_CHECKPOINT_INVALID", "unbound dense files already exist")
        matrix = np.lib.format.open_memmap(
            vector_partial,
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), identity.dimension),
        )
        ids_partial.write_bytes(b"")
        completed = 0
        ids_digest = hashlib.sha256()
        vector_digest = hashlib.sha256()
        checkpoint_checksum = _write_checkpoint(
            checkpoint_path,
            _checkpoint_value(
                identity,
                total_count=len(rows),
                completed_count=0,
                ids_prefix_checksum=checksum_bytes(b""),
                ids_prefix_bytes=0,
                vector_prefix_checksum=f"sha256:{vector_digest.hexdigest()}",
                status="INCOMPLETE",
            ),
        )

    batches = 0
    with ids_partial.open("ab") as ids_stream:
        while completed < len(rows):
            stop = min(completed + batch_size, len(rows))
            batch_rows = rows[completed:stop]
            batch_ids, batch_texts = _read_texts(chunks_path, batch_rows)
            vectors = np.asarray(backend.encode_documents(batch_texts), dtype=np.float32)
            if (
                vectors.shape != (len(batch_rows), identity.dimension)
                or not np.isfinite(vectors).all()
            ):
                _fail("DENSE_OUTPUT_SHAPE", "embedding backend returned invalid vectors")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0.0):
                _fail("DENSE_ZERO_VECTOR", "embedding output contains zero")
            matrix[completed:stop] = vectors / norms
            vector_digest.update(
                np.asarray(matrix[completed:stop], dtype=np.float16).tobytes(order="C")
            )
            for row_index, chunk_id in enumerate(batch_ids, start=completed):
                line = content_json_bytes({"row": row_index, "chunk_id": chunk_id})
                ids_stream.write(line)
                ids_digest.update(line)
            completed = stop
            batches += 1
            stopping = maximum_batches is not None and batches >= maximum_batches
            checkpoint_due = (
                completed == len(rows)
                or stopping
                or completed % checkpoint_interval_rows < batch_size
            )
            if checkpoint_due:
                matrix.flush()
                ids_stream.flush()
                os.fsync(ids_stream.fileno())
                checkpoint_checksum = _write_checkpoint(
                    checkpoint_path,
                    _checkpoint_value(
                        identity,
                        total_count=len(rows),
                        completed_count=completed,
                        ids_prefix_checksum=f"sha256:{ids_digest.hexdigest()}",
                        ids_prefix_bytes=ids_stream.tell(),
                        vector_prefix_checksum=f"sha256:{vector_digest.hexdigest()}",
                        status="INCOMPLETE",
                    ),
                )
            if progress is not None:
                progress(completed, len(rows))
            if stopping:
                del matrix
                return DenseBuildResult(
                    status="INCOMPLETE",
                    completed_count=completed,
                    total_count=len(rows),
                    checkpoint_checksum=checkpoint_checksum,
                    manifest_checksum=None,
                )

    del matrix
    os.replace(vector_partial, final_vector)
    os.replace(ids_partial, final_ids)
    vector_checksum = _streaming_checksum(final_vector)
    ids_checksum = _streaming_checksum(final_ids)
    manifest_data = content_json_bytes(
        {
            "schema_version": "dense.store.manifest.v2",
            "identity": asdict(identity),
            "chunk_count": len(rows),
            "vector_checksum": vector_checksum,
            "ids_checksum": ids_checksum,
        }
    )
    manifest_checksum = write_immutable_bytes(manifest_path, manifest_data)
    checkpoint_checksum = _write_checkpoint(
        checkpoint_path,
        _checkpoint_value(
            identity,
            total_count=len(rows),
            completed_count=len(rows),
            ids_prefix_checksum=ids_checksum,
            ids_prefix_bytes=final_ids.stat().st_size,
            vector_prefix_checksum=f"sha256:{vector_digest.hexdigest()}",
            status="COMPLETE",
        ),
    )
    return DenseBuildResult(
        status="COMPLETE",
        completed_count=len(rows),
        total_count=len(rows),
        checkpoint_checksum=checkpoint_checksum,
        manifest_checksum=manifest_checksum,
    )


__all__ = [
    "DenseBuildIdentity",
    "DenseBuildResult",
    "DenseSourceRow",
    "build_resumable_dense_store",
]
