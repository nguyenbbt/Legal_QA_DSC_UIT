"""Streaming float16 dense-index storage for the real 641k-chunk corpus."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from legal_rag.domain.checksums import checksum_file
from legal_rag.retrieval.dense import DenseRetrievalError, EmbeddingBackend


@dataclass(frozen=True, slots=True)
class DenseHit:
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class DenseStoreManifest:
    schema_version: str
    model_id: str
    model_revision: str
    source_chunk_checksum: str
    chunk_count: int
    dimension: int
    storage_dtype: str
    vector_checksum: str
    ids_checksum: str


def _chunk_rows(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                chunk_id = row["chunk_id"]
                retrieval_text = row["retrieval_text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DenseRetrievalError(
                    "DENSE_SOURCE_INVALID", f"chunk source row {line_number} is invalid"
                ) from error
            if not isinstance(chunk_id, str) or not isinstance(retrieval_text, str):
                raise DenseRetrievalError(
                    "DENSE_SOURCE_INVALID", f"chunk source row {line_number} is invalid"
                )
            yield chunk_id, retrieval_text


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def build_dense_store(
    chunks_path: Path,
    output_directory: Path,
    backend: EmbeddingBackend,
    *,
    batch_size: int,
) -> DenseStoreManifest:
    """Encode in source order and atomically publish vectors, IDs, and manifest."""

    if batch_size < 1:
        raise DenseRetrievalError("DENSE_BATCH_INVALID", "dense batch size must be positive")
    chunk_count = sum(1 for _ in _chunk_rows(chunks_path))
    if chunk_count < 1:
        raise DenseRetrievalError("DENSE_INDEX_EMPTY", "chunk source is empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    vector_path = output_directory / "vectors.f16.npy"
    ids_path = output_directory / "chunk-ids.jsonl"
    manifest_path = output_directory / "manifest.json"
    temporary_vector = output_directory / "vectors.f16.npy.partial"
    temporary_ids = output_directory / "chunk-ids.jsonl.partial"
    matrix = np.lib.format.open_memmap(
        temporary_vector,
        mode="w+",
        dtype=np.float16,
        shape=(chunk_count, backend.dimension),
    )
    offset = 0
    seen: set[str] = set()
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    with temporary_ids.open("wb") as ids_stream:
        for chunk_id, retrieval_text in _chunk_rows(chunks_path):
            if chunk_id in seen:
                raise DenseRetrievalError(
                    "DENSE_CHUNK_ID_DUPLICATE", "chunk source contains duplicate IDs"
                )
            seen.add(chunk_id)
            batch_ids.append(chunk_id)
            batch_texts.append(retrieval_text)
            if len(batch_ids) < batch_size and offset + len(batch_ids) < chunk_count:
                continue
            staged = sorted(
                zip(batch_ids, batch_texts, strict=True),
                key=lambda item: (len(item[1]), item[0].encode("utf-8")),
            )
            batch_ids[:] = (item[0] for item in staged)
            batch_texts[:] = (item[1] for item in staged)
            vectors = np.asarray(backend.encode_documents(tuple(batch_texts)), dtype=np.float32)
            expected = (len(batch_ids), backend.dimension)
            if vectors.shape != expected or not np.isfinite(vectors).all():
                raise DenseRetrievalError(
                    "DENSE_OUTPUT_SHAPE", "embedding backend returned invalid vectors"
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0.0):
                raise DenseRetrievalError("DENSE_ZERO_VECTOR", "embedding output contains zero")
            matrix[offset : offset + len(batch_ids)] = vectors / norms
            for row_id, value in enumerate(batch_ids, start=offset):
                ids_stream.write(_json_bytes({"row": row_id, "chunk_id": value}))
            offset += len(batch_ids)
            matrix.flush()
            batch_ids.clear()
            batch_texts.clear()
    del matrix
    os.replace(temporary_vector, vector_path)
    os.replace(temporary_ids, ids_path)
    manifest = DenseStoreManifest(
        schema_version="dense.store.manifest.v1",
        model_id=backend.model_id,
        model_revision=backend.model_revision,
        source_chunk_checksum=checksum_file(chunks_path),
        chunk_count=chunk_count,
        dimension=backend.dimension,
        storage_dtype="float16",
        vector_checksum=checksum_file(vector_path),
        ids_checksum=checksum_file(ids_path),
    )
    manifest_path.write_bytes(_json_bytes(asdict(manifest)))
    return manifest


class MemmapDenseIndex:
    """Blockwise exact cosine search over normalized float16 vectors."""

    def __init__(self, directory: Path, *, block_rows: int = 16_384) -> None:
        if block_rows < 1:
            raise ValueError("dense search block size must be positive")
        manifest_row = json.loads((directory / "manifest.json").read_bytes())
        self.manifest = DenseStoreManifest(**manifest_row)
        vector_path = directory / "vectors.f16.npy"
        ids_path = directory / "chunk-ids.jsonl"
        if (
            checksum_file(vector_path) != self.manifest.vector_checksum
            or checksum_file(ids_path) != self.manifest.ids_checksum
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_CHECKSUM_MISMATCH", "dense index differs from its manifest"
            )
        self._vectors = np.load(vector_path, mmap_mode="r")
        self._ids = tuple(
            json.loads(line)["chunk_id"] for line in ids_path.read_bytes().splitlines()
        )
        if (
            self._vectors.shape != (self.manifest.chunk_count, self.manifest.dimension)
            or len(self._ids) != self.manifest.chunk_count
        ):
            raise DenseRetrievalError("DENSE_INDEX_SHAPE", "dense index shape is inconsistent")
        self._block_rows = block_rows

    def retrieve(self, query_embedding: list[float], *, limit: int) -> tuple[DenseHit, ...]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self.manifest.dimension,) or not np.isfinite(query).all():
            raise DenseRetrievalError("DENSE_DIMENSION_MISMATCH", "query vector is invalid")
        norm = np.linalg.norm(query)
        if norm == 0.0 or limit < 1:
            raise DenseRetrievalError("DENSE_QUERY_INVALID", "dense query or limit is invalid")
        query /= norm
        best: list[tuple[float, str]] = []
        for start in range(0, self.manifest.chunk_count, self._block_rows):
            stop = min(start + self._block_rows, self.manifest.chunk_count)
            scores = np.asarray(self._vectors[start:stop], dtype=np.float32) @ query
            local_limit = min(limit, len(scores))
            indices = np.argpartition(scores, -local_limit)[-local_limit:]
            best.extend((float(scores[index]), self._ids[start + int(index)]) for index in indices)
            best = sorted(best, key=lambda item: (-item[0], item[1].encode("utf-8")))[:limit]
        return tuple(DenseHit(chunk_id, score) for score, chunk_id in best)


__all__ = ["DenseHit", "DenseStoreManifest", "MemmapDenseIndex", "build_dense_store"]
