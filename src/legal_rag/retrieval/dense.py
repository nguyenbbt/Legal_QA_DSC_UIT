"""Provider-neutral deterministic dense retrieval primitives.

Framework/model adapters live outside this module.  This boundary deliberately
accepts plain vectors so its ranking and validation behavior remains CPU-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.models import RetrievalCandidate


class DenseRetrievalError(Exception):
    """Stable failure raised before a malformed dense result can be published."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EmbeddingBackend(Protocol):
    """Minimal contract implemented by local embedding model adapters."""

    model_id: str
    model_revision: str
    dimension: int

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class DenseDocument:
    chunk: ChunkRecord
    embedding: tuple[float, ...]


def _unit_vector(values: Sequence[float], *, dimension: int) -> tuple[float, ...]:
    if len(values) != dimension or dimension < 1:
        raise DenseRetrievalError(
            "DENSE_DIMENSION_MISMATCH", "dense vector has an unexpected dimension"
        )
    vector = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in vector):
        raise DenseRetrievalError("DENSE_SCORE_NONFINITE", "dense vector must be finite")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise DenseRetrievalError("DENSE_ZERO_VECTOR", "dense vector must have a positive norm")
    return tuple(value / norm for value in vector)


class DenseIndex:
    """Exact cosine index with deterministic UTF-8 tie-breaking.

    This in-memory implementation is the reference oracle. Large-corpus adapters
    may stream/memmap the same normalized-vector contract without changing ranking.
    """

    def __init__(self, documents: Sequence[DenseDocument], *, dimension: int) -> None:
        if not documents:
            raise DenseRetrievalError("DENSE_INDEX_EMPTY", "dense index contains no documents")
        ids = tuple(document.chunk.chunk_id for document in documents)
        if len(ids) != len(set(ids)):
            raise DenseRetrievalError(
                "DENSE_CHUNK_ID_DUPLICATE", "dense index contains duplicate chunk IDs"
            )
        self.dimension = dimension
        self._documents = tuple(
            DenseDocument(document.chunk, _unit_vector(document.embedding, dimension=dimension))
            for document in documents
        )

    def retrieve(
        self, query_embedding: Sequence[float], *, limit: int
    ) -> tuple[RetrievalCandidate, ...]:
        if limit < 1:
            raise DenseRetrievalError("DENSE_LIMIT_INVALID", "dense limit must be positive")
        query = _unit_vector(query_embedding, dimension=self.dimension)
        scored = (
            (
                sum(left * right for left, right in zip(query, document.embedding, strict=True)),
                document,
            )
            for document in self._documents
        )
        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk.chunk_id.encode("utf-8")))
        return tuple(
            RetrievalCandidate(
                chunk=document.chunk,
                exact_reference_match=False,
                sparse_score=None,
                dense_score=score,
            )
            for score, document in ranked[:limit]
        )


def encode_dense_documents(
    chunks: Sequence[ChunkRecord],
    backend: EmbeddingBackend,
) -> tuple[DenseDocument, ...]:
    """Encode a stable chunk order and prove backend output cardinality."""

    ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id.encode("utf-8")))
    vectors = tuple(backend.encode_documents(tuple(chunk.retrieval_text for chunk in ordered)))
    if len(vectors) != len(ordered):
        raise DenseRetrievalError(
            "DENSE_OUTPUT_CARDINALITY", "embedding backend returned the wrong number of vectors"
        )
    return tuple(
        DenseDocument(chunk, _unit_vector(vector, dimension=backend.dimension))
        for chunk, vector in zip(ordered, vectors, strict=True)
    )


__all__ = [
    "DenseDocument",
    "DenseIndex",
    "DenseRetrievalError",
    "EmbeddingBackend",
    "encode_dense_documents",
]
