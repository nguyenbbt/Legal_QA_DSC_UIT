"""Fail-closed execution-only contracts for sparse resource remediation."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult

if TYPE_CHECKING:
    from legal_rag.retrieval.disk_bm25 import DiskBm25Index

SCORE_ABSOLUTE_TOLERANCE = 1e-12


def open_worker_read_only_connection(database_path: Path) -> sqlite3.Connection:
    """Open immutable SQLite data in a worker and permit orchestrator cleanup."""

    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


class BoundedTopKDiskBm25Index:
    """Expose the remediated execution behind the unchanged sparse interface."""

    def __init__(self, source: DiskBm25Index) -> None:
        self._source = source
        self.index_checksum = source.index_checksum

    def retrieve(self, query: str, *, candidate_limit: int = 12) -> SparseRetrievalResult:
        return self._source.retrieve_bounded_top_k(query, candidate_limit=candidate_limit)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return self._source.chunks_for_context(context_id)

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return self._source.chunks_for_coordinate(hierarchy_kind, hierarchy_ordinal)


class SparseExecutionError(Exception):
    """Stable fail-closed error for D066-R1 execution evidence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise SparseExecutionError(code, message)


@dataclass(frozen=True, slots=True)
class SparseExecutionContract:
    index_checksum: str
    retrieval_version: str
    corpus_checksum: str
    chunks_artifact_checksum: str
    tokenizer_id: str
    tokenizer_revision: str
    k1: float
    b: float
    document_count: int
    candidate_limit: int

    @classmethod
    def from_index(cls, index: DiskBm25Index, *, candidate_limit: int) -> SparseExecutionContract:
        if candidate_limit < 1 or candidate_limit > 200:
            raise ValueError("sparse candidate limit must be within [1, 200]")
        manifest = index.manifest
        return cls(
            index_checksum=index.index_checksum,
            retrieval_version=manifest.retrieval_version,
            corpus_checksum=manifest.corpus_checksum,
            chunks_artifact_checksum=manifest.chunks_artifact_checksum,
            tokenizer_id=manifest.tokenizer_id,
            tokenizer_revision=manifest.tokenizer_revision,
            k1=manifest.k1,
            b=manifest.b,
            document_count=manifest.document_count,
            candidate_limit=candidate_limit,
        )


@dataclass(frozen=True, slots=True)
class SparseCompatibilityReport:
    schema_version: Literal["retrieval.sparse-execution-compatibility.v1"]
    status: Literal["PASS"]
    candidate_limit: int
    mismatched_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SparseParityReport:
    schema_version: Literal["retrieval.sparse-execution-parity.v1"]
    candidate_count: int
    score_absolute_tolerance: float
    maximum_absolute_score_delta: float
    identical_chunk_ids_and_order: Literal[True]


def audit_sparse_execution_contract(
    index: DiskBm25Index,
    expected: SparseExecutionContract,
    *,
    candidate_limit: int,
) -> SparseCompatibilityReport:
    """Reject execution reuse unless every semantic identity remains exact."""

    actual = SparseExecutionContract.from_index(index, candidate_limit=candidate_limit)
    actual_fields = asdict(actual)
    expected_fields = asdict(expected)
    mismatches = tuple(
        name for name in actual_fields if actual_fields[name] != expected_fields[name]
    )
    if mismatches:
        _fail(
            "SPARSE_EXECUTION_CONTRACT_MISMATCH",
            "sparse execution identities differ: " + ", ".join(mismatches),
        )
    return SparseCompatibilityReport(
        schema_version="retrieval.sparse-execution-compatibility.v1",
        status="PASS",
        candidate_limit=candidate_limit,
        mismatched_fields=(),
    )


def assert_sparse_result_parity(
    reference: SparseRetrievalResult,
    candidate: SparseRetrievalResult,
    *,
    score_tolerance: float = SCORE_ABSOLUTE_TOLERANCE,
) -> SparseParityReport:
    """Prove identical ranking identity and bounded binary64 score drift."""

    if not math.isfinite(score_tolerance) or score_tolerance < 0.0:
        raise ValueError("score tolerance must be finite and non-negative")
    reference_ids = tuple(item.chunk.chunk_id for item in reference.candidates)
    candidate_ids = tuple(item.chunk.chunk_id for item in candidate.candidates)
    if (
        reference.query != candidate.query
        or reference.query_terms != candidate.query_terms
        or reference.diagnostics != candidate.diagnostics
        or reference.index_checksum != candidate.index_checksum
        or reference_ids != candidate_ids
    ):
        _fail(
            "SPARSE_EXECUTION_PARITY_MISMATCH",
            "sparse execution changed query, diagnostics, index, or ranked chunk order",
        )
    deltas: list[float] = []
    for reference_item, candidate_item in zip(
        reference.candidates, candidate.candidates, strict=True
    ):
        reference_score = reference_item.sparse_score
        candidate_score = candidate_item.sparse_score
        if reference_score is None or candidate_score is None:
            if reference_score != candidate_score:
                _fail("SPARSE_EXECUTION_PARITY_MISMATCH", "sparse score presence changed")
            deltas.append(0.0)
            continue
        delta = abs(reference_score - candidate_score)
        if not math.isfinite(delta) or delta > score_tolerance:
            _fail(
                "SPARSE_EXECUTION_PARITY_MISMATCH",
                "sparse score exceeds the declared absolute tolerance",
            )
        deltas.append(delta)
    return SparseParityReport(
        schema_version="retrieval.sparse-execution-parity.v1",
        candidate_count=len(reference_ids),
        score_absolute_tolerance=score_tolerance,
        maximum_absolute_score_delta=max(deltas, default=0.0),
        identical_chunk_ids_and_order=True,
    )


__all__ = [
    "BoundedTopKDiskBm25Index",
    "SCORE_ABSOLUTE_TOLERANCE",
    "SparseCompatibilityReport",
    "SparseExecutionContract",
    "SparseExecutionError",
    "SparseParityReport",
    "assert_sparse_result_parity",
    "audit_sparse_execution_contract",
    "open_worker_read_only_connection",
]
