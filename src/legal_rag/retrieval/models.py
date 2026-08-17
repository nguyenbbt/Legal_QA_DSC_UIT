"""Typed provider-neutral retrieval results shared by exact and sparse paths."""

from __future__ import annotations

from dataclasses import dataclass

from legal_rag.ingestion.chunking import ChunkRecord


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    chunk: ChunkRecord
    exact_reference_match: bool
    sparse_score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostic:
    code: str
    message: str
    parser_version: str = "legal-reference-parser.v1"
    document_key_version: str = "legal-document-number-key.v1"
    alias_manifest_checksum: str | None = None
    candidate_count: int = 0
