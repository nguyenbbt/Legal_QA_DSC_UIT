"""Exact deterministic bm25.v1 reference implementation."""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, NoReturn

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.exact import DOCUMENT_KEY_VERSION, REFERENCE_PARSER_VERSION
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic
from legal_rag.retrieval.tokenizer import (
    RETRIEVAL_TOKENIZER_ID,
    RETRIEVAL_TOKENIZER_REVISION,
    retrieval_token_values,
)

APPROVED_BM25_RUNTIME_ID = "bm25rt.v1-cpython-3.12.7-windows-10.0.26200-x86_64-ucrt-10.0.26100.8875"
BM25_VERSION = "bm25.v1"
BM25_K1 = 1.2
BM25_B = 0.75


class SparseRetrievalError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise SparseRetrievalError(code, message)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _normalized_terms(text: str) -> tuple[str, ...]:
    view = unicodedata.normalize("NFC", text).casefold()
    return retrieval_token_values(view)


def ordered_unique_query_terms(query: str) -> tuple[str, ...]:
    return tuple(sorted(set(_normalized_terms(query)), key=lambda value: value.encode("utf-8")))


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    chunk: ChunkRecord
    term_frequencies: Counter[str]
    length: int


@dataclass(frozen=True, slots=True)
class SparseRetrievalResult:
    query: str
    query_terms: tuple[str, ...]
    candidates: tuple[RetrievalCandidate, ...]
    diagnostics: tuple[RetrievalDiagnostic, ...]
    index_checksum: str

    def json_bytes(self) -> bytes:
        return _json_bytes(
            {
                "schema_version": "sparse.retrieval.v1",
                "query": self.query,
                "query_terms": list(self.query_terms),
                "retrieval_version": BM25_VERSION,
                "index_checksum": self.index_checksum,
                "candidates": [
                    {
                        "chunk_id": candidate.chunk.chunk_id,
                        "exact_reference_match": candidate.exact_reference_match,
                        "sparse_score": candidate.sparse_score,
                    }
                    for candidate in self.candidates
                ],
                "diagnostics": [
                    {
                        "code": diagnostic.code,
                        "message": diagnostic.message,
                        "candidate_count": diagnostic.candidate_count,
                    }
                    for diagnostic in self.diagnostics
                ],
            }
        )


class Bm25Index:
    def __init__(
        self,
        documents: tuple[_IndexedDocument, ...],
        *,
        corpus_checksum: str,
        alias_manifest_checksum: str,
        runtime_compatibility_id: str,
    ) -> None:
        self._documents = documents
        self.corpus_checksum = corpus_checksum
        self.alias_manifest_checksum = alias_manifest_checksum
        self.runtime_compatibility_id = runtime_compatibility_id
        self.document_count = len(documents)
        total_length = sum(document.length for document in documents)
        self.average_length = (
            float(total_length) / float(self.document_count) if self.document_count else 0.0
        )
        self._document_frequencies: Counter[str] = Counter()
        for document in documents:
            self._document_frequencies.update(document.term_frequencies.keys())
        self.index_checksum = checksum_bytes(self.manifest_bytes())

    def manifest_bytes(self) -> bytes:
        return _json_bytes(
            {
                "schema_version": "bm25.index.manifest.v1",
                "retrieval_version": BM25_VERSION,
                "parameters": {"k1": BM25_K1, "b": BM25_B},
                "tokenizer_id": RETRIEVAL_TOKENIZER_ID,
                "tokenizer_revision": RETRIEVAL_TOKENIZER_REVISION,
                "corpus_checksum": self.corpus_checksum,
                "chunking_version": "chunking.v1",
                "document_count": self.document_count,
                "average_length": self.average_length,
                "legal_reference_parser": REFERENCE_PARSER_VERSION,
                "document_key_version": DOCUMENT_KEY_VERSION,
                "unicode_version": unicodedata.unidata_version,
                "alias_manifest_checksum": self.alias_manifest_checksum,
                "runtime_compatibility_id": self.runtime_compatibility_id,
                "documents": [
                    {
                        "chunk_id": document.chunk.chunk_id,
                        "chunk_checksum": document.chunk.chunk_checksum,
                        "document_length": document.length,
                    }
                    for document in self._documents
                ],
            }
        )

    def _score(self, document: _IndexedDocument, query_terms: tuple[str, ...]) -> float:
        score = 0.0
        for term in query_terms:
            tf_value = float(document.term_frequencies.get(term, 0))
            df_value = float(self._document_frequencies.get(term, 0))
            ratio = (float(self.document_count) - df_value + 0.5) / (df_value + 0.5)
            idf = math.log1p(ratio)
            length_ratio = float(document.length) / self.average_length
            length_norm = (1.0 - BM25_B) + (BM25_B * length_ratio)
            numerator = tf_value * (BM25_K1 + 1.0)
            denominator = tf_value + (BM25_K1 * length_norm)
            term_score = idf * (numerator / denominator)
            score = score + term_score
            if not all(
                math.isfinite(value)
                for value in (
                    tf_value,
                    df_value,
                    ratio,
                    idf,
                    length_ratio,
                    length_norm,
                    numerator,
                    denominator,
                    term_score,
                    score,
                )
            ):
                _fail("SPARSE_SCORE_NONFINITE", "BM25 produced a non-finite value")
        return 0.0 if score == 0.0 else score

    def retrieve(self, query: str) -> SparseRetrievalResult:
        canonical_query = unicodedata.normalize("NFC", query)
        if self.document_count == 0:
            return SparseRetrievalResult(
                query=canonical_query,
                query_terms=(),
                candidates=(),
                diagnostics=(
                    RetrievalDiagnostic(
                        code="SPARSE_INDEX_EMPTY",
                        message="BM25 index has no documents",
                    ),
                ),
                index_checksum=self.index_checksum,
            )
        query_terms = ordered_unique_query_terms(canonical_query)
        if not query_terms:
            return SparseRetrievalResult(
                query=canonical_query,
                query_terms=(),
                candidates=(),
                diagnostics=(
                    RetrievalDiagnostic(
                        code="SPARSE_QUERY_EMPTY",
                        message="query contains no retrieval tokens",
                    ),
                ),
                index_checksum=self.index_checksum,
            )
        scored = [(document, self._score(document, query_terms)) for document in self._documents]
        ranked = sorted(
            ((document, score) for document, score in scored if score > 0.0),
            key=lambda item: (-item[1], item[0].chunk.chunk_id),
        )[:12]
        candidates = tuple(
            RetrievalCandidate(
                chunk=document.chunk,
                exact_reference_match=False,
                sparse_score=score,
            )
            for document, score in ranked
        )
        return SparseRetrievalResult(
            query=canonical_query,
            query_terms=query_terms,
            candidates=candidates,
            diagnostics=(),
            index_checksum=self.index_checksum,
        )


def build_bm25_index(
    chunks: tuple[ChunkRecord, ...],
    *,
    corpus_checksum: str,
    alias_manifest_checksum: str,
    runtime_compatibility_id: str,
) -> Bm25Index:
    if runtime_compatibility_id != APPROVED_BM25_RUNTIME_ID:
        _fail(
            "BM25_RUNTIME_UNAPPROVED",
            "BM25 runtime compatibility ID is not owner-approved",
        )
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(chunk_ids) != len(set(chunk_ids)):
        _fail("SPARSE_CHUNK_DUPLICATE", "BM25 input contains a duplicate chunk ID")
    documents = tuple(
        _IndexedDocument(
            chunk=chunk,
            term_frequencies=Counter(_normalized_terms(chunk.retrieval_text)),
            length=len(_normalized_terms(chunk.retrieval_text)),
        )
        for chunk in chunks
    )
    if documents and sum(document.length for document in documents) == 0:
        _fail(
            "SPARSE_INDEX_ZERO_AVGDL",
            "non-empty BM25 index cannot have zero average document length",
        )
    return Bm25Index(
        documents,
        corpus_checksum=corpus_checksum,
        alias_manifest_checksum=alias_manifest_checksum,
        runtime_compatibility_id=runtime_compatibility_id,
    )
