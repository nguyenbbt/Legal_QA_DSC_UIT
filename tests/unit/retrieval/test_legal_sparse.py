from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, chunk_context
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.legal_sparse import LegalSparseRetriever
from legal_rag.retrieval.models import RetrievalCandidate


def _candidate(
    context_id: int, text: str, score: float, *, exact: bool = False
) -> RetrievalCandidate:
    context = ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": str(context_id),
            "original_id": str(context_id),
            "original_id_kind": "json_integer",
            "source_position": context_id,
            "source_artifact": f"fixtures/context_{context_id}.json",
            "source_checksum": checksum_bytes(text.encode()),
            "name": None,
            "source_url": f"https://example.invalid/{context_id}",
            "passage": text,
            "indexable": True,
            "quarantine_reason": None,
        }
    )
    chunk = chunk_context(
        context,
        config=ChunkingConfig(minimum_fragment_tokens=1),
    ).chunks[0]
    return RetrievalCandidate(chunk, exact, score)


class _Index:
    index_checksum = "sha256:" + "a" * 64

    def __init__(self, candidates: tuple[RetrievalCandidate, ...]) -> None:
        self.candidates = candidates
        self.limit = 0

    def retrieve(self, query: str, *, candidate_limit: int = 12) -> SparseRetrievalResult:
        self.limit = candidate_limit
        return SparseRetrievalResult(query, ("điều", "5"), self.candidates, (), self.index_checksum)


def test_legal_sparse_expands_pool_and_boosts_matching_hierarchy() -> None:
    body_winner = _candidate(1, "nội dung chung", 10.0)
    heading_winner = _candidate(2, "Điều 5. nội dung", 9.9)
    index = _Index((body_winner, heading_winner))

    result = LegalSparseRetriever(index, discovery_limit=100).retrieve("Điều 5 quy định gì?")

    assert index.limit == 100
    assert result.candidates[0].chunk.chunk_id == heading_winner.chunk.chunk_id
    assert result.index_checksum != index.index_checksum
    assert result.diagnostics[0].code == "LEGAL_SPARSE_BM25F_LITE"


def test_legal_sparse_preserves_exact_reference_precedence() -> None:
    lexical = _candidate(1, "nội dung rất phù hợp", 100.0)
    exact = _candidate(2, "Điều 5.", 0.1, exact=True)

    result = LegalSparseRetriever(_Index((lexical, exact)), discovery_limit=100).retrieve(
        "Điều 5 quy định gì?"
    )

    assert result.candidates[0].chunk.chunk_id == exact.chunk.chunk_id
    assert result.candidates[0].exact_reference_match is True


def test_legal_sparse_zero_scores_remain_finite_and_replayable() -> None:
    index = _Index((_candidate(1, "không khớp", 0.0),))
    retriever = LegalSparseRetriever(index, discovery_limit=200)

    first = retriever.retrieve("Điều 5")
    second = retriever.retrieve("Điều 5")

    assert first == second
    assert first.candidates[0].sparse_score is not None
    assert first.candidates[0].sparse_score >= 0


def test_legal_sparse_candidate_pool_is_bounded() -> None:
    index = _Index(())
    for invalid in (99, 201):
        try:
            LegalSparseRetriever(index, discovery_limit=invalid)
        except ValueError as error:
            assert "[100, 200]" in str(error)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("invalid sparse discovery limit was accepted")
