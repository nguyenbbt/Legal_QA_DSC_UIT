"""Local model-backed LegalQA pipeline over immutable exact/BM25 evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from legal_rag.domain.models import AnswerRecord, Evidence, GeneratedAnswer, QuestionRecord
from legal_rag.evaluation.real_retrieval import RealRetrievalIndex, retrieve_question
from legal_rag.retrieval.exact import AliasIndex
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.retrieval.reranker import RerankerBackend, rerank_candidates


class AnswerGenerator(Protocol):
    generator_id: str

    def generate(
        self, question: QuestionRecord, evidence: Sequence[Evidence]
    ) -> GeneratedAnswer: ...


def _evidence(candidate: RetrievalCandidate, rank: int) -> Evidence:
    chunk = candidate.chunk
    return Evidence.model_validate(
        {
            "schema_version": "internal.evidence.v1",
            "evidence_id": chunk.chunk_id,
            "context_id": chunk.context_id,
            "source_url": chunk.source_url,
            "hierarchy_path": chunk.hierarchy_path,
            "canonical_start": chunk.canonical_start,
            "canonical_end": chunk.canonical_end,
            "display_text": chunk.display_text,
            "retrieval_text": chunk.retrieval_text,
            "rank": rank,
            "component_scores": {
                "exact_reference_match": candidate.exact_reference_match,
                "sparse_score": candidate.sparse_score,
                "dense_score": candidate.dense_score,
                "reranker_score": candidate.reranker_score,
            },
            "chunk_checksum": chunk.chunk_checksum,
            "context_checksum": chunk.context_checksum,
            "integrity_status": "valid",
            "claim_support": "unknown",
            "version_validity": "unknown",
        }
    )


def run_model_questions(
    questions: Sequence[QuestionRecord],
    *,
    index: RealRetrievalIndex,
    aliases: AliasIndex,
    reranker: RerankerBackend,
    generator: AnswerGenerator,
    run_id: str,
    evidence_limit: int = 3,
) -> tuple[AnswerRecord, ...]:
    """Run deterministic retrieval→rerank→generation for ordered questions."""

    if evidence_limit < 1:
        raise ValueError("pipeline evidence limit must be positive")
    answers: list[AnswerRecord] = []
    for question in questions:
        retrieved = retrieve_question(question, index=index, aliases=aliases)
        reranked = (
            rerank_candidates(
                question.question,
                retrieved.candidates,
                reranker,
                limit=min(12, max(1, len(retrieved.candidates))),
            )
            if retrieved.candidates
            else ()
        )
        evidence = tuple(
            _evidence(candidate, rank)
            for rank, candidate in enumerate(reranked[:evidence_limit], start=1)
        )
        generated = generator.generate(question, evidence)
        answers.append(
            AnswerRecord.model_validate(
                {
                    "schema_version": "internal.answer.v1",
                    "question_id": question.question_id,
                    "answer": generated.answer_text,
                    "generator_id": generated.generator_id,
                    "evidence_ids": generated.used_evidence_ids,
                    "run_id": run_id,
                }
            )
        )
    return tuple(answers)


__all__ = ["AnswerGenerator", "run_model_questions"]
