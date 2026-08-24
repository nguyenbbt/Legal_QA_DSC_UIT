"""Deterministic high-confidence evidence mining for official RAG-SFT rows."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.training.rag_sft import answer_token_coverage


class EvidenceMiningError(Exception):
    """Stable failure at the model-backed evidence mining boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EvidenceSupportBackend(Protocol):
    model_id: str
    model_revision: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class EvidenceMiningConfig:
    minimum_support_score: float
    minimum_answer_token_coverage: float
    maximum_candidates: int = 8
    maximum_evidence: int = 3
    support_policy_version: str = "official-reranker-plus-lexical.v1"

    def __post_init__(self) -> None:
        if not self.support_policy_version:
            raise ValueError("support policy version must be non-empty")
        for threshold in (self.minimum_support_score, self.minimum_answer_token_coverage):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("evidence thresholds must be finite values in [0, 1]")
        if self.maximum_candidates < 1 or not 1 <= self.maximum_evidence <= 3:
            raise ValueError("evidence candidate limits are invalid")


@dataclass(frozen=True, slots=True)
class EvidenceMiningReport:
    candidate_rows: int
    accepted_rows: int
    rejected_rows: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    support_policy_version: str
    model_id: str
    model_revision: str


@dataclass(frozen=True, slots=True)
class EvidenceMiningResult:
    selection_data: bytes
    report: EvidenceMiningReport


SplitName = Literal["train", "development", "local_test"]


def _ordered_candidates(
    answer: str,
    candidates: tuple[RetrievalCandidate, ...],
    backend: EvidenceSupportBackend,
) -> tuple[tuple[RetrievalCandidate, float], ...]:
    scores = tuple(
        float(score)
        for score in backend.score(
            answer, tuple(candidate.chunk.retrieval_text for candidate in candidates)
        )
    )
    if len(scores) != len(candidates):
        raise EvidenceMiningError(
            "EVIDENCE_MINING_OUTPUT_CARDINALITY", "support scorer returned wrong cardinality"
        )
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise EvidenceMiningError(
            "EVIDENCE_MINING_SCORE_INVALID", "support scorer returned an invalid probability"
        )
    return tuple(
        sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk.chunk_id.encode()),
        )
    )


def _select_by_coverage(
    answer: str,
    supported: tuple[tuple[RetrievalCandidate, float], ...],
    *,
    maximum_evidence: int,
    required_coverage: float,
) -> tuple[tuple[RetrievalCandidate, float], ...]:
    selected: list[tuple[RetrievalCandidate, float]] = []
    remaining = list(supported)
    while remaining and len(selected) < maximum_evidence:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -answer_token_coverage(
                    answer,
                    tuple(candidate.chunk for candidate, _ in (*selected, item)),
                ),
                -item[1],
                item[0].chunk.chunk_id.encode(),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        remaining.remove(chosen)
        if (
            answer_token_coverage(answer, tuple(candidate.chunk for candidate, _ in selected))
            >= required_coverage
        ):
            break
    return tuple(selected)


def mine_evidence_selections(
    *,
    questions: tuple[QuestionRecord, ...],
    split_by_question: Mapping[str, SplitName],
    retrieve: Callable[[QuestionRecord], Sequence[RetrievalCandidate]],
    backend: EvidenceSupportBackend,
    config: EvidenceMiningConfig,
) -> EvidenceMiningResult:
    """Select up to three answer-supporting chunks; unsupported rows stay excluded."""

    rows: list[bytes] = []
    reasons: Counter[str] = Counter()
    train_questions = tuple(
        sorted(
            (
                question
                for question in questions
                if split_by_question.get(question.question_id) == "train"
            ),
            key=lambda question: question.question_id.encode(),
        )
    )
    for question in train_questions:
        answer = question.answer
        if answer is None:
            raise EvidenceMiningError(
                "EVIDENCE_MINING_TARGET_MISSING", "official train answer is missing"
            )
        retrieved = tuple(retrieve(question))
        if not retrieved:
            reasons["retrieval_empty"] += 1
            continue
        if len(retrieved) > 50:
            raise EvidenceMiningError(
                "EVIDENCE_MINING_CANDIDATE_LIMIT", "retrieval candidate universe exceeds 50"
            )
        candidate_ids = tuple(candidate.chunk.chunk_id for candidate in retrieved)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise EvidenceMiningError(
                "EVIDENCE_MINING_CANDIDATE_DUPLICATE", "retrieval candidates are not unique"
            )
        admitted = tuple(
            candidate
            for _, candidate in sorted(
                enumerate(retrieved),
                key=lambda item: (
                    -answer_token_coverage(answer, (item[1].chunk,)),
                    item[0],
                    item[1].chunk.chunk_id.encode(),
                ),
            )[: config.maximum_candidates]
        )
        ordered = _ordered_candidates(answer, admitted, backend)
        supported = tuple(item for item in ordered if item[1] >= config.minimum_support_score)
        if not supported:
            reasons["support_below_threshold"] += 1
            continue
        supported = _select_by_coverage(
            answer,
            supported,
            maximum_evidence=config.maximum_evidence,
            required_coverage=config.minimum_answer_token_coverage,
        )
        evidence = tuple(candidate.chunk for candidate, _ in supported)
        if answer_token_coverage(answer, evidence) < config.minimum_answer_token_coverage:
            reasons["coverage_below_threshold"] += 1
            continue
        rows.append(
            content_json_bytes(
                {
                    "schema_version": "training.evidence.selection.v1",
                    "question_id": question.question_id,
                    "question_checksum": checksum_bytes(question.question.encode()),
                    "evidence_ids": [chunk.chunk_id for chunk in evidence],
                    "evidence_checksums": [chunk.chunk_checksum for chunk in evidence],
                    "support_score": min(score for _, score in supported),
                    "support_policy_version": config.support_policy_version,
                }
            )
        )
    accepted = len(rows)
    candidate_count = len(train_questions)
    report = EvidenceMiningReport(
        candidate_rows=candidate_count,
        accepted_rows=accepted,
        rejected_rows=candidate_count - accepted,
        rejected_by_reason=tuple(sorted(reasons.items())),
        support_policy_version=config.support_policy_version,
        model_id=backend.model_id,
        model_revision=backend.model_revision,
    )
    return EvidenceMiningResult(b"".join(rows), report)


__all__ = [
    "EvidenceMiningConfig",
    "EvidenceMiningError",
    "EvidenceMiningReport",
    "EvidenceMiningResult",
    "EvidenceSupportBackend",
    "mine_evidence_selections",
]
