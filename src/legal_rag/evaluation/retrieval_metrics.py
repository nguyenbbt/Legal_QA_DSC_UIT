"""Exact label-backed retrieval metrics for the private MIL-004 benchmark."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


class RetrievalEvaluationError(Exception):
    """Stable failure at the retrieval-evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RetrievalLabelRow:
    question_id: str
    relevant_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalOutputRow:
    question_id: str
    retrieved_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMetricReport:
    benchmark_question_count: int
    retrieval_evaluable_count: int
    retrieval_unevaluable_count: int
    unevaluable_question_ids: tuple[str, ...]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    evidence_set_recall_at_10: float


@dataclass(frozen=True, slots=True)
class ContainmentInputRow:
    question_id: str
    gold_answer: str
    retrieved_display_texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContainmentMetricReport:
    metric_namespace: str
    total_question_count: int
    eligible_question_count: int
    excluded_question_count: int
    excluded: tuple[tuple[str, str], ...]
    containment_at_1: float
    containment_at_5: float
    containment_at_10: float


def _ordered_rows(rows: tuple[RetrievalLabelRow, ...]) -> tuple[RetrievalLabelRow, ...]:
    ordered = tuple(sorted(rows, key=lambda row: row.question_id.encode("utf-8")))
    if any(not row.question_id for row in ordered):
        raise RetrievalEvaluationError(
            "RETRIEVAL_LABEL_ID_INVALID", "question IDs must be non-empty"
        )
    if len({row.question_id for row in ordered}) != len(ordered):
        raise RetrievalEvaluationError(
            "RETRIEVAL_LABEL_ID_DUPLICATE", "label question IDs must be unique"
        )
    if any(
        len(set(row.relevant_evidence_ids)) != len(row.relevant_evidence_ids) for row in ordered
    ):
        raise RetrievalEvaluationError(
            "RETRIEVAL_LABEL_DUPLICATE", "label rows cannot contain duplicate evidence IDs"
        )
    return ordered


def _outputs_by_id(rows: tuple[RetrievalOutputRow, ...]) -> dict[str, RetrievalOutputRow]:
    values: dict[str, RetrievalOutputRow] = {}
    for row in rows:
        if not row.question_id or row.question_id in values:
            raise RetrievalEvaluationError(
                "RETRIEVAL_OUTPUT_ID_DUPLICATE", "retrieval output question IDs must be unique"
            )
        if len(set(row.retrieved_evidence_ids)) != len(row.retrieved_evidence_ids):
            raise RetrievalEvaluationError(
                "RETRIEVAL_OUTPUT_DUPLICATE",
                "retrieval output rows cannot contain duplicate evidence IDs",
            )
        values[row.question_id] = row
    return values


def _recall(relevant: frozenset[str], retrieved: tuple[str, ...], k: int) -> float:
    return float(len(relevant.intersection(retrieved[:k]))) / float(len(relevant))


def _reciprocal_rank(relevant: frozenset[str], retrieved: tuple[str, ...]) -> float:
    for rank, evidence_id in enumerate(retrieved[:10], start=1):
        if evidence_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def evaluate_retrieval(
    labels: tuple[RetrievalLabelRow, ...],
    outputs: tuple[RetrievalOutputRow, ...],
) -> RetrievalMetricReport:
    """Evaluate identical labeled/output IDs, excluding empty relevance sets."""

    ordered_labels = _ordered_rows(labels)
    output_by_id = _outputs_by_id(outputs)
    label_ids = {row.question_id for row in ordered_labels}
    if label_ids != set(output_by_id):
        raise RetrievalEvaluationError(
            "RETRIEVAL_EVAL_ID_MISMATCH",
            "labels and retrieval outputs must cover identical question IDs",
        )
    evaluable = tuple(row for row in ordered_labels if row.relevant_evidence_ids)
    unevaluable = tuple(row.question_id for row in ordered_labels if not row.relevant_evidence_ids)
    if not evaluable:
        raise RetrievalEvaluationError(
            "RETRIEVAL_EVAL_EMPTY", "retrieval evaluation has no evaluable questions"
        )
    recall_1 = 0.0
    recall_5 = 0.0
    recall_10 = 0.0
    reciprocal_ranks = 0.0
    evidence_set_hits = 0.0
    for label in evaluable:
        relevant = frozenset(label.relevant_evidence_ids)
        retrieved = output_by_id[label.question_id].retrieved_evidence_ids
        recall_1 += _recall(relevant, retrieved, 1)
        recall_5 += _recall(relevant, retrieved, 5)
        recall_10 += _recall(relevant, retrieved, 10)
        reciprocal_ranks += _reciprocal_rank(relevant, retrieved)
        evidence_set_hits += float(relevant.issubset(retrieved[:10]))
    denominator = float(len(evaluable))
    return RetrievalMetricReport(
        benchmark_question_count=len(ordered_labels),
        retrieval_evaluable_count=len(evaluable),
        retrieval_unevaluable_count=len(unevaluable),
        unevaluable_question_ids=unevaluable,
        recall_at_1=recall_1 / denominator,
        recall_at_5=recall_5 / denominator,
        recall_at_10=recall_10 / denominator,
        mrr_at_10=reciprocal_ranks / denominator,
        evidence_set_recall_at_10=evidence_set_hits / denominator,
    )


def _containment_view(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def evaluate_answer_containment(
    rows: tuple[ContainmentInputRow, ...],
) -> ContainmentMetricReport:
    """Report answer substring containment without treating it as relevance gold."""

    ordered = tuple(sorted(rows, key=lambda row: row.question_id.encode("utf-8")))
    ids = tuple(row.question_id for row in ordered)
    if any(not question_id for question_id in ids) or len(ids) != len(set(ids)):
        raise RetrievalEvaluationError(
            "CONTAINMENT_ID_INVALID", "containment question IDs must be non-empty and unique"
        )
    eligible: list[tuple[str, tuple[str, ...]]] = []
    excluded: list[tuple[str, str]] = []
    for row in ordered:
        answer = _containment_view(row.gold_answer)
        if not answer:
            excluded.append((row.question_id, "EMPTY_GOLD_ANSWER"))
            continue
        displays = tuple(_containment_view(text) for text in row.retrieved_display_texts)
        eligible.append((answer, displays))

    def rate(k: int) -> float:
        if not eligible:
            return 0.0
        hits = sum(
            any(answer in display for display in displays[:k]) for answer, displays in eligible
        )
        return float(hits) / float(len(eligible))

    return ContainmentMetricReport(
        metric_namespace="diagnostic_answer_containment",
        total_question_count=len(ordered),
        eligible_question_count=len(eligible),
        excluded_question_count=len(excluded),
        excluded=tuple(excluded),
        containment_at_1=rate(1),
        containment_at_5=rate(5),
        containment_at_10=rate(10),
    )
