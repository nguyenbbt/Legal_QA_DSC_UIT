"""Exact label-backed retrieval metrics for the private MIL-004 benchmark."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from math import log2, sqrt
from typing import Literal


class RetrievalEvaluationError(Exception):
    """Stable failure at the retrieval-evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class GradedEvidenceLabel:
    evidence_id: str
    relevance: Literal["relevant", "partially_relevant", "not_relevant"]


@dataclass(frozen=True, slots=True)
class RetrievalLabelRow:
    question_id: str
    relevant_evidence_ids: tuple[str, ...]
    graded_evidence: tuple[GradedEvidenceLabel, ...] = ()
    hierarchy_context_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCandidateMetadata:
    evidence_id: str
    context_id: str
    hierarchy_path: tuple[str, ...]
    canonical_start: int
    canonical_end: int
    token_cost: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalOutputRow:
    question_id: str
    retrieved_evidence_ids: tuple[str, ...]
    candidate_metadata: tuple[RetrievalCandidateMetadata, ...] = ()


@dataclass(frozen=True, slots=True)
class UnevaluableRetrievalQuestion:
    question_id: str
    reason: Literal["NO_RELEVANT_EVIDENCE"]


@dataclass(frozen=True, slots=True)
class RetrievalMetricReport:
    benchmark_question_count: int
    retrieval_evaluable_count: int
    retrieval_unevaluable_count: int
    unevaluable_question_ids: tuple[str, ...]
    unevaluable: tuple[UnevaluableRetrievalQuestion, ...]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    evidence_set_recall_at_10: float


SetFailureClass = Literal[
    "UNRESOLVED_LABEL_OR_GOLD",
    "DISCOVERY_MISS",
    "RANKING_MISS",
    "SET_COVERAGE_MISS",
    "REDUNDANCY_ERROR",
    "HIERARCHY_CONTEXT_MISS",
    "CORRECT_EVIDENCE_GENERATION_ERROR",
    "NO_FAILURE",
]
CorrelationReason = Literal["INSUFFICIENT_PAIRS", "ZERO_VARIANCE"]


@dataclass(frozen=True, slots=True)
class RetrievalSetQuestionMetrics:
    question_id: str
    precision_at_1: float
    precision_at_3: float
    evidence_set_recall_at_3: float
    ndcg_at_10: float
    required_evidence_coverage_at_1: float | None
    required_evidence_coverage_at_3: float | None
    required_evidence_unavailable_reason: Literal["NO_REQUIRED_EVIDENCE"] | None
    duplicate_span_pair_rate_at_3: float | None
    positive_overlap_pair_rate_at_3: float | None
    parent_child_pair_rate_at_3: float | None
    hierarchy_diversity_at_3: float | None
    metadata_unavailable_reason: Literal["CANDIDATE_METADATA_UNAVAILABLE"] | None
    unique_legal_coordinate_coverage_at_3: float | None
    document_context_hit_at_1: float | None
    document_context_hit_at_3: float | None
    coordinate_metadata_unavailable_reason: (
        Literal["RELEVANT_COORDINATE_METADATA_INCOMPLETE"] | None
    )
    evidence_count_at_3: int
    token_cost_at_3: int | None
    token_cost_unavailable_reason: Literal["TOKEN_COST_UNAVAILABLE"] | None
    primary_failure_class: SetFailureClass


@dataclass(frozen=True, slots=True)
class MetricCorrelation:
    retrieval_metric: str
    answer_metric: Literal["meteor", "rouge_l"]
    pair_count: int
    value: float | None
    reason: CorrelationReason | None


@dataclass(frozen=True, slots=True)
class RetrievalSetMetricReport:
    schema_version: Literal["retrieval.set-evaluation.v1"]
    benchmark_question_count: int
    retrieval_evaluable_count: int
    retrieval_unevaluable_count: int
    unevaluable_question_ids: tuple[str, ...]
    questions: tuple[RetrievalSetQuestionMetrics, ...]
    mean_precision_at_1: float
    mean_precision_at_3: float
    mean_evidence_set_recall_at_3: float
    mean_ndcg_at_10: float
    mean_unique_legal_coordinate_coverage_at_3: float | None
    mean_document_context_hit_at_1: float | None
    mean_document_context_hit_at_3: float | None
    required_evidence_eligible_count: int
    metadata_available_count: int
    coordinate_metric_available_count: int
    evidence_count_distribution_at_3: tuple[int, int, int, int]
    token_cost_available_count: int
    token_cost_unavailable_count: int
    token_cost_total_at_3: int | None
    token_cost_mean_at_3: float | None
    token_cost_minimum_at_3: int | None
    token_cost_maximum_at_3: int | None
    correlations: tuple[MetricCorrelation, ...]


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
    for row in ordered:
        graded_ids = tuple(label.evidence_id for label in row.graded_evidence)
        if len(graded_ids) != len(set(graded_ids)):
            raise RetrievalEvaluationError(
                "RETRIEVAL_GRADED_LABEL_DUPLICATE",
                "graded label rows cannot contain duplicate evidence IDs",
            )
        if row.graded_evidence:
            positive_ids = tuple(
                label.evidence_id
                for label in row.graded_evidence
                if label.relevance in {"relevant", "partially_relevant"}
            )
            if frozenset(positive_ids) != frozenset(row.relevant_evidence_ids):
                raise RetrievalEvaluationError(
                    "RETRIEVAL_GRADED_LABEL_MISMATCH",
                    "graded positive IDs must match relevant evidence IDs",
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
        if row.candidate_metadata:
            metadata_ids = tuple(item.evidence_id for item in row.candidate_metadata)
            if metadata_ids != row.retrieved_evidence_ids:
                raise RetrievalEvaluationError(
                    "RETRIEVAL_OUTPUT_METADATA_MISMATCH",
                    "candidate metadata must cover retrieved IDs in identical order",
                )
            if any(
                not item.evidence_id
                or not item.context_id
                or not item.hierarchy_path
                or item.canonical_start < 0
                or item.canonical_start >= item.canonical_end
                or (item.token_cost is not None and item.token_cost < 0)
                for item in row.candidate_metadata
            ):
                raise RetrievalEvaluationError(
                    "RETRIEVAL_OUTPUT_METADATA_INVALID",
                    "candidate metadata contains an invalid identity, span, hierarchy, or cost",
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
        unevaluable=tuple(
            UnevaluableRetrievalQuestion(question_id, "NO_RELEVANT_EVIDENCE")
            for question_id in unevaluable
        ),
        recall_at_1=recall_1 / denominator,
        recall_at_5=recall_5 / denominator,
        recall_at_10=recall_10 / denominator,
        mrr_at_10=reciprocal_ranks / denominator,
        evidence_set_recall_at_10=evidence_set_hits / denominator,
    )


def _gain_by_id(label: RetrievalLabelRow) -> dict[str, int]:
    return {
        item.evidence_id: {
            "relevant": 2,
            "partially_relevant": 1,
            "not_relevant": 0,
        }[item.relevance]
        for item in label.graded_evidence
    }


def _ndcg_at_10(label: RetrievalLabelRow, retrieved: tuple[str, ...]) -> float:
    gains = _gain_by_id(label)
    dcg = sum(
        float(gains.get(evidence_id, 0)) / log2(float(rank + 1))
        for rank, evidence_id in enumerate(retrieved[:10], start=1)
    )
    ideal = sorted(gains.values(), reverse=True)[:10]
    idcg = sum(float(gain) / log2(float(rank + 1)) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _is_parent_child(left: RetrievalCandidateMetadata, right: RetrievalCandidateMetadata) -> bool:
    if left.context_id != right.context_id or left.hierarchy_path == right.hierarchy_path:
        return False
    shorter, longer = sorted((left.hierarchy_path, right.hierarchy_path), key=len)
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def _pair_metrics(
    metadata: tuple[RetrievalCandidateMetadata, ...],
) -> tuple[float, float, float, float]:
    selected = metadata[:3]
    pair_count = len(selected) * (len(selected) - 1) // 2
    duplicate_spans = 0
    overlaps = 0
    parent_children = 0
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            same_context = left.context_id == right.context_id
            if same_context and (left.canonical_start, left.canonical_end) == (
                right.canonical_start,
                right.canonical_end,
            ):
                duplicate_spans += 1
            if same_context and min(left.canonical_end, right.canonical_end) > max(
                left.canonical_start, right.canonical_start
            ):
                overlaps += 1
            if _is_parent_child(left, right):
                parent_children += 1
    denominator = float(pair_count) if pair_count else 1.0
    diversity = (
        float(len({(item.context_id, item.hierarchy_path) for item in selected}))
        / float(len(selected))
        if selected
        else 0.0
    )
    return (
        float(duplicate_spans) / denominator,
        float(overlaps) / denominator,
        float(parent_children) / denominator,
        diversity,
    )


def _coordinate_metrics(
    relevant: frozenset[str],
    retrieved: tuple[str, ...],
    metadata: tuple[RetrievalCandidateMetadata, ...],
) -> tuple[float | None, float | None, float | None]:
    by_id = {item.evidence_id: item for item in metadata}
    if any(evidence_id not in by_id for evidence_id in relevant):
        return None, None, None
    positive_coordinates = {
        (by_id[evidence_id].context_id, by_id[evidence_id].hierarchy_path)
        for evidence_id in relevant
    }
    hit_coordinates = {
        (by_id[evidence_id].context_id, by_id[evidence_id].hierarchy_path)
        for evidence_id in retrieved[:3]
        if evidence_id in relevant
    }
    positive_contexts = {coordinate[0] for coordinate in positive_coordinates}
    return (
        float(len(hit_coordinates)) / float(len(positive_coordinates)),
        float(
            any(by_id[evidence_id].context_id in positive_contexts for evidence_id in retrieved[:1])
        ),
        float(
            any(by_id[evidence_id].context_id in positive_contexts for evidence_id in retrieved[:3])
        ),
    )


def _failure_class(
    *,
    label: RetrievalLabelRow,
    output: RetrievalOutputRow,
    label_scope_establishes_candidate_absence: bool,
    generator_answer_failed: bool,
    redundancy_present: bool,
) -> SetFailureClass:
    relevant = frozenset(label.relevant_evidence_ids)
    if not relevant or not label.graded_evidence:
        return "UNRESOLVED_LABEL_OR_GOLD"
    universe = frozenset(output.retrieved_evidence_ids)
    top_three = frozenset(output.retrieved_evidence_ids[:3])
    if label_scope_establishes_candidate_absence and not relevant.issubset(universe):
        return "DISCOVERY_MISS"
    if not relevant.intersection(top_three) and relevant.intersection(universe):
        return "RANKING_MISS"
    if relevant.intersection(top_three) and not relevant.issubset(top_three):
        return "SET_COVERAGE_MISS"
    if redundancy_present and any(item in universe - top_three for item in relevant):
        return "REDUNDANCY_ERROR"
    hierarchy_context = frozenset(label.hierarchy_context_evidence_ids)
    if hierarchy_context.intersection(universe) and not hierarchy_context.intersection(top_three):
        return "HIERARCHY_CONTEXT_MISS"
    if relevant.issubset(top_three) and generator_answer_failed:
        return "CORRECT_EVIDENCE_GENERATION_ERROR"
    return "NO_FAILURE"


_CORRELATION_FIELDS = (
    "precision_at_1",
    "precision_at_3",
    "evidence_set_recall_at_3",
    "ndcg_at_10",
    "required_evidence_coverage_at_1",
    "required_evidence_coverage_at_3",
    "duplicate_span_pair_rate_at_3",
    "positive_overlap_pair_rate_at_3",
    "parent_child_pair_rate_at_3",
    "hierarchy_diversity_at_3",
    "unique_legal_coordinate_coverage_at_3",
    "document_context_hit_at_1",
    "document_context_hit_at_3",
)


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    available = tuple(value for value in values if value is not None)
    return sum(available) / float(len(available)) if available else None


def _pearson(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float | None, CorrelationReason | None]:
    if len(left) < 2:
        return None, "INSUFFICIENT_PAIRS"
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    left_delta = tuple(value - left_mean for value in left)
    right_delta = tuple(value - right_mean for value in right)
    denominator = sqrt(sum(value * value for value in left_delta)) * sqrt(
        sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None, "ZERO_VARIANCE"
    return sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator, None


def evaluate_retrieval_set(
    labels: tuple[RetrievalLabelRow, ...],
    outputs: tuple[RetrievalOutputRow, ...],
    *,
    answer_metrics: Mapping[str, tuple[float, float]] | None = None,
    label_scope_establishes_candidate_absence: bool = False,
    generator_answer_failures: frozenset[str] = frozenset(),
) -> RetrievalSetMetricReport:
    """Evaluate deterministic graded/set-aware diagnostics without replacing legacy metrics."""

    ordered_labels = _ordered_rows(labels)
    output_by_id = _outputs_by_id(outputs)
    label_ids = {row.question_id for row in ordered_labels}
    if label_ids != set(output_by_id):
        raise RetrievalEvaluationError(
            "RETRIEVAL_EVAL_ID_MISMATCH",
            "labels and retrieval outputs must cover identical question IDs",
        )
    if answer_metrics is not None and not set(answer_metrics).issubset(label_ids):
        raise RetrievalEvaluationError(
            "RETRIEVAL_ANSWER_METRIC_ID_MISMATCH",
            "answer metric IDs must be a subset of retrieval evaluation IDs",
        )
    evaluable = tuple(row for row in ordered_labels if row.relevant_evidence_ids)
    unevaluable = tuple(row.question_id for row in ordered_labels if not row.relevant_evidence_ids)
    if not evaluable:
        raise RetrievalEvaluationError(
            "RETRIEVAL_EVAL_EMPTY", "retrieval evaluation has no evaluable questions"
        )
    rows: list[RetrievalSetQuestionMetrics] = []
    for label in evaluable:
        output = output_by_id[label.question_id]
        relevant = frozenset(label.relevant_evidence_ids)
        top_one = output.retrieved_evidence_ids[:1]
        top_three = output.retrieved_evidence_ids[:3]
        required = frozenset(
            item.evidence_id for item in label.graded_evidence if item.relevance == "relevant"
        )
        if required:
            coverage_1 = float(len(required.intersection(top_one))) / float(len(required))
            coverage_3 = float(len(required.intersection(top_three))) / float(len(required))
            required_reason: Literal["NO_REQUIRED_EVIDENCE"] | None = None
        else:
            coverage_1 = None
            coverage_3 = None
            required_reason = "NO_REQUIRED_EVIDENCE"
        if output.candidate_metadata:
            duplicate, overlap, parent_child, diversity = _pair_metrics(output.candidate_metadata)
            metadata_reason: Literal["CANDIDATE_METADATA_UNAVAILABLE"] | None = None
            costs = tuple(item.token_cost for item in output.candidate_metadata[:3])
            if all(value is not None for value in costs):
                token_cost = sum(value for value in costs if value is not None)
                token_reason: Literal["TOKEN_COST_UNAVAILABLE"] | None = None
            else:
                token_cost = None
                token_reason = "TOKEN_COST_UNAVAILABLE"
            coordinate_coverage, context_hit_1, context_hit_3 = _coordinate_metrics(
                relevant,
                output.retrieved_evidence_ids,
                output.candidate_metadata,
            )
            coordinate_reason: Literal["RELEVANT_COORDINATE_METADATA_INCOMPLETE"] | None = (
                None
                if coordinate_coverage is not None
                else "RELEVANT_COORDINATE_METADATA_INCOMPLETE"
            )
        else:
            duplicate = None
            overlap = None
            parent_child = None
            diversity = None
            metadata_reason = "CANDIDATE_METADATA_UNAVAILABLE"
            token_cost = None
            token_reason = "TOKEN_COST_UNAVAILABLE"
            coordinate_coverage = None
            context_hit_1 = None
            context_hit_3 = None
            coordinate_reason = "RELEVANT_COORDINATE_METADATA_INCOMPLETE"
        rows.append(
            RetrievalSetQuestionMetrics(
                question_id=label.question_id,
                precision_at_1=float(len(relevant.intersection(top_one))),
                precision_at_3=float(len(relevant.intersection(top_three))) / 3.0,
                evidence_set_recall_at_3=float(relevant.issubset(top_three)),
                ndcg_at_10=_ndcg_at_10(label, output.retrieved_evidence_ids),
                required_evidence_coverage_at_1=coverage_1,
                required_evidence_coverage_at_3=coverage_3,
                required_evidence_unavailable_reason=required_reason,
                duplicate_span_pair_rate_at_3=duplicate,
                positive_overlap_pair_rate_at_3=overlap,
                parent_child_pair_rate_at_3=parent_child,
                hierarchy_diversity_at_3=diversity,
                metadata_unavailable_reason=metadata_reason,
                unique_legal_coordinate_coverage_at_3=coordinate_coverage,
                document_context_hit_at_1=context_hit_1,
                document_context_hit_at_3=context_hit_3,
                coordinate_metadata_unavailable_reason=coordinate_reason,
                evidence_count_at_3=len(top_three),
                token_cost_at_3=token_cost,
                token_cost_unavailable_reason=token_reason,
                primary_failure_class=_failure_class(
                    label=label,
                    output=output,
                    label_scope_establishes_candidate_absence=(
                        label_scope_establishes_candidate_absence
                    ),
                    generator_answer_failed=label.question_id in generator_answer_failures,
                    redundancy_present=bool(
                        duplicate is not None
                        and overlap is not None
                        and parent_child is not None
                        and (duplicate > 0.0 or overlap > 0.0 or parent_child > 0.0)
                    ),
                ),
            )
        )
    question_rows = tuple(rows)
    correlations: list[MetricCorrelation] = []
    answer_values = answer_metrics or {}
    for field in _CORRELATION_FIELDS:
        answer_names: tuple[Literal["meteor", "rouge_l"], ...] = ("meteor", "rouge_l")
        for answer_index, answer_name in enumerate(answer_names):
            pairs = tuple(
                (float(value), float(answer_values[row.question_id][answer_index]))
                for row in question_rows
                if row.question_id in answer_values and (value := getattr(row, field)) is not None
            )
            correlation, reason = _pearson(
                tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
            )
            correlations.append(
                MetricCorrelation(
                    retrieval_metric=field,
                    answer_metric=answer_name,
                    pair_count=len(pairs),
                    value=correlation,
                    reason=reason,
                )
            )
    denominator = float(len(question_rows))
    evidence_count_distribution = (
        sum(row.evidence_count_at_3 == 0 for row in question_rows),
        sum(row.evidence_count_at_3 == 1 for row in question_rows),
        sum(row.evidence_count_at_3 == 2 for row in question_rows),
        sum(row.evidence_count_at_3 == 3 for row in question_rows),
    )
    token_costs = tuple(
        row.token_cost_at_3 for row in question_rows if row.token_cost_at_3 is not None
    )
    return RetrievalSetMetricReport(
        schema_version="retrieval.set-evaluation.v1",
        benchmark_question_count=len(ordered_labels),
        retrieval_evaluable_count=len(question_rows),
        retrieval_unevaluable_count=len(unevaluable),
        unevaluable_question_ids=unevaluable,
        questions=question_rows,
        mean_precision_at_1=sum(row.precision_at_1 for row in question_rows) / denominator,
        mean_precision_at_3=sum(row.precision_at_3 for row in question_rows) / denominator,
        mean_evidence_set_recall_at_3=(
            sum(row.evidence_set_recall_at_3 for row in question_rows) / denominator
        ),
        mean_ndcg_at_10=sum(row.ndcg_at_10 for row in question_rows) / denominator,
        mean_unique_legal_coordinate_coverage_at_3=_mean_optional(
            tuple(row.unique_legal_coordinate_coverage_at_3 for row in question_rows)
        ),
        mean_document_context_hit_at_1=_mean_optional(
            tuple(row.document_context_hit_at_1 for row in question_rows)
        ),
        mean_document_context_hit_at_3=_mean_optional(
            tuple(row.document_context_hit_at_3 for row in question_rows)
        ),
        required_evidence_eligible_count=sum(
            row.required_evidence_coverage_at_3 is not None for row in question_rows
        ),
        metadata_available_count=sum(
            row.metadata_unavailable_reason is None for row in question_rows
        ),
        coordinate_metric_available_count=sum(
            row.coordinate_metadata_unavailable_reason is None for row in question_rows
        ),
        evidence_count_distribution_at_3=evidence_count_distribution,
        token_cost_available_count=len(token_costs),
        token_cost_unavailable_count=len(question_rows) - len(token_costs),
        token_cost_total_at_3=sum(token_costs) if token_costs else None,
        token_cost_mean_at_3=(
            float(sum(token_costs)) / float(len(token_costs)) if token_costs else None
        ),
        token_cost_minimum_at_3=min(token_costs) if token_costs else None,
        token_cost_maximum_at_3=max(token_costs) if token_costs else None,
        correlations=tuple(correlations),
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
