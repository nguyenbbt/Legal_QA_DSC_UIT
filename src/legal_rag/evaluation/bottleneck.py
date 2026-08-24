"""Deterministic retrieval bottleneck classification from approved metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RetrievalMetricsSnapshot:
    run_id: str
    recall_at_1: float
    recall_at_10: float
    mrr_at_10: float
    evidence_set_recall_at_10: float


@dataclass(frozen=True, slots=True)
class RetrievalBottleneckDecision:
    action: Literal["FT-EMBED", "FT-RERANK", "NO_RETRIEVAL_FT", "FIX_CODE_OR_DATA"]
    reason_code: str
    selected_run_id: str


def diagnose_retrieval_bottleneck(
    metrics: RetrievalMetricsSnapshot,
    *,
    parser_or_identity_failures: int,
    minimum_recall_at_10: float,
    minimum_mrr_at_10: float,
    minimum_evidence_set_recall_at_10: float,
) -> RetrievalBottleneckDecision:
    """Select exactly one action using owner-locked thresholds.

    Numeric thresholds are explicit inputs because OQ-005 forbids inventing them
    inside implementation code.
    """

    thresholds = (
        minimum_recall_at_10,
        minimum_mrr_at_10,
        minimum_evidence_set_recall_at_10,
    )
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("retrieval bottleneck thresholds must be within [0, 1]")
    if parser_or_identity_failures > 0:
        return RetrievalBottleneckDecision(
            "FIX_CODE_OR_DATA", "PARSER_OR_IDENTITY_FAILURE", metrics.run_id
        )
    if (
        metrics.recall_at_10 < minimum_recall_at_10
        or metrics.evidence_set_recall_at_10 < minimum_evidence_set_recall_at_10
    ):
        return RetrievalBottleneckDecision(
            "FT-EMBED", "TOP_10_EVIDENCE_COVERAGE_WEAK", metrics.run_id
        )
    if metrics.mrr_at_10 < minimum_mrr_at_10:
        return RetrievalBottleneckDecision("FT-RERANK", "TOP_RANK_ORDERING_WEAK", metrics.run_id)
    return RetrievalBottleneckDecision(
        "NO_RETRIEVAL_FT", "RETRIEVAL_GATES_SATISFIED", metrics.run_id
    )


__all__ = [
    "RetrievalBottleneckDecision",
    "RetrievalMetricsSnapshot",
    "diagnose_retrieval_bottleneck",
]
