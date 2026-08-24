from __future__ import annotations

import pytest

from legal_rag.evaluation.bottleneck import (
    RetrievalMetricsSnapshot,
    diagnose_retrieval_bottleneck,
)


def _metrics(**changes: float) -> RetrievalMetricsSnapshot:
    values: dict[str, float | str] = {
        "run_id": "R1",
        "recall_at_1": 0.8,
        "recall_at_10": 0.95,
        "mrr_at_10": 0.8,
        "evidence_set_recall_at_10": 0.9,
    }
    values.update(changes)
    return RetrievalMetricsSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("metrics", "failures", "action"),
    (
        (_metrics(), 1, "FIX_CODE_OR_DATA"),
        (_metrics(recall_at_10=0.7), 0, "FT-EMBED"),
        (_metrics(evidence_set_recall_at_10=0.5), 0, "FT-EMBED"),
        (_metrics(mrr_at_10=0.4), 0, "FT-RERANK"),
        (_metrics(), 0, "NO_RETRIEVAL_FT"),
    ),
)
def test_bottleneck_decision_is_exclusive(
    metrics: RetrievalMetricsSnapshot, failures: int, action: str
) -> None:
    decision = diagnose_retrieval_bottleneck(
        metrics,
        parser_or_identity_failures=failures,
        minimum_recall_at_10=0.9,
        minimum_mrr_at_10=0.7,
        minimum_evidence_set_recall_at_10=0.8,
    )

    assert decision.action == action


def test_bottleneck_does_not_invent_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        diagnose_retrieval_bottleneck(
            _metrics(),
            parser_or_identity_failures=0,
            minimum_recall_at_10=1.01,
            minimum_mrr_at_10=0.7,
            minimum_evidence_set_recall_at_10=0.8,
        )
