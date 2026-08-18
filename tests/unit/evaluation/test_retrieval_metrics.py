from __future__ import annotations

import pytest

from legal_rag.evaluation.retrieval_metrics import (
    ContainmentInputRow,
    RetrievalEvaluationError,
    RetrievalLabelRow,
    RetrievalOutputRow,
    evaluate_answer_containment,
    evaluate_retrieval,
)


def test_retrieval_metrics_match_hand_computed_multi_evidence_fixture() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a", "b")),
        RetrievalLabelRow("q2", ("c",)),
        RetrievalLabelRow("q3", ()),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a", "x", "b")),
        RetrievalOutputRow("q2", ("x", "c")),
        RetrievalOutputRow("q3", ("z",)),
    )

    report = evaluate_retrieval(labels, outputs)

    assert report.benchmark_question_count == 3
    assert report.retrieval_evaluable_count == 2
    assert report.retrieval_unevaluable_count == 1
    assert report.unevaluable_question_ids == ("q3",)
    assert report.recall_at_1 == 0.25
    assert report.recall_at_5 == 1.0
    assert report.recall_at_10 == 1.0
    assert report.mrr_at_10 == 0.75
    assert report.evidence_set_recall_at_10 == 1.0


def test_retrieval_metrics_reject_empty_evaluation_and_id_mismatch() -> None:
    with pytest.raises(RetrievalEvaluationError, match="no evaluable") as empty:
        evaluate_retrieval((RetrievalLabelRow("q", ()),), (RetrievalOutputRow("q", ()),))
    assert empty.value.code == "RETRIEVAL_EVAL_EMPTY"

    with pytest.raises(RetrievalEvaluationError, match="identical question IDs") as mismatch:
        evaluate_retrieval(
            (RetrievalLabelRow("q1", ("a",)),),
            (RetrievalOutputRow("q2", ("a",)),),
        )
    assert mismatch.value.code == "RETRIEVAL_EVAL_ID_MISMATCH"


def test_retrieval_metrics_reject_duplicate_ranked_evidence() -> None:
    with pytest.raises(RetrievalEvaluationError, match="duplicate evidence") as duplicate:
        evaluate_retrieval(
            (RetrievalLabelRow("q", ("a",)),),
            (RetrievalOutputRow("q", ("a", "a")),),
        )
    assert duplicate.value.code == "RETRIEVAL_OUTPUT_DUPLICATE"


def test_answer_containment_is_separate_and_reports_all_denominators() -> None:
    report = evaluate_answer_containment(
        (
            ContainmentInputRow("q1", "Nội   dung", ("không khớp", "NỘI DUNG")),
            ContainmentInputRow("q2", "", ("bất kỳ",)),
        )
    )

    assert report.metric_namespace == "diagnostic_answer_containment"
    assert report.total_question_count == 2
    assert report.eligible_question_count == 1
    assert report.excluded_question_count == 1
    assert report.excluded == (("q2", "EMPTY_GOLD_ANSWER"),)
    assert report.containment_at_1 == 0.0
    assert report.containment_at_5 == 1.0
    assert report.containment_at_10 == 1.0
