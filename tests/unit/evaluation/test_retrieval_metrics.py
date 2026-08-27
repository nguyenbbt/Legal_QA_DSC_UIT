from __future__ import annotations

import pytest

from legal_rag.evaluation.retrieval_metrics import (
    ContainmentInputRow,
    GradedEvidenceLabel,
    RetrievalCandidateMetadata,
    RetrievalEvaluationError,
    RetrievalLabelRow,
    RetrievalOutputRow,
    evaluate_answer_containment,
    evaluate_retrieval,
    evaluate_retrieval_set,
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
    assert tuple((row.question_id, row.reason) for row in report.unevaluable) == (
        ("q3", "NO_RELEVANT_EVIDENCE"),
    )
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


def test_set_metrics_match_hand_computed_grades_and_fixed_denominators() -> None:
    labels = (
        RetrievalLabelRow(
            "q2",
            ("required", "partial"),
            graded_evidence=(
                GradedEvidenceLabel("required", "relevant"),
                GradedEvidenceLabel("partial", "partially_relevant"),
            ),
        ),
        RetrievalLabelRow("q1", ()),
    )
    outputs = (
        RetrievalOutputRow("q1", ()),
        RetrievalOutputRow("q2", ("partial", "wrong", "required")),
    )

    report = evaluate_retrieval_set(labels, outputs)

    assert report.schema_version == "retrieval.set-evaluation.v1"
    assert report.benchmark_question_count == 2
    assert report.retrieval_evaluable_count == 1
    assert report.retrieval_unevaluable_count == 1
    assert report.unevaluable_question_ids == ("q1",)
    row = report.questions[0]
    assert row.question_id == "q2"
    assert row.precision_at_1 == 1.0
    assert row.precision_at_3 == pytest.approx(2 / 3)
    assert row.evidence_set_recall_at_3 == 1.0
    assert row.required_evidence_coverage_at_1 == 0.0
    assert row.required_evidence_coverage_at_3 == 1.0
    expected_dcg = 1.0 + 2.0 / 2.0
    expected_idcg = 2.0 + 1.0 / 1.584962500721156
    assert row.ndcg_at_10 == pytest.approx(expected_dcg / expected_idcg)
    assert row.primary_failure_class == "NO_FAILURE"


def test_set_metrics_measure_span_and_hierarchy_redundancy() -> None:
    metadata = (
        RetrievalCandidateMetadata("a", "ctx", ("Điều 1",), 0, 100, 40),
        RetrievalCandidateMetadata("b", "ctx", ("Điều 1", "Khoản 1"), 0, 100, 40),
        RetrievalCandidateMetadata("c", "ctx", ("Điều 2",), 50, 150, 40),
    )
    report = evaluate_retrieval_set(
        (
            RetrievalLabelRow(
                "q",
                ("a", "c"),
                graded_evidence=(
                    GradedEvidenceLabel("a", "relevant"),
                    GradedEvidenceLabel("c", "relevant"),
                ),
            ),
        ),
        (RetrievalOutputRow("q", ("a", "b", "c"), candidate_metadata=metadata),),
    )

    row = report.questions[0]
    assert row.duplicate_span_pair_rate_at_3 == pytest.approx(1 / 3)
    assert row.positive_overlap_pair_rate_at_3 == 1.0
    assert row.parent_child_pair_rate_at_3 == pytest.approx(1 / 3)
    assert row.hierarchy_diversity_at_3 == 1.0
    assert row.unique_legal_coordinate_coverage_at_3 == 1.0
    assert row.document_context_hit_at_1 == 1.0
    assert row.document_context_hit_at_3 == 1.0
    assert row.coordinate_metadata_unavailable_reason is None
    assert row.evidence_count_at_3 == 3
    assert row.token_cost_at_3 == 120
    assert report.mean_unique_legal_coordinate_coverage_at_3 == 1.0
    assert report.mean_document_context_hit_at_1 == 1.0
    assert report.mean_document_context_hit_at_3 == 1.0
    assert report.coordinate_metric_available_count == 1
    assert report.evidence_count_distribution_at_3 == (0, 0, 0, 1)
    assert report.token_cost_total_at_3 == 120
    assert report.token_cost_mean_at_3 == 120.0
    assert report.token_cost_minimum_at_3 == 120
    assert report.token_cost_maximum_at_3 == 120
    assert report.token_cost_unavailable_count == 0


def test_set_metrics_report_unavailable_metadata_and_correlation_reason() -> None:
    report = evaluate_retrieval_set(
        (
            RetrievalLabelRow(
                "q",
                ("a",),
                graded_evidence=(GradedEvidenceLabel("a", "partially_relevant"),),
            ),
        ),
        (RetrievalOutputRow("q", ("a",)),),
        answer_metrics={"q": (0.2, 0.3)},
    )

    row = report.questions[0]
    assert row.required_evidence_coverage_at_1 is None
    assert row.required_evidence_unavailable_reason == "NO_REQUIRED_EVIDENCE"
    assert row.duplicate_span_pair_rate_at_3 is None
    assert row.metadata_unavailable_reason == "CANDIDATE_METADATA_UNAVAILABLE"
    assert row.unique_legal_coordinate_coverage_at_3 is None
    assert row.document_context_hit_at_1 is None
    assert row.document_context_hit_at_3 is None
    assert row.coordinate_metadata_unavailable_reason == "RELEVANT_COORDINATE_METADATA_INCOMPLETE"
    assert row.token_cost_at_3 is None
    assert report.coordinate_metric_available_count == 0
    assert report.token_cost_total_at_3 is None
    assert report.token_cost_mean_at_3 is None
    assert report.token_cost_minimum_at_3 is None
    assert report.token_cost_maximum_at_3 is None
    assert report.token_cost_unavailable_count == 1
    assert all(item.value is None for item in report.correlations)
    assert {item.reason for item in report.correlations} == {"INSUFFICIENT_PAIRS"}


def test_set_metrics_report_partial_coordinate_coverage_and_document_hit() -> None:
    metadata = (
        RetrievalCandidateMetadata("wrong", "ctx-gold", ("Điều 9",), 20, 30, 5),
        RetrievalCandidateMetadata("gold-a", "ctx-gold", ("Điều 1",), 0, 10, 7),
        RetrievalCandidateMetadata("gold-b", "ctx-other", ("Điều 2",), 0, 10, 11),
    )
    report = evaluate_retrieval_set(
        (
            RetrievalLabelRow(
                "q",
                ("gold-a", "gold-b"),
                graded_evidence=(
                    GradedEvidenceLabel("gold-a", "relevant"),
                    GradedEvidenceLabel("gold-b", "partially_relevant"),
                ),
            ),
        ),
        (
            RetrievalOutputRow(
                "q",
                ("wrong", "gold-a", "gold-b"),
                candidate_metadata=metadata,
            ),
        ),
    )

    row = report.questions[0]
    assert row.unique_legal_coordinate_coverage_at_3 == 1.0
    assert row.document_context_hit_at_1 == 1.0
    assert row.document_context_hit_at_3 == 1.0
    assert report.token_cost_total_at_3 == 23


def test_set_metrics_fail_closed_when_positive_coordinate_metadata_is_incomplete() -> None:
    report = evaluate_retrieval_set(
        (
            RetrievalLabelRow(
                "q",
                ("gold-a", "gold-missing"),
                graded_evidence=(
                    GradedEvidenceLabel("gold-a", "relevant"),
                    GradedEvidenceLabel("gold-missing", "partially_relevant"),
                ),
            ),
        ),
        (
            RetrievalOutputRow(
                "q",
                ("gold-a",),
                candidate_metadata=(
                    RetrievalCandidateMetadata("gold-a", "ctx", ("Điều 1",), 0, 10, 3),
                ),
            ),
        ),
    )

    row = report.questions[0]
    assert row.unique_legal_coordinate_coverage_at_3 is None
    assert row.document_context_hit_at_1 is None
    assert row.document_context_hit_at_3 is None
    assert row.coordinate_metadata_unavailable_reason == "RELEVANT_COORDINATE_METADATA_INCOMPLETE"


def test_set_metrics_failure_precedence_distinguishes_discovery_ranking_and_generation() -> None:
    label = RetrievalLabelRow(
        "q",
        ("gold",),
        graded_evidence=(GradedEvidenceLabel("gold", "relevant"),),
    )
    discovery = evaluate_retrieval_set(
        (label,),
        (RetrievalOutputRow("q", ("x",)),),
        label_scope_establishes_candidate_absence=True,
    )
    ranking = evaluate_retrieval_set(
        (label,),
        (RetrievalOutputRow("q", ("x1", "x2", "x3", "gold")),),
    )
    generation = evaluate_retrieval_set(
        (label,),
        (RetrievalOutputRow("q", ("gold",)),),
        generator_answer_failures=frozenset({"q"}),
    )

    assert discovery.questions[0].primary_failure_class == "DISCOVERY_MISS"
    assert ranking.questions[0].primary_failure_class == "RANKING_MISS"
    assert generation.questions[0].primary_failure_class == "CORRECT_EVIDENCE_GENERATION_ERROR"


def test_set_metrics_reject_misaligned_candidate_metadata() -> None:
    with pytest.raises(RetrievalEvaluationError, match="metadata") as error:
        evaluate_retrieval_set(
            (
                RetrievalLabelRow(
                    "q",
                    ("a",),
                    graded_evidence=(GradedEvidenceLabel("a", "relevant"),),
                ),
            ),
            (
                RetrievalOutputRow(
                    "q",
                    ("a",),
                    candidate_metadata=(RetrievalCandidateMetadata("b", "ctx", ("Điều 1",), 0, 1),),
                ),
            ),
        )
    assert error.value.code == "RETRIEVAL_OUTPUT_METADATA_MISMATCH"
