"""Pure scorer-parity comparison and rendering tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.evaluation.official_exact import EvaluationResult, PerQueryScore, ScorerError
from legal_rag.evaluation.parity import build_parity_report, parity_json_bytes, parity_markdown

SCORING_PATH = Path("Scoring-Program-Task-LegalQA/scoring.py")


def evaluation(
    question_id: str = "q1", *, rouge_l: float = 0.75, meteor: float = 0.5
) -> EvaluationResult:
    return EvaluationResult(
        question_ids=(question_id,),
        per_query=(PerQueryScore(question_id=question_id, rouge_l=rouge_l, meteor=meteor),),
        macro_rouge_l=rouge_l,
        macro_meteor=meteor,
    )


def test_equal_results_build_passing_deterministic_report() -> None:
    project = evaluation()

    report = build_parity_report(
        project,
        project,
        fixed_case_count=1,
        sampled_case_count=0,
        scoring_path=SCORING_PATH,
    )

    assert report["status"] == "pass"
    assert report["metrics"]["per_query_max_absolute_difference"] == 0.0
    assert report["supplied_scorer"]["checksum"] == (
        "sha256:f04843fbfad26d41356506d8e49692a7c8a0ed1b9f065a3a8472fa6398a5aa95"
    )
    assert parity_json_bytes(report).endswith(b"\n")
    assert "status: pass" in parity_markdown(report)


def test_difference_over_threshold_builds_failure_report() -> None:
    report = build_parity_report(
        evaluation(),
        evaluation(rouge_l=0.7, meteor=0.4),
        fixed_case_count=1,
        sampled_case_count=0,
        scoring_path=SCORING_PATH,
    )

    assert report["status"] == "fail"
    assert report["metrics"]["rouge_l"]["absolute_difference"] == pytest.approx(0.05)
    assert report["metrics"]["meteor"]["absolute_difference"] == pytest.approx(0.1)


def test_parity_report_rejects_different_question_ordering() -> None:
    with pytest.raises(ScorerError) as captured:
        build_parity_report(
            evaluation("q1"),
            evaluation("q2"),
            fixed_case_count=1,
            sampled_case_count=0,
            scoring_path=SCORING_PATH,
        )

    assert captured.value.code == "SCORER_PARITY_ID_MISMATCH"
