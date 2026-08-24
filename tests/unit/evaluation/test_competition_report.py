"""Deterministic development report around the reviewed official scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.competition import (
    CompetitionEvaluationError,
    evaluate_competition_bytes,
    write_competition_evaluation,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value) + "\n").encode()


def test_competition_report_records_order_scores_checksums_and_versions() -> None:
    predictions = _json_bytes({"q-z": {"answer": "exact answer"}, "q-a": {"answer": "different"}})
    references = _json_bytes({"q-a": "reference", "q-z": "exact answer"})

    evaluation = evaluate_competition_bytes(
        predictions,
        references,
        scorer_root=Path("Scoring-Program-Task-LegalQA"),
        nltk_data_root=Path("resources/nltk_data"),
    )
    report = json.loads(evaluation.report_bytes)
    per_query = [json.loads(line) for line in evaluation.per_query_bytes.splitlines()]

    assert report["schema_version"] == "competition.evaluation.report.v1"
    assert report["question_order"] == ["q-z", "q-a"]
    assert report["predictions_checksum"] == checksum_bytes(predictions)
    assert report["references_checksum"] == checksum_bytes(references)
    assert report["question_count"] == 2
    assert report["dependencies"]["nltk"]
    assert report["dependencies"]["numpy"]
    assert tuple(row["question_id"] for row in per_query) == ("q-z", "q-a")
    assert per_query[0]["rouge_l"] == 1.0


def test_competition_report_accepts_model_specific_provenance() -> None:
    evaluation = evaluate_competition_bytes(
        _json_bytes({"q": {"answer": "one"}}),
        _json_bytes({"q": "one"}),
        scorer_root=Path("Scoring-Program-Task-LegalQA"),
        nltk_data_root=Path("resources/nltk_data"),
        baseline_kind="zero_shot_grounded_generator",
        limitation="exploratory_non_promotable_btc_pending",
    )

    report = json.loads(evaluation.report_bytes)
    assert report["baseline_kind"] == "zero_shot_grounded_generator"
    assert report["limitation"] == "exploratory_non_promotable_btc_pending"


def test_competition_report_rejects_id_mismatch() -> None:
    with pytest.raises(CompetitionEvaluationError) as captured:
        evaluate_competition_bytes(
            _json_bytes({"q1": {"answer": "one"}}),
            _json_bytes({"q2": "two"}),
            scorer_root=Path("Scoring-Program-Task-LegalQA"),
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "EVAL_QUESTION_ID_MISMATCH"


def test_competition_report_writes_both_outputs_immutably(tmp_path: Path) -> None:
    evaluation = evaluate_competition_bytes(
        _json_bytes({"q": {"answer": "one"}}),
        _json_bytes({"q": "one"}),
        scorer_root=Path("Scoring-Program-Task-LegalQA"),
        nltk_data_root=Path("resources/nltk_data"),
    )
    per_query = tmp_path / "per-query.jsonl"
    report = tmp_path / "report.json"

    checksums = write_competition_evaluation(
        evaluation, per_query_path=per_query, report_path=report
    )

    assert (
        write_competition_evaluation(evaluation, per_query_path=per_query, report_path=report)
        == checksums
    )
    report.write_bytes(b"different\n")
    with pytest.raises(CompetitionEvaluationError) as captured:
        write_competition_evaluation(evaluation, per_query_path=per_query, report_path=report)
    assert captured.value.code == "EVAL_REPORT_IMMUTABLE"
