"""Unit contracts for the offline exact competition evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.evaluation.official_exact import (
    ScorerError,
    evaluate_official_exact,
    evaluate_supplied_scorer,
)
from legal_rag.evaluation.parity import run_parity_suite


def test_missing_nltk_archive_fails_before_scoring(tmp_path: Path) -> None:
    scorer_root = Path("Scoring-Program-Task-LegalQA")

    with pytest.raises(ScorerError) as captured:
        evaluate_official_exact(
            {"q1": {"answer": "answer"}},
            {"q1": "answer"},
            scorer_root=scorer_root,
            nltk_data_root=tmp_path,
        )

    assert captured.value.code == "OFFLINE_RESOURCE_MISSING"
    assert "wordnet" in captured.value.message


def test_exact_match_scores_one_and_preserves_prediction_order() -> None:
    predictions = {
        "q-z": {"answer": "the exact answer"},
        "q-a": {"answer": "another exact answer"},
    }
    references = {
        "q-a": "another exact answer",
        "q-z": "the exact answer",
    }

    result = evaluate_official_exact(
        predictions,
        references,
        scorer_root=Path("Scoring-Program-Task-LegalQA"),
        nltk_data_root=Path("resources/nltk_data"),
    )

    assert result.question_ids == ("q-z", "q-a")
    assert result.macro_rouge_l == 1.0
    assert result.macro_meteor == 0.9814814814814815
    assert tuple(row.question_id for row in result.per_query) == result.question_ids


def test_prediction_and_reference_counts_must_match() -> None:
    with pytest.raises(ScorerError) as captured:
        evaluate_official_exact(
            {"q1": {"answer": "one"}},
            {"q1": "one", "q2": "two"},
            scorer_root=Path("Scoring-Program-Task-LegalQA"),
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "SCORER_SAMPLE_COUNT_MISMATCH"


def test_missing_prediction_answer_has_typed_error() -> None:
    with pytest.raises(ScorerError) as captured:
        evaluate_official_exact(
            {"q1": {"question": "not an answer"}},
            {"q1": "one"},
            scorer_root=Path("Scoring-Program-Task-LegalQA"),
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "SCORER_PREDICTION_INVALID"


def test_supplied_scorer_rejects_unreviewed_source_bytes(tmp_path: Path) -> None:
    scoring_path = tmp_path / "scoring.py"
    original = Path("Scoring-Program-Task-LegalQA/scoring.py").read_bytes()
    scoring_path.write_bytes(original + b"\n# unreviewed change\n")

    with pytest.raises(ScorerError) as captured:
        evaluate_supplied_scorer(
            {"q1": {"answer": "answer"}},
            {"q1": "answer"},
            scoring_path=scoring_path,
            scorer_root=Path("Scoring-Program-Task-LegalQA"),
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "SCORER_SOURCE_CHECKSUM_MISMATCH"


def test_fixture_loader_rejects_duplicate_json_members(tmp_path: Path) -> None:
    (tmp_path / "cases.v1.json").write_text(
        '{"schema_version":"scorer.fixtures.v1",'
        '"schema_version":"scorer.fixtures.v1",'
        '"cases":[{"question_id":"q1","reference":"one","prediction":"one"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ScorerError) as captured:
        run_parity_suite(
            scoring_path=Path("Scoring-Program-Task-LegalQA/scoring.py"),
            fixtures_directory=tmp_path,
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "SCORER_FIXTURE_INVALID"


def test_fixture_loader_rejects_oversized_json_before_parse(tmp_path: Path) -> None:
    (tmp_path / "cases.v1.json").write_bytes(b" " * (1024 * 1024 + 1))

    with pytest.raises(ScorerError) as captured:
        run_parity_suite(
            scoring_path=Path("Scoring-Program-Task-LegalQA/scoring.py"),
            fixtures_directory=tmp_path,
            nltk_data_root=Path("resources/nltk_data"),
        )

    assert captured.value.code == "SCORER_FIXTURE_INVALID"
    assert "1048576" in captured.value.message
