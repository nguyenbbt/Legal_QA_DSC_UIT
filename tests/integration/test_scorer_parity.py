"""Offline parity against the exact supplied ``eval_qa`` function."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import nltk
import pytest

from legal_rag.cli import main
from legal_rag.evaluation.official_exact import evaluate_official_exact, evaluate_supplied_scorer
from legal_rag.evaluation.parity import build_parity_report, parity_json_bytes, parity_markdown

pytestmark = pytest.mark.integration

SCORER_ROOT = Path("Scoring-Program-Task-LegalQA")
NLTK_DATA_ROOT = Path("resources/nltk_data")
FIXTURE_PATH = Path("data/fixtures/scoring/cases.v1.json")


def _forbidden(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("runtime attempted nltk.download")


def _load_fixed_cases() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    predictions: dict[str, dict[str, str]] = {}
    references: dict[str, str] = {}
    for case in value["cases"]:
        question_id = case["question_id"]
        predictions[question_id] = {"answer": case["prediction"]}
        references[question_id] = case["reference"]
    return predictions, references


def _sampled_cases() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    generator = random.Random(20260814)
    vocabulary = ("court", "law", "worker", "right", "duty", "article", "remedy")
    predictions: dict[str, dict[str, str]] = {}
    references: dict[str, str] = {}
    for index in range(24):
        reference_tokens = [generator.choice(vocabulary) for _ in range(7)]
        prediction_tokens = reference_tokens.copy()
        left = generator.randrange(len(prediction_tokens))
        right = generator.randrange(len(prediction_tokens))
        prediction_tokens[left], prediction_tokens[right] = (
            prediction_tokens[right],
            prediction_tokens[left],
        )
        question_id = f"sample-{index:02d}"
        references[question_id] = " ".join(reference_tokens)
        predictions[question_id] = {"answer": " ".join(prediction_tokens)}
    return predictions, references


@pytest.mark.parametrize("case_loader", [_load_fixed_cases, _sampled_cases])
def test_project_matches_supplied_scorer_to_1e_12_without_download(
    case_loader: Callable[[], tuple[dict[str, dict[str, str]], dict[str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nltk, "download", _forbidden)
    predictions, references = case_loader()

    project = evaluate_official_exact(
        predictions,
        references,
        scorer_root=SCORER_ROOT,
        nltk_data_root=NLTK_DATA_ROOT,
    )
    supplied = evaluate_supplied_scorer(
        predictions,
        references,
        scoring_path=SCORER_ROOT / "scoring.py",
        scorer_root=SCORER_ROOT,
        nltk_data_root=NLTK_DATA_ROOT,
    )

    assert abs(project.macro_rouge_l - supplied.macro_rouge_l) <= 1e-12
    assert abs(project.macro_meteor - supplied.macro_meteor) <= 1e-12
    for project_row, supplied_row in zip(project.per_query, supplied.per_query, strict=True):
        assert project_row.question_id == supplied_row.question_id
        assert abs(project_row.rouge_l - supplied_row.rouge_l) <= 1e-12
        assert abs(project_row.meteor - supplied_row.meteor) <= 1e-12


def test_parity_report_bytes_are_deterministic_and_pass() -> None:
    fixed_predictions, fixed_references = _load_fixed_cases()
    sampled_predictions, sampled_references = _sampled_cases()
    predictions = fixed_predictions | sampled_predictions
    references = fixed_references | sampled_references
    project = evaluate_official_exact(
        predictions,
        references,
        scorer_root=SCORER_ROOT,
        nltk_data_root=NLTK_DATA_ROOT,
    )
    supplied = evaluate_supplied_scorer(
        predictions,
        references,
        scoring_path=SCORER_ROOT / "scoring.py",
        scorer_root=SCORER_ROOT,
        nltk_data_root=NLTK_DATA_ROOT,
    )

    report = build_parity_report(
        project,
        supplied,
        fixed_case_count=len(fixed_predictions),
        sampled_case_count=len(sampled_predictions),
        scoring_path=SCORER_ROOT / "scoring.py",
    )

    assert report["status"] == "pass"
    assert report["threshold"] == 1e-12
    assert parity_json_bytes(report) == parity_json_bytes(report)
    assert parity_json_bytes(report).endswith(b"\n")
    assert "status: pass" in parity_markdown(report)


def test_scorer_parity_cli_writes_offline_deterministic_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(nltk, "download", _forbidden)
    json_report = tmp_path / "parity.json"
    markdown_report = tmp_path / "parity.md"

    exit_code = main(
        [
            "scorer",
            "parity",
            "--official",
            str(SCORER_ROOT / "scoring.py"),
            "--fixtures",
            str(FIXTURE_PATH.parent),
            "--nltk-data",
            str(NLTK_DATA_ROOT),
            "--json-report",
            str(json_report),
            "--markdown-report",
            str(markdown_report),
            "--output-format",
            "json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads(json_report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output["status"] == "pass"
    assert output == written
    assert written["case_counts"] == {"fixed": 5, "sampled": 24, "total": 29}
    assert written["metrics"]["per_query_max_absolute_difference"] <= 1e-12
    assert markdown_report.read_text(encoding="utf-8").startswith(
        "# MIL-001 Official Scorer Parity\n"
    )
