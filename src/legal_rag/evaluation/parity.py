"""Deterministic fixed/sampled scorer-parity orchestration and reports."""

from __future__ import annotations

import importlib.metadata
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

from legal_rag.evaluation.official_exact import (
    EvaluationResult,
    ScorerError,
    evaluate_official_exact,
    evaluate_supplied_scorer,
    supplied_scorer_provenance,
)

_PARITY_THRESHOLD = 1e-12
_SAMPLED_CASE_SEED = 20260814
_SAMPLED_CASE_COUNT = 24
_MAX_FIXTURE_BYTES = 1024 * 1024


def _fail(code: str, message: str) -> NoReturn:
    raise ScorerError(code, message)


def _absolute_differences(
    project: EvaluationResult, supplied: EvaluationResult
) -> tuple[float, float, float]:
    if project.question_ids != supplied.question_ids:
        _fail("SCORER_PARITY_ID_MISMATCH", "parity inputs use different question ordering")
    rouge_difference = abs(project.macro_rouge_l - supplied.macro_rouge_l)
    meteor_difference = abs(project.macro_meteor - supplied.macro_meteor)
    per_query_max = max(
        max(
            abs(project_row.rouge_l - supplied_row.rouge_l),
            abs(project_row.meteor - supplied_row.meteor),
        )
        for project_row, supplied_row in zip(project.per_query, supplied.per_query, strict=True)
    )
    return rouge_difference, meteor_difference, per_query_max


def build_parity_report(
    project: EvaluationResult,
    supplied: EvaluationResult,
    *,
    fixed_case_count: int,
    sampled_case_count: int,
    scoring_path: Path,
) -> dict[str, Any]:
    """Build a deterministic report with no operational telemetry."""

    rouge_difference, meteor_difference, per_query_max = _absolute_differences(project, supplied)
    passed = max(rouge_difference, meteor_difference, per_query_max) <= _PARITY_THRESHOLD
    return {
        "schema_id": "scorer.parity.v1",
        "status": "pass" if passed else "fail",
        "execution_mode": "local-offline",
        "threshold": _PARITY_THRESHOLD,
        "case_counts": {
            "fixed": fixed_case_count,
            "sampled": sampled_case_count,
            "total": fixed_case_count + sampled_case_count,
        },
        "dependencies": {
            "numpy": importlib.metadata.version("numpy"),
            "nltk": importlib.metadata.version("nltk"),
        },
        "supplied_scorer": supplied_scorer_provenance(scoring_path),
        "metrics": {
            "rouge_l": {
                "project": project.macro_rouge_l,
                "supplied": supplied.macro_rouge_l,
                "absolute_difference": rouge_difference,
            },
            "meteor": {
                "project": project.macro_meteor,
                "supplied": supplied.macro_meteor,
                "absolute_difference": meteor_difference,
            },
            "per_query_max_absolute_difference": per_query_max,
        },
    }


def _reject_duplicate_fixture_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError("duplicate fixture key")
        value[key] = member
    return value


def _load_fixed_cases(
    fixtures_directory: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    fixture_path = fixtures_directory / "cases.v1.json"
    try:
        if fixture_path.stat().st_size > _MAX_FIXTURE_BYTES:
            _fail("SCORER_FIXTURE_INVALID", "scorer fixture exceeds the 1048576-byte limit")
        value = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fixture_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ScorerError(
            "SCORER_FIXTURE_INVALID", "scorer fixture file is missing or invalid"
        ) from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "cases"}:
        _fail("SCORER_FIXTURE_INVALID", "scorer fixture root does not match its schema")
    if value["schema_version"] != "scorer.fixtures.v1" or not isinstance(value["cases"], list):
        _fail("SCORER_FIXTURE_INVALID", "scorer fixture schema version or cases are invalid")
    predictions: dict[str, dict[str, str]] = {}
    references: dict[str, str] = {}
    for case in value["cases"]:
        if not isinstance(case, dict) or set(case) != {
            "question_id",
            "reference",
            "prediction",
        }:
            _fail("SCORER_FIXTURE_INVALID", "scorer fixture case has invalid members")
        question_id = case["question_id"]
        reference = case["reference"]
        prediction = case["prediction"]
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in predictions
            or not isinstance(reference, str)
            or not reference
            or not isinstance(prediction, str)
            or not prediction
        ):
            _fail("SCORER_FIXTURE_INVALID", "scorer fixture case values are invalid")
        predictions[question_id] = {"answer": prediction}
        references[question_id] = reference
    if not predictions:
        _fail("SCORER_FIXTURE_INVALID", "scorer fixtures must contain at least one case")
    return predictions, references


def _build_sampled_cases() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    generator = random.Random(_SAMPLED_CASE_SEED)
    vocabulary = ("court", "law", "worker", "right", "duty", "article", "remedy")
    predictions: dict[str, dict[str, str]] = {}
    references: dict[str, str] = {}
    for index in range(_SAMPLED_CASE_COUNT):
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


def run_parity_suite(
    *,
    scoring_path: Path,
    fixtures_directory: Path,
    nltk_data_root: Path,
) -> dict[str, Any]:
    """Run the fixed plus sampled local-offline suite and return its report."""

    fixed_predictions, fixed_references = _load_fixed_cases(fixtures_directory)
    sampled_predictions, sampled_references = _build_sampled_cases()
    predictions = fixed_predictions | sampled_predictions
    references = fixed_references | sampled_references
    scorer_root = scoring_path.parent
    project = evaluate_official_exact(
        predictions,
        references,
        scorer_root=scorer_root,
        nltk_data_root=nltk_data_root,
    )
    supplied = evaluate_supplied_scorer(
        predictions,
        references,
        scoring_path=scoring_path,
        scorer_root=scorer_root,
        nltk_data_root=nltk_data_root,
    )
    report = build_parity_report(
        project,
        supplied,
        fixed_case_count=len(fixed_predictions),
        sampled_case_count=len(sampled_predictions),
        scoring_path=scoring_path,
    )
    report["sample_generation"] = {
        "algorithm": "python-random-v1-swap-two-of-seven",
        "seed": _SAMPLED_CASE_SEED,
    }
    return report


def parity_json_bytes(report: Mapping[str, Any]) -> bytes:
    """Render stable human-reviewable JSON with one final LF."""

    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def parity_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact deterministic MIL-001 parity evidence page."""

    counts = cast(Mapping[str, object], report["case_counts"])
    metrics = cast(Mapping[str, Any], report["metrics"])
    rouge = cast(Mapping[str, object], metrics["rouge_l"])
    meteor = cast(Mapping[str, object], metrics["meteor"])
    scorer = cast(Mapping[str, object], report["supplied_scorer"])
    return (
        "# MIL-001 Official Scorer Parity\n\n"
        f"status: {report['status']}\n\n"
        f"execution_mode: {report['execution_mode']}\n\n"
        f"threshold: {report['threshold']}\n\n"
        f"cases: {counts['fixed']} fixed + {counts['sampled']} sampled = {counts['total']}\n\n"
        "| Metric | Project | Supplied | Absolute difference |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| ROUGE-L | {rouge['project']} | {rouge['supplied']} | "
        f"{rouge['absolute_difference']} |\n"
        f"| METEOR | {meteor['project']} | {meteor['supplied']} | "
        f"{meteor['absolute_difference']} |\n\n"
        f"Per-query maximum absolute difference: {metrics['per_query_max_absolute_difference']}\n\n"
        f"Supplied scorer: `{scorer['path']}` (`{scorer['checksum']}`); "
        "top-level download calls executed: 0.\n"
    )
