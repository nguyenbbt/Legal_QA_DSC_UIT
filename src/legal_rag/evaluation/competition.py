"""Deterministic official-exact development evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn

from legal_rag.domain.artifacts import ImmutableArtifactError, write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.official_exact import (
    evaluate_official_exact,
    supplied_scorer_provenance,
)


class CompetitionEvaluationError(Exception):
    """Stable safe failure at the competition evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class CompetitionEvaluation:
    per_query_bytes: bytes
    report_bytes: bytes
    question_count: int
    macro_rouge_l: float
    macro_meteor: float


class _ObjectPairs(list[tuple[str, Any]]):
    pass


def _fail(code: str, message: str) -> NoReturn:
    raise CompetitionEvaluationError(code, message)


def _convert(value: Any) -> Any:
    if isinstance(value, _ObjectPairs):
        converted: dict[str, Any] = {}
        for key, member in value:
            if key in converted:
                _fail("EVAL_JSON_DUPLICATE_KEY", "evaluation input contains a duplicate key")
            converted[key] = _convert(member)
        return converted
    if isinstance(value, list):
        return [_convert(member) for member in value]
    return value


def _load_json_object(data: bytes, *, label: str) -> dict[str, Any]:
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        _fail("EVAL_INPUT_ENCODING_INVALID", f"{label} has invalid UTF-8 framing")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_ObjectPairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CompetitionEvaluationError(
            "EVAL_INPUT_JSON_INVALID", f"{label} is not valid strict JSON"
        ) from error
    converted = _convert(value)
    if not isinstance(converted, dict):
        _fail("EVAL_INPUT_ROOT_INVALID", f"{label} root must be an object")
    return converted


def _validated_inputs(
    predictions_bytes: bytes, references_bytes: bytes
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    raw_predictions = _load_json_object(predictions_bytes, label="predictions")
    raw_references = _load_json_object(references_bytes, label="references")
    predictions: dict[str, dict[str, object]] = {}
    references: dict[str, str] = {}
    for question_id, raw_record in raw_predictions.items():
        if (
            type(question_id) is not str
            or not isinstance(raw_record, dict)
            or set(raw_record) != {"answer"}
            or type(raw_record["answer"]) is not str
            or not raw_record["answer"].strip()
        ):
            _fail("EVAL_PREDICTION_INVALID", "prediction records require one non-empty answer")
        predictions[question_id] = raw_record
    for question_id, reference in raw_references.items():
        if type(question_id) is not str or type(reference) is not str or not reference.strip():
            _fail("EVAL_REFERENCE_INVALID", "references require non-empty answer strings")
        references[question_id] = reference
    if not predictions:
        _fail("EVAL_INPUT_EMPTY", "competition evaluation requires at least one question")
    if set(predictions) != set(references):
        _fail("EVAL_QUESTION_ID_MISMATCH", "prediction and reference IDs differ")
    return predictions, references


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=pretty,
            indent=indent,
            separators=separators,
        )
        + "\n"
    ).encode()


def evaluate_competition_bytes(
    predictions_bytes: bytes,
    references_bytes: bytes,
    *,
    scorer_root: Path,
    nltk_data_root: Path,
) -> CompetitionEvaluation:
    """Score strict inputs and render stable per-query and macro artifacts."""

    predictions, references = _validated_inputs(predictions_bytes, references_bytes)
    result = evaluate_official_exact(
        predictions,
        references,
        scorer_root=scorer_root,
        nltk_data_root=nltk_data_root,
    )
    per_query_bytes = _jsonl_rows(
        tuple(
            {
                "schema_version": "competition.per_query.v1",
                "question_id": row.question_id,
                "rouge_l": row.rouge_l,
                "meteor": row.meteor,
            }
            for row in result.per_query
        )
    )
    report = {
        "schema_version": "competition.evaluation.report.v1",
        "mode": "official_exact",
        "execution_mode": "local-offline",
        "baseline_kind": "plumbing_baseline",
        "question_count": len(result.question_ids),
        "question_order": list(result.question_ids),
        "ordering": "prediction_input_order",
        "predictions_checksum": checksum_bytes(predictions_bytes),
        "references_checksum": checksum_bytes(references_bytes),
        "per_query_checksum": checksum_bytes(per_query_bytes),
        "metrics": {
            "macro_rouge_l": result.macro_rouge_l,
            "macro_meteor": result.macro_meteor,
        },
        "dependencies": {"numpy": version("numpy"), "nltk": version("nltk")},
        "supplied_scorer": supplied_scorer_provenance(scorer_root / "scoring.py"),
        "limitation": "fixed_refusal_without_real_context_index",
    }
    return CompetitionEvaluation(
        per_query_bytes=per_query_bytes,
        report_bytes=_json_bytes(report, pretty=True),
        question_count=len(result.question_ids),
        macro_rouge_l=result.macro_rouge_l,
        macro_meteor=result.macro_meteor,
    )


def _jsonl_rows(rows: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(_json_bytes(row, pretty=False) for row in rows)


def write_competition_evaluation(
    evaluation: CompetitionEvaluation, *, per_query_path: Path, report_path: Path
) -> dict[str, str]:
    """Preflight and create both immutable evaluation outputs."""

    outputs = {
        "per_query": (per_query_path, evaluation.per_query_bytes),
        "report": (report_path, evaluation.report_bytes),
    }
    for path, data in outputs.values():
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            _fail("EVAL_REPORT_IMMUTABLE", "an existing evaluation report cannot be replaced")
    try:
        return {label: write_immutable_bytes(path, data) for label, (path, data) in outputs.items()}
    except ImmutableArtifactError as error:
        raise CompetitionEvaluationError("EVAL_REPORT_WRITE_FAILED", error.message) from error
