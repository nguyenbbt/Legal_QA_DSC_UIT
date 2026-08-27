"""Strict deterministic organizer prediction writer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import AnswerRecord, QuestionRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.ingestion.organizer import (
    OrganizerDataError,
    OrganizerQuestionReader,
    OrganizerQuestionSource,
)

SUBMISSION_SCHEMA_VERSION = "organizer.prediction.answer-only.v2"


class _RawObject(list[tuple[str, Any]]):
    pass


class SubmissionError(Exception):
    """Stable failure at the organizer submission boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    count: int
    checksum: str

    @property
    def summary(self) -> str:
        return f"VALID SUBMISSION count={self.count} {self.checksum}"


def _fail(code: str, message: str) -> NoReturn:
    raise SubmissionError(code, message)


def _source_records(
    source_bytes: bytes,
) -> tuple[tuple[QuestionRecord, ...], tuple[OrganizerQuestionSource, ...]]:
    try:
        imported = OrganizerQuestionReader().read_bytes(
            source_bytes,
            kind="public",
            artifact_path="questions/source.json",
        )
    except OrganizerDataError as exc:
        raise SubmissionError("SUB_SOURCE_INVALID", exc.message) from exc
    return imported.records, imported.source_records


def _render(value: dict[str, dict[str, str]]) -> bytes:
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    return (text + "\n").encode("utf-8")


def answers_jsonl_bytes(answers: tuple[AnswerRecord, ...]) -> bytes:
    """Serialize ordered internal answers as strict deterministic JSONL."""

    lines = (
        json.dumps(
            answer.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for answer in answers
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def load_answers_jsonl(
    data: bytes,
    *,
    artifact_path: str = "answers.jsonl",
) -> tuple[AnswerRecord, ...]:
    """Load a complete internal answer artifact or fail atomically."""

    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("SUB_ANSWER_ARTIFACT_INVALID", "answer JSONL has invalid UTF-8 framing")
    lines = data.splitlines()
    if not lines or any(not line for line in lines):
        _fail("SUB_ANSWER_ARTIFACT_INVALID", "answer JSONL contains an empty record")
    try:
        return tuple(
            parse_record_json(
                line + b"\n",
                AnswerRecord,
                artifact_path=artifact_path,
                record_identity=str(line_number),
            )
            for line_number, line in enumerate(lines, start=1)
        )
    except RecordValidationError as exc:
        raise SubmissionError(
            "SUB_ANSWER_ARTIFACT_INVALID",
            "answer JSONL contains an invalid record",
        ) from exc


def build_submission(source_bytes: bytes, answers: tuple[AnswerRecord, ...]) -> bytes:
    """Render answers over the immutable source object without changing its identity."""

    records, source = _source_records(source_bytes)
    expected_ids = tuple(record.question_id for record in records)
    actual_ids = tuple(answer.question_id for answer in answers)
    if actual_ids != expected_ids:
        _fail("SUB_ID_MISMATCH", "answer IDs must exactly match source ID order")

    output: dict[str, dict[str, str]] = {}
    for source_record, answer in zip(source, answers, strict=True):
        output[source_record.original_id] = {"answer": answer.answer}
    rendered = _render(output)
    validate_submission(source_bytes, rendered)
    return rendered


def build_submission_zip(submission_bytes: bytes) -> bytes:
    """Package one canonical root `submission.json` with deterministic ZIP metadata."""

    if not submission_bytes:
        _fail("SUB_SCHEMA_INVALID", "submission bytes must be non-empty")
    output = io.BytesIO()
    entry = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 0
    entry.external_attr = 0
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(entry, submission_bytes)
    return output.getvalue()


def _decode_predictions(data: bytes) -> _RawObject:
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        _fail("SUB_ENCODING_INVALID", "predictions must be UTF-8 without BOM and use LF")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("SUB_ENCODING_INVALID", "predictions must be strict UTF-8")
    try:
        parsed = json.loads(text, object_pairs_hook=_RawObject)
    except json.JSONDecodeError:
        _fail("SUB_SCHEMA_INVALID", "predictions must be valid JSON")
    if not isinstance(parsed, _RawObject):
        _fail("SUB_SCHEMA_INVALID", "prediction root must be an object")
    return parsed


def _reject_duplicates(value: Any) -> None:
    if isinstance(value, _RawObject):
        seen: set[str] = set()
        for key, member in value:
            if key in seen:
                _fail("SUB_DUPLICATE_KEY", "predictions contain a duplicate object member")
            seen.add(key)
            _reject_duplicates(member)
    elif isinstance(value, list):
        for member in value:
            _reject_duplicates(member)


def validate_submission(source_bytes: bytes, predictions_bytes: bytes) -> SubmissionValidation:
    """Validate schema, identity, source text, and exact serialization bytes."""

    _, source = _source_records(source_bytes)
    parsed = _decode_predictions(predictions_bytes)
    _reject_duplicates(parsed)
    normalized: dict[str, dict[str, str]] = {}

    for question_id, raw_record in parsed:
        if not isinstance(raw_record, _RawObject):
            _fail("SUB_SCHEMA_INVALID", "each prediction must be an object")
        fields = dict(raw_record)
        if set(fields) != {"answer"}:
            _fail("SUB_SCHEMA_INVALID", "prediction fields must contain exactly answer")
        answer = fields["answer"]
        if type(answer) is not str or not answer.strip():
            _fail("SUB_EMPTY_ANSWER", "prediction answer must be a non-empty string")
        normalized[question_id] = dict(raw_record)

    if tuple(normalized) != tuple(record.original_id for record in source):
        _fail("SUB_ID_MISMATCH", "prediction IDs must exactly match source ID order")
    if any(tuple(fields) != ("answer",) for fields in normalized.values()):
        _fail("SUB_SCHEMA_INVALID", "answer must be the only prediction field")
    if predictions_bytes != _render(normalized):
        _fail("SUB_ENCODING_INVALID", "predictions do not use the canonical byte format")

    checksum = checksum_bytes(predictions_bytes)
    return SubmissionValidation(count=len(normalized), checksum=checksum)


def write_submission(
    destination: Path,
    source_bytes: bytes,
    answers: tuple[AnswerRecord, ...],
) -> SubmissionValidation:
    """Validate fully, then atomically replace the destination with complete bytes."""

    rendered = build_submission(source_bytes, answers)
    result = validate_submission(source_bytes, rendered)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return result
