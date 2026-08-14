"""Duplicate-aware atomic validation for internal v1 JSON records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from legal_rag.domain.models import AnswerRecord, ContextRecord, Evidence, FrozenStrictModel

_SIMPLE_MEMBER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One safe validation issue with a stable code and JSON path."""

    code: str
    json_path: str
    message: str


class RecordValidationError(Exception):
    """Atomic record failure carrying immutable provenance and all issues."""

    def __init__(
        self,
        *,
        artifact_path: str,
        record_identity: str | None,
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        super().__init__(issues[0].message if issues else "record validation failed")
        self.artifact_path = artifact_path
        self.record_identity = record_identity
        self.issues = issues


class _ObjectPairs(list[tuple[str, Any]]):
    pass


class _DuplicateMemberError(ValueError):
    def __init__(self, json_path: str) -> None:
        super().__init__(json_path)
        self.json_path = json_path


def _member_path(parent: str, member: str) -> str:
    if _SIMPLE_MEMBER.fullmatch(member) is not None:
        return f"{parent}.{member}"
    return f"{parent}[{json.dumps(member, ensure_ascii=False)}]"


def _convert_pairs(value: Any, path: str = "$") -> Any:
    if isinstance(value, _ObjectPairs):
        converted: dict[str, Any] = {}
        for key, member in value:
            member_path = _member_path(path, key)
            if key in converted:
                raise _DuplicateMemberError(member_path)
            converted[key] = _convert_pairs(member, member_path)
        return converted
    if isinstance(value, list):
        return [_convert_pairs(member, f"{path}[{index}]") for index, member in enumerate(value)]
    return value


def _raise_one(
    code: str,
    message: str,
    *,
    artifact_path: str,
    record_identity: str | None,
    json_path: str = "$",
) -> None:
    raise RecordValidationError(
        artifact_path=artifact_path,
        record_identity=record_identity,
        issues=(ValidationIssue(code=code, json_path=json_path, message=message),),
    )


def _pydantic_path(location: tuple[int | str, ...]) -> str:
    path = "$"
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path = _member_path(path, part)
    return path


_ERROR_CODES = {
    "extra_forbidden": "SCHEMA_UNKNOWN_FIELD",
    "missing": "SCHEMA_FIELD_MISSING",
    "literal_error": "SCHEMA_ENUM_INVALID",
    "finite_number": "SCHEMA_NUMBER_NONFINITE",
    "int_type": "SCHEMA_TYPE_INVALID",
    "string_type": "SCHEMA_TYPE_INVALID",
    "bool_type": "SCHEMA_TYPE_INVALID",
    "list_type": "SCHEMA_TYPE_INVALID",
    "tuple_type": "SCHEMA_TYPE_INVALID",
    "dict_type": "SCHEMA_TYPE_INVALID",
    "model_type": "SCHEMA_TYPE_INVALID",
    "string_pattern_mismatch": "SCHEMA_PATTERN_INVALID",
    "greater_than_equal": "SCHEMA_RANGE_INVALID",
}


def _map_validation_error(error: ValidationError) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        error_type = str(item["type"])
        code = _ERROR_CODES.get(error_type, f"SCHEMA_{error_type.upper()}")
        issues.append(
            ValidationIssue(
                code=code,
                json_path=_pydantic_path(tuple(item["loc"])),
                message=str(item["msg"]),
            )
        )
    return tuple(issues)


def parse_record_json[RecordT: FrozenStrictModel](
    data: bytes,
    model: type[RecordT],
    *,
    artifact_path: str,
    record_identity: str | None = None,
) -> RecordT:
    """Validate one UTF-8 internal JSON record and return it only on total success."""

    if data.startswith(b"\xef\xbb\xbf"):
        _raise_one(
            "INTERNAL_UTF8_BOM_FORBIDDEN",
            "internal JSON must not contain a UTF-8 BOM",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        _raise_one(
            "INTERNAL_FINAL_LF_REQUIRED",
            "internal JSON must end with exactly one LF",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
    if b"\r" in data:
        _raise_one(
            "INTERNAL_NEWLINE_INVALID",
            "internal JSON must use LF newlines",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _raise_one(
            "INTERNAL_UTF8_INVALID",
            "internal JSON is not strict UTF-8",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
        raise AssertionError("unreachable") from exc

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite constant")

    try:
        pairs = json.loads(
            text,
            object_pairs_hook=_ObjectPairs,
            parse_constant=reject_constant,
        )
        converted = _convert_pairs(pairs)
    except _DuplicateMemberError as exc:
        _raise_one(
            "INTERNAL_DUPLICATE_KEY",
            "internal JSON contains a duplicate object key",
            artifact_path=artifact_path,
            record_identity=record_identity,
            json_path=exc.json_path,
        )
        raise AssertionError("unreachable") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        _raise_one(
            "INTERNAL_JSON_INVALID",
            "internal JSON syntax or number is invalid",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
        raise AssertionError("unreachable") from exc
    if not isinstance(converted, dict):
        _raise_one(
            "INTERNAL_ROOT_TYPE",
            "internal JSON record root must be an object",
            artifact_path=artifact_path,
            record_identity=record_identity,
        )
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        raise RecordValidationError(
            artifact_path=artifact_path,
            record_identity=record_identity,
            issues=_map_validation_error(exc),
        ) from None


def validate_evidence_span(
    evidence: Evidence,
    context: ContextRecord,
    *,
    artifact_path: str,
) -> Evidence:
    """Prove Evidence identity and display text against canonical NFC passage."""

    if evidence.context_id != context.context_id or evidence.source_url != context.source_url:
        _raise_one(
            "EVIDENCE_CONTEXT_MISMATCH",
            "evidence context identity does not match the canonical context",
            artifact_path=artifact_path,
            record_identity=evidence.evidence_id,
            json_path="$.context_id",
        )
    start = evidence.canonical_start
    end = evidence.canonical_end
    if end > len(context.passage) or context.passage[start:end] != evidence.display_text:
        _raise_one(
            "EVIDENCE_OFFSET_INVALID",
            "canonical offsets do not reconstruct display_text exactly",
            artifact_path=artifact_path,
            record_identity=evidence.evidence_id,
            json_path="$.canonical_start",
        )
    return evidence


def validate_answer_evidence_order(
    answer: AnswerRecord,
    evidence: tuple[Evidence, ...],
    *,
    artifact_path: str,
) -> AnswerRecord:
    """Require answer references to equal the accepted evidence sequence."""

    accepted_ids = tuple(item.evidence_id for item in evidence)
    if answer.evidence_ids != accepted_ids:
        _raise_one(
            "ANSWER_EVIDENCE_ORDER_MISMATCH",
            "answer evidence_ids do not equal accepted evidence order",
            artifact_path=artifact_path,
            record_identity=answer.question_id,
            json_path="$.evidence_ids",
        )
    return answer
