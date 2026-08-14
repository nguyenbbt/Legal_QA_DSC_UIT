from __future__ import annotations

import json
import unicodedata

import pytest
from pydantic import ValidationError

from legal_rag.domain.models import ContextRecord, QuestionRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json

CHECKSUM = "sha256:" + "0" * 64


def valid_question(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.question.v1",
        "question_id": "001",
        "original_id": "001",
        "original_id_kind": "object_key_string",
        "source_position": 0,
        "source_artifact": "data/fixtures/questions.json",
        "source_checksum": CHECKSUM,
        "question": "Mức xử phạt là bao nhiêu?",
        "answer": None,
        "answer_state": "unlabeled",
    }
    value.update(changes)
    return value


def valid_context(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.context.v1",
        "context_id": "740",
        "original_id": "740",
        "original_id_kind": "json_integer",
        "source_position": 0,
        "source_artifact": "data/fixtures/context_740.json",
        "source_checksum": CHECKSUM,
        "name": "Quyet-dinh-740",
        "source_url": "https://example.invalid/legal/740",
        "passage": "Điều 1. Phạm vi điều chỉnh.",
        "indexable": True,
        "quarantine_reason": None,
    }
    value.update(changes)
    return value


def test_question_validates_exact_fields_round_trips_and_is_frozen() -> None:
    record = QuestionRecord.model_validate(valid_question())

    assert record.model_dump(mode="json") == valid_question()
    with pytest.raises(ValidationError):
        record.question = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"source_position": "0"}, "int_type"),
        ({"unexpected": True}, "extra_forbidden"),
        ({"source_artifact": "../private.json"}, "safe_relative_path"),
        ({"source_checksum": "0" * 64}, "string_pattern_mismatch"),
        ({"question_id": "2"}, "question_id_mismatch"),
        ({"answer_state": "gold"}, "question_answer_state"),
        ({"answer": "có", "answer_state": "unlabeled"}, "question_answer_state"),
    ],
)
def test_question_rejects_invalid_contracts(changes: dict[str, object], error_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        QuestionRecord.model_validate(valid_question(**changes))
    assert error_type in {error["type"] for error in exc_info.value.errors()}


def test_question_gold_requires_non_empty_answer() -> None:
    record = QuestionRecord.model_validate(
        valid_question(answer="  Có hiệu lực.  ", answer_state="gold")
    )
    assert record.answer == "  Có hiệu lực.  "

    with pytest.raises(ValidationError) as exc_info:
        QuestionRecord.model_validate(valid_question(answer="   ", answer_state="gold"))
    assert "question_answer_state" in {error["type"] for error in exc_info.value.errors()}


def test_context_enforces_indexable_and_empty_passage_invariants() -> None:
    indexable = ContextRecord.model_validate(valid_context())
    quarantined = ContextRecord.model_validate(
        valid_context(passage="", indexable=False, quarantine_reason="EMPTY_PASSAGE")
    )

    assert indexable.indexable is True
    assert quarantined.indexable is False
    with pytest.raises(ValidationError) as exc_info:
        ContextRecord.model_validate(valid_context(passage="  "))
    assert "context_quarantine_state" in {error["type"] for error in exc_info.value.errors()}


@pytest.mark.parametrize(
    "changes",
    [
        {"indexable": True, "quarantine_reason": "EMPTY_PASSAGE"},
        {"indexable": False, "quarantine_reason": None},
        {"passage": "", "indexable": False, "quarantine_reason": "OTHER_REASON"},
        {"context_id": "0740"},
        {"context_id": "741"},
        {"source_url": "ftp://example.invalid/legal/740"},
        {"source_url": "https://user:secret@example.invalid/legal/740"},
    ],
)
def test_context_rejects_invalid_contracts(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ContextRecord.model_validate(valid_context(**changes))


def test_context_preserves_raw_integer_lexeme_but_requires_canonical_id() -> None:
    record = ContextRecord.model_validate(valid_context(context_id="0", original_id="-0"))
    assert record.context_id == "0"
    assert record.original_id == "-0"


def test_all_operational_strings_require_nfc() -> None:
    composed = "Nội dung"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed

    with pytest.raises(ValidationError) as exc_info:
        ContextRecord.model_validate(valid_context(passage=decomposed))
    assert "nfc_required" in {error["type"] for error in exc_info.value.errors()}


def test_duplicate_aware_json_validation_is_atomic_and_has_provenance() -> None:
    raw = b'{"schema_version":"internal.question.v1","question_id":"001","question_id":"002"}\n'

    with pytest.raises(RecordValidationError) as exc_info:
        parse_record_json(
            raw,
            QuestionRecord,
            artifact_path="artifacts/internal/questions.jsonl",
            record_identity="001",
        )

    error = exc_info.value
    assert error.artifact_path == "artifacts/internal/questions.jsonl"
    assert error.record_identity == "001"
    assert [(issue.code, issue.json_path) for issue in error.issues] == [
        ("INTERNAL_DUPLICATE_KEY", "$.question_id")
    ]


def test_json_validation_requires_utf8_no_bom_and_exact_final_lf() -> None:
    encoded = json.dumps(valid_question(), ensure_ascii=False, separators=(",", ":")).encode()

    for raw, code in [
        (encoded, "INTERNAL_FINAL_LF_REQUIRED"),
        (b"\xef\xbb\xbf" + encoded + b"\n", "INTERNAL_UTF8_BOM_FORBIDDEN"),
        (b"\xff\n", "INTERNAL_UTF8_INVALID"),
    ]:
        with pytest.raises(RecordValidationError) as exc_info:
            parse_record_json(raw, QuestionRecord, artifact_path="fixture.jsonl")
        assert exc_info.value.issues[0].code == code


def test_json_validation_maps_schema_errors_to_stable_code_and_path() -> None:
    value = valid_question(unexpected=True)
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()

    with pytest.raises(RecordValidationError) as exc_info:
        parse_record_json(raw, QuestionRecord, artifact_path="fixture.jsonl")

    assert any(
        issue.code == "SCHEMA_UNKNOWN_FIELD" and issue.json_path == "$.unexpected"
        for issue in exc_info.value.issues
    )
