from __future__ import annotations

import math
import unicodedata

import pytest
from pydantic import ValidationError

from legal_rag.domain.models import (
    AnswerRecord,
    ContextRecord,
    Evidence,
    GeneratedAnswer,
)
from legal_rag.domain.validation import (
    RecordValidationError,
    validate_answer_evidence_order,
    validate_evidence_span,
)

CHECKSUM = "sha256:" + "0" * 64
EVIDENCE_ID = "chunk_aaaaaaaaaaaaaaaaaaaaaaaa"


def valid_context(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.context.v1",
        "context_id": "740",
        "original_id": "740",
        "original_id_kind": "json_integer",
        "source_position": 0,
        "source_artifact": "data/fixtures/context_740.json",
        "source_checksum": CHECKSUM,
        "name": None,
        "source_url": "https://example.invalid/legal/740",
        "passage": "Nội dung Điều 1.",
        "indexable": True,
        "quarantine_reason": None,
    }
    value.update(changes)
    return value


def valid_evidence(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.evidence.v1",
        "evidence_id": EVIDENCE_ID,
        "context_id": "740",
        "source_url": "https://example.invalid/legal/740",
        "hierarchy_path": ("Điều 1",),
        "canonical_start": 0,
        "canonical_end": 3,
        "display_text": "Nội",
        "retrieval_text": "nội",
        "rank": 1,
        "component_scores": {
            "exact_reference_match": True,
            "sparse_score": 3.25,
            "dense_score": None,
            "reranker_score": None,
        },
        "chunk_checksum": CHECKSUM,
        "context_checksum": CHECKSUM,
        "integrity_status": "valid",
        "claim_support": "unknown",
        "version_validity": "unknown",
    }
    value.update(changes)
    return value


def valid_generated_answer(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.generated_answer.v1",
        "question_id": "001",
        "answer_text": "Nội dung trả lời.",
        "generator_id": "fixture-extractive-v1",
        "competition_policy": "baseline.v1",
        "used_evidence_ids": (EVIDENCE_ID,),
        "material_claims": (),
    }
    value.update(changes)
    return value


def valid_answer(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "internal.answer.v1",
        "question_id": "001",
        "answer": "Nội dung trả lời.",
        "generator_id": "fixture-extractive-v1",
        "evidence_ids": (EVIDENCE_ID,),
        "run_id": "run_4f12b96c4b87a168759b3ca0",
    }
    value.update(changes)
    return value


def test_evidence_is_exact_closed_and_deeply_immutable() -> None:
    record = Evidence.model_validate(valid_evidence())

    assert record.hierarchy_path == ("Điều 1",)
    assert record.component_scores.sparse_score == 3.25
    with pytest.raises(ValidationError):
        record.rank = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        record.component_scores.sparse_score = 4.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"canonical_start": 3, "canonical_end": 3},
        {"canonical_start": -1},
        {"hierarchy_path": ()},
        {"hierarchy_path": ("",)},
        {"display_text": "   "},
        {"rank": 0},
        {"integrity_status": "invalid"},
        {"claim_support": "valid"},
        {"version_validity": "supported"},
        {"component_scores": {"exact_reference_match": 1}},
        {"unexpected": True},
    ],
)
def test_evidence_rejects_invalid_contracts(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate(valid_evidence(**changes))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_evidence_rejects_non_finite_scores(value: float) -> None:
    scores = dict(valid_evidence()["component_scores"])  # type: ignore[arg-type]
    scores["sparse_score"] = value
    with pytest.raises(ValidationError):
        Evidence.model_validate(valid_evidence(component_scores=scores))


def test_evidence_span_reconstructs_composed_nfc_exactly() -> None:
    context = ContextRecord.model_validate(valid_context())
    evidence = Evidence.model_validate(valid_evidence())

    assert (
        validate_evidence_span(evidence, context, artifact_path="fixture.evidence.jsonl")
        is evidence
    )
    assert context.passage[evidence.canonical_start : evidence.canonical_end] == "Nội"

    decomposed = unicodedata.normalize("NFD", "Nội")
    with pytest.raises(ValidationError):
        Evidence.model_validate(valid_evidence(display_text=decomposed))


@pytest.mark.parametrize(
    "changes",
    [
        {"canonical_end": 30},
        {"display_text": "Nồi"},
        {"context_id": "741"},
        {"source_url": "https://example.invalid/legal/741"},
    ],
)
def test_evidence_span_failure_has_stable_atomic_error(changes: dict[str, object]) -> None:
    context = ContextRecord.model_validate(valid_context())
    evidence = Evidence.model_validate(valid_evidence(**changes))

    with pytest.raises(RecordValidationError) as exc_info:
        validate_evidence_span(evidence, context, artifact_path="fixture.evidence.jsonl")
    assert len(exc_info.value.issues) == 1
    assert exc_info.value.issues[0].code in {
        "EVIDENCE_CONTEXT_MISMATCH",
        "EVIDENCE_OFFSET_INVALID",
    }


def test_verified_generated_answer_accepts_exact_material_claim() -> None:
    claim = {
        "claim_id": "claim_1",
        "text": "Quy định áp dụng.",
        "evidence_ids": [EVIDENCE_ID],
        "claim_support": "supported",
        "version_validity": "unknown",
        "confidence": 0.75,
    }
    record = GeneratedAnswer.model_validate_json(
        __import__("json").dumps(
            valid_generated_answer(competition_policy="verified.v1", material_claims=[claim]),
            ensure_ascii=False,
        )
    )

    assert record.material_claims[0].claim_id == "claim_1"
    assert record.material_claims[0].evidence_ids == (EVIDENCE_ID,)


@pytest.mark.parametrize(
    "changes",
    [
        {"answer_text": "   "},
        {"generator_id": "Invalid Generator"},
        {"competition_policy": "research.v1"},
        {"used_evidence_ids": (EVIDENCE_ID, EVIDENCE_ID)},
        {"material_claims": ({"claim_id": "claim_1"},)},
        {"unexpected": True},
    ],
)
def test_generated_answer_rejects_invalid_contracts(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(valid_generated_answer(**changes))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.inf, math.nan])
def test_material_claim_rejects_invalid_confidence(confidence: float) -> None:
    claim = {
        "claim_id": "claim_1",
        "text": "Quy định áp dụng.",
        "evidence_ids": (EVIDENCE_ID,),
        "claim_support": "supported",
        "version_validity": "valid",
        "confidence": confidence,
    }
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(
            valid_generated_answer(competition_policy="verified.v1", material_claims=(claim,))
        )


def test_answer_record_enforces_pattern_uniqueness_and_ordered_reference() -> None:
    evidence = Evidence.model_validate(valid_evidence())
    answer = AnswerRecord.model_validate(valid_answer())

    assert (
        validate_answer_evidence_order(answer, (evidence,), artifact_path="fixture.answers.jsonl")
        is answer
    )
    for changes in [
        {"answer": "   "},
        {"generator_id": "Fixture V1"},
        {"run_id": "run_ABC"},
        {"evidence_ids": (EVIDENCE_ID, EVIDENCE_ID)},
        {"unexpected": True},
    ]:
        with pytest.raises(ValidationError):
            AnswerRecord.model_validate(valid_answer(**changes))


def test_answer_record_allows_empty_evidence_for_later_refusal_policy_validation() -> None:
    record = AnswerRecord.model_validate(valid_answer(evidence_ids=()))
    assert record.evidence_ids == ()


def test_answer_evidence_order_mismatch_is_typed() -> None:
    evidence = Evidence.model_validate(valid_evidence())
    answer = AnswerRecord.model_validate(valid_answer(evidence_ids=("chunk_other",)))

    with pytest.raises(RecordValidationError) as exc_info:
        validate_answer_evidence_order(answer, (evidence,), artifact_path="fixture.answers.jsonl")
    assert exc_info.value.issues[0].code == "ANSWER_EVIDENCE_ORDER_MISMATCH"
