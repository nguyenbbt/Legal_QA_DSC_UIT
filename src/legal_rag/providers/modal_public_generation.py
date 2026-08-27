"""Typed allowlist for the owner-approved Modal public-generation transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field, ValidationError, model_validator

from legal_rag.domain.models import (
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
)
from legal_rag.evaluation.public_dry_run import load_public_evidence_queue
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.ingestion.organizer import OrganizerDataError, OrganizerQuestionReader


class ModalPublicGenerationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModalPublicGenerationRequest(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    question: NonEmptyString
    evidence: tuple[NonEmptyString, ...] = Field(max_length=3)
    system_prompt: NonEmptyString

    @model_validator(mode="after")
    def _validate_approval_scope(self) -> ModalPublicGenerationRequest:
        if self.system_prompt != PROMPT_A:
            raise ValueError("only the owner-approved prompt A may leave the local host")
        return self


class ModalPublicGenerationResponse(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    answer: NonEmptyString
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    peak_cuda_bytes: NonNegativeInt


def _expected_ids(values: Sequence[str]) -> tuple[str, ...]:
    identifiers = tuple(values)
    if (
        not identifiers
        or len(identifiers) != len(set(identifiers))
        or any(not item for item in identifiers)
    ):
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_QUESTION_ID_MISMATCH",
            "approved public question IDs must be non-empty and unique",
        )
    return identifiers


def build_modal_public_requests(
    evidence_queue_data: bytes,
    *,
    public_source_data: bytes,
    expected_question_ids: Sequence[str],
    system_prompt: str,
) -> tuple[ModalPublicGenerationRequest, ...]:
    """Project only the four explicitly owner-approved fields into remote requests."""

    if system_prompt != PROMPT_A:
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_APPROVAL_SCOPE_VIOLATION",
            "Modal public generation permits only prompt A",
        )
    rows = load_public_evidence_queue(evidence_queue_data)
    expected = _expected_ids(expected_question_ids)
    try:
        public_questions = (
            OrganizerQuestionReader()
            .read_bytes(
                public_source_data,
                kind="public",
                artifact_path="questions/public-source.json",
            )
            .records
        )
    except OrganizerDataError as error:
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_QUESTION_SOURCE_INVALID",
            "approved public question source is invalid",
        ) from error
    if tuple(question.question_id for question in public_questions) != expected:
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_QUESTION_ID_MISMATCH",
            "approved IDs differ from the immutable public source",
        )
    if tuple(row.question_id for row in rows) != expected:
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_QUESTION_ID_MISMATCH",
            "evidence queue IDs differ from the approved public source",
        )
    if tuple((row.question_id, row.question) for row in rows) != tuple(
        (question.question_id, question.question) for question in public_questions
    ):
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_QUESTION_SOURCE_MISMATCH",
            "evidence queue questions differ from the immutable public source",
        )
    if any(len(row.evidence) > 3 for row in rows):
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_APPROVAL_SCOPE_VIOLATION",
            "Modal public generation permits at most three evidence passages",
        )
    return tuple(
        ModalPublicGenerationRequest(
            question_id=row.question_id,
            question=row.question,
            evidence=tuple(item.display_text for item in row.evidence),
            system_prompt=system_prompt,
        )
        for row in rows
    )


def validate_modal_public_responses(
    values: Sequence[Mapping[str, object]],
    *,
    expected_question_ids: Sequence[str],
) -> tuple[ModalPublicGenerationResponse, ...]:
    """Treat every Modal/model response as untrusted and enforce exact ID order."""

    try:
        responses = tuple(ModalPublicGenerationResponse.model_validate(value) for value in values)
    except ValidationError as error:
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_RESPONSE_INVALID",
            "Modal returned a response outside the approved answer/telemetry contract",
        ) from error
    if tuple(response.question_id for response in responses) != _expected_ids(
        expected_question_ids
    ):
        raise ModalPublicGenerationError(
            "MODAL_PUBLIC_RESPONSE_ID_MISMATCH",
            "Modal response IDs or order differ from the submitted public questions",
        )
    return responses


__all__ = [
    "ModalPublicGenerationError",
    "ModalPublicGenerationRequest",
    "ModalPublicGenerationResponse",
    "build_modal_public_requests",
    "validate_modal_public_responses",
]
