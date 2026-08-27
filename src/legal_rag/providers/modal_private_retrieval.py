"""Pure offline preflight for the closed D-054 private retrieval profile.

This module deliberately has no Modal dependency.  Provider code may consume only a
bundle and configuration that have already passed these closed contracts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, computed_field, model_validator

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import (
    FiniteScore,
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)

_ABSOLUTE_PATH = re.compile(r"(?:^|[\s='\"])(?:[A-Za-z]:[\\/]|\\\\|/[^/\s])")
_SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{6,}|ghp_[A-Za-z0-9]{6,}|"
    r"(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=])",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ModalPrivateRetrievalError(Exception):
    """Stable safe pre-provider rejection for D-054."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModalPrivateRetrievalChunk(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    canonical_text: NonEmptyString
    context_id: NonEmptyString
    canonical_start: NonNegativeInt
    canonical_end: PositiveInt
    hierarchy_path: tuple[NonEmptyString, ...] = Field(min_length=1)
    chunk_checksum: Sha256
    context_checksum: Sha256
    corpus_checksum: Sha256
    parent_evidence_id: NonEmptyString | None
    sibling_evidence_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def _validate_span_and_identity(self) -> Self:
        if self.canonical_start >= self.canonical_end:
            raise ValueError("canonical span must be non-empty")
        if self.evidence_id in self.sibling_evidence_ids:
            raise ValueError("evidence cannot be its own sibling")
        if len(self.sibling_evidence_ids) != len(set(self.sibling_evidence_ids)):
            raise ValueError("sibling evidence identities must be unique")
        return self


class ModalPrivateRetrievalQuestion(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    question: NonEmptyString
    question_checksum: Sha256


class ModalPrivateRetrievalBundle(FrozenStrictModel, frozen=True):
    schema_version: Literal["modal.private-retrieval.bundle.v1"]
    chunks: tuple[ModalPrivateRetrievalChunk, ...] = Field(min_length=1)
    questions: tuple[ModalPrivateRetrievalQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_identities(self) -> Self:
        chunk_ids = tuple(item.evidence_id for item in self.chunks)
        question_ids = tuple(item.question_id for item in self.questions)
        if len(chunk_ids) != len(set(chunk_ids)) or len(question_ids) != len(set(question_ids)):
            raise ValueError("bundle identities must be unique")
        corpus_checksums = {item.corpus_checksum for item in self.chunks}
        if len(corpus_checksums) != 1:
            raise ValueError("all chunks must address one frozen corpus")
        return self


class ModalPrivateRetrievalConfig(FrozenStrictModel, frozen=True):
    schema_version: Literal["modal.private-retrieval.config.v1"] = (
        "modal.private-retrieval.config.v1"
    )
    volume_private: Literal[True]
    teammate_sharing: Literal[False]
    backup_enabled: Literal[False]
    egress_enabled: Literal[False]
    max_containers: Literal[1]
    encrypted_io_retention_days: int = Field(ge=0, le=7)
    maximum_run_cost_usd: float = Field(gt=0, le=10, allow_inf_nan=False)
    campaign_spend_before_usd: float = Field(ge=0, allow_inf_nan=False)
    projected_run_cost_usd: float = Field(gt=0, le=10, allow_inf_nan=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def projected_campaign_cost_usd(self) -> float:
        return self.campaign_spend_before_usd + self.projected_run_cost_usd

    @model_validator(mode="after")
    def _validate_cost_ceiling(self) -> Self:
        if self.projected_run_cost_usd > self.maximum_run_cost_usd:
            raise ValueError("projected run cost exceeds the configured per-run limit")
        if self.projected_campaign_cost_usd >= 30:
            raise ValueError("D-054 campaign spend must remain strictly below USD 30")
        return self


class ModalPrivateRetrievalResultRow(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    evidence_id: NonEmptyString
    score: FiniteScore
    rank: PositiveInt


class ModalPrivateRetrievalTelemetry(FrozenStrictModel, frozen=True):
    question_count: PositiveInt
    candidate_count: NonNegativeInt
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_cuda_bytes: NonNegativeInt | None = None


class ModalPrivateRetrievalResponse(FrozenStrictModel, frozen=True):
    schema_version: Literal["modal.private-retrieval.response.v1"]
    model_id: NonEmptyString
    model_revision: NonEmptyString
    configuration_checksum: Sha256
    bundle_checksum: Sha256
    rows: tuple[ModalPrivateRetrievalResultRow, ...]
    aggregate_telemetry: ModalPrivateRetrievalTelemetry

    @model_validator(mode="after")
    def _validate_rows(self) -> Self:
        pairs = tuple((row.question_id, row.evidence_id) for row in self.rows)
        ranks = tuple((row.question_id, row.rank) for row in self.rows)
        if len(pairs) != len(set(pairs)) or len(ranks) != len(set(ranks)):
            raise ValueError("returned evidence identities and ranks must be unique per question")
        for question_id in dict.fromkeys(row.question_id for row in self.rows):
            question_ranks = tuple(row.rank for row in self.rows if row.question_id == question_id)
            if question_ranks != tuple(range(1, len(question_ranks) + 1)):
                raise ValueError("returned ranks must be consecutive and ordered per question")
        if self.aggregate_telemetry.candidate_count != len(self.rows):
            raise ValueError("aggregate candidate count differs from returned rows")
        return self


def _contains_forbidden_content(value: str) -> bool:
    return _SECRET.search(value) is not None or _ABSOLUTE_PATH.search(value) is not None


def build_modal_private_retrieval_bundle(
    *,
    chunks: Sequence[Mapping[str, object]],
    questions: Sequence[Mapping[str, object]],
    expected_chunk_checksums: Mapping[str, str],
    expected_context_checksums: Mapping[str, str],
    expected_canonical_text_checksums: Mapping[str, str],
    expected_corpus_checksum: str,
) -> ModalPrivateRetrievalBundle:
    """Validate and project the exact outbound D-054 bundle without provider access."""

    try:
        parsed_chunks = tuple(ModalPrivateRetrievalChunk.model_validate(item) for item in chunks)
        parsed_questions = tuple(
            ModalPrivateRetrievalQuestion.model_validate(item) for item in questions
        )
        bundle = ModalPrivateRetrievalBundle(
            schema_version="modal.private-retrieval.bundle.v1",
            chunks=parsed_chunks,
            questions=parsed_questions,
        )
    except ValidationError as error:
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_BUNDLE_INVALID", "D-054 outbound bundle violates its closed schema"
        ) from error
    outbound_text = tuple(
        value
        for item in bundle.chunks
        for value in (
            item.evidence_id,
            item.canonical_text,
            item.context_id,
            *item.hierarchy_path,
            *((item.parent_evidence_id,) if item.parent_evidence_id is not None else ()),
            *item.sibling_evidence_ids,
        )
    ) + tuple(value for item in bundle.questions for value in (item.question_id, item.question))
    if any(_contains_forbidden_content(value) for value in outbound_text):
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_CONTENT_FORBIDDEN",
            "D-054 outbound text contains a secret-like value or absolute path",
        )
    chunk_ids = {item.evidence_id for item in bundle.chunks}
    context_ids = {item.context_id for item in bundle.chunks}
    checksum_inputs_valid = (
        set(expected_chunk_checksums) == chunk_ids
        and set(expected_canonical_text_checksums) == chunk_ids
        and set(expected_context_checksums) == context_ids
        and _SHA256.fullmatch(expected_corpus_checksum) is not None
    )
    if (
        not checksum_inputs_valid
        or any(
            item.chunk_checksum != expected_chunk_checksums[item.evidence_id]
            or item.context_checksum != expected_context_checksums[item.context_id]
            or item.corpus_checksum != expected_corpus_checksum
            or expected_canonical_text_checksums[item.evidence_id]
            != checksum_bytes(item.canonical_text.encode("utf-8"))
            for item in bundle.chunks
        )
        or any(
            item.question_checksum != checksum_bytes(item.question.encode("utf-8"))
            for item in bundle.questions
        )
    ):
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_CHECKSUM_MISMATCH", "D-054 outbound content checksum mismatch"
        )
    return bundle


def validate_modal_private_retrieval_response(
    value: Mapping[str, Any], *, expected_question_ids: Sequence[str]
) -> ModalPrivateRetrievalResponse:
    """Validate the text-free closed return contract and exact question order."""

    try:
        response = ModalPrivateRetrievalResponse.model_validate(value)
    except ValidationError as error:
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_RESPONSE_INVALID", "D-054 return violates its closed schema"
        ) from error
    response_strings = (
        response.model_id,
        response.model_revision,
        *(row.question_id for row in response.rows),
        *(row.evidence_id for row in response.rows),
    )
    if any(_contains_forbidden_content(item) for item in response_strings):
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_RESPONSE_INVALID",
            "D-054 return contains a secret-like value or absolute path",
        )
    observed = tuple(dict.fromkeys(row.question_id for row in response.rows))
    expected = tuple(expected_question_ids)
    if not expected or len(expected) != len(set(expected)) or observed != expected:
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_RESPONSE_ID_MISMATCH",
            "D-054 return question identities/order differ from the submitted bundle",
        )
    if response.aggregate_telemetry.question_count != len(expected):
        raise ModalPrivateRetrievalError(
            "MODAL_PRIVATE_RESPONSE_ID_MISMATCH",
            "D-054 aggregate question count differs from the submitted bundle",
        )
    return response


LifecycleState = Literal["declared", "created", "deleted", "absence_verified"]


@dataclass(frozen=True, slots=True)
class ModalPrivateRetrievalLifecycle:
    campaign_id: str
    state: LifecycleState
    deletion_receipt_checksum: str | None = None

    @classmethod
    def initial(cls, campaign_id: str) -> ModalPrivateRetrievalLifecycle:
        if not campaign_id.strip() or _contains_forbidden_content(campaign_id):
            raise ModalPrivateRetrievalError(
                "MODAL_PRIVATE_LIFECYCLE_INVALID", "campaign identity is invalid"
            )
        return cls(campaign_id=campaign_id, state="declared")

    def mark_created(self) -> ModalPrivateRetrievalLifecycle:
        if self.state != "declared":
            raise ModalPrivateRetrievalError(
                "MODAL_PRIVATE_LIFECYCLE_INVALID", "Volume creation state is not monotonic"
            )
        return replace(self, state="created")

    def mark_deleted(self, *, deletion_receipt_checksum: str) -> ModalPrivateRetrievalLifecycle:
        if self.state != "created" or _SHA256.fullmatch(deletion_receipt_checksum) is None:
            raise ModalPrivateRetrievalError(
                "MODAL_PRIVATE_LIFECYCLE_INVALID", "Volume deletion requires a valid receipt"
            )
        return replace(
            self,
            state="deleted",
            deletion_receipt_checksum=deletion_receipt_checksum,
        )

    def mark_absence_verified(self) -> ModalPrivateRetrievalLifecycle:
        if self.state != "deleted" or self.deletion_receipt_checksum is None:
            raise ModalPrivateRetrievalError(
                "MODAL_PRIVATE_LIFECYCLE_INVALID",
                "Volume absence may be verified only after receipt-backed deletion",
            )
        return replace(self, state="absence_verified")


__all__ = [
    "ModalPrivateRetrievalBundle",
    "ModalPrivateRetrievalChunk",
    "ModalPrivateRetrievalConfig",
    "ModalPrivateRetrievalError",
    "ModalPrivateRetrievalLifecycle",
    "ModalPrivateRetrievalQuestion",
    "ModalPrivateRetrievalResponse",
    "build_modal_private_retrieval_bundle",
    "validate_modal_private_retrieval_response",
]
