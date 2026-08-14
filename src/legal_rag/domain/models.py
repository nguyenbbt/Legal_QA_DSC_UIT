"""Exact immutable v1 domain schemas."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

_JSON_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_CANONICAL_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_QUARANTINE_CODE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_GENERATOR_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


def _require_nfc(value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise PydanticCustomError("nfc_required", "string must be NFC-normalized")
    return value


def _require_non_empty(value: str) -> str:
    if not value.strip():
        raise PydanticCustomError("non_empty_string", "string must be non-empty after trimming")
    return value


def _require_safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"^[A-Za-z]:", value) is not None
    ):
        raise PydanticCustomError(
            "safe_relative_path", "path must be normalized repository-relative POSIX"
        )
    return value


def _require_http_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PydanticCustomError("absolute_http_url", "value must be an absolute HTTP(S) URL")
    return value


def _require_json_integer_lexeme(value: str) -> str:
    if _JSON_INTEGER.fullmatch(value) is None:
        raise PydanticCustomError("json_integer_lexeme", "value must be a JSON integer lexeme")
    return value


def _require_canonical_integer(value: str) -> str:
    if _CANONICAL_INTEGER.fullmatch(value) is None:
        raise PydanticCustomError(
            "canonical_integer_string", "value must be a canonical base-10 integer string"
        )
    return value


def _require_quarantine_code(value: str) -> str:
    if _QUARANTINE_CODE.fullmatch(value) is None:
        raise PydanticCustomError(
            "quarantine_code", "quarantine reason must be an uppercase typed code"
        )
    return value


NfcString = Annotated[str, AfterValidator(_require_nfc)]
NonEmptyString = Annotated[NfcString, AfterValidator(_require_non_empty)]
SafeRelativePath = Annotated[NfcString, AfterValidator(_require_safe_relative_path)]
AbsoluteHttpUrl = Annotated[NfcString, AfterValidator(_require_http_url)]
JsonIntegerLexeme = Annotated[NfcString, AfterValidator(_require_json_integer_lexeme)]
CanonicalIntegerString = Annotated[NfcString, AfterValidator(_require_canonical_integer)]
QuarantineCode = Annotated[NfcString, AfterValidator(_require_quarantine_code)]
Sha256 = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
    AfterValidator(_require_nfc),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
GeneratorId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
    AfterValidator(_require_nfc),
]
RunId = Annotated[
    str,
    StringConstraints(pattern=r"^run_[0-9a-f]{24}$"),
    AfterValidator(_require_nfc),
]
CodeRevision = Annotated[
    str,
    StringConstraints(pattern=r"^(?:git:[0-9a-f]{7,64}|tree:sha256:[0-9a-f]{64})$"),
    AfterValidator(_require_nfc),
]


def _require_unique(values: tuple[str, ...], *, error_type: str, label: str) -> None:
    if len(values) != len(set(values)):
        raise PydanticCustomError(error_type, f"{label} must be unique")


class FrozenStrictModel(BaseModel, frozen=True):
    """Common configuration for every closed immutable operational model."""

    model_config = ConfigDict(extra="forbid", strict=True)


class QuestionRecord(FrozenStrictModel, frozen=True):
    """`internal.question.v1`."""

    schema_version: Literal["internal.question.v1"]
    question_id: NonEmptyString
    original_id: NonEmptyString
    original_id_kind: Literal["object_key_string"]
    source_position: NonNegativeInt
    source_artifact: SafeRelativePath
    source_checksum: Sha256
    question: NonEmptyString
    answer: NfcString | None
    answer_state: Literal["gold", "unlabeled"]

    @model_validator(mode="after")
    def _validate_identity_and_answer(self) -> Self:
        if self.question_id != self.original_id:
            raise PydanticCustomError("question_id_mismatch", "question_id must equal original_id")
        is_valid_gold = (
            self.answer_state == "gold" and self.answer is not None and bool(self.answer.strip())
        )
        is_valid_unlabeled = self.answer_state == "unlabeled" and self.answer is None
        if not (is_valid_gold or is_valid_unlabeled):
            raise PydanticCustomError(
                "question_answer_state", "answer and answer_state are inconsistent"
            )
        return self


class ContextRecord(FrozenStrictModel, frozen=True):
    """`internal.context.v1` in canonical NFC offset space."""

    schema_version: Literal["internal.context.v1"]
    context_id: CanonicalIntegerString
    original_id: JsonIntegerLexeme
    original_id_kind: Literal["json_integer"]
    source_position: NonNegativeInt
    source_artifact: SafeRelativePath
    source_checksum: Sha256
    name: NfcString | None
    source_url: AbsoluteHttpUrl
    passage: NfcString
    indexable: bool
    quarantine_reason: QuarantineCode | None

    @model_validator(mode="after")
    def _validate_identity_and_quarantine(self) -> Self:
        if self.context_id != str(int(self.original_id)):
            raise PydanticCustomError(
                "context_id_mismatch", "context_id must be the canonical original_id"
            )
        has_passage = bool(self.passage.strip())
        valid_indexable = self.indexable and self.quarantine_reason is None and has_passage
        valid_quarantined = (
            not self.indexable
            and self.quarantine_reason is not None
            and (
                (not has_passage and self.quarantine_reason == "EMPTY_PASSAGE")
                or (has_passage and self.quarantine_reason != "EMPTY_PASSAGE")
            )
        )
        if not (valid_indexable or valid_quarantined):
            raise PydanticCustomError(
                "context_quarantine_state",
                "indexable, passage, and quarantine_reason are inconsistent",
            )
        return self


class ComponentScores(FrozenStrictModel, frozen=True):
    """Closed retrieval-component score object embedded in Evidence."""

    exact_reference_match: bool
    sparse_score: FiniteScore | None
    dense_score: FiniteScore | None
    reranker_score: FiniteScore | None


class Evidence(FrozenStrictModel, frozen=True):
    """Accepted `internal.evidence.v1` record."""

    schema_version: Literal["internal.evidence.v1"]
    evidence_id: NonEmptyString
    context_id: CanonicalIntegerString
    source_url: AbsoluteHttpUrl
    hierarchy_path: tuple[NonEmptyString, ...]
    canonical_start: NonNegativeInt
    canonical_end: NonNegativeInt
    display_text: NonEmptyString
    retrieval_text: NonEmptyString
    rank: PositiveInt
    component_scores: ComponentScores
    chunk_checksum: Sha256
    context_checksum: Sha256
    integrity_status: Literal["valid"]
    claim_support: Literal["supported", "contradicted", "unknown"]
    version_validity: Literal["valid", "invalid", "unknown"]

    @model_validator(mode="after")
    def _validate_interval_and_hierarchy(self) -> Self:
        if self.canonical_start >= self.canonical_end:
            raise PydanticCustomError(
                "canonical_interval", "canonical_start must be less than canonical_end"
            )
        if not self.hierarchy_path:
            raise PydanticCustomError(
                "hierarchy_path_empty", "hierarchy_path must contain at least one item"
            )
        return self


class MaterialClaim(FrozenStrictModel, frozen=True):
    """Exact material-claim object for `verified.v1`."""

    claim_id: NonEmptyString
    text: NonEmptyString
    evidence_ids: tuple[NonEmptyString, ...]
    claim_support: Literal["supported", "contradicted", "unknown"]
    version_validity: Literal["valid", "invalid", "unknown"]
    confidence: Confidence

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> Self:
        if not self.evidence_ids:
            raise PydanticCustomError(
                "claim_evidence_empty", "material-claim evidence_ids must be non-empty"
            )
        _require_unique(
            self.evidence_ids,
            error_type="claim_evidence_duplicate",
            label="material-claim evidence_ids",
        )
        return self


class GeneratedAnswer(FrozenStrictModel, frozen=True):
    """`internal.generated_answer.v1` before competition rendering."""

    schema_version: Literal["internal.generated_answer.v1"]
    question_id: NonEmptyString
    answer_text: NonEmptyString
    generator_id: GeneratorId
    competition_policy: Literal["baseline.v1", "verified.v1"]
    used_evidence_ids: tuple[NonEmptyString, ...]
    material_claims: tuple[MaterialClaim, ...]

    @model_validator(mode="after")
    def _validate_used_evidence(self) -> Self:
        _require_unique(
            self.used_evidence_ids,
            error_type="generated_evidence_duplicate",
            label="used_evidence_ids",
        )
        return self


class AnswerRecord(FrozenStrictModel, frozen=True):
    """`internal.answer.v1` competition answer record."""

    schema_version: Literal["internal.answer.v1"]
    question_id: NonEmptyString
    answer: NonEmptyString
    generator_id: GeneratorId
    evidence_ids: tuple[NonEmptyString, ...]
    run_id: RunId

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> Self:
        _require_unique(
            self.evidence_ids,
            error_type="answer_evidence_duplicate",
            label="evidence_ids",
        )
        return self


class RunManifest(FrozenStrictModel, frozen=True):
    """Exact deterministic `run.manifest.v1` schema.

    T06 owns canonical byte serialization and recomputation of ``run_id``.
    """

    schema_version: Literal["run.manifest.v1"]
    run_id: RunId
    pipeline_version: GeneratorId
    code_revision: CodeRevision
    source_tree_checksum: Sha256
    scoped_source_paths: tuple[SafeRelativePath, ...]
    config_checksum: Sha256
    question_checksum: Sha256
    corpus_checksum: Sha256
    index_checksum: Sha256 | None
    split_checksum: Sha256 | None
    model_id: NonEmptyString | None
    model_revision: NonEmptyString | None
    tokenizer_id: NonEmptyString | None
    tokenizer_revision: NonEmptyString | None
    prompt_revision: NonEmptyString | None
    seed: NonEmptyString
    execution_mode: Literal["prepare-online", "local-offline", "private-modal"]
    competition_policy: Literal["baseline.v1", "verified.v1"]
    comparison_type: Literal[
        "baseline",
        "generator_only",
        "retrieval_only",
        "joint_chunking_retrieval_generation",
        "bug_fix",
    ]
    resolved_as_of_date: NfcString | None
    as_of_timezone: Literal["Asia/Ho_Chi_Minh"] | None
    resource_manifest_checksum: Sha256
    evidence_diagnostics_checksum: Sha256
    answer_artifact_checksum: Sha256

    @model_validator(mode="after")
    def _validate_paths_and_date(self) -> Self:
        paths = self.scoped_source_paths
        if not paths:
            raise PydanticCustomError(
                "scoped_source_paths_empty", "scoped_source_paths must be non-empty"
            )
        if len(paths) != len(set(paths)) or paths != tuple(
            sorted(paths, key=lambda value: value.encode("utf-8"))
        ):
            raise PydanticCustomError(
                "scoped_source_paths_order",
                "scoped_source_paths must be unique and ordered by raw UTF-8 bytes",
            )
        resolved = self.resolved_as_of_date
        timezone = self.as_of_timezone
        if (resolved is None) != (timezone is None):
            raise PydanticCustomError(
                "resolved_date_pair", "resolved_as_of_date and as_of_timezone must form a pair"
            )
        if resolved is not None:
            try:
                parsed = date.fromisoformat(resolved)
            except ValueError as exc:
                raise PydanticCustomError(
                    "resolved_date_invalid", "resolved_as_of_date must be an ISO calendar date"
                ) from exc
            if len(resolved) != 10 or parsed.isoformat() != resolved:
                raise PydanticCustomError(
                    "resolved_date_invalid", "resolved_as_of_date must use YYYY-MM-DD"
                )
        return self


class OperationalTelemetry(FrozenStrictModel, frozen=True):
    """Minimal linkage envelope for excluded operational observations."""

    schema_version: Literal["operational.telemetry.v1"]
    run_id: RunId
    run_instance_id: UUID
    run_manifest_checksum: Sha256
