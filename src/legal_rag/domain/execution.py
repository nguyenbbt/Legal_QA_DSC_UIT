"""Closed execution-mode schemas and workload-free preflight policies."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Annotated, Literal, NoReturn, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from legal_rag.domain.models import (
    AbsoluteHttpUrl,
    FrozenStrictModel,
    GeneratorId,
    Sha256,
)

ArtifactClass = Literal[
    "fixture",
    "project_software",
    "public_model",
    "public_tokenizer",
    "non_sensitive_config",
    "fixture_output",
    "log",
    "billing",
    "checksummed_manifest",
    "organizer_question",
    "organizer_context",
    "organizer_answer",
    "derived_chunk",
    "embedding",
    "index",
    "cache",
]
TransferDirection = Literal["local-to-modal", "modal-to-local"]

_REAL_OR_DERIVED_CLASSES = frozenset(
    {
        "organizer_question",
        "organizer_context",
        "organizer_answer",
        "derived_chunk",
        "embedding",
        "index",
        "cache",
    }
)
_LOCAL_TO_MODAL_CLASSES = frozenset(
    {
        "fixture",
        "project_software",
        "public_model",
        "public_tokenizer",
        "non_sensitive_config",
    }
)
_MODAL_TO_LOCAL_CLASSES = frozenset({"fixture_output", "log", "billing", "checksummed_manifest"})
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_BEARER_VALUE = re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\s]+")


class ExecutionModeError(Exception):
    """Safe typed execution-policy failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise ExecutionModeError(code, message)


def _require_unique_ordered(values: tuple[str, ...], *, label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")
    if values != tuple(sorted(values, key=lambda item: item.encode("utf-8"))):
        raise ValueError(f"{label} must be ordered by raw UTF-8 bytes")


def _require_origin(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("configured endpoint must be an HTTPS origin")


class ArtifactTransfer(FrozenStrictModel, frozen=True):
    """One immutable checksummed transfer allowlist entry."""

    artifact_id: GeneratorId
    artifact_class: ArtifactClass
    direction: TransferDirection
    checksum: Sha256

    @model_validator(mode="after")
    def _validate_interim_direction(self) -> Self:
        if self.artifact_class in _REAL_OR_DERIVED_CLASSES:
            return self
        allowed = (
            self.direction == "local-to-modal" and self.artifact_class in _LOCAL_TO_MODAL_CLASSES
        ) or (self.direction == "modal-to-local" and self.artifact_class in _MODAL_TO_LOCAL_CLASSES)
        if not allowed:
            raise ValueError("artifact class is forbidden for the requested transfer direction")
        return self


class LoopbackDependency(FrozenStrictModel, frozen=True):
    """Manifested executable/model identity for allowed local loopback IPC."""

    dependency_id: GeneratorId
    executable_checksum: Sha256
    model_checksum: Sha256 | None


class PrepareOnlineConfig(FrozenStrictModel, frozen=True):
    """Resource-acquisition-only execution mode."""

    schema_version: Literal["execution.mode.v1"]
    mode: Literal["prepare-online"]
    accepted_origins: tuple[AbsoluteHttpUrl, ...]
    resource_manifest_updates_required: Literal[True]
    accepted_competition_answers_allowed: Literal[False]

    @model_validator(mode="after")
    def _validate_origins(self) -> Self:
        for origin in self.accepted_origins:
            _require_origin(origin)
        _require_unique_ordered(self.accepted_origins, label="accepted_origins")
        return self


class LocalOfflineConfig(FrozenStrictModel, frozen=True):
    """No-egress local execution over immutable manifested resources."""

    schema_version: Literal["execution.mode.v1"]
    mode: Literal["local-offline"]
    resource_manifest_checksum: Sha256
    required_resource_ids: tuple[GeneratorId, ...]
    outbound_network: Literal["deny-non-loopback"]
    loopback_dependencies: tuple[LoopbackDependency, ...]
    evaluation_wall_clock: Literal["forbidden"]

    @model_validator(mode="after")
    def _validate_resources(self) -> Self:
        _require_unique_ordered(self.required_resource_ids, label="required_resource_ids")
        dependency_ids = tuple(item.dependency_id for item in self.loopback_dependencies)
        if dependency_ids:
            _require_unique_ordered(dependency_ids, label="loopback dependency IDs")
        return self


class PrivateModalConfig(FrozenStrictModel, frozen=True):
    """Interim OQ-003-blocked private Modal preflight configuration."""

    schema_version: Literal["execution.mode.v1"]
    mode: Literal["private-modal"]
    control_plane_origin: AbsoluteHttpUrl
    control_plane_origin_allowlist: tuple[AbsoluteHttpUrl, ...]
    workload_egress_disabled: Literal[True]
    workload_egress_verified: bool
    private_storage_ids: tuple[GeneratorId, ...]
    required_resource_ids: tuple[GeneratorId, ...]
    transfer_allowlist: tuple[ArtifactTransfer, ...]
    real_data_approved: Literal[False]
    max_submission_retries: Literal[3]
    submission_backoff_seconds: tuple[Literal[1], Literal[2], Literal[4]]
    declared_job_identity: GeneratorId
    secret_policy: Literal["credential-store-only-redacted"]

    @model_validator(mode="after")
    def _validate_private_policy(self) -> Self:
        _require_origin(self.control_plane_origin)
        for origin in self.control_plane_origin_allowlist:
            _require_origin(origin)
        _require_unique_ordered(
            self.control_plane_origin_allowlist,
            label="control_plane_origin_allowlist",
        )
        if self.control_plane_origin not in self.control_plane_origin_allowlist:
            raise ValueError("control_plane_origin must be explicitly allowlisted")
        _require_unique_ordered(self.private_storage_ids, label="private_storage_ids")
        _require_unique_ordered(self.required_resource_ids, label="required_resource_ids")
        if not self.transfer_allowlist or len(self.transfer_allowlist) != len(
            set(self.transfer_allowlist)
        ):
            raise ValueError("transfer_allowlist must be non-empty and unique")
        if any(
            transfer.artifact_class in _REAL_OR_DERIVED_CLASSES
            for transfer in self.transfer_allowlist
        ):
            raise ValueError("real and derived data cannot enter the interim allowlist")
        return self


ExecutionModeConfig = Annotated[
    PrepareOnlineConfig | LocalOfflineConfig | PrivateModalConfig,
    Field(discriminator="mode"),
]


def _first_missing(required: tuple[str, ...], available: tuple[str, ...]) -> str | None:
    available_set = set(available)
    return next((resource_id for resource_id in required if resource_id not in available_set), None)


def preflight_execution[ConfigT: (PrepareOnlineConfig, LocalOfflineConfig, PrivateModalConfig)](
    config: ConfigT,
    *,
    available_resource_ids: tuple[str, ...],
    requested_transfers: tuple[ArtifactTransfer, ...] = (),
) -> ConfigT:
    """Validate only local policy state; this function cannot submit a workload."""

    if isinstance(config, PrepareOnlineConfig):
        if requested_transfers:
            _fail(
                "PREPARE_TRANSFER_UNSUPPORTED",
                "prepare-online does not accept run-artifact transfers",
            )
        return config

    missing = _first_missing(config.required_resource_ids, available_resource_ids)
    if isinstance(config, LocalOfflineConfig):
        if requested_transfers:
            _fail(
                "OFFLINE_TRANSFER_FORBIDDEN",
                "local-offline does not permit artifact transfer",
            )
        if missing is not None:
            _fail(
                "OFFLINE_RESOURCE_MISSING",
                f"required manifested resource is missing: {missing}",
            )
        return config

    if not config.workload_egress_verified:
        _fail("MODAL_EGRESS_UNVERIFIED", "private Modal no-egress setting is unverified")
    if missing is not None:
        _fail("MODAL_RESOURCE_MISSING", f"required manifested resource is missing: {missing}")
    for requested in requested_transfers:
        if requested.artifact_class in _REAL_OR_DERIVED_CLASSES:
            _fail(
                "MODAL_REAL_DATA_NOT_APPROVED",
                "real organizer and derived data transfer is not approved",
            )
        if requested not in config.transfer_allowlist:
            _fail(
                "MODAL_TRANSFER_NOT_ALLOWLISTED",
                "private Modal transfer is not exactly allowlisted",
            )
    return config


@dataclass(frozen=True, slots=True)
class ModalSubmissionState:
    """Pure single-job retry guard; it performs no sleep, network, or submission."""

    submission_attempts: int = 0
    job_id: str | None = None

    def record_submission_attempt(self) -> ModalSubmissionState:
        if self.job_id is not None:
            _fail("MODAL_JOB_ALREADY_DECLARED", "a job ID already exists; poll instead")
        if self.submission_attempts >= 4:
            _fail("MODAL_RETRY_LIMIT", "private Modal submission retry limit is exhausted")
        return replace(self, submission_attempts=self.submission_attempts + 1)

    def retry_delay_seconds(self) -> int:
        if self.job_id is not None:
            _fail("MODAL_JOB_ALREADY_DECLARED", "a job ID already exists; poll instead")
        if self.submission_attempts not in {1, 2, 3}:
            _fail("MODAL_RETRY_LIMIT", "no additional private Modal retry is allowed")
        return (1, 2, 4)[self.submission_attempts - 1]

    def record_job_id(self, job_id: str) -> ModalSubmissionState:
        if self.job_id is not None:
            _fail("MODAL_JOB_ALREADY_DECLARED", "a job ID is already declared")
        if self.submission_attempts == 0 or _JOB_ID.fullmatch(job_id) is None:
            _fail("MODAL_JOB_ID_INVALID", "job ID requires one preceding submission attempt")
        return replace(self, job_id=job_id)


def redact_sensitive_message(message: str, *, secret_values: tuple[str, ...]) -> str:
    """Redact configured secrets and bearer credentials from operational messages."""

    redacted = message
    for secret in sorted((value for value in secret_values if value), key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return _BEARER_VALUE.sub(r"\1[REDACTED]", redacted)
