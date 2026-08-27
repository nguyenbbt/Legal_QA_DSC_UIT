"""Fail-closed official-profile and acquisition preflight checks."""

from __future__ import annotations

from typing import Literal

from legal_rag.models.manifest import (
    ModelComponentManifest,
    ModelManifestError,
    ModelParameterManifest,
    validate_parameter_limit,
)

ExecutionMode = Literal["prepare-online", "local-offline", "private-modal"]


def _fail(code: str, model: ModelComponentManifest, reason: str) -> None:
    raise ModelManifestError(code, f"model {model.model_id} {reason}")


def validate_model_revision(model: ModelComponentManifest) -> None:
    """Require pinned model/tokenizer revisions and exact local artifact hashes."""
    missing_pin = (
        model.model_revision is None
        or model.tokenizer_revision is None
        or model.local_model_hash is None
        or model.local_tokenizer_hash is None
    )
    if missing_pin:
        _fail("MODEL_REVISION_UNPINNED", model, "has an unpinned revision or artifact")


def validate_license(model: ModelComponentManifest) -> None:
    """Require a declared license."""
    if model.license is None:
        _fail("MODEL_LICENSE_MISSING", model, "has no declared license")


def validate_parameter_audit(model: ModelComponentManifest) -> None:
    """Require exact positive base numel and its immutable audit checksum."""
    if model.exact_parameter_count <= 0 or model.parameter_audit_checksum is None:
        _fail("MODEL_PARAMETER_AUDIT_MISSING", model, "has no exact parameter audit")


def validate_competition_registration(model: ModelComponentManifest) -> None:
    """Require registration evidence for the exact final competition checkpoint."""
    if model.btc_approval_state != "approved" or model.btc_approval_evidence is None:
        _fail(
            "MODEL_COMPETITION_REGISTRATION_MISSING",
            model,
            "has no competition registration evidence",
        )


def validate_btc_approval(model: ModelComponentManifest) -> None:
    """Compatibility alias for v1 callers; registration is now a final gate."""
    validate_competition_registration(model)


def validate_model_governance(model: ModelComponentManifest) -> None:
    """Validate one component for bounded experimentation or fitting."""
    validate_model_revision(model)
    validate_license(model)
    validate_parameter_audit(model)


def validate_experiment_profile(manifest: ModelParameterManifest) -> None:
    """Accept an unregistered profile only after every technical audit passes."""
    validators = (validate_model_revision, validate_license, validate_parameter_audit)
    for validator in validators:
        for model in manifest.models:
            validator(model)
    validate_parameter_limit(manifest.system_parameter_count)


def validate_official_profile(manifest: ModelParameterManifest) -> None:
    """Fail before inference unless every component and the system are approved."""
    validators = (
        validate_model_revision,
        validate_license,
        validate_parameter_audit,
        validate_competition_registration,
    )
    for validator in validators:
        for model in manifest.models:
            validator(model)
    validate_parameter_limit(manifest.system_parameter_count)


def validate_acquisition_mode(execution_mode: ExecutionMode) -> None:
    """Permit model acquisition only during an explicit online preparation step."""
    if execution_mode != "prepare-online":
        raise ModelManifestError(
            "MODEL_ACQUISITION_MODE_INVALID",
            "model acquisition requires prepare-online execution mode",
        )


__all__ = [
    "ModelManifestError",
    "validate_acquisition_mode",
    "validate_btc_approval",
    "validate_competition_registration",
    "validate_experiment_profile",
    "validate_license",
    "validate_model_governance",
    "validate_model_revision",
    "validate_official_profile",
    "validate_parameter_audit",
]
