"""Closed whole-system model manifest and exact competition parameter gate."""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from legal_rag.domain.checksums import canonical_json_bytes
from legal_rag.domain.models import (
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    Sha256,
)

COMPETITION_PARAMETER_LIMIT = 4_000_000_000
ModelRole = Literal["embedding", "generator", "reranker"]


class ModelManifestError(Exception):
    """Stable safe failure at the model-governance boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelComponentManifest(FrozenStrictModel, frozen=True):
    """One enabled learned component in a whole-system manifest."""

    role: ModelRole
    model_id: NonEmptyString
    model_revision: NonEmptyString | None
    tokenizer_id: NonEmptyString
    tokenizer_revision: NonEmptyString | None
    license: NonEmptyString | None
    exact_parameter_count: NonNegativeInt
    trainable_parameter_count: NonNegativeInt
    adapter_parameter_count: NonNegativeInt
    quantization: NonEmptyString | None
    parameter_audit_checksum: Sha256 | None
    btc_approval_state: Literal["approved", "pending", "rejected"]
    btc_approval_evidence: NonEmptyString | None
    local_model_hash: Sha256 | None
    local_tokenizer_hash: Sha256 | None

    @model_validator(mode="after")
    def _validate_trainable_count(self) -> Self:
        available = self.exact_parameter_count + self.adapter_parameter_count
        if self.trainable_parameter_count > available:
            raise PydanticCustomError(
                "model_trainable_parameter_count",
                "trainable parameter count cannot exceed base plus adapter count",
            )
        return self


class ModelParameterManifest(FrozenStrictModel, frozen=True):
    """`model.parameter_manifest.v1` whole-system governance artifact."""

    schema_version: Literal["model.parameter_manifest.v1"]
    models: tuple[ModelComponentManifest, ...]
    system_parameter_count: NonNegativeInt
    competition_limit_exclusive: Literal[4_000_000_000]
    passes_parameter_gate: bool

    @model_validator(mode="after")
    def _validate_system_accounting(self) -> Self:
        if not self.models:
            raise PydanticCustomError(
                "model_manifest_empty", "models must contain at least one component"
            )

        keys = tuple((model.role, model.model_id) for model in self.models)
        expected_order = tuple(sorted(keys, key=lambda key: (key[0].encode(), key[1].encode())))
        if keys != expected_order:
            raise PydanticCustomError(
                "model_manifest_order",
                "models must be ordered by role and model ID UTF-8 bytes",
            )
        roles = tuple(model.role for model in self.models)
        if len(roles) != len(set(roles)):
            raise PydanticCustomError(
                "model_manifest_role_duplicate", "enabled model roles must be unique"
            )

        exact_total = compute_system_parameter_total(self.models)
        if self.system_parameter_count != exact_total:
            raise PydanticCustomError(
                "model_parameter_total_mismatch",
                "system_parameter_count does not equal exact component total",
            )
        expected_result = exact_total < self.competition_limit_exclusive
        if self.passes_parameter_gate is not expected_result:
            raise PydanticCustomError(
                "model_parameter_gate_mismatch",
                "passes_parameter_gate does not match the exclusive limit",
            )
        return self


class ModelRunFingerprintInputs(FrozenStrictModel, frozen=True):
    """Checksums that every future model-backed run identity must bind."""

    schema_version: Literal["model.run_fingerprint_inputs.v1"]
    model_parameter_manifest_checksum: Sha256
    model_hashes: tuple[Sha256, ...]
    tokenizer_hashes: tuple[Sha256, ...]
    adapter_hashes: tuple[Sha256, ...]
    prompt_checksum: Sha256 | None
    training_recipe_checksum: Sha256 | None

    @model_validator(mode="after")
    def _validate_required_assets(self) -> Self:
        if not self.model_hashes or not self.tokenizer_hashes:
            raise PydanticCustomError(
                "model_run_inputs_missing",
                "model and tokenizer hashes must be non-empty",
            )
        if len(self.model_hashes) != len(self.tokenizer_hashes):
            raise PydanticCustomError(
                "model_run_inputs_cardinality",
                "model and tokenizer hash counts must match",
            )
        return self


def compute_system_parameter_total(
    models: tuple[ModelComponentManifest, ...],
) -> int:
    """Count every enabled base and separately learned adapter parameter."""
    return sum(model.exact_parameter_count + model.adapter_parameter_count for model in models)


def validate_parameter_limit(total_system_parameters: int) -> None:
    """Enforce the strict, exclusive whole-system competition limit."""
    if total_system_parameters >= COMPETITION_PARAMETER_LIMIT:
        raise ModelManifestError(
            "MODEL_PARAMETER_LIMIT",
            "whole-system learned parameter count reaches the exclusive limit",
        )


def validate_system_parameters(models: tuple[ModelComponentManifest, ...]) -> int:
    """Compute and validate a component set without constructing an artifact."""
    total = compute_system_parameter_total(models)
    validate_parameter_limit(total)
    return total


def _model_checksum(model: FrozenStrictModel) -> str:
    payload = model.model_dump(mode="json")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def compute_manifest_checksum(manifest: ModelParameterManifest) -> str:
    """Return the canonical whole-system manifest checksum."""
    return _model_checksum(manifest)


def compute_model_run_inputs_checksum(inputs: ModelRunFingerprintInputs) -> str:
    """Return the material fingerprint input for a future model-backed run."""
    return _model_checksum(inputs)


__all__ = [
    "COMPETITION_PARAMETER_LIMIT",
    "ModelComponentManifest",
    "ModelManifestError",
    "ModelParameterManifest",
    "ModelRunFingerprintInputs",
    "compute_manifest_checksum",
    "compute_model_run_inputs_checksum",
    "compute_system_parameter_total",
    "validate_parameter_limit",
    "validate_system_parameters",
]
