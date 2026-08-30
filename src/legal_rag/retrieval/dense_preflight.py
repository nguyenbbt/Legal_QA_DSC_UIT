"""Deterministic runtime, storage, and checkpoint preflight for D-066 dense discovery."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from legal_rag.domain.checksums import content_json_bytes


@dataclass(frozen=True, slots=True)
class DenseCheckpointIdentity:
    namespace: str
    model_id: str
    model_revision: str
    corpus_checksum: str
    model_artifact_checksum: str
    tokenizer_checksum: str
    indexing_config_checksum: str
    chunk_count: int
    dimension: int


@dataclass(frozen=True, slots=True)
class DensePreflightInput:
    arm_id: str
    model_id: str
    model_revision: str
    corpus_checksum: str
    model_artifact_checksum: str
    tokenizer_checksum: str
    indexing_config_checksum: str
    chunk_count: int
    dimension: int
    bytes_per_component: int
    observed_documents: int
    observed_seconds_millis: int
    maximum_runtime_seconds: int
    available_storage_bytes: int
    required_free_storage_bytes: int

    def __post_init__(self) -> None:
        integer_values = (
            self.chunk_count,
            self.dimension,
            self.bytes_per_component,
            self.observed_documents,
            self.observed_seconds_millis,
            self.maximum_runtime_seconds,
            self.available_storage_bytes,
            self.required_free_storage_bytes,
        )
        if (
            not self.arm_id
            or not self.model_id
            or not self.model_revision
            or any(
                not checksum.startswith("sha256:") or len(checksum) != 71
                for checksum in (
                    self.corpus_checksum,
                    self.model_artifact_checksum,
                    self.tokenizer_checksum,
                    self.indexing_config_checksum,
                )
            )
            or any(value < 1 for value in integer_values)
        ):
            raise ValueError("dense preflight inputs must be complete and positive")


@dataclass(frozen=True, slots=True)
class DensePreflightReport:
    schema_version: str
    status: Literal["PASS", "BLOCKED"]
    blocker_codes: tuple[str, ...]
    resumable_namespace: str
    checkpoint_disposition: Literal["NEW", "RESUME_EXACT", "REJECT_STALE_PARTIAL"]
    projected_runtime_seconds: int
    projected_vector_bytes: int
    projected_required_storage_bytes: int
    observed_documents_per_second_milli: int


def _namespace(value: DensePreflightInput) -> str:
    payload = {
        "arm_id": value.arm_id,
        "model_id": value.model_id,
        "model_revision": value.model_revision,
        "corpus_checksum": value.corpus_checksum,
        "model_artifact_checksum": value.model_artifact_checksum,
        "tokenizer_checksum": value.tokenizer_checksum,
        "indexing_config_checksum": value.indexing_config_checksum,
        "chunk_count": value.chunk_count,
        "dimension": value.dimension,
        "bytes_per_component": value.bytes_per_component,
    }
    digest = hashlib.sha256(content_json_bytes(payload)).hexdigest()
    return f"d066-{value.arm_id.casefold()}-{digest[:16]}"


def evaluate_dense_preflight(
    value: DensePreflightInput,
    *,
    checkpoint: DenseCheckpointIdentity | None = None,
) -> DensePreflightReport:
    """Fail closed unless the measured full build and exact checkpoint fit OPS-002."""

    namespace = _namespace(value)
    projected_runtime = math.ceil(
        value.chunk_count * value.observed_seconds_millis / value.observed_documents / 1000.0
    )
    projected_vectors = value.chunk_count * value.dimension * value.bytes_per_component
    projected_storage = projected_vectors + value.required_free_storage_bytes
    blockers: list[str] = []
    if projected_runtime > value.maximum_runtime_seconds:
        blockers.append("OPS002_RUNTIME_LIMIT")
    if projected_storage > value.available_storage_bytes:
        blockers.append("OPS002_STORAGE_LIMIT")

    disposition: Literal["NEW", "RESUME_EXACT", "REJECT_STALE_PARTIAL"] = "NEW"
    if checkpoint is not None:
        expected = (
            namespace,
            value.model_id,
            value.model_revision,
            value.corpus_checksum,
            value.model_artifact_checksum,
            value.tokenizer_checksum,
            value.indexing_config_checksum,
            value.chunk_count,
            value.dimension,
        )
        actual = (
            checkpoint.namespace,
            checkpoint.model_id,
            checkpoint.model_revision,
            checkpoint.corpus_checksum,
            checkpoint.model_artifact_checksum,
            checkpoint.tokenizer_checksum,
            checkpoint.indexing_config_checksum,
            checkpoint.chunk_count,
            checkpoint.dimension,
        )
        if actual == expected:
            disposition = "RESUME_EXACT"
        else:
            disposition = "REJECT_STALE_PARTIAL"
            blockers.append("DENSE_CHECKPOINT_FINGERPRINT_MISMATCH")
    return DensePreflightReport(
        schema_version="retrieval.dense-preflight.v1",
        status="BLOCKED" if blockers else "PASS",
        blocker_codes=tuple(blockers),
        resumable_namespace=namespace,
        checkpoint_disposition=disposition,
        projected_runtime_seconds=projected_runtime,
        projected_vector_bytes=projected_vectors,
        projected_required_storage_bytes=projected_storage,
        observed_documents_per_second_milli=(
            value.observed_documents * 1_000_000 // value.observed_seconds_millis
        ),
    )


__all__ = [
    "DenseCheckpointIdentity",
    "DensePreflightInput",
    "DensePreflightReport",
    "evaluate_dense_preflight",
]
