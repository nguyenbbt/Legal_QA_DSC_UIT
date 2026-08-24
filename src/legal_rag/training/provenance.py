"""Closed official-data-only `training.example.v1` provenance boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import ValidationError, model_validator
from pydantic_core import PydanticCustomError

from legal_rag.domain.models import FrozenStrictModel, NonEmptyString, Sha256

ALLOWED_TARGET_SOURCES = frozenset({"official_train_answer", "deterministic_relevance"})


class ProvenanceError(Exception):
    """Stable safe failure at the untrusted training-provenance boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class TrainingExample(FrozenStrictModel, frozen=True):
    """One official train-derived fitting example with immutable provenance."""

    schema_version: Literal["training.example.v1"]
    example_id: NonEmptyString
    task: Literal["embedding", "reranking", "generation"]
    question_id: NonEmptyString
    split: Literal["train"]
    question_source_checksum: Sha256
    evidence_ids: tuple[NonEmptyString, ...]
    target_source: Literal["official_train_answer", "deterministic_relevance"]
    target_checksum: Sha256
    contains_generated_text: Literal[False]
    construction_version: NonEmptyString

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        if not self.evidence_ids:
            raise PydanticCustomError("training_evidence_missing", "evidence_ids must be non-empty")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise PydanticCustomError("training_evidence_duplicate", "evidence_ids must be unique")
        return self


def _reject_known_policy_violations(payload: Mapping[str, object]) -> None:
    split = payload.get("split")
    if split is not None and split != "train":
        raise ProvenanceError(
            "PROVENANCE_SPLIT_REJECTED", "training example does not use train split"
        )
    generated = payload.get("contains_generated_text")
    if "contains_generated_text" in payload and generated is not False:
        raise ProvenanceError(
            "PROVENANCE_GENERATED_TEXT", "training example contains generated text"
        )
    target_source = payload.get("target_source")
    if target_source is not None and target_source not in ALLOWED_TARGET_SOURCES:
        raise ProvenanceError(
            "PROVENANCE_TARGET_SOURCE_REJECTED",
            "training example uses a forbidden target source",
        )
    task = payload.get("task")
    valid_task_target = (task == "generation" and target_source == "official_train_answer") or (
        task in {"embedding", "reranking"} and target_source == "deterministic_relevance"
    )
    if task is not None and target_source is not None and not valid_task_target:
        raise ProvenanceError(
            "PROVENANCE_TARGET_SOURCE_REJECTED",
            "training task and target source are inconsistent",
        )
    evidence_ids = payload.get("evidence_ids")
    if isinstance(evidence_ids, (list, tuple)) and not evidence_ids:
        raise ProvenanceError("PROVENANCE_EVIDENCE_MISSING", "training example has no evidence IDs")


def parse_training_example(payload: Mapping[str, object]) -> TrainingExample:
    """Parse an untrusted JSON-shaped row and translate failures to stable codes."""
    _reject_known_policy_violations(payload)
    prepared = dict(payload)
    evidence_ids = prepared.get("evidence_ids")
    if isinstance(evidence_ids, list):
        prepared["evidence_ids"] = tuple(evidence_ids)
    try:
        return TrainingExample.model_validate(prepared)
    except ValidationError as error:
        raise ProvenanceError(
            "PROVENANCE_SCHEMA_INVALID", "training example schema is invalid"
        ) from error


def validate_example_provenance(example: TrainingExample) -> None:
    """Recheck semantic policy at typed internal boundaries."""
    valid_task_target = (
        example.task == "generation" and example.target_source == "official_train_answer"
    ) or (
        example.task in {"embedding", "reranking"}
        and example.target_source == "deterministic_relevance"
    )
    if not valid_task_target:
        raise ProvenanceError(
            "PROVENANCE_TARGET_SOURCE_REJECTED",
            "training task and target source are inconsistent",
        )
    if not example.evidence_ids:
        raise ProvenanceError("PROVENANCE_EVIDENCE_MISSING", "training example has no evidence IDs")


__all__ = [
    "ALLOWED_TARGET_SOURCES",
    "ProvenanceError",
    "TrainingExample",
    "parse_training_example",
    "validate_example_provenance",
]
