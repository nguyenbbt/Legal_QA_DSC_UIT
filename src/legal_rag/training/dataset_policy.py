"""Single fail-closed training dataset policy boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from legal_rag.training.provenance import (
    ProvenanceError,
    TrainingExample,
    parse_training_example,
    validate_example_provenance,
)


class DatasetPolicyError(Exception):
    """Stable safe failure at the training dataset policy boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class DatasetBuildReport:
    """Aggregate metadata emitted only for a fully accepted dataset."""

    candidate_rows: int
    accepted_rows: int
    rejected_rows: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    unique_question_ids: int
    unique_evidence_ids: int
    construction_version: str


TrainingRow = TrainingExample | Mapping[str, object]


def _translate(error: ProvenanceError) -> DatasetPolicyError:
    return DatasetPolicyError(error.code, error.message)


def validate_training_example(example: TrainingExample) -> None:
    """Validate one already typed internal example."""
    try:
        validate_example_provenance(example)
    except ProvenanceError as error:
        raise _translate(error) from error


def _parse_row(row: TrainingRow) -> TrainingExample:
    if isinstance(row, TrainingExample):
        validate_training_example(row)
        return row
    try:
        return parse_training_example(row)
    except ProvenanceError as error:
        raise _translate(error) from error


def validate_training_dataset(
    rows: tuple[TrainingRow, ...],
) -> DatasetBuildReport:
    """Validate the whole artifact; any rejected row rejects the complete build."""
    if not rows:
        raise DatasetPolicyError("DATASET_EMPTY", "training dataset contains no rows")

    examples = tuple(_parse_row(row) for row in rows)
    versions = {example.construction_version for example in examples}
    if len(versions) != 1:
        raise DatasetPolicyError(
            "DATASET_CONSTRUCTION_VERSION_MISMATCH",
            "training examples use different construction versions",
        )
    question_ids = {example.question_id for example in examples}
    evidence_ids = {evidence_id for example in examples for evidence_id in example.evidence_ids}
    return DatasetBuildReport(
        candidate_rows=len(rows),
        accepted_rows=len(examples),
        rejected_rows=0,
        rejected_by_reason=(),
        unique_question_ids=len(question_ids),
        unique_evidence_ids=len(evidence_ids),
        construction_version=examples[0].construction_version,
    )


__all__ = [
    "DatasetBuildReport",
    "DatasetPolicyError",
    "validate_training_dataset",
    "validate_training_example",
]
