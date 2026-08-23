"""Pure grounding assessment validation service.

Implements ``validate_grounding_assessments()`` per Section 8.2 of the guide.
Responsibilities: schema, closed-field, approval-state, benchmark identity,
question coverage, checksum binding, evidence existence/integrity,
duplicate detection, enum/range, atomic failure, deterministic output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.grounding_labels import (
    ApprovedGroundingBenchmark,
    GroundingLabelError,
    load_approved_grounding_benchmark,
)


class GroundingValidationError(Exception):
    """Stable failure at the grounding validation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class GroundingValidationReport:
    """Deterministic, metadata-only validation report."""

    schema_version: str
    benchmark_question_count: int
    annotation_status: str
    benchmark_checksum: str
    manifest_checksum: str
    label_version: str
    validation_result: Literal["valid", "invalid"]
    validation_errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_question_count": self.benchmark_question_count,
            "annotation_status": self.annotation_status,
            "benchmark_checksum": self.benchmark_checksum,
            "manifest_checksum": self.manifest_checksum,
            "label_version": self.label_version,
            "validation_result": self.validation_result,
            "validation_errors": list(self.validation_errors),
        }

    def json_bytes(self) -> bytes:
        """Render the report with the project canonical JSON contract."""

        return content_json_bytes(self.as_dict())


def validate_grounding_assessments(
    *,
    manifest_data: bytes,
    benchmark_data: bytes,
) -> GroundingValidationReport:
    """Validate grounding assessments and return a metadata-only report.

    This is a pure function — it does not write files or produce side effects.
    On validation failure, it raises ``GroundingValidationError`` with the
    specific failure code. No partial output is produced on failure.
    """
    manifest_checksum = checksum_bytes(manifest_data)
    benchmark_checksum = checksum_bytes(benchmark_data)

    try:
        approved: ApprovedGroundingBenchmark = load_approved_grounding_benchmark(
            manifest_data, benchmark_data
        )
    except GroundingLabelError as error:
        raise GroundingValidationError(error.code, error.message) from error

    return GroundingValidationReport(
        schema_version="grounding.validation.report.v1",
        benchmark_question_count=len(approved.records),
        annotation_status=approved.manifest.annotation_status,
        benchmark_checksum=benchmark_checksum,
        manifest_checksum=manifest_checksum,
        label_version=approved.manifest.label_version,
        validation_result="valid",
        validation_errors=(),
    )
