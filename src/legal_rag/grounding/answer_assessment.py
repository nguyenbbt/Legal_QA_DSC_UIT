"""Strict import and validation for answer-level grounding assessments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import (
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.grounding_labels import (
    GroundingFile,
    GroundingLabelError,
    load_approved_grounding_benchmark,
)

AnswerSupport = Literal[
    "fully_supported",
    "partially_supported",
    "unsupported",
    "not_answerable",
]
_MUTABLE_FIELDS = frozenset(
    {
        "answer_support",
        "unsupported_claim_count",
        "annotator_id",
        "adjudicator_id",
        "adjudication_state",
    }
)


class AnswerAssessmentError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AnswerAssessmentEvidence(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    display_text: NonEmptyString


class AnswerAssessmentWorkItem(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.answer-assessment.work-item.v1"]
    question_id: NonEmptyString
    evaluated_run_id: NonEmptyString
    answer_checksum: Sha256
    question: NonEmptyString
    gold_answer: NonEmptyString
    generated_answer: NonEmptyString
    evidence: tuple[AnswerAssessmentEvidence, ...]
    answer_support: AnswerSupport | None
    unsupported_claim_count: NonNegativeInt | None
    annotator_id: NonEmptyString | None
    adjudicator_id: NonEmptyString | None
    adjudication_state: Literal["pending", "approved", "rejected"]
    label_version: Literal["grounding.v1"]

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("assessment evidence IDs must be non-empty and unique")
        if checksum_bytes(self.generated_answer.encode()) != self.answer_checksum:
            raise ValueError("assessment answer checksum does not match generated answer")
        labels = (self.answer_support, self.unsupported_claim_count, self.annotator_id)
        if self.adjudication_state == "pending":
            if any(value is not None for value in (*labels, self.adjudicator_id)):
                raise ValueError("pending assessment fields must remain empty")
            return self
        if any(value is None for value in labels):
            raise ValueError("reviewed assessment fields must be complete")
        if self.answer_support != "fully_supported" and self.adjudicator_id is None:
            raise ValueError("non-fully-supported assessments require adjudication")
        if self.answer_support in {"fully_supported", "not_answerable"}:
            if self.unsupported_claim_count != 0:
                raise ValueError("supported/not-answerable rows cannot have unsupported claims")
        elif self.unsupported_claim_count == 0:
            raise ValueError("partial/unsupported rows require an unsupported claim")
        return self


@dataclass(frozen=True, slots=True)
class ImportedAnswerAssessments:
    evaluated_run_id: str
    records: tuple[AnswerAssessmentWorkItem, ...]
    source_labeled_checksum: str


class GroundingAssessmentRecord(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.assessment.v1"]
    question_id: NonEmptyString
    evaluated_run_id: NonEmptyString
    answer_checksum: Sha256
    answer_support: AnswerSupport
    unsupported_claim_count: NonNegativeInt
    annotator_id: NonEmptyString
    adjudicator_id: NonEmptyString | None
    adjudication_state: Literal["approved"]
    label_version: Literal["grounding.v1"]


class GroundingAssessmentManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.assessment.manifest.v1"]
    label_version: Literal["grounding.v1"]
    train_split_checksum: Sha256
    development_split_checksum: Sha256
    sampling_version: Literal["grounding-sample.v1"]
    sampling_seed: Literal["dsc2026-grounding-sample-v1"]
    ordered_question_ids: tuple[NonEmptyString, ...]
    chunk_artifact_checksum: Sha256
    index_checksum: Sha256
    benchmark_manifest_checksum: Sha256
    benchmark_checksum: Sha256
    evaluated_run_id: NonEmptyString
    source_work_queue_checksum: Sha256
    source_labeled_checksum: Sha256
    annotation_status: Literal["approved", "rejected"]
    ordered_files: tuple[GroundingFile, ...]

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if (
            len(self.ordered_question_ids) != 60
            or len(set(self.ordered_question_ids)) != 60
            or len(self.ordered_files) != 1
        ):
            raise ValueError("assessment manifest requires 60 unique IDs and one file")
        return self


@dataclass(frozen=True, slots=True)
class GroundingRates:
    unsupported_answer_rate: float
    fully_supported_rate: float
    partially_supported_rate: float
    not_answerable_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "unsupported_answer_rate": self.unsupported_answer_rate,
            "fully_supported_rate": self.fully_supported_rate,
            "partially_supported_rate": self.partially_supported_rate,
            "not_answerable_rate": self.not_answerable_rate,
        }


@dataclass(frozen=True, slots=True)
class ExportedAnswerAssessments:
    assessment_data: bytes
    manifest_data: bytes
    report_data: bytes


@dataclass(frozen=True, slots=True)
class ApprovedAnswerAssessments:
    manifest: GroundingAssessmentManifest
    records: tuple[GroundingAssessmentRecord, ...]
    rates: GroundingRates


def _schema_error(error: RecordValidationError) -> AnswerAssessmentError:
    message = error.issues[0].message if error.issues else "assessment schema is invalid"
    return AnswerAssessmentError("GROUNDING_ASSESSMENT_SCHEMA_INVALID", message)


def _load_rows(data: bytes, *, artifact_name: str) -> tuple[AnswerAssessmentWorkItem, ...]:
    rows: list[AnswerAssessmentWorkItem] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        if not line.strip():
            raise AnswerAssessmentError(
                "GROUNDING_ASSESSMENT_SCHEMA_INVALID",
                "answer assessment contains an empty record",
            )
        try:
            rows.append(
                parse_record_json(
                    line,
                    AnswerAssessmentWorkItem,
                    artifact_path=artifact_name,
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            raise _schema_error(error) from error
    question_ids = tuple(row.question_id for row in rows)
    if len(rows) != 60 or len(set(question_ids)) != 60:
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_COVERAGE_INVALID",
            "answer assessment requires exactly 60 unique questions",
        )
    if len({row.evaluated_run_id for row in rows}) != 1:
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_RUN_MISMATCH",
            "answer assessment rows must identify one run",
        )
    return tuple(rows)


def _identity(item: AnswerAssessmentWorkItem) -> dict[str, object]:
    return {
        key: value
        for key, value in item.model_dump(mode="json").items()
        if key not in _MUTABLE_FIELDS
    }


def import_labeled_answer_assessment_queue(
    queue_data: bytes,
    labeled_data: bytes,
) -> ImportedAnswerAssessments:
    """Bind approved labels to every immutable byte-addressed work-item field."""

    queue = _load_rows(queue_data, artifact_name="answer-grounding-work-queue.v1.jsonl")
    labeled = _load_rows(labeled_data, artifact_name="answer-grounding-work-queue.v1.labeled.jsonl")
    if tuple(_identity(item) for item in queue) != tuple(_identity(item) for item in labeled):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_QUEUE_MISMATCH",
            "labeled answer assessment differs from the frozen work queue",
        )
    if any(item.adjudication_state != "approved" for item in labeled):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_APPROVAL_MISSING",
            "all answer assessments must be approved",
        )
    fully_supported = tuple(item for item in labeled if item.answer_support == "fully_supported")
    adjudication_count = math.ceil(len(fully_supported) * 0.2)
    required_adjudication = sorted(
        fully_supported,
        key=lambda item: (
            hashlib.sha256(f"dsc2026-grounding-sample-v1\n{item.question_id}".encode()).digest(),
            item.question_id.encode(),
        ),
    )[:adjudication_count]
    if any(item.adjudicator_id is None for item in required_adjudication):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_ADJUDICATION_INCOMPLETE",
            "the smallest-digest 20% fully-supported sample requires adjudication",
        )
    return ImportedAnswerAssessments(
        evaluated_run_id=labeled[0].evaluated_run_id,
        records=labeled,
        source_labeled_checksum=checksum_bytes(labeled_data),
    )


def _rates(records: tuple[GroundingAssessmentRecord, ...]) -> GroundingRates:
    denominator = len(records)
    counts = {
        label: sum(record.answer_support == label for record in records)
        for label in (
            "unsupported",
            "fully_supported",
            "partially_supported",
            "not_answerable",
        )
    }
    return GroundingRates(
        unsupported_answer_rate=counts["unsupported"] / denominator,
        fully_supported_rate=counts["fully_supported"] / denominator,
        partially_supported_rate=counts["partially_supported"] / denominator,
        not_answerable_rate=counts["not_answerable"] / denominator,
    )


def _assessment_records(
    imported: ImportedAnswerAssessments,
) -> tuple[GroundingAssessmentRecord, ...]:
    return tuple(
        GroundingAssessmentRecord(
            schema_version="grounding.assessment.v1",
            question_id=item.question_id,
            evaluated_run_id=item.evaluated_run_id,
            answer_checksum=item.answer_checksum,
            answer_support=item.answer_support,
            unsupported_claim_count=item.unsupported_claim_count,
            annotator_id=item.annotator_id,
            adjudicator_id=item.adjudicator_id,
            adjudication_state="approved",
            label_version="grounding.v1",
        )
        for item in imported.records
        if item.answer_support is not None
        and item.unsupported_claim_count is not None
        and item.annotator_id is not None
    )


def export_answer_assessments(
    imported: ImportedAnswerAssessments,
    *,
    queue_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
    assessment_path: str,
) -> ExportedAnswerAssessments:
    """Export minimal approved assessment rows plus a benchmark-bound manifest."""

    try:
        benchmark = load_approved_grounding_benchmark(benchmark_manifest_data, benchmark_data)
    except GroundingLabelError as error:
        raise AnswerAssessmentError(error.code, error.message) from error
    records = _assessment_records(imported)
    record_ids = tuple(record.question_id for record in records)
    if record_ids != benchmark.manifest.ordered_question_ids:
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_BENCHMARK_MISMATCH",
            "assessment IDs differ from the approved grounding benchmark",
        )
    assessment_data = b"".join(
        content_json_bytes(record.model_dump(mode="json")) for record in records
    )
    manifest = GroundingAssessmentManifest(
        schema_version="grounding.assessment.manifest.v1",
        label_version="grounding.v1",
        train_split_checksum=benchmark.manifest.train_split_checksum,
        development_split_checksum=benchmark.manifest.development_split_checksum,
        sampling_version=benchmark.manifest.sampling_version,
        sampling_seed=benchmark.manifest.sampling_seed,
        ordered_question_ids=record_ids,
        chunk_artifact_checksum=benchmark.manifest.chunk_artifact_checksum,
        index_checksum=benchmark.manifest.index_checksum,
        benchmark_manifest_checksum=checksum_bytes(benchmark_manifest_data),
        benchmark_checksum=checksum_bytes(benchmark_data),
        evaluated_run_id=imported.evaluated_run_id,
        source_work_queue_checksum=checksum_bytes(queue_data),
        source_labeled_checksum=imported.source_labeled_checksum,
        annotation_status="approved",
        ordered_files=(
            GroundingFile(path=assessment_path, checksum=checksum_bytes(assessment_data)),
        ),
    )
    rates = _rates(records)
    report_data = content_json_bytes(
        {
            "schema_version": "grounding.assessment.report.v1",
            "evaluated_run_id": imported.evaluated_run_id,
            "annotation_status": "approved",
            "question_count": len(records),
            "assessment_checksum": checksum_bytes(assessment_data),
            "benchmark_checksum": checksum_bytes(benchmark_data),
            "rates": rates.as_dict(),
        }
    )
    return ExportedAnswerAssessments(
        assessment_data=assessment_data,
        manifest_data=content_json_bytes(manifest.model_dump(mode="json")),
        report_data=report_data,
    )


def _load_manifest(data: bytes) -> GroundingAssessmentManifest:
    try:
        return parse_record_json(
            data,
            GroundingAssessmentManifest,
            artifact_path="grounding.assessment.manifest.v1.json",
            record_identity="manifest",
        )
    except RecordValidationError as error:
        raise _schema_error(error) from error


def _load_assessment_records(
    data: bytes, *, artifact_path: str
) -> tuple[GroundingAssessmentRecord, ...]:
    records: list[GroundingAssessmentRecord] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        try:
            records.append(
                parse_record_json(
                    line,
                    GroundingAssessmentRecord,
                    artifact_path=artifact_path,
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            raise _schema_error(error) from error
    return tuple(records)


def load_approved_answer_assessments(
    *,
    manifest_data: bytes,
    assessment_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
) -> ApprovedAnswerAssessments:
    """Load answer grounding only after every approval and checksum identity passes."""

    manifest = _load_manifest(manifest_data)
    if manifest.annotation_status != "approved":
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_APPROVAL_MISSING",
            "answer grounding requires an approved manifest",
        )
    if manifest.benchmark_manifest_checksum != checksum_bytes(
        benchmark_manifest_data
    ) or manifest.benchmark_checksum != checksum_bytes(benchmark_data):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_BENCHMARK_MISMATCH",
            "answer grounding references another benchmark",
        )
    try:
        benchmark = load_approved_grounding_benchmark(benchmark_manifest_data, benchmark_data)
    except GroundingLabelError as error:
        raise AnswerAssessmentError(error.code, error.message) from error
    if manifest.ordered_files[0].checksum != checksum_bytes(assessment_data):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_CHECKSUM_MISMATCH",
            "assessment checksum differs from its approved manifest",
        )
    records = _load_assessment_records(
        assessment_data, artifact_path=manifest.ordered_files[0].path
    )
    record_ids = tuple(record.question_id for record in records)
    if (
        record_ids != manifest.ordered_question_ids
        or record_ids != benchmark.manifest.ordered_question_ids
        or any(record.evaluated_run_id != manifest.evaluated_run_id for record in records)
    ):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_ID_MISMATCH",
            "assessment identity or order differs from its manifest",
        )
    return ApprovedAnswerAssessments(manifest, records, _rates(records))


def compare_answer_grounding(
    baseline: ApprovedAnswerAssessments,
    candidate: ApprovedAnswerAssessments,
) -> bytes:
    """Apply the exact unsupported/fully-supported EVAL-005 grounding guard."""

    if (
        baseline.manifest.benchmark_manifest_checksum
        != candidate.manifest.benchmark_manifest_checksum
        or baseline.manifest.benchmark_checksum != candidate.manifest.benchmark_checksum
        or baseline.manifest.ordered_question_ids != candidate.manifest.ordered_question_ids
    ):
        raise AnswerAssessmentError(
            "GROUNDING_ASSESSMENT_BENCHMARK_MISMATCH",
            "baseline and candidate grounding use different benchmark identities",
        )
    unsupported_delta = (
        candidate.rates.unsupported_answer_rate - baseline.rates.unsupported_answer_rate
    )
    fully_supported_delta = (
        candidate.rates.fully_supported_rate - baseline.rates.fully_supported_rate
    )
    blockers: list[str] = []
    if unsupported_delta > 0:
        blockers.append("UNSUPPORTED_ANSWER_RATE_REGRESSION")
    if fully_supported_delta < -0.02:
        blockers.append("FULLY_SUPPORTED_RATE_REGRESSION_EXCEEDS_0_02")
    return content_json_bytes(
        {
            "schema_version": "grounding.assessment.comparison.v1",
            "baseline_run_id": baseline.manifest.evaluated_run_id,
            "candidate_run_id": candidate.manifest.evaluated_run_id,
            "benchmark_manifest_checksum": baseline.manifest.benchmark_manifest_checksum,
            "benchmark_checksum": baseline.manifest.benchmark_checksum,
            "question_count": len(baseline.records),
            "rates": {
                "baseline_unsupported_answer_rate": baseline.rates.unsupported_answer_rate,
                "candidate_unsupported_answer_rate": candidate.rates.unsupported_answer_rate,
                "unsupported_answer_rate_delta": unsupported_delta,
                "baseline_fully_supported_rate": baseline.rates.fully_supported_rate,
                "candidate_fully_supported_rate": candidate.rates.fully_supported_rate,
                "fully_supported_delta": fully_supported_delta,
            },
            "grounding_gate": "failed" if blockers else "passed",
            "promotion_blockers": blockers,
        }
    )


__all__ = [
    "AnswerAssessmentError",
    "AnswerAssessmentEvidence",
    "AnswerAssessmentWorkItem",
    "AnswerSupport",
    "ApprovedAnswerAssessments",
    "ExportedAnswerAssessments",
    "GroundingAssessmentManifest",
    "GroundingAssessmentRecord",
    "GroundingRates",
    "ImportedAnswerAssessments",
    "compare_answer_grounding",
    "export_answer_assessments",
    "import_labeled_answer_assessment_queue",
    "load_approved_answer_assessments",
]
