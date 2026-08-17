"""Strict owner-approved grounding benchmark and retrieval-label contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import (
    FrozenStrictModel,
    NfcString,
    NonEmptyString,
    SafeRelativePath,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.retrieval_metrics import RetrievalLabelRow


class GroundingLabelError(Exception):
    """Stable failure at the private grounding-label boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EvidenceRelevance(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    relevance: Literal["relevant", "partially_relevant", "not_relevant"]


class GroundingBenchmarkRecord(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.benchmark.v1"]
    question_id: NonEmptyString
    split: Literal["development"]
    question_checksum: Sha256
    relevant_evidence: tuple[EvidenceRelevance, ...]
    required_claims: tuple[NfcString, ...]
    question_answerability: Literal["answerable", "not_answerable", "unknown"]
    temporal_assessment: Literal["valid", "invalid", "unknown", "not_applicable"]
    label_version: Literal["grounding.v1"]

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.relevant_evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("relevant_evidence IDs must be unique")
        return self


class GroundingFile(FrozenStrictModel, frozen=True):
    path: SafeRelativePath
    checksum: Sha256


class GroundingBenchmarkManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.benchmark.manifest.v1"]
    label_version: Literal["grounding.v1"]
    train_split_checksum: Sha256
    development_split_checksum: Sha256
    sampling_version: Literal["grounding-sample.v1"]
    sampling_seed: Literal["dsc2026-grounding-sample-v1"]
    ordered_question_ids: tuple[NonEmptyString, ...]
    chunk_artifact_checksum: Sha256
    index_checksum: Sha256
    annotation_status: Literal["draft", "approved", "rejected"]
    ordered_files: tuple[GroundingFile, ...]

    @model_validator(mode="after")
    def _validate_complete_manifest(self) -> Self:
        if len(self.ordered_question_ids) != 60:
            raise ValueError("grounding benchmark must contain exactly 60 question IDs")
        if len(set(self.ordered_question_ids)) != 60:
            raise ValueError("grounding benchmark question IDs must be unique")
        if len(self.ordered_files) != 1:
            raise ValueError("grounding benchmark manifest requires exactly one label file")
        return self


@dataclass(frozen=True, slots=True)
class ApprovedGroundingBenchmark:
    manifest: GroundingBenchmarkManifest
    records: tuple[GroundingBenchmarkRecord, ...]
    retrieval_labels: tuple[RetrievalLabelRow, ...]


def _schema_error(error: RecordValidationError) -> GroundingLabelError:
    message = error.issues[0].message if error.issues else "grounding label schema is invalid"
    return GroundingLabelError("GROUNDING_LABEL_SCHEMA_INVALID", message)


def load_approved_grounding_benchmark(
    manifest_data: bytes,
    benchmark_data: bytes,
) -> ApprovedGroundingBenchmark:
    """Load labels only when the exact manifest is complete and owner-approved."""

    try:
        manifest = parse_record_json(
            manifest_data,
            GroundingBenchmarkManifest,
            artifact_path="grounding.benchmark.manifest.json",
            record_identity="manifest",
        )
    except RecordValidationError as error:
        raise _schema_error(error) from error
    if manifest.annotation_status != "approved":
        raise GroundingLabelError(
            "GROUNDING_LABEL_APPROVAL_MISSING",
            "grounding labels require an owner-approved manifest",
        )
    if checksum_bytes(benchmark_data) != manifest.ordered_files[0].checksum:
        raise GroundingLabelError(
            "GROUNDING_LABEL_CHECKSUM_MISMATCH",
            "grounding label checksum does not match its approved manifest",
        )
    records: list[GroundingBenchmarkRecord] = []
    for line_number, line in enumerate(benchmark_data.splitlines(keepends=True), start=1):
        if not line.strip():
            raise GroundingLabelError(
                "GROUNDING_LABEL_SCHEMA_INVALID", "grounding labels contain an empty record"
            )
        try:
            records.append(
                parse_record_json(
                    line,
                    GroundingBenchmarkRecord,
                    artifact_path=manifest.ordered_files[0].path,
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            raise _schema_error(error) from error
    record_ids = tuple(record.question_id for record in records)
    if record_ids != manifest.ordered_question_ids:
        raise GroundingLabelError(
            "GROUNDING_LABEL_ID_MISMATCH",
            "grounding labels must follow the approved ordered question IDs",
        )
    retrieval_labels = tuple(
        RetrievalLabelRow(
            question_id=record.question_id,
            relevant_evidence_ids=tuple(
                evidence.evidence_id
                for evidence in record.relevant_evidence
                if evidence.relevance in {"relevant", "partially_relevant"}
            ),
        )
        for record in records
    )
    return ApprovedGroundingBenchmark(
        manifest=manifest,
        records=tuple(records),
        retrieval_labels=retrieval_labels,
    )
