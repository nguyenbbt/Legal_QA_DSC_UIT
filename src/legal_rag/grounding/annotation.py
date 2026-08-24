"""Local-only, resumable relevance annotation for the frozen MIL-004 queue."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import (
    CanonicalIntegerString,
    FiniteScore,
    FrozenStrictModel,
    NfcString,
    NonEmptyString,
    NonNegativeInt,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.grounding_labels import (
    EvidenceRelevance,
    GroundingBenchmarkManifest,
    GroundingBenchmarkRecord,
    GroundingFile,
)

APPROVAL_CONFIRMATION = "APPROVE_GROUNDING_V1"
MAX_QUEUE_BYTES = 16 * 1024 * 1024
MAX_PROGRESS_BYTES = 4 * 1024 * 1024
Relevance = Literal["relevant", "partially_relevant", "not_relevant"]
LabeledRelevance = Literal[
    "r",
    "p",
    "n",
    "relevant",
    "partially_relevant",
    "not_relevant",
]
ApprovalState = Literal["draft", "approved"]


class GroundingAnnotationError(Exception):
    """Stable safe failure at the local annotation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AnnotationCandidate(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    context_id: CanonicalIntegerString
    hierarchy_path: tuple[NonEmptyString, ...]
    canonical_start: NonNegativeInt
    canonical_end: NonNegativeInt
    chunk_checksum: Sha256
    exact_reference_match: bool
    sparse_score: FiniteScore | None
    display_text: NonEmptyString

    @model_validator(mode="after")
    def _validate_candidate(self) -> Self:
        if not self.hierarchy_path or self.canonical_start >= self.canonical_end:
            raise ValueError("candidate hierarchy or canonical span is invalid")
        return self


class LabeledAnnotationCandidate(AnnotationCandidate, frozen=True):
    relevance_label: LabeledRelevance


class AnnotationDiagnostic(FrozenStrictModel, frozen=True):
    code: NonEmptyString
    candidate_count: NonNegativeInt
    parser_version: NonEmptyString
    document_key_version: NonEmptyString
    alias_manifest_checksum: Sha256 | None


class AnnotationQueueItem(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.annotation.work-item.v1"]
    question_id: NonEmptyString
    split: Literal["development"]
    question_checksum: Sha256
    question: NonEmptyString
    gold_answer: NonEmptyString
    split_checksum: Sha256
    chunk_artifact_checksum: Sha256
    index_checksum: Sha256
    candidates: tuple[AnnotationCandidate, ...]
    diagnostics: tuple[AnnotationDiagnostic, ...]
    relevant_evidence: None
    required_claims: None
    question_answerability: None
    temporal_assessment: None
    annotation_state: Literal["pending_primary_annotation"]

    @model_validator(mode="after")
    def _validate_item(self) -> Self:
        evidence_ids = tuple(candidate.evidence_id for candidate in self.candidates)
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("annotation candidates must be non-empty and unique")
        if checksum_bytes(self.question.encode("utf-8")) != self.question_checksum:
            raise ValueError("question checksum does not match question text")
        return self


class LabeledAnnotationQueueItem(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.annotation.work-item.v1"]
    question_id: NonEmptyString
    split: Literal["development"]
    question_checksum: Sha256
    question: NonEmptyString
    gold_answer: NonEmptyString
    split_checksum: Sha256
    chunk_artifact_checksum: Sha256
    index_checksum: Sha256
    candidates: tuple[LabeledAnnotationCandidate, ...]
    diagnostics: tuple[AnnotationDiagnostic, ...]
    relevant_evidence: None
    required_claims: None
    question_answerability: None
    temporal_assessment: None
    annotation_state: Literal["pending_primary_annotation"]

    @model_validator(mode="after")
    def _validate_item(self) -> Self:
        evidence_ids = tuple(candidate.evidence_id for candidate in self.candidates)
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("labeled annotation candidates must be non-empty and unique")
        if checksum_bytes(self.question.encode("utf-8")) != self.question_checksum:
            raise ValueError("question checksum does not match question text")
        return self


@dataclass(frozen=True, slots=True)
class AnnotationQueue:
    items: tuple[AnnotationQueueItem, ...]
    checksum: str
    split_checksum: str
    chunk_artifact_checksum: str
    index_checksum: str


@dataclass(frozen=True, slots=True)
class LabeledQueueImport:
    progress: AnnotationProgress
    defaulted_metadata_question_count: int


class QuestionAnnotation(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    evidence_labels: tuple[tuple[NonEmptyString, Relevance], ...]
    required_claims: tuple[NfcString, ...]
    question_answerability: Literal["answerable", "not_answerable", "unknown"]
    temporal_assessment: Literal["valid", "invalid", "unknown", "not_applicable"]

    @model_validator(mode="after")
    def _validate_labels(self) -> Self:
        evidence_ids = tuple(evidence_id for evidence_id, _label in self.evidence_labels)
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence labels must be non-empty and unique")
        if len(self.required_claims) != len(set(self.required_claims)):
            raise ValueError("required claims must be unique")
        return self


class AnnotationProgress(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.annotation.progress.v1"]
    queue_checksum: Sha256
    annotator_id: NonEmptyString
    ordered_question_ids: tuple[NonEmptyString, ...]
    annotations: tuple[QuestionAnnotation, ...]
    annotation_origin: Literal["interactive_human", "imported_labeled_queue"] = "interactive_human"
    source_labeled_checksum: Sha256 | None = None
    metadata_completion: Literal["complete", "retrieval_only_defaults"] = "complete"

    @model_validator(mode="after")
    def _validate_progress(self) -> Self:
        if len(self.ordered_question_ids) != 60 or len(set(self.ordered_question_ids)) != 60:
            raise ValueError("annotation progress requires exactly 60 ordered question IDs")
        annotation_ids = tuple(annotation.question_id for annotation in self.annotations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation progress question IDs must be unique")
        imported = self.annotation_origin == "imported_labeled_queue"
        if imported != (self.source_labeled_checksum is not None) or imported != (
            self.metadata_completion == "retrieval_only_defaults"
        ):
            raise ValueError("annotation progress provenance fields are inconsistent")
        return self


def _schema_error(error: RecordValidationError) -> GroundingAnnotationError:
    message = error.issues[0].message if error.issues else "annotation artifact is invalid"
    return GroundingAnnotationError("GROUNDING_ANNOTATION_SCHEMA_INVALID", message)


def load_annotation_queue(data: bytes) -> AnnotationQueue:
    """Parse and bind the exact private 60-question work queue."""

    if not data or len(data) > MAX_QUEUE_BYTES:
        raise GroundingAnnotationError(
            "GROUNDING_QUEUE_SIZE_INVALID", "annotation queue is empty or exceeds 16 MiB"
        )
    items: list[AnnotationQueueItem] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        if not line.strip():
            raise GroundingAnnotationError(
                "GROUNDING_ANNOTATION_SCHEMA_INVALID",
                "annotation queue contains an empty record",
            )
        try:
            items.append(
                parse_record_json(
                    line,
                    AnnotationQueueItem,
                    artifact_path="annotation-work-queue.v1.jsonl",
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            raise _schema_error(error) from error
    if len(items) != 60 or len({item.question_id for item in items}) != 60:
        raise GroundingAnnotationError(
            "GROUNDING_QUEUE_COVERAGE_INVALID",
            "annotation queue must contain exactly 60 unique questions",
        )
    identities = {
        (item.split_checksum, item.chunk_artifact_checksum, item.index_checksum) for item in items
    }
    if len(identities) != 1:
        raise GroundingAnnotationError(
            "GROUNDING_QUEUE_IDENTITY_MISMATCH",
            "annotation queue artifact identities are inconsistent",
        )
    split_checksum, chunk_checksum, index_checksum = identities.pop()
    return AnnotationQueue(
        items=tuple(items),
        checksum=checksum_bytes(data),
        split_checksum=split_checksum,
        chunk_artifact_checksum=chunk_checksum,
        index_checksum=index_checksum,
    )


def _unlabeled_item(item: LabeledAnnotationQueueItem) -> AnnotationQueueItem:
    data = item.model_dump(mode="python")
    data["candidates"] = tuple(
        {key: value for key, value in candidate.items() if key != "relevance_label"}
        for candidate in data["candidates"]
    )
    return AnnotationQueueItem.model_validate(data)


def _canonical_relevance(label: LabeledRelevance) -> Relevance:
    labels: dict[LabeledRelevance, Relevance] = {
        "r": "relevant",
        "p": "partially_relevant",
        "n": "not_relevant",
        "relevant": "relevant",
        "partially_relevant": "partially_relevant",
        "not_relevant": "not_relevant",
    }
    return labels[label]


def import_labeled_annotation_queue(
    queue: AnnotationQueue,
    labeled_data: bytes,
    *,
    annotator_id: str,
) -> LabeledQueueImport:
    """Import relevance-only labels while binding every byte-addressed queue field."""

    if not labeled_data or len(labeled_data) > MAX_QUEUE_BYTES:
        raise GroundingAnnotationError(
            "GROUNDING_QUEUE_SIZE_INVALID", "labeled annotation queue is empty or exceeds 16 MiB"
        )
    labeled_items: list[LabeledAnnotationQueueItem] = []
    for line_number, line in enumerate(labeled_data.splitlines(keepends=True), start=1):
        if not line.strip():
            raise GroundingAnnotationError(
                "GROUNDING_ANNOTATION_SCHEMA_INVALID",
                "labeled annotation queue contains an empty record",
            )
        try:
            labeled_items.append(
                parse_record_json(
                    line,
                    LabeledAnnotationQueueItem,
                    artifact_path="annotation-work-queue.v1.labeled.jsonl",
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            raise _schema_error(error) from error

    if len(labeled_items) != 60:
        raise GroundingAnnotationError(
            "GROUNDING_QUEUE_COVERAGE_INVALID",
            "labeled annotation queue must contain exactly 60 questions",
        )
    if tuple(_unlabeled_item(item) for item in labeled_items) != queue.items:
        raise GroundingAnnotationError(
            "GROUNDING_LABELED_QUEUE_MISMATCH",
            "labeled annotation queue differs from the frozen work queue",
        )

    annotations = tuple(
        QuestionAnnotation(
            question_id=item.question_id,
            evidence_labels=tuple(
                (candidate.evidence_id, _canonical_relevance(candidate.relevance_label))
                for candidate in item.candidates
            ),
            required_claims=(),
            question_answerability="unknown",
            temporal_assessment="unknown",
        )
        for item in labeled_items
    )
    progress = AnnotationProgress(
        schema_version="grounding.annotation.progress.v1",
        queue_checksum=queue.checksum,
        annotator_id=annotator_id,
        ordered_question_ids=tuple(item.question_id for item in queue.items),
        annotations=annotations,
        annotation_origin="imported_labeled_queue",
        source_labeled_checksum=checksum_bytes(labeled_data),
        metadata_completion="retrieval_only_defaults",
    )
    for annotation in progress.annotations:
        _validate_annotation(annotation, {item.question_id: item for item in queue.items})
    return LabeledQueueImport(
        progress=progress,
        defaulted_metadata_question_count=len(labeled_items),
    )


def new_annotation_progress(queue: AnnotationQueue, *, annotator_id: str) -> AnnotationProgress:
    return AnnotationProgress(
        schema_version="grounding.annotation.progress.v1",
        queue_checksum=queue.checksum,
        annotator_id=annotator_id,
        ordered_question_ids=tuple(item.question_id for item in queue.items),
        annotations=(),
    )


def _validate_progress_identity(progress: AnnotationProgress, queue: AnnotationQueue) -> None:
    expected_ids = tuple(item.question_id for item in queue.items)
    if progress.queue_checksum != queue.checksum or progress.ordered_question_ids != expected_ids:
        raise GroundingAnnotationError(
            "GROUNDING_PROGRESS_QUEUE_MISMATCH",
            "annotation progress belongs to a different work queue",
        )


def _validate_annotation(
    annotation: QuestionAnnotation, queue_by_id: dict[str, AnnotationQueueItem]
) -> None:
    item = queue_by_id.get(annotation.question_id)
    if item is None:
        raise GroundingAnnotationError(
            "GROUNDING_ANNOTATION_QUESTION_UNKNOWN",
            "annotation question is absent from the work queue",
        )
    expected_ids = tuple(candidate.evidence_id for candidate in item.candidates)
    labeled_ids = tuple(evidence_id for evidence_id, _label in annotation.evidence_labels)
    if labeled_ids != expected_ids:
        raise GroundingAnnotationError(
            "GROUNDING_ANNOTATION_EVIDENCE_MISMATCH",
            "every candidate must be labeled once in queue order",
        )


def load_annotation_progress(data: bytes, queue: AnnotationQueue) -> AnnotationProgress:
    if not data or len(data) > MAX_PROGRESS_BYTES:
        raise GroundingAnnotationError(
            "GROUNDING_PROGRESS_SIZE_INVALID", "annotation progress is empty or exceeds 4 MiB"
        )
    try:
        progress = parse_record_json(
            data,
            AnnotationProgress,
            artifact_path="annotation-progress.v1.json",
            record_identity="progress",
        )
    except RecordValidationError as error:
        raise _schema_error(error) from error
    _validate_progress_identity(progress, queue)
    queue_by_id = {item.question_id: item for item in queue.items}
    for annotation in progress.annotations:
        _validate_annotation(annotation, queue_by_id)
    return progress


def annotation_progress_bytes(progress: AnnotationProgress) -> bytes:
    return content_json_bytes(progress.model_dump(mode="json"))


def require_private_grounding_path(path: Path, *, project_root: Path) -> Path:
    """Resolve a path only inside the ignored private grounding directory."""

    private_root = (project_root / "artifacts" / "evaluations" / "grounding").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as error:
        raise GroundingAnnotationError(
            "GROUNDING_PRIVATE_PATH_REQUIRED",
            "grounding annotation artifacts must stay under artifacts/evaluations/grounding",
        ) from error
    return resolved


def write_annotation_progress(path: Path, progress: AnnotationProgress) -> str:
    """Atomically replace mutable private progress and return its checksum."""

    data = annotation_progress_bytes(progress)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise GroundingAnnotationError(
            "GROUNDING_PROGRESS_WRITE_FAILED", "annotation progress could not be saved"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return checksum_bytes(data)


def record_question_annotation(
    queue: AnnotationQueue,
    progress: AnnotationProgress,
    annotation: QuestionAnnotation,
) -> AnnotationProgress:
    """Add or replace one annotation while preserving frozen queue order."""

    _validate_progress_identity(progress, queue)
    _validate_annotation(annotation, {item.question_id: item for item in queue.items})
    by_id = {row.question_id: row for row in progress.annotations}
    by_id[annotation.question_id] = annotation
    return progress.model_copy(
        update={
            "annotations": tuple(
                by_id[question_id]
                for question_id in progress.ordered_question_ids
                if question_id in by_id
            )
        }
    )


def export_grounding_benchmark(
    queue: AnnotationQueue,
    progress: AnnotationProgress,
    *,
    benchmark_path: str,
    approval_state: ApprovalState,
    owner_confirmation: str | None,
) -> tuple[bytes, bytes]:
    """Export complete relevance labels and their deterministic manifest."""

    _validate_progress_identity(progress, queue)
    if approval_state == "approved" and owner_confirmation != APPROVAL_CONFIRMATION:
        raise GroundingAnnotationError(
            "GROUNDING_APPROVAL_CONFIRMATION_REQUIRED",
            f"approved export requires exact confirmation {APPROVAL_CONFIRMATION}",
        )
    by_id = {annotation.question_id: annotation for annotation in progress.annotations}
    if tuple(by_id) != progress.ordered_question_ids:
        raise GroundingAnnotationError(
            "GROUNDING_ANNOTATION_INCOMPLETE",
            "all 60 questions must be completely labeled before export",
        )
    records = tuple(
        GroundingBenchmarkRecord(
            schema_version="grounding.benchmark.v1",
            question_id=item.question_id,
            split="development",
            question_checksum=item.question_checksum,
            relevant_evidence=tuple(
                EvidenceRelevance(evidence_id=evidence_id, relevance=relevance)
                for evidence_id, relevance in by_id[item.question_id].evidence_labels
            ),
            required_claims=by_id[item.question_id].required_claims,
            question_answerability=by_id[item.question_id].question_answerability,
            temporal_assessment=by_id[item.question_id].temporal_assessment,
            label_version="grounding.v1",
        )
        for item in queue.items
    )
    benchmark_data = b"".join(
        content_json_bytes(record.model_dump(mode="json")) for record in records
    )
    manifest = GroundingBenchmarkManifest(
        schema_version="grounding.benchmark.manifest.v1",
        label_version="grounding.v1",
        train_split_checksum=queue.split_checksum,
        development_split_checksum=queue.split_checksum,
        sampling_version="grounding-sample.v1",
        sampling_seed="dsc2026-grounding-sample-v1",
        ordered_question_ids=progress.ordered_question_ids,
        chunk_artifact_checksum=queue.chunk_artifact_checksum,
        index_checksum=queue.index_checksum,
        annotation_status=approval_state,
        ordered_files=(
            GroundingFile(path=benchmark_path, checksum=checksum_bytes(benchmark_data)),
        ),
    )
    return benchmark_data, content_json_bytes(manifest.model_dump(mode="json"))
