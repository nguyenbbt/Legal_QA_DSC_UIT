"""Approved-label evaluation for the fixed MIL-004 retrieval artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import (
    CanonicalIntegerString,
    FiniteScore,
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.grounding_labels import (
    GroundingLabelError,
    load_approved_grounding_benchmark,
)
from legal_rag.evaluation.retrieval_metrics import (
    ContainmentInputRow,
    RetrievalEvaluationError,
    RetrievalOutputRow,
    evaluate_answer_containment,
    evaluate_retrieval,
)


class LabeledRetrievalError(Exception):
    """Stable safe failure at the labeled retrieval artifact boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _RetrievalCandidate(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    context_id: CanonicalIntegerString
    hierarchy_path: tuple[NonEmptyString, ...]
    canonical_start: NonNegativeInt
    canonical_end: NonNegativeInt
    chunk_checksum: Sha256
    exact_reference_match: bool
    sparse_score: FiniteScore | None

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if not self.hierarchy_path or self.canonical_start >= self.canonical_end:
            raise ValueError("retrieval candidate hierarchy or canonical span is invalid")
        return self


class _AnnotationCandidate(_RetrievalCandidate, frozen=True):
    display_text: NonEmptyString


class _RetrievalDiagnostic(FrozenStrictModel, frozen=True):
    code: NonEmptyString
    candidate_count: NonNegativeInt
    parser_version: NonEmptyString
    document_key_version: NonEmptyString
    alias_manifest_checksum: Sha256 | None


class _RetrievalOutputRecord(FrozenStrictModel, frozen=True):
    schema_version: Literal["retrieval.output.v1"]
    question_id: NonEmptyString
    question_checksum: Sha256
    candidates: tuple[_RetrievalCandidate, ...]
    diagnostics: tuple[_RetrievalDiagnostic, ...]

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> Self:
        evidence_ids = tuple(candidate.evidence_id for candidate in self.candidates)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("retrieval candidate evidence IDs must be unique")
        return self


class _AnnotationWorkItem(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.annotation.work-item.v1"]
    question_id: NonEmptyString
    split: Literal["development"]
    question_checksum: Sha256
    question: NonEmptyString
    gold_answer: NonEmptyString
    split_checksum: Sha256
    chunk_artifact_checksum: Sha256
    index_checksum: Sha256
    candidates: tuple[_AnnotationCandidate, ...]
    diagnostics: tuple[_RetrievalDiagnostic, ...]
    relevant_evidence: None
    required_claims: None
    question_answerability: None
    temporal_assessment: None
    annotation_state: Literal["pending_primary_annotation"]

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> Self:
        evidence_ids = tuple(candidate.evidence_id for candidate in self.candidates)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("annotation candidate evidence IDs must be unique")
        return self


def _load_jsonl[RecordT: FrozenStrictModel](
    data: bytes,
    model: type[RecordT],
    *,
    artifact_path: str,
) -> tuple[RecordT, ...]:
    if not data:
        raise LabeledRetrievalError(
            "RETRIEVAL_EVAL_INPUT_INVALID", "retrieval evaluation input is empty"
        )
    records: list[RecordT] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        if not line.strip():
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_INPUT_INVALID",
                "retrieval evaluation input contains an empty record",
            )
        try:
            records.append(
                parse_record_json(
                    line,
                    model,
                    artifact_path=artifact_path,
                    record_identity=str(line_number),
                )
            )
        except RecordValidationError as error:
            message = error.issues[0].message if error.issues else "input schema is invalid"
            raise LabeledRetrievalError("RETRIEVAL_EVAL_INPUT_INVALID", message) from error
    return tuple(records)


def _unique_by_question_id[RecordT: _RetrievalOutputRecord | _AnnotationWorkItem](
    records: Sequence[RecordT],
) -> dict[str, RecordT]:
    by_id: dict[str, RecordT] = {}
    for record in records:
        if record.question_id in by_id:
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_ID_DUPLICATE",
                "retrieval evaluation question IDs must be unique",
            )
        by_id[record.question_id] = record
    return by_id


def _candidate_identity(
    candidate: _RetrievalCandidate | _AnnotationCandidate,
) -> tuple[object, ...]:
    return (
        candidate.evidence_id,
        candidate.context_id,
        candidate.hierarchy_path,
        candidate.canonical_start,
        candidate.canonical_end,
        candidate.chunk_checksum,
        candidate.exact_reference_match,
        candidate.sparse_score,
    )


def _validate_artifact_identity(
    *,
    question_ids: tuple[str, ...],
    split_checksum: str,
    chunks_checksum: str,
    index_checksum: str,
    outputs: dict[str, _RetrievalOutputRecord],
    queue: tuple[_AnnotationWorkItem, ...],
) -> None:
    if tuple(item.question_id for item in queue) != question_ids:
        raise LabeledRetrievalError(
            "RETRIEVAL_EVAL_ID_MISMATCH",
            "annotation queue must follow the approved benchmark question IDs",
        )
    if not set(question_ids).issubset(outputs):
        raise LabeledRetrievalError(
            "RETRIEVAL_EVAL_ID_MISMATCH",
            "retrieval output does not cover the approved benchmark question IDs",
        )
    for work_item in queue:
        if work_item.split_checksum != split_checksum:
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_SPLIT_MISMATCH",
                "annotation queue split checksum differs from the approved benchmark",
            )
        if work_item.chunk_artifact_checksum != chunks_checksum:
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_CHUNK_MISMATCH",
                "annotation queue chunk checksum differs from the approved benchmark",
            )
        if work_item.index_checksum != index_checksum:
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_INDEX_MISMATCH",
                "annotation queue index checksum differs from the approved benchmark",
            )
        output = outputs[work_item.question_id]
        if (
            checksum_bytes(work_item.question.encode("utf-8")) != work_item.question_checksum
            or output.question_checksum != work_item.question_checksum
        ):
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_QUESTION_MISMATCH",
                "question checksum differs across retrieval evaluation artifacts",
            )
        output_candidates = tuple(_candidate_identity(item) for item in output.candidates)
        queue_candidates = tuple(_candidate_identity(item) for item in work_item.candidates)
        if output_candidates != queue_candidates or output.diagnostics != work_item.diagnostics:
            raise LabeledRetrievalError(
                "RETRIEVAL_EVAL_CANDIDATE_MISMATCH",
                "retrieval candidates differ from the annotation queue",
            )


def _failure_taxonomy(
    *,
    question_ids: tuple[str, ...],
    labels_by_id: dict[str, tuple[str, ...]],
    outputs: dict[str, _RetrievalOutputRecord],
) -> list[dict[str, object]]:
    parser_or_alias_errors = 0
    missing_top_10 = 0
    poor_top_rank = 0
    for question_id in question_ids:
        output = outputs[question_id]
        if any(
            diagnostic.code.startswith("EXACT_") and diagnostic.code != "EXACT_COORDINATE_ABSENT"
            for diagnostic in output.diagnostics
        ):
            parser_or_alias_errors += 1
        relevant = frozenset(labels_by_id[question_id])
        if not relevant:
            continue
        retrieved = tuple(candidate.evidence_id for candidate in output.candidates[:10])
        if not relevant.issubset(retrieved):
            missing_top_10 += 1
        if retrieved and retrieved[0] not in relevant and relevant.intersection(retrieved):
            poor_top_rank += 1
    return [
        {
            "category": "parser_or_alias_identity_error",
            "status": "observed",
            "question_count": parser_or_alias_errors,
        },
        {
            "category": "missing_top_10_evidence",
            "status": "observed_from_owner_approved_labels",
            "question_count": missing_top_10,
        },
        {
            "category": "poor_top_rank_ordering",
            "status": "observed_from_owner_approved_labels",
            "question_count": poor_top_rank,
        },
        {
            "category": "evidence_packing_error",
            "status": "not_evaluated_in_retrieval_only_baseline",
            "question_count": None,
        },
        {
            "category": "answer_generation_error",
            "status": "not_applicable_no_learned_generator",
            "question_count": None,
        },
    ]


def evaluate_labeled_retrieval(
    *,
    retrieval_output_data: bytes,
    annotation_queue_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
) -> bytes:
    """Bind approved labels to the fixed retrieval run and emit exact metrics."""

    try:
        approved = load_approved_grounding_benchmark(benchmark_manifest_data, benchmark_data)
        output_records = _load_jsonl(
            retrieval_output_data,
            _RetrievalOutputRecord,
            artifact_path="retrieval.output.v1.jsonl",
        )
        queue = _load_jsonl(
            annotation_queue_data,
            _AnnotationWorkItem,
            artifact_path="annotation-work-queue.v1.jsonl",
        )
        outputs = _unique_by_question_id(output_records)
        question_ids = tuple(approved.manifest.ordered_question_ids)
        _validate_artifact_identity(
            question_ids=question_ids,
            split_checksum=approved.manifest.development_split_checksum,
            chunks_checksum=approved.manifest.chunk_artifact_checksum,
            index_checksum=approved.manifest.index_checksum,
            outputs=outputs,
            queue=queue,
        )
        labels_by_id = {
            row.question_id: row.relevant_evidence_ids for row in approved.retrieval_labels
        }
        for record in approved.records:
            output = outputs[record.question_id]
            if output.question_checksum != record.question_checksum:
                raise LabeledRetrievalError(
                    "RETRIEVAL_EVAL_QUESTION_MISMATCH",
                    "approved label question checksum differs from retrieval output",
                )
        metric_rows = tuple(
            RetrievalOutputRow(
                question_id=question_id,
                retrieved_evidence_ids=tuple(
                    candidate.evidence_id for candidate in outputs[question_id].candidates
                ),
            )
            for question_id in question_ids
        )
        metrics = evaluate_retrieval(approved.retrieval_labels, metric_rows)
        containment = evaluate_answer_containment(
            tuple(
                ContainmentInputRow(
                    question_id=item.question_id,
                    gold_answer=item.gold_answer,
                    retrieved_display_texts=tuple(
                        candidate.display_text for candidate in item.candidates
                    ),
                )
                for item in queue
            )
        )
    except GroundingLabelError as error:
        raise LabeledRetrievalError(error.code, error.message) from error
    except RetrievalEvaluationError as error:
        raise LabeledRetrievalError(error.code, error.message) from error

    return content_json_bytes(
        {
            "schema_version": "retrieval.evaluation.report.v1",
            "metrics_status": "complete_owner_approved_labels",
            "benchmark_manifest_checksum": checksum_bytes(benchmark_manifest_data),
            "benchmark_checksum": checksum_bytes(benchmark_data),
            "retrieval_output_checksum": checksum_bytes(retrieval_output_data),
            "annotation_queue_checksum": checksum_bytes(annotation_queue_data),
            "split_checksum": approved.manifest.development_split_checksum,
            "chunk_artifact_checksum": approved.manifest.chunk_artifact_checksum,
            "index_checksum": approved.manifest.index_checksum,
            "metrics": asdict(metrics),
            "containment": asdict(containment),
            "failure_taxonomy": _failure_taxonomy(
                question_ids=question_ids,
                labels_by_id=labels_by_id,
                outputs=outputs,
            ),
        }
    )
