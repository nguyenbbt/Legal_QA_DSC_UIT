"""Local-only MIL-004 exact/BM25 retrieval and private annotation artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, Self

from pydantic import model_validator

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import (
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    QuestionRecord,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.retrieval_metrics import (
    ContainmentInputRow,
    evaluate_answer_containment,
)
from legal_rag.evaluation.split import (
    SplitError,
    load_split_manifest_rows,
    load_split_questions_jsonl,
)
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.exact import (
    AliasIndex,
    LegalReference,
    document_number_key,
    parse_legal_reference,
    resolve_exact_reference,
)
from legal_rag.retrieval.fusion import union_rank_candidates
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic


class RealRetrievalIndex(Protocol):
    def retrieve(self, query: str) -> SparseRetrievalResult: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_coordinate(
        self,
        hierarchy_kind: str,
        hierarchy_ordinal: str | None,
    ) -> tuple[ChunkRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalQuestionResult:
    question: QuestionRecord
    candidates: tuple[RetrievalCandidate, ...]
    diagnostics: tuple[RetrievalDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class RealRetrievalArtifacts:
    retrieval_output: bytes
    annotation_queue: bytes
    report: bytes


class RealRetrievalError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _GroundingSampleRow(FrozenStrictModel, frozen=True):
    question_id: NonEmptyString
    exact_reference_syntax: bool
    normalized_length: NonNegativeInt
    tercile: Literal[0, 1, 2]
    sample_digest: NonEmptyString
    selected: bool
    fill_reason: Literal[
        "stratum_primary", "same_tercile_other_branch", "global_underfill", "not_selected"
    ]
    final_position: NonNegativeInt | None


class _GroundingSampleManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["grounding.sample.manifest.v1"]
    sampling_version: Literal["grounding-sample.v1"]
    seed: Literal["dsc2026-grounding-sample-v1"]
    split_checksum: Sha256
    annotation_status: Literal["unlabeled"]
    eligible_question_count: NonNegativeInt
    sample_size: Literal[60]
    selected_question_ids: tuple[NonEmptyString, ...]
    rows: tuple[_GroundingSampleRow, ...]

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        selected_rows = tuple(
            row.question_id
            for row in sorted(
                (row for row in self.rows if row.selected),
                key=lambda row: row.final_position if row.final_position is not None else -1,
            )
        )
        if (
            len(self.rows) != self.eligible_question_count
            or len(self.selected_question_ids) != 60
            or len(set(self.selected_question_ids)) != 60
            or selected_rows != self.selected_question_ids
        ):
            raise ValueError("grounding sample selection is inconsistent")
        return self


def _coordinate_chunks(
    reference: LegalReference,
    *,
    index: RealRetrievalIndex,
    aliases: AliasIndex,
) -> tuple[ChunkRecord, ...]:
    if reference.document_number is not None:
        resolved_ids = aliases.context_ids_for(document_number_key(reference.document_number))
        if len(resolved_ids) == 1:
            return index.chunks_for_context(resolved_ids[0])
        return ()
    if reference.point is not None:
        return index.chunks_for_coordinate("point", reference.point)
    if reference.clause is not None:
        return index.chunks_for_coordinate("clause", reference.clause)
    return index.chunks_for_coordinate("article", reference.article)


def retrieve_question(
    question: QuestionRecord,
    *,
    index: RealRetrievalIndex,
    aliases: AliasIndex,
) -> RetrievalQuestionResult:
    """Run fail-closed exact resolution and deterministic sparse union for one question."""

    parsed = parse_legal_reference(question.question)
    exact_candidates: tuple[RetrievalCandidate, ...] = ()
    exact_diagnostics = parsed.diagnostics
    if parsed.reference is not None:
        exact = resolve_exact_reference(
            parsed.reference,
            aliases=aliases,
            chunks=_coordinate_chunks(parsed.reference, index=index, aliases=aliases),
        )
        exact_candidates = exact.candidates
        exact_diagnostics = exact.diagnostics
    sparse = index.retrieve(question.question)
    return RetrievalQuestionResult(
        question=question,
        candidates=union_rank_candidates(exact=exact_candidates, sparse=sparse.candidates),
        diagnostics=(*exact_diagnostics, *sparse.diagnostics),
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _candidate_row(candidate: RetrievalCandidate, *, include_text: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "evidence_id": candidate.chunk.chunk_id,
        "context_id": candidate.chunk.context_id,
        "hierarchy_path": list(candidate.chunk.hierarchy_path),
        "canonical_start": candidate.chunk.canonical_start,
        "canonical_end": candidate.chunk.canonical_end,
        "chunk_checksum": candidate.chunk.chunk_checksum,
        "exact_reference_match": candidate.exact_reference_match,
        "sparse_score": candidate.sparse_score,
    }
    if include_text:
        row["display_text"] = candidate.chunk.display_text
    return row


def _diagnostic_row(diagnostic: RetrievalDiagnostic) -> dict[str, object]:
    return {
        "code": diagnostic.code,
        "candidate_count": diagnostic.candidate_count,
        "parser_version": diagnostic.parser_version,
        "document_key_version": diagnostic.document_key_version,
        "alias_manifest_checksum": diagnostic.alias_manifest_checksum,
    }


def _retrieval_output_bytes(results: tuple[RetrievalQuestionResult, ...]) -> bytes:
    return b"".join(
        _json_bytes(
            {
                "schema_version": "retrieval.output.v1",
                "question_id": result.question.question_id,
                "question_checksum": checksum_bytes(result.question.question.encode("utf-8")),
                "candidates": [
                    _candidate_row(candidate, include_text=False) for candidate in result.candidates
                ],
                "diagnostics": [_diagnostic_row(diagnostic) for diagnostic in result.diagnostics],
            }
        )
        for result in results
    )


def _annotation_queue_bytes(
    selected: tuple[RetrievalQuestionResult, ...],
    *,
    split_checksum: str,
    index_checksum: str,
    chunks_checksum: str,
) -> bytes:
    return b"".join(
        _json_bytes(
            {
                "schema_version": "grounding.annotation.work-item.v1",
                "question_id": result.question.question_id,
                "split": "development",
                "question_checksum": checksum_bytes(result.question.question.encode("utf-8")),
                "question": result.question.question,
                "gold_answer": result.question.answer,
                "split_checksum": split_checksum,
                "chunk_artifact_checksum": chunks_checksum,
                "index_checksum": index_checksum,
                "candidates": [
                    _candidate_row(candidate, include_text=True) for candidate in result.candidates
                ],
                "diagnostics": [_diagnostic_row(diagnostic) for diagnostic in result.diagnostics],
                "relevant_evidence": None,
                "required_claims": None,
                "question_answerability": None,
                "temporal_assessment": None,
                "annotation_state": "pending_primary_annotation",
            }
        )
        for result in selected
    )


def build_failure_taxonomy(
    results: tuple[RetrievalQuestionResult, ...],
) -> list[dict[str, object]]:
    diagnostic_counts = Counter(
        diagnostic.code for result in results for diagnostic in result.diagnostics
    )
    parser_alias_count = sum(
        count
        for code, count in diagnostic_counts.items()
        if code.startswith("EXACT_") and code != "EXACT_COORDINATE_ABSENT"
    )
    return [
        {
            "category": "parser_or_alias_identity_error",
            "status": "observed",
            "question_count": parser_alias_count,
        },
        {
            "category": "missing_top_10_evidence",
            "status": "blocked_pending_owner_approved_labels",
            "question_count": None,
        },
        {
            "category": "poor_top_rank_ordering",
            "status": "blocked_pending_owner_approved_labels",
            "question_count": None,
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


def build_real_retrieval_artifacts(
    results: tuple[RetrievalQuestionResult, ...],
    *,
    selected_question_ids: tuple[str, ...],
    split_checksum: str,
    index_checksum: str,
    chunks_checksum: str,
    alias_manifest_checksum: str,
) -> RealRetrievalArtifacts:
    """Serialize private retrieval outputs without inventing human relevance labels."""

    ordered = tuple(sorted(results, key=lambda result: result.question.question_id.encode("utf-8")))
    by_id = {result.question.question_id: result for result in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("retrieval question IDs must be unique")
    if len(selected_question_ids) != 60 or len(set(selected_question_ids)) != 60:
        raise ValueError("grounding annotation queue requires exactly 60 unique IDs")
    try:
        selected = tuple(by_id[question_id] for question_id in selected_question_ids)
    except KeyError as error:
        raise ValueError("grounding selection is absent from retrieval outputs") from error
    output = _retrieval_output_bytes(ordered)
    queue = _annotation_queue_bytes(
        selected,
        split_checksum=split_checksum,
        index_checksum=index_checksum,
        chunks_checksum=chunks_checksum,
    )
    containment = evaluate_answer_containment(
        tuple(
            ContainmentInputRow(
                result.question.question_id,
                result.question.answer or "",
                tuple(candidate.chunk.display_text for candidate in result.candidates),
            )
            for result in ordered
        )
    )
    diagnostic_counts = Counter(
        diagnostic.code for result in ordered for diagnostic in result.diagnostics
    )
    report = _json_bytes(
        {
            "schema_version": "mil-004.retrieval.report.v1",
            "retrieval_question_count": len(ordered),
            "questions_without_candidates": sum(not result.candidates for result in ordered),
            "questions_with_exact_hit": sum(
                any(candidate.exact_reference_match for candidate in result.candidates)
                for result in ordered
            ),
            "split_checksum": split_checksum,
            "index_checksum": index_checksum,
            "chunk_artifact_checksum": chunks_checksum,
            "alias_manifest_checksum": alias_manifest_checksum,
            "retrieval_output_checksum": checksum_bytes(output),
            "annotation_queue_checksum": checksum_bytes(queue),
            "annotation_queue_size": len(selected),
            "annotation_status": "pending_primary_annotation",
            "metrics_status": "blocked_pending_owner_approved_labels",
            "diagnostic_counts": [
                {"code": code, "count": count} for code, count in sorted(diagnostic_counts.items())
            ],
            "containment": asdict(containment),
            "failure_taxonomy": build_failure_taxonomy(ordered),
        }
    )
    return RealRetrievalArtifacts(output, queue, report)


def run_development_retrieval(
    *,
    question_data: bytes,
    split_manifest_data: bytes,
    grounding_sample_data: bytes,
    index: RealRetrievalIndex,
    aliases: AliasIndex,
    index_checksum: str,
    chunks_checksum: str,
    alias_manifest_checksum: str,
) -> RealRetrievalArtifacts:
    """Run the fixed real-corpus baseline over all and only development questions."""

    try:
        split_questions = load_split_questions_jsonl(question_data, expected_answer_state="gold")
        questions = tuple(
            parse_record_json(
                line,
                QuestionRecord,
                artifact_path="train.questions.jsonl",
                record_identity=str(line_number),
            )
            for line_number, line in enumerate(question_data.splitlines(keepends=True), start=1)
        )
        if any(question.answer_state != "gold" for question in questions):
            raise RealRetrievalError(
                "SPLIT_QUESTION_ROLE_INVALID", "retrieval questions must have gold answers"
            )
        split_rows = load_split_manifest_rows(
            split_manifest_data,
            expected_source_checksum=checksum_bytes(question_data),
            expected_question_ids=tuple(question.question_id for question in split_questions),
        )
        sample = parse_record_json(
            grounding_sample_data,
            _GroundingSampleManifest,
            artifact_path="grounding-sample.v1.json",
            record_identity="manifest",
        )
    except SplitError as error:
        raise RealRetrievalError(error.code, error.message) from error
    except RecordValidationError as error:
        message = error.issues[0].message if error.issues else "grounding sample is invalid"
        raise RealRetrievalError("GROUNDING_SAMPLE_INVALID", message) from error
    split_checksum = checksum_bytes(split_manifest_data)
    if sample.split_checksum != split_checksum:
        raise RealRetrievalError(
            "GROUNDING_SPLIT_MISMATCH", "grounding sample differs from the active split"
        )
    development_ids = {row.question_id for row in split_rows if row.split == "development"}
    if not set(sample.selected_question_ids).issubset(development_ids):
        raise RealRetrievalError(
            "GROUNDING_SAMPLE_INVALID", "grounding selection is outside development"
        )
    development = tuple(
        question for question in questions if question.question_id in development_ids
    )
    results = tuple(
        retrieve_question(question, index=index, aliases=aliases) for question in development
    )
    return build_real_retrieval_artifacts(
        results,
        selected_question_ids=sample.selected_question_ids,
        split_checksum=split_checksum,
        index_checksum=index_checksum,
        chunks_checksum=chunks_checksum,
        alias_manifest_checksum=alias_manifest_checksum,
    )
