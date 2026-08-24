"""Checkpointed local-only public dry run over frozen retrieval evidence."""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from legal_rag.domain.artifacts import ImmutableArtifactError, write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import (
    AnswerRecord,
    FiniteScore,
    FrozenStrictModel,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    QuestionRecord,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.real_retrieval import RealRetrievalIndex, retrieve_question
from legal_rag.generation.fixture import FIXED_REFUSAL
from legal_rag.ingestion.organizer import OrganizerDataError, OrganizerQuestionReader
from legal_rag.retrieval.exact import AliasIndex
from legal_rag.retrieval.reranker import RerankerBackend, RerankerError, rerank_candidates
from legal_rag.submission.writer import SubmissionError, answers_jsonl_bytes, build_submission

_INPUT_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class PublicGeneratorBackend(Protocol):
    model_id: str
    model_revision: str

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str: ...


class PublicDryRunError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PublicEvidenceItem(FrozenStrictModel, frozen=True):
    evidence_id: NonEmptyString
    context_id: NonEmptyString
    hierarchy_path: tuple[NonEmptyString, ...]
    canonical_start: NonNegativeInt
    canonical_end: PositiveInt
    display_text: NonEmptyString
    chunk_checksum: Sha256
    exact_reference_match: bool
    sparse_score: FiniteScore | None
    reranker_score: FiniteScore | None
    rank: PositiveInt

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.canonical_start >= self.canonical_end or not self.hierarchy_path:
            raise ValueError("public evidence span and hierarchy must be valid")
        return self


class PublicEvidenceRow(FrozenStrictModel, frozen=True):
    schema_version: Literal["public.evidence.v1"]
    retrieval_run_id: NonEmptyString
    retrieval_fingerprint: Sha256
    question_id: NonEmptyString
    question_checksum: Sha256
    question: NonEmptyString
    evidence: tuple[PublicEvidenceItem, ...]

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if self.question_checksum != checksum_bytes(self.question.encode("utf-8")):
            raise ValueError("public evidence question checksum is invalid")
        ids = tuple(item.evidence_id for item in self.evidence)
        ranks = tuple(item.rank for item in self.evidence)
        if len(ids) != len(set(ids)) or ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("public evidence IDs and ranks must be unique and consecutive")
        return self


class PublicAnswerCheckpoint(FrozenStrictModel, frozen=True):
    schema_version: Literal["public.answer.checkpoint.v1"]
    run_id: NonEmptyString
    run_fingerprint: Sha256
    question_id: NonEmptyString
    question_checksum: Sha256
    evidence_ids: tuple[NonEmptyString, ...]
    answer: NonEmptyString
    generator_id: NonEmptyString
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> Self:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("checkpoint evidence IDs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class PublicEvidenceArtifacts:
    queue_data: bytes
    report_data: bytes


@dataclass(frozen=True, slots=True)
class PublicGenerationArtifacts:
    answers_data: bytes
    predictions_data: bytes
    manifest_data: bytes
    telemetry_data: bytes
    generated_question_count: int
    resumed_question_count: int


def _load_evidence_queue(data: bytes) -> tuple[PublicEvidenceRow, ...]:
    if not data or b"\r" in data or not data.endswith(b"\n"):
        raise PublicDryRunError(
            "PUBLIC_EVIDENCE_QUEUE_INVALID", "public evidence queue framing is invalid"
        )
    try:
        rows = tuple(
            parse_record_json(
                line + b"\n",
                PublicEvidenceRow,
                artifact_path="public.evidence.v1.jsonl",
                record_identity=str(line_number),
            )
            for line_number, line in enumerate(data.splitlines(), start=1)
        )
    except RecordValidationError as error:
        raise PublicDryRunError(
            "PUBLIC_EVIDENCE_QUEUE_INVALID", "public evidence queue contains an invalid row"
        ) from error
    ids = tuple(row.question_id for row in rows)
    if (
        not rows
        or len(ids) != len(set(ids))
        or len({row.retrieval_run_id for row in rows}) != 1
        or len({row.retrieval_fingerprint for row in rows}) != 1
    ):
        raise PublicDryRunError(
            "PUBLIC_EVIDENCE_QUEUE_INVALID", "public evidence queue identity is invalid"
        )
    return rows


def build_public_evidence_queue(
    questions: Sequence[QuestionRecord],
    *,
    index: RealRetrievalIndex,
    aliases: AliasIndex,
    reranker: RerankerBackend | None,
    retrieval_run_id: str,
    evidence_limit: int = 3,
    reranker_candidate_limit: int = 12,
    checkpoint_directory: Path | None = None,
    frozen_inputs: Mapping[str, str] | None = None,
) -> PublicEvidenceArtifacts:
    """Freeze top evidence for every public question without copying an answer field."""

    ordered = tuple(questions)
    ids = tuple(question.question_id for question in ordered)
    if (
        not ordered
        or len(ids) != len(set(ids))
        or any(question.answer_state != "unlabeled" for question in ordered)
    ):
        raise PublicDryRunError(
            "PUBLIC_QUESTION_SET_INVALID", "public questions must be unique and unlabeled"
        )
    if not retrieval_run_id.strip() or not 1 <= evidence_limit <= reranker_candidate_limit <= 50:
        raise PublicDryRunError(
            "PUBLIC_RETRIEVAL_CONFIG_INVALID", "public retrieval bounds are invalid"
        )
    frozen = _frozen_inputs(frozen_inputs) if frozen_inputs is not None else {}
    if checkpoint_directory is not None and not frozen:
        raise PublicDryRunError(
            "PUBLIC_RUN_FINGERPRINT_INVALID",
            "checkpointed public retrieval requires frozen input checksums",
        )
    retrieval_mode = "exact_bm25" if reranker is None else "exact_bm25_reranker"
    reranker_model_id = reranker.model_id if reranker is not None else None
    reranker_model_revision = reranker.model_revision if reranker is not None else None
    fingerprint_value = {
        "schema_version": "public.retrieval-fingerprint.v1",
        "retrieval_run_id": retrieval_run_id,
        "retrieval_mode": retrieval_mode,
        "reranker_model_id": reranker_model_id,
        "reranker_model_revision": reranker_model_revision,
        "evidence_limit": evidence_limit,
        "reranker_candidate_limit": reranker_candidate_limit,
        "frozen_inputs": frozen,
    }
    retrieval_fingerprint = checksum_bytes(content_json_bytes(fingerprint_value))

    rows: list[PublicEvidenceRow] = []
    generated_count = 0
    resumed_count = 0
    try:
        for question in ordered:
            checkpoint_path = (
                _checkpoint_path(checkpoint_directory, question.question_id)
                if checkpoint_directory is not None
                else None
            )
            if checkpoint_path is not None and checkpoint_path.exists():
                row = _load_evidence_checkpoint(checkpoint_path)
                if (
                    row.retrieval_run_id != retrieval_run_id
                    or row.retrieval_fingerprint != retrieval_fingerprint
                    or row.question_id != question.question_id
                    or row.question != question.question
                    or row.question_checksum != checksum_bytes(question.question.encode("utf-8"))
                ):
                    raise PublicDryRunError(
                        "PUBLIC_EVIDENCE_CHECKPOINT_MISMATCH",
                        "public evidence checkpoint differs from frozen inputs",
                    )
                rows.append(row)
                resumed_count += 1
                continue
            retrieved = retrieve_question(question, index=index, aliases=aliases)
            admitted = retrieved.candidates[:reranker_candidate_limit]
            reranked = (
                rerank_candidates(
                    question.question,
                    admitted,
                    reranker,
                    limit=evidence_limit,
                    maximum_candidate_count=reranker_candidate_limit,
                )
                if admitted and reranker is not None
                else admitted[:evidence_limit]
            )
            evidence = tuple(
                PublicEvidenceItem(
                    evidence_id=candidate.chunk.chunk_id,
                    context_id=candidate.chunk.context_id,
                    hierarchy_path=candidate.chunk.hierarchy_path,
                    canonical_start=candidate.chunk.canonical_start,
                    canonical_end=candidate.chunk.canonical_end,
                    display_text=candidate.chunk.display_text,
                    chunk_checksum=candidate.chunk.chunk_checksum,
                    exact_reference_match=candidate.exact_reference_match,
                    sparse_score=candidate.sparse_score,
                    reranker_score=candidate.reranker_score,
                    rank=rank,
                )
                for rank, candidate in enumerate(reranked, start=1)
            )
            row = PublicEvidenceRow(
                schema_version="public.evidence.v1",
                retrieval_run_id=retrieval_run_id,
                retrieval_fingerprint=retrieval_fingerprint,
                question_id=question.question_id,
                question_checksum=checksum_bytes(question.question.encode("utf-8")),
                question=question.question,
                evidence=evidence,
            )
            if checkpoint_path is not None:
                write_immutable_bytes(
                    checkpoint_path, content_json_bytes(row.model_dump(mode="json"))
                )
            rows.append(row)
            generated_count += 1
    except PublicDryRunError:
        raise
    except ImmutableArtifactError as error:
        raise PublicDryRunError(error.code, error.message) from error
    except (RerankerError, ValueError) as error:
        raise PublicDryRunError(
            "PUBLIC_RETRIEVAL_FAILED", "public retrieval or reranking failed"
        ) from error

    queue_data = b"".join(content_json_bytes(row.model_dump(mode="json")) for row in rows)
    report_data = content_json_bytes(
        {
            "schema_version": "public.evidence.report.v1",
            "retrieval_run_id": retrieval_run_id,
            "question_count": len(rows),
            "questions_without_evidence": sum(not row.evidence for row in rows),
            "evidence_limit": evidence_limit,
            "reranker_candidate_limit": reranker_candidate_limit,
            "retrieval_mode": retrieval_mode,
            "reranker_model_id": reranker_model_id,
            "reranker_model_revision": reranker_model_revision,
            "retrieval_fingerprint": retrieval_fingerprint,
            "frozen_inputs": frozen,
            "generated_question_count": generated_count,
            "resumed_question_count": resumed_count,
            "queue_checksum": checksum_bytes(queue_data),
        }
    )
    return PublicEvidenceArtifacts(queue_data, report_data)


def _frozen_inputs(value: Mapping[str, str]) -> dict[str, str]:
    ordered = dict(sorted(value.items(), key=lambda item: item[0].encode("utf-8")))
    if not ordered or any(
        _INPUT_NAME.fullmatch(name) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", checksum) is None
        for name, checksum in ordered.items()
    ):
        raise PublicDryRunError(
            "PUBLIC_RUN_FINGERPRINT_INVALID", "frozen inputs must be named SHA-256 values"
        )
    return ordered


def _checkpoint_path(directory: Path, question_id: str) -> Path:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _load_evidence_checkpoint(path: Path) -> PublicEvidenceRow:
    try:
        return parse_record_json(
            path.read_bytes(),
            PublicEvidenceRow,
            artifact_path="public.evidence.v1.json",
            record_identity=path.name,
        )
    except (OSError, RecordValidationError) as error:
        raise PublicDryRunError(
            "PUBLIC_EVIDENCE_CHECKPOINT_INVALID",
            "public evidence checkpoint is invalid",
        ) from error


def _load_checkpoint(path: Path) -> PublicAnswerCheckpoint:
    try:
        return parse_record_json(
            path.read_bytes(),
            PublicAnswerCheckpoint,
            artifact_path="public.answer.checkpoint.v1.json",
            record_identity=path.name,
        )
    except (OSError, RecordValidationError) as error:
        raise PublicDryRunError(
            "PUBLIC_CHECKPOINT_INVALID", "public generation checkpoint is invalid"
        ) from error


def run_checkpointed_public_generation(
    *,
    public_source_data: bytes,
    evidence_queue_data: bytes,
    backend: PublicGeneratorBackend,
    system_prompt: str,
    run_id: str,
    generator_id: str,
    checkpoint_directory: Path,
    maximum_input_tokens: int,
    maximum_new_tokens: int,
    frozen_inputs: Mapping[str, str],
) -> PublicGenerationArtifacts:
    """Generate all public answers locally and resume only checksum-identical checkpoints."""

    rows = _load_evidence_queue(evidence_queue_data)
    try:
        source_questions = (
            OrganizerQuestionReader()
            .read_bytes(
                public_source_data,
                kind="public",
                artifact_path="questions/source.json",
            )
            .records
        )
    except OrganizerDataError as error:
        raise PublicDryRunError("PUBLIC_SOURCE_INVALID", error.message) from error
    if tuple((row.question_id, row.question) for row in rows) != tuple(
        (question.question_id, question.question) for question in source_questions
    ):
        raise PublicDryRunError(
            "PUBLIC_QUESTION_SOURCE_MISMATCH",
            "public evidence questions differ from the immutable organizer source",
        )
    frozen = _frozen_inputs(frozen_inputs)
    if (
        not system_prompt.strip()
        or not run_id.strip()
        or not generator_id.strip()
        or maximum_input_tokens < 1
        or maximum_new_tokens < 1
    ):
        raise PublicDryRunError(
            "PUBLIC_GENERATION_CONFIG_INVALID", "public generation configuration is invalid"
        )
    fingerprint_value = {
        "schema_version": "public.run-fingerprint.v1",
        "run_id": run_id,
        "retrieval_run_id": rows[0].retrieval_run_id,
        "public_source_checksum": checksum_bytes(public_source_data),
        "evidence_queue_checksum": checksum_bytes(evidence_queue_data),
        "model_id": backend.model_id,
        "model_revision": backend.model_revision,
        "generator_id": generator_id,
        "prompt_checksum": checksum_bytes(system_prompt.encode("utf-8")),
        "maximum_input_tokens": maximum_input_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "do_sample": False,
        "enable_thinking": False,
        "frozen_inputs": frozen,
    }
    run_fingerprint = checksum_bytes(content_json_bytes(fingerprint_value))
    checkpoints: list[PublicAnswerCheckpoint] = []
    generated_count = 0
    resumed_count = 0
    invocation_started = time.perf_counter()
    for row in rows:
        evidence_ids = tuple(item.evidence_id for item in row.evidence)
        path = _checkpoint_path(checkpoint_directory, row.question_id)
        if path.exists():
            checkpoint = _load_checkpoint(path)
            if (
                checkpoint.run_id != run_id
                or checkpoint.run_fingerprint != run_fingerprint
                or checkpoint.question_id != row.question_id
                or checkpoint.question_checksum != row.question_checksum
                or checkpoint.evidence_ids != evidence_ids
                or checkpoint.generator_id != generator_id
            ):
                raise PublicDryRunError(
                    "PUBLIC_CHECKPOINT_MISMATCH",
                    "public generation checkpoint differs from frozen inputs",
                )
            resumed_count += 1
        else:
            started = time.perf_counter()
            answer = (
                backend.generate(
                    system_prompt=system_prompt,
                    question=row.question,
                    evidence=tuple(item.display_text for item in row.evidence),
                ).strip()
                if row.evidence
                else FIXED_REFUSAL
            )
            answer = unicodedata.normalize("NFC", answer)
            if not answer:
                raise PublicDryRunError(
                    "PUBLIC_GENERATION_EMPTY", "public generator returned an empty answer"
                )
            checkpoint = PublicAnswerCheckpoint(
                schema_version="public.answer.checkpoint.v1",
                run_id=run_id,
                run_fingerprint=run_fingerprint,
                question_id=row.question_id,
                question_checksum=row.question_checksum,
                evidence_ids=evidence_ids,
                answer=answer,
                generator_id=generator_id,
                elapsed_seconds=time.perf_counter() - started,
            )
            try:
                write_immutable_bytes(path, content_json_bytes(checkpoint.model_dump(mode="json")))
            except ImmutableArtifactError as error:
                raise PublicDryRunError(error.code, error.message) from error
            generated_count += 1
        checkpoints.append(checkpoint)

    answers = tuple(
        AnswerRecord.model_validate(
            {
                "schema_version": "internal.answer.v1",
                "question_id": checkpoint.question_id,
                "answer": checkpoint.answer,
                "generator_id": checkpoint.generator_id,
                "evidence_ids": checkpoint.evidence_ids,
                "run_id": "run_" + run_fingerprint.removeprefix("sha256:")[:24],
            }
        )
        for checkpoint in checkpoints
    )
    answers_data = answers_jsonl_bytes(answers)
    try:
        predictions_data = build_submission(public_source_data, answers)
    except SubmissionError as error:
        raise PublicDryRunError(error.code, error.message) from error
    manifest_data = content_json_bytes(
        {
            **fingerprint_value,
            "schema_version": "public.dry-run.manifest.v1",
            "run_fingerprint": run_fingerprint,
            "question_count": len(rows),
            "ordered_question_ids": [row.question_id for row in rows],
            "answers_checksum": checksum_bytes(answers_data),
            "predictions_checksum": checksum_bytes(predictions_data),
            "public_results_usage": "validation_only_not_fitting_feedback",
            "profile_state": "promoted_dry_run",
        }
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "public.dry-run.telemetry.v1",
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "question_count": len(rows),
            "generated_question_count": generated_count,
            "resumed_question_count": resumed_count,
            "checkpoint_elapsed_seconds": sum(item.elapsed_seconds for item in checkpoints),
            "invocation_elapsed_seconds": time.perf_counter() - invocation_started,
            "paid_service_used": False,
            "execution_mode": "local-offline",
        }
    )
    return PublicGenerationArtifacts(
        answers_data=answers_data,
        predictions_data=predictions_data,
        manifest_data=manifest_data,
        telemetry_data=telemetry_data,
        generated_question_count=generated_count,
        resumed_question_count=resumed_count,
    )


__all__ = [
    "PublicDryRunError",
    "PublicEvidenceArtifacts",
    "PublicEvidenceItem",
    "PublicEvidenceRow",
    "PublicGenerationArtifacts",
    "build_public_evidence_queue",
    "run_checkpointed_public_generation",
]
