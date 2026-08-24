"""Fail-closed construction of local official-train RAG-SFT artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, Protocol, TypeGuard

from pydantic import ValidationError

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes, content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.evaluation.split import (
    SplitError,
    load_split_manifest_rows,
    load_split_questions_jsonl,
)
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.tokenizer import retrieval_token_values
from legal_rag.training.dataset_policy import DatasetPolicyError, validate_training_dataset
from legal_rag.training.provenance import TrainingExample


class RagSftBuildError(Exception):
    """Stable safe failure at the RAG-SFT construction boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ChunkResolver(Protocol):
    """Minimum local index interface needed to resolve approved evidence."""

    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class RagSftBuildConfig:
    """Checksummed policy inputs controlling deterministic dataset construction."""

    construction_version: str
    support_policy_checksum: str
    chunks_checksum: str
    index_checksum: str
    minimum_support_score: float
    minimum_answer_token_coverage: float
    support_policy_version: str = "official-reranker-plus-lexical.v1"

    def __post_init__(self) -> None:
        if not self.construction_version or not self.support_policy_version:
            raise ValueError("construction and support policy versions must be non-empty")
        for checksum in (
            self.support_policy_checksum,
            self.chunks_checksum,
            self.index_checksum,
        ):
            if not _is_checksum(checksum):
                raise ValueError("RAG-SFT configuration checksums must be typed SHA-256 values")
        for threshold in (self.minimum_support_score, self.minimum_answer_token_coverage):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("RAG-SFT thresholds must be finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class RagSftArtifacts:
    """Exact private material, public-safe provenance, and build manifest bytes."""

    provenance_data: bytes
    material_data: bytes
    manifest_data: bytes


@dataclass(frozen=True, slots=True)
class _Selection:
    question_id: str
    question_checksum: str
    evidence_ids: tuple[str, ...]
    evidence_checksums: tuple[str, ...]
    support_score: float
    support_policy_version: str


_SELECTION_FIELDS = {
    "schema_version",
    "question_id",
    "question_checksum",
    "evidence_ids",
    "evidence_checksums",
    "support_score",
    "support_policy_version",
}


def _fail(code: str, message: str) -> NoReturn:
    raise RagSftBuildError(code, message)


def _is_checksum(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def load_gold_questions(data: bytes) -> tuple[QuestionRecord, ...]:
    try:
        ordered = load_split_questions_jsonl(data, expected_answer_state="gold")
        parsed = tuple(
            parse_record_json(line, QuestionRecord, artifact_path="train.questions.jsonl")
            for line in data.splitlines(keepends=True)
        )
    except (SplitError, RecordValidationError) as error:
        raise RagSftBuildError(
            "RAG_SFT_QUESTION_ARTIFACT_INVALID", "official train question artifact is invalid"
        ) from error
    by_id = {question.question_id: question for question in parsed}
    if len(by_id) != len(parsed):
        _fail("RAG_SFT_QUESTION_ARTIFACT_INVALID", "official train question IDs are not unique")
    return tuple(by_id[item.question_id] for item in ordered)


def _selection_from_value(value: object) -> _Selection:
    if not isinstance(value, dict) or set(value) != _SELECTION_FIELDS:
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection row shape is invalid")
    if value.get("schema_version") != "training.evidence.selection.v1":
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection schema is invalid")
    question_id = value.get("question_id")
    question_checksum = value.get("question_checksum")
    raw_ids = value.get("evidence_ids")
    raw_checksums = value.get("evidence_checksums")
    score = value.get("support_score")
    policy_version = value.get("support_policy_version")
    if (
        not isinstance(question_id, str)
        or not question_id
        or not _is_checksum(question_checksum)
        or not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > 3
        or not all(isinstance(item, str) and item for item in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or not isinstance(raw_checksums, list)
        or len(raw_checksums) != len(raw_ids)
        or not all(_is_checksum(item) for item in raw_checksums)
        or not isinstance(score, float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        or not isinstance(policy_version, str)
        or not policy_version
    ):
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection row values are invalid")
    return _Selection(
        question_id,
        question_checksum,
        tuple(raw_ids),
        tuple(raw_checksums),
        score,
        policy_version,
    )


def _load_selections(data: bytes) -> tuple[_Selection, ...]:
    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection JSONL framing is invalid")
    selections: list[_Selection] = []
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
            if content_json_bytes(value) != line:
                _fail("RAG_SFT_SELECTION_INVALID", "evidence selection row is not canonical")
        except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as error:
            raise RagSftBuildError(
                "RAG_SFT_SELECTION_INVALID", "evidence selection row is invalid JSON"
            ) from error
        selections.append(_selection_from_value(value))
    question_ids = [selection.question_id for selection in selections]
    if len(question_ids) != len(set(question_ids)):
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection question IDs are not unique")
    if question_ids != sorted(question_ids, key=str.encode):
        _fail("RAG_SFT_SELECTION_INVALID", "evidence selection rows are not bytewise ordered")
    return tuple(selections)


def answer_token_coverage(answer: str, evidence: tuple[ChunkRecord, ...]) -> float:
    answer_tokens = {
        token.casefold()
        for token in retrieval_token_values(answer)
        if any(character.isalnum() for character in token)
    }
    if not answer_tokens:
        return 0.0
    evidence_tokens = {
        token.casefold()
        for chunk in evidence
        for token in retrieval_token_values(chunk.display_text)
        if any(character.isalnum() for character in token)
    }
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _example_id(construction_version: str, question_id: str, evidence_ids: tuple[str, ...]) -> str:
    identity = "\n".join((construction_version, question_id, *evidence_ids)).encode()
    return "sft_" + hashlib.sha256(identity).hexdigest()[:24]


def _resolve_evidence(selection: _Selection, chunks: ChunkResolver) -> tuple[ChunkRecord, ...]:
    resolved = chunks.chunks_by_ids(selection.evidence_ids)
    by_id = {chunk.chunk_id: chunk for chunk in resolved}
    if len(by_id) != len(selection.evidence_ids) or set(by_id) != set(selection.evidence_ids):
        _fail("RAG_SFT_EVIDENCE_UNKNOWN", "selected evidence is absent from the frozen chunk index")
    ordered = tuple(by_id[evidence_id] for evidence_id in selection.evidence_ids)
    if tuple(chunk.chunk_checksum for chunk in ordered) != selection.evidence_checksums:
        _fail("RAG_SFT_EVIDENCE_CHECKSUM_MISMATCH", "selected evidence checksum changed")
    return ordered


def build_rag_sft_dataset(
    *,
    question_data: bytes,
    split_manifest_data: bytes,
    selection_data: bytes,
    chunks: ChunkResolver,
    config: RagSftBuildConfig,
) -> RagSftArtifacts:
    """Build deterministic local RAG-SFT bytes or reject the complete artifact."""

    questions = load_gold_questions(question_data)
    question_checksum = checksum_bytes(question_data)
    question_ids = tuple(question.question_id for question in questions)
    try:
        split_rows = load_split_manifest_rows(
            split_manifest_data,
            expected_source_checksum=question_checksum,
            expected_question_ids=question_ids,
        )
    except SplitError as error:
        raise RagSftBuildError(
            "RAG_SFT_SPLIT_MANIFEST_INVALID", "split manifest does not match train questions"
        ) from error
    split_by_id = {row.question_id: row.split for row in split_rows}
    questions_by_id = {question.question_id: question for question in questions}
    selections = _load_selections(selection_data)

    examples: list[TrainingExample] = []
    materials: list[Mapping[str, object]] = []
    coverage_values: list[float] = []
    for selection in selections:
        question = questions_by_id.get(selection.question_id)
        if question is None:
            _fail("RAG_SFT_QUESTION_UNKNOWN", "evidence selection references an unknown question")
        if split_by_id[selection.question_id] != "train":
            _fail("RAG_SFT_SPLIT_REJECTED", "RAG-SFT examples must belong to the train split")
        if selection.support_policy_version != config.support_policy_version:
            _fail("RAG_SFT_SUPPORT_POLICY_MISMATCH", "support policy version changed")
        if selection.question_checksum != checksum_bytes(question.question.encode()):
            _fail("RAG_SFT_QUESTION_CHECKSUM_MISMATCH", "selected question text changed")
        if selection.support_score < config.minimum_support_score:
            _fail("RAG_SFT_SUPPORT_REJECTED", "selected evidence is below the support threshold")
        evidence = _resolve_evidence(selection, chunks)
        answer = question.answer
        if answer is None:
            _fail("RAG_SFT_TARGET_MISSING", "official train answer is missing")
        coverage = answer_token_coverage(answer, evidence)
        if coverage < config.minimum_answer_token_coverage:
            _fail("RAG_SFT_COVERAGE_REJECTED", "selected evidence has insufficient answer coverage")
        coverage_values.append(coverage)
        example_id = _example_id(
            config.construction_version, question.question_id, selection.evidence_ids
        )
        examples.append(
            TrainingExample.model_validate(
                {
                    "schema_version": "training.example.v1",
                    "example_id": example_id,
                    "task": "generation",
                    "question_id": question.question_id,
                    "split": "train",
                    "question_source_checksum": question_checksum,
                    "evidence_ids": selection.evidence_ids,
                    "target_source": "official_train_answer",
                    "target_checksum": checksum_bytes(answer.encode()),
                    "contains_generated_text": False,
                    "construction_version": config.construction_version,
                }
            )
        )
        materials.append(
            {
                "schema_version": "training.rag_sft.material.v1",
                "example_id": example_id,
                "question_id": question.question_id,
                "question": question.question,
                "evidence": [
                    {
                        "evidence_id": chunk.chunk_id,
                        "evidence_checksum": chunk.chunk_checksum,
                        "display_text": chunk.display_text,
                    }
                    for chunk in evidence
                ],
                "target": answer,
            }
        )

    try:
        report = validate_training_dataset(tuple(examples))
    except (DatasetPolicyError, ValidationError) as error:
        raise RagSftBuildError(
            "RAG_SFT_PROVENANCE_INVALID", "constructed provenance failed dataset policy"
        ) from error
    provenance_data = b"".join(
        content_json_bytes(example.model_dump(mode="json")) for example in examples
    )
    material_data = b"".join(content_json_bytes(material) for material in materials)
    manifest_data = canonical_json_bytes(
        {
            "schema_version": "training.rag_sft.manifest.v1",
            "construction_version": config.construction_version,
            "accepted_rows": report.accepted_rows,
            "unique_question_ids": report.unique_question_ids,
            "unique_evidence_ids": report.unique_evidence_ids,
            "question_source_checksum": question_checksum,
            "split_manifest_checksum": checksum_bytes(split_manifest_data),
            "selection_checksum": checksum_bytes(selection_data),
            "support_policy_version": config.support_policy_version,
            "support_policy_checksum": config.support_policy_checksum,
            "chunks_checksum": config.chunks_checksum,
            "index_checksum": config.index_checksum,
            "minimum_support_score": format(config.minimum_support_score, ".17g"),
            "minimum_answer_token_coverage": format(config.minimum_answer_token_coverage, ".17g"),
            "minimum_observed_answer_token_coverage": format(min(coverage_values), ".17g"),
            "provenance_checksum": checksum_bytes(provenance_data),
            "material_checksum": checksum_bytes(material_data),
            "contains_generated_text": False,
        }
    )
    return RagSftArtifacts(provenance_data, material_data, manifest_data)


__all__ = [
    "ChunkResolver",
    "RagSftArtifacts",
    "RagSftBuildConfig",
    "RagSftBuildError",
    "answer_token_coverage",
    "build_rag_sft_dataset",
    "load_gold_questions",
]
