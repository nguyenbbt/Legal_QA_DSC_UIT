"""Local official-artifact orchestration for R-008 reranker groups."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import NoReturn, Protocol

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import SplitError, load_split_manifest_rows
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.training.rag_sft import load_gold_questions
from legal_rag.training.reranker_dataset import (
    RerankerTrainingArtifacts,
    RerankerTrainingSeed,
    build_reranker_training_artifacts,
)

_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "question_id",
        "question_checksum",
        "evidence_ids",
        "evidence_checksums",
        "support_score",
        "support_policy_version",
    }
)


class RerankerChunkResolver(Protocol):
    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...


class LocalRerankerDatasetError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RerankerSeedBuildConfig:
    construction_version: str = "r008-reranker-groups.v1"
    support_policy_version: str = "official-reranker-plus-lexical.v1"
    minimum_support_score: float = 0.95
    maximum_negatives: int = 8
    maximum_context_chunks: int = 5000

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_support_score <= 1.0:
            raise ValueError("minimum support score must be within [0, 1]")
        if not 1 <= self.maximum_negatives <= 16:
            raise ValueError("maximum negatives must be within [1, 16]")
        if self.maximum_context_chunks < 1:
            raise ValueError("maximum context chunks must be positive")


@dataclass(frozen=True, slots=True)
class _Selection:
    question_id: str
    question_checksum: str
    evidence_ids: tuple[str, ...]
    evidence_checksums: tuple[str, ...]
    support_score: float


def _fail(code: str, message: str) -> NoReturn:
    raise LocalRerankerDatasetError(code, message)


def _load_selections(data: bytes, config: RerankerSeedBuildConfig) -> tuple[_Selection, ...]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        _fail("RERANKER_SELECTION_INVALID", "selection JSONL framing is invalid")
    selections: list[_Selection] = []
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LocalRerankerDatasetError(
                "RERANKER_SELECTION_INVALID", "selection JSONL is invalid"
            ) from error
        if (
            not isinstance(value, dict)
            or set(value) != _SELECTION_FIELDS
            or value.get("schema_version") != "training.evidence.selection.v1"
            or content_json_bytes(value) != line
        ):
            _fail("RERANKER_SELECTION_INVALID", "selection row schema is invalid")
        question_id = value.get("question_id")
        question_checksum = value.get("question_checksum")
        evidence_ids = value.get("evidence_ids")
        evidence_checksums = value.get("evidence_checksums")
        support_score = value.get("support_score")
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or not isinstance(question_checksum, str)
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(not isinstance(item, str) or not item for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))
            or not isinstance(evidence_checksums, list)
            or len(evidence_checksums) != len(evidence_ids)
            or any(not isinstance(item, str) for item in evidence_checksums)
            or isinstance(support_score, bool)
            or not isinstance(support_score, (int, float))
            or not math.isfinite(float(support_score))
            or value.get("support_policy_version") != config.support_policy_version
        ):
            _fail("RERANKER_SELECTION_INVALID", "selection row value is invalid")
        if float(support_score) < config.minimum_support_score:
            _fail("RERANKER_SELECTION_UNSUPPORTED", "selection is below the support gate")
        selections.append(
            _Selection(
                question_id=question_id,
                question_checksum=question_checksum,
                evidence_ids=tuple(evidence_ids),
                evidence_checksums=tuple(evidence_checksums),
                support_score=float(support_score),
            )
        )
    ids = tuple(selection.question_id for selection in selections)
    if len(ids) != len(set(ids)):
        _fail("RERANKER_SELECTION_DUPLICATE", "selection question IDs must be unique")
    return tuple(selections)


def build_reranker_dataset_from_selections(
    *,
    question_data: bytes,
    split_manifest_data: bytes,
    selection_data: bytes,
    chunks: RerankerChunkResolver,
    chunks_checksum: str,
    index_checksum: str,
    config: RerankerSeedBuildConfig,
) -> RerankerTrainingArtifacts:
    """Convert approved positive mappings into closed answer-free training groups."""

    questions = load_gold_questions(question_data)
    question_source_checksum = checksum_bytes(question_data)
    try:
        split_rows = load_split_manifest_rows(
            split_manifest_data,
            expected_source_checksum=question_source_checksum,
            expected_question_ids=tuple(question.question_id for question in questions),
        )
    except SplitError as error:
        raise LocalRerankerDatasetError(
            "RERANKER_SPLIT_MANIFEST_INVALID",
            "split manifest does not match official questions",
        ) from error
    questions_by_id = {question.question_id: question for question in questions}
    split_by_id = {row.question_id: row.split for row in split_rows}
    selections = _load_selections(selection_data, config)

    seeds: list[RerankerTrainingSeed] = []
    for selection in selections:
        question = questions_by_id.get(selection.question_id)
        if question is None:
            _fail("RERANKER_SELECTION_QUESTION_UNKNOWN", "selection question is unknown")
        if selection.question_checksum != checksum_bytes(question.question.encode()):
            _fail("RERANKER_SELECTION_QUESTION_MISMATCH", "selection question changed")
        positives = chunks.chunks_by_ids(selection.evidence_ids)
        if (
            tuple(chunk.chunk_id for chunk in positives) != selection.evidence_ids
            or tuple(chunk.chunk_checksum for chunk in positives) != selection.evidence_checksums
        ):
            _fail("RERANKER_SELECTION_EVIDENCE_MISMATCH", "selection evidence changed")
        candidates_by_id: dict[str, ChunkRecord] = {}
        for context_id in sorted({chunk.context_id for chunk in positives}):
            context_chunks = chunks.chunks_for_context(context_id)
            if len(context_chunks) > config.maximum_context_chunks:
                _fail(
                    "RERANKER_CONTEXT_CANDIDATE_LIMIT",
                    "positive context exceeds the deterministic candidate bound",
                )
            for chunk in context_chunks:
                candidates_by_id[chunk.chunk_id] = chunk
        seeds.append(
            RerankerTrainingSeed(
                question_id=question.question_id,
                question=question.question,
                split=split_by_id[question.question_id],
                positives=positives,
                candidate_pool=tuple(
                    candidates_by_id[key] for key in sorted(candidates_by_id, key=str.encode)
                ),
            )
        )
    return build_reranker_training_artifacts(
        seeds=tuple(seeds),
        question_source_checksum=question_source_checksum,
        split_manifest_checksum=checksum_bytes(split_manifest_data),
        selection_checksum=checksum_bytes(selection_data),
        chunks_checksum=chunks_checksum,
        index_checksum=index_checksum,
        construction_version=config.construction_version,
        maximum_negatives=config.maximum_negatives,
    )


__all__ = [
    "LocalRerankerDatasetError",
    "RerankerChunkResolver",
    "RerankerSeedBuildConfig",
    "build_reranker_dataset_from_selections",
]
