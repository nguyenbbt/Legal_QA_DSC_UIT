"""Deterministic official-train pairwise groups for R-008 reranker LoRA."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.tokenizer import retrieval_tokens
from legal_rag.training.dataset_policy import DatasetPolicyError, validate_training_dataset
from legal_rag.training.provenance import TrainingExample

NegativeType = Literal[
    "SAME_DOCUMENT_WRONG_ARTICLE",
    "SAME_ARTICLE_WRONG_CLAUSE",
    "SAME_CLAUSE_WRONG_POINT",
    "ADJACENT_COORDINATE",
]


class RerankerDatasetError(Exception):
    """Stable fail-closed error for R-008 training material."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RerankerTrainingSeed:
    """One high-confidence official-train positive mapping and candidate universe."""

    question_id: str
    question: str
    split: str
    positives: tuple[ChunkRecord, ...]
    candidate_pool: tuple[ChunkRecord, ...]


@dataclass(frozen=True, slots=True)
class RerankerTrainingArtifacts:
    groups_data: bytes
    provenance_data: bytes
    manifest_data: bytes
    group_count: int
    pair_count: int


def _fail(code: str, message: str) -> None:
    raise RerankerDatasetError(code, message)


def _coordinate(path: tuple[str, ...], prefix: str) -> str | None:
    folded_prefix = prefix.casefold()
    for item in path:
        if item.casefold().startswith(folded_prefix):
            return item.casefold()
    return None


def _negative_type(positive: ChunkRecord, candidate: ChunkRecord) -> NegativeType | None:
    if positive.context_id != candidate.context_id:
        return None
    if max(positive.canonical_start, candidate.canonical_start) < min(
        positive.canonical_end, candidate.canonical_end
    ):
        return None
    if positive.hierarchy_path == candidate.hierarchy_path:
        return None

    positive_article = _coordinate(positive.hierarchy_path, "Điều")
    candidate_article = _coordinate(candidate.hierarchy_path, "Điều")
    if positive_article is not None and candidate_article is not None:
        if positive_article != candidate_article:
            return "SAME_DOCUMENT_WRONG_ARTICLE"
        positive_clause = _coordinate(positive.hierarchy_path, "Khoản")
        candidate_clause = _coordinate(candidate.hierarchy_path, "Khoản")
        if positive_clause is not None and candidate_clause is not None:
            if positive_clause != candidate_clause:
                return "SAME_ARTICLE_WRONG_CLAUSE"
            positive_point = _coordinate(positive.hierarchy_path, "Điểm")
            candidate_point = _coordinate(candidate.hierarchy_path, "Điểm")
            if (
                positive_point is not None
                and candidate_point is not None
                and positive_point != candidate_point
            ):
                return "SAME_CLAUSE_WRONG_POINT"
    return "ADJACENT_COORDINATE"


def _passage(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "evidence_id": chunk.chunk_id,
        "context_id": chunk.context_id,
        "evidence_checksum": chunk.chunk_checksum,
        "hierarchy_path": list(chunk.hierarchy_path),
        "canonical_start": chunk.canonical_start,
        "canonical_end": chunk.canonical_end,
        "text": chunk.retrieval_text,
    }


def _group_id(construction_version: str, question_id: str) -> str:
    digest = hashlib.sha256(f"{construction_version}\n{question_id}".encode()).hexdigest()
    return f"r008_group_{digest[:24]}"


def _example_id(construction_version: str, group_id: str) -> str:
    digest = hashlib.sha256(f"{construction_version}\n{group_id}".encode()).hexdigest()
    return f"r008_example_{digest[:24]}"


def _select_negatives(
    seed: RerankerTrainingSeed, *, maximum_negatives: int
) -> tuple[tuple[ChunkRecord, NegativeType], ...]:
    positive_ids = {chunk.chunk_id for chunk in seed.positives}
    query_terms = set(retrieval_tokens(seed.question))
    admitted: dict[str, tuple[ChunkRecord, NegativeType, int, int]] = {}
    priority = {
        "SAME_CLAUSE_WRONG_POINT": 0,
        "SAME_ARTICLE_WRONG_CLAUSE": 1,
        "SAME_DOCUMENT_WRONG_ARTICLE": 2,
        "ADJACENT_COORDINATE": 3,
    }
    for candidate in seed.candidate_pool:
        if candidate.chunk_id in positive_ids:
            continue
        classifications = tuple(
            value
            for positive in seed.positives
            if (value := _negative_type(positive, candidate)) is not None
        )
        if not classifications:
            continue
        classification = min(classifications, key=lambda item: priority[item])
        lexical_overlap = len(query_terms & set(retrieval_tokens(candidate.retrieval_text)))
        distance = min(
            abs(candidate.canonical_start - positive.canonical_start) for positive in seed.positives
        )
        admitted[candidate.chunk_id] = (
            candidate,
            classification,
            lexical_overlap,
            distance,
        )
    ordered = sorted(
        admitted.values(),
        key=lambda item: (
            priority[item[1]],
            -item[2],
            item[3],
            item[0].chunk_id.encode(),
        ),
    )[:maximum_negatives]
    return tuple((chunk, classification) for chunk, classification, _, _ in ordered)


def build_reranker_training_artifacts(
    *,
    seeds: tuple[RerankerTrainingSeed, ...],
    question_source_checksum: str,
    split_manifest_checksum: str,
    selection_checksum: str,
    chunks_checksum: str,
    index_checksum: str,
    construction_version: str,
    maximum_negatives: int,
) -> RerankerTrainingArtifacts:
    """Build immutable pairwise groups without answers or generated training text."""

    if not seeds or not construction_version.strip():
        _fail("RERANKER_TRAIN_DATASET_EMPTY", "reranker training seeds are empty")
    if maximum_negatives < 1 or maximum_negatives > 16:
        _fail(
            "RERANKER_TRAIN_NEGATIVE_LIMIT_INVALID",
            "maximum negatives must be within [1, 16]",
        )
    question_ids = tuple(seed.question_id for seed in seeds)
    if len(question_ids) != len(set(question_ids)):
        _fail("RERANKER_TRAIN_QUESTION_DUPLICATE", "training question IDs must be unique")

    groups: list[bytes] = []
    examples: list[TrainingExample] = []
    pair_count = 0
    rejected_no_negative = 0
    for seed in sorted(seeds, key=lambda item: item.question_id.encode()):
        if seed.split != "train":
            _fail(
                "RERANKER_TRAIN_SPLIT_REJECTED",
                "reranker training examples must use the train split",
            )
        if not seed.question_id.strip() or not seed.question.strip() or not seed.positives:
            _fail("RERANKER_TRAIN_SEED_INVALID", "reranker training seed is incomplete")
        positive_ids = tuple(chunk.chunk_id for chunk in seed.positives)
        if len(positive_ids) != len(set(positive_ids)):
            _fail("RERANKER_TRAIN_EVIDENCE_DUPLICATE", "positive evidence IDs must be unique")
        negatives = _select_negatives(seed, maximum_negatives=maximum_negatives)
        if not negatives:
            rejected_no_negative += 1
            continue

        group_id = _group_id(construction_version, seed.question_id)
        negative_targets = [
            {
                "evidence_id": chunk.chunk_id,
                "relevance": "not_relevant",
                "negative_type": classification,
            }
            for chunk, classification in negatives
        ]
        target = {
            "positives": [
                {"evidence_id": chunk.chunk_id, "relevance": "relevant"} for chunk in seed.positives
            ],
            "negatives": negative_targets,
        }
        target_checksum = checksum_bytes(content_json_bytes(target))
        evidence_ids = (*positive_ids, *(chunk.chunk_id for chunk, _ in negatives))
        example = TrainingExample.model_validate(
            {
                "schema_version": "training.example.v1",
                "example_id": _example_id(construction_version, group_id),
                "task": "reranking",
                "question_id": seed.question_id,
                "split": "train",
                "question_source_checksum": question_source_checksum,
                "evidence_ids": evidence_ids,
                "target_source": "deterministic_relevance",
                "target_checksum": target_checksum,
                "contains_generated_text": False,
                "construction_version": construction_version,
            }
        )
        examples.append(example)
        pair_count += len(seed.positives) * len(negatives)
        groups.append(
            content_json_bytes(
                {
                    "schema_version": "reranker.training-group.v1",
                    "group_id": group_id,
                    "question_id": seed.question_id,
                    "split": "train",
                    "question": seed.question,
                    "question_checksum": checksum_bytes(seed.question.encode()),
                    "positives": [_passage(chunk) for chunk in seed.positives],
                    "negatives": [
                        {**_passage(chunk), "negative_type": classification}
                        for chunk, classification in negatives
                    ],
                    "target_checksum": target_checksum,
                    "construction_version": construction_version,
                    "contains_generated_text": False,
                }
            )
        )

    if not groups:
        _fail(
            "RERANKER_TRAIN_DATASET_EMPTY",
            "no deterministic official-train reranker group has a hard negative",
        )
    try:
        policy = validate_training_dataset(tuple(examples))
    except DatasetPolicyError as error:
        raise RerankerDatasetError(
            "RERANKER_TRAIN_PROVENANCE_INVALID",
            "reranker training provenance failed policy validation",
        ) from error
    groups_data = b"".join(groups)
    provenance_data = b"".join(
        content_json_bytes(example.model_dump(mode="json")) for example in examples
    )
    manifest_data = content_json_bytes(
        {
            "schema_version": "reranker.training-manifest.v1",
            "construction_version": construction_version,
            "group_count": len(groups),
            "pair_count": pair_count,
            "rejected_no_negative_count": rejected_no_negative,
            "unique_question_ids": policy.unique_question_ids,
            "unique_evidence_ids": policy.unique_evidence_ids,
            "maximum_negatives": maximum_negatives,
            "question_source_checksum": question_source_checksum,
            "split_manifest_checksum": split_manifest_checksum,
            "selection_checksum": selection_checksum,
            "chunks_checksum": chunks_checksum,
            "index_checksum": index_checksum,
            "groups_checksum": checksum_bytes(groups_data),
            "provenance_checksum": checksum_bytes(provenance_data),
            "contains_generated_text": False,
        }
    )
    return RerankerTrainingArtifacts(
        groups_data=groups_data,
        provenance_data=provenance_data,
        manifest_data=manifest_data,
        group_count=len(groups),
        pair_count=pair_count,
    )


__all__ = [
    "NegativeType",
    "RerankerDatasetError",
    "RerankerTrainingArtifacts",
    "RerankerTrainingSeed",
    "build_reranker_training_artifacts",
]
