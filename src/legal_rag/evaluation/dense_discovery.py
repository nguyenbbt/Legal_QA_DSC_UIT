"""Paired sparse/dense diagnostics and fixed D-066 discovery fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryGroup,
    DiscoveryRanking,
    DiscoveryTournamentError,
)

_CUTOFFS = (5, 10, 20, 50)
_CLASSIFICATIONS = ("SPARSE_ONLY", "DENSE_ONLY", "BOTH", "NEITHER")
DiscoveryClassification = Literal["SPARSE_ONLY", "DENSE_ONLY", "BOTH", "NEITHER"]


@dataclass(frozen=True, slots=True)
class SparseDenseQuestionDiagnostics:
    question_id: str
    classification: DiscoveryClassification
    sparse_first_positive_rank: int | None
    dense_first_positive_rank: int | None
    first_positive_rank_delta_dense_minus_sparse: int | None
    sparse_positive_hits_at: dict[int, int]
    dense_positive_hits_at: dict[int, int]
    overlap_positive_hits_at: dict[int, int]
    union_positive_hits_at: dict[int, int]
    sparse_evidence_set_at: dict[int, bool]
    dense_evidence_set_at: dict[int, bool]
    union_evidence_set_at: dict[int, bool]


@dataclass(frozen=True, slots=True)
class SparseDenseDiagnostics:
    schema_version: str
    question_count: int
    classification_counts: dict[str, int]
    dense_novel_positive_group_ids: tuple[str, ...]
    sparse_only_positive_group_ids: tuple[str, ...]
    both_positive_group_ids: tuple[str, ...]
    neither_positive_group_ids: tuple[str, ...]
    rows: tuple[SparseDenseQuestionDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class SparseDenseUnionEvaluation:
    schema_version: str
    question_count: int
    positive_assignment_count: int
    recall_at: dict[int, float]
    evidence_set_recall_at: dict[int, float]
    positive_assignment_recall_at: dict[int, float]
    mean_candidate_pool_size_at: dict[int, float]


def _paired(
    groups: Sequence[DiscoveryGroup],
    sparse: Sequence[DiscoveryRanking],
    dense: Sequence[DiscoveryRanking],
) -> tuple[tuple[DiscoveryGroup, DiscoveryRanking, DiscoveryRanking], ...]:
    ordered_groups = tuple(sorted(groups, key=lambda item: item.question_id.encode("utf-8")))
    sparse_by_id = _rankings_by_id(sparse)
    dense_by_id = _rankings_by_id(dense)
    expected = {group.question_id for group in ordered_groups}
    if not ordered_groups or set(sparse_by_id) != expected or set(dense_by_id) != expected:
        raise DiscoveryTournamentError(
            "D066_COMPARISON_ID_MISMATCH", "sparse and dense rankings must cover identical groups"
        )
    return tuple(
        (group, sparse_by_id[group.question_id], dense_by_id[group.question_id])
        for group in ordered_groups
    )


def _rankings_by_id(rankings: Sequence[DiscoveryRanking]) -> dict[str, DiscoveryRanking]:
    result: dict[str, DiscoveryRanking] = {}
    for ranking in rankings:
        ids = tuple(candidate.chunk_id for candidate in ranking.candidates)
        if (
            not ranking.question_id
            or ranking.question_id in result
            or any(not chunk_id for chunk_id in ids)
            or len(ids) != len(set(ids))
        ):
            raise DiscoveryTournamentError(
                "D066_RANKING_INVALID", "ranking identity or candidates are invalid"
            )
        result[ranking.question_id] = ranking
    return result


def _first_positive(candidate_ids: tuple[str, ...], positives: frozenset[str]) -> int | None:
    return next(
        (
            rank
            for rank, chunk_id in enumerate(candidate_ids[:50], start=1)
            if chunk_id in positives
        ),
        None,
    )


def diagnose_sparse_dense(
    groups: Sequence[DiscoveryGroup],
    sparse: Sequence[DiscoveryRanking],
    dense: Sequence[DiscoveryRanking],
) -> SparseDenseDiagnostics:
    """Classify complementary Top-50 recovery and preserve paired rank evidence."""

    rows: list[SparseDenseQuestionDiagnostics] = []
    by_class: dict[str, list[str]] = {name: [] for name in _CLASSIFICATIONS}
    for group, sparse_ranking, dense_ranking in _paired(groups, sparse, dense):
        positives = frozenset(group.positive_chunk_ids)
        sparse_ids = tuple(candidate.chunk_id for candidate in sparse_ranking.candidates)
        dense_ids = tuple(candidate.chunk_id for candidate in dense_ranking.candidates)
        sparse_first = _first_positive(sparse_ids, positives)
        dense_first = _first_positive(dense_ids, positives)
        if sparse_first is not None and dense_first is not None:
            classification: DiscoveryClassification = "BOTH"
        elif sparse_first is not None:
            classification = "SPARSE_ONLY"
        elif dense_first is not None:
            classification = "DENSE_ONLY"
        else:
            classification = "NEITHER"
        by_class[classification].append(group.question_id)
        sparse_hits_at: dict[int, int] = {}
        dense_hits_at: dict[int, int] = {}
        overlap_hits_at: dict[int, int] = {}
        union_hits_at: dict[int, int] = {}
        sparse_set_at: dict[int, bool] = {}
        dense_set_at: dict[int, bool] = {}
        union_set_at: dict[int, bool] = {}
        for cutoff in _CUTOFFS:
            sparse_hits = positives.intersection(sparse_ids[:cutoff])
            dense_hits = positives.intersection(dense_ids[:cutoff])
            union_hits = sparse_hits | dense_hits
            sparse_hits_at[cutoff] = len(sparse_hits)
            dense_hits_at[cutoff] = len(dense_hits)
            overlap_hits_at[cutoff] = len(sparse_hits & dense_hits)
            union_hits_at[cutoff] = len(union_hits)
            sparse_set_at[cutoff] = positives.issubset(sparse_ids[:cutoff])
            dense_set_at[cutoff] = positives.issubset(dense_ids[:cutoff])
            union_set_at[cutoff] = positives.issubset(union_hits)
        rows.append(
            SparseDenseQuestionDiagnostics(
                question_id=group.question_id,
                classification=classification,
                sparse_first_positive_rank=sparse_first,
                dense_first_positive_rank=dense_first,
                first_positive_rank_delta_dense_minus_sparse=(
                    dense_first - sparse_first
                    if dense_first is not None and sparse_first is not None
                    else None
                ),
                sparse_positive_hits_at=sparse_hits_at,
                dense_positive_hits_at=dense_hits_at,
                overlap_positive_hits_at=overlap_hits_at,
                union_positive_hits_at=union_hits_at,
                sparse_evidence_set_at=sparse_set_at,
                dense_evidence_set_at=dense_set_at,
                union_evidence_set_at=union_set_at,
            )
        )
    return SparseDenseDiagnostics(
        schema_version="evaluation.sparse-dense-diagnostics.v1",
        question_count=len(rows),
        classification_counts={name: len(by_class[name]) for name in _CLASSIFICATIONS},
        dense_novel_positive_group_ids=tuple(by_class["DENSE_ONLY"]),
        sparse_only_positive_group_ids=tuple(by_class["SPARSE_ONLY"]),
        both_positive_group_ids=tuple(by_class["BOTH"]),
        neither_positive_group_ids=tuple(by_class["NEITHER"]),
        rows=tuple(rows),
    )


def evaluate_sparse_dense_union(
    groups: Sequence[DiscoveryGroup],
    sparse: Sequence[DiscoveryRanking],
    dense: Sequence[DiscoveryRanking],
) -> SparseDenseUnionEvaluation:
    """Evaluate unordered Top-K-per-arm union coverage without inventing a rank."""

    paired = _paired(groups, sparse, dense)
    assignments = sum(len(group.positive_chunk_ids) for group, _, _ in paired)
    recall_totals = {cutoff: 0 for cutoff in _CUTOFFS}
    set_totals = {cutoff: 0 for cutoff in _CUTOFFS}
    hit_totals = {cutoff: 0 for cutoff in _CUTOFFS}
    pool_totals = {cutoff: 0 for cutoff in _CUTOFFS}
    for group, sparse_ranking, dense_ranking in paired:
        positives = frozenset(group.positive_chunk_ids)
        sparse_ids = tuple(candidate.chunk_id for candidate in sparse_ranking.candidates)
        dense_ids = tuple(candidate.chunk_id for candidate in dense_ranking.candidates)
        for cutoff in _CUTOFFS:
            pool = set(sparse_ids[:cutoff]) | set(dense_ids[:cutoff])
            hits = positives & pool
            recall_totals[cutoff] += int(bool(hits))
            set_totals[cutoff] += int(positives.issubset(pool))
            hit_totals[cutoff] += len(hits)
            pool_totals[cutoff] += len(pool)
    question_count = len(paired)
    return SparseDenseUnionEvaluation(
        schema_version="evaluation.sparse-dense-union.v1",
        question_count=question_count,
        positive_assignment_count=assignments,
        recall_at={cutoff: recall_totals[cutoff] / question_count for cutoff in _CUTOFFS},
        evidence_set_recall_at={cutoff: set_totals[cutoff] / question_count for cutoff in _CUTOFFS},
        positive_assignment_recall_at={
            cutoff: hit_totals[cutoff] / assignments for cutoff in _CUTOFFS
        },
        mean_candidate_pool_size_at={
            cutoff: pool_totals[cutoff] / question_count for cutoff in _CUTOFFS
        },
    )


def build_fixed_rrf_60_rankings(
    sparse: Sequence[DiscoveryRanking],
    dense: Sequence[DiscoveryRanking],
    *,
    limit: int = 50,
) -> tuple[DiscoveryRanking, ...]:
    """Fuse complete paired rankings with the frozen unweighted RRF constant 60."""

    if limit < 1:
        raise DiscoveryTournamentError("D066_RANKING_INVALID", "RRF limit must be positive")
    sparse_by_id = _rankings_by_id(sparse)
    dense_by_id = _rankings_by_id(dense)
    if set(sparse_by_id) != set(dense_by_id):
        raise DiscoveryTournamentError(
            "D066_COMPARISON_ID_MISMATCH", "RRF inputs must cover identical questions"
        )
    output: list[DiscoveryRanking] = []
    for question_id in sorted(sparse_by_id, key=str.encode):
        candidates: dict[str, DiscoveryCandidate] = {}
        scores: dict[str, float] = {}
        for ranking in (sparse_by_id[question_id], dense_by_id[question_id]):
            for rank, candidate in enumerate(ranking.candidates[:50], start=1):
                existing = candidates.get(candidate.chunk_id)
                if existing is not None and existing.display_text != candidate.display_text:
                    raise DiscoveryTournamentError(
                        "D066_RANKING_INVALID", "one RRF chunk has conflicting display text"
                    )
                candidates[candidate.chunk_id] = candidate
                scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (60 + rank)
        ranked_ids = sorted(scores, key=lambda key: (-scores[key], key.encode("utf-8")))[:limit]
        output.append(DiscoveryRanking(question_id, tuple(candidates[key] for key in ranked_ids)))
    return tuple(output)


def serialize_dense_diagnostics(value: SparseDenseDiagnostics) -> bytes:
    return content_json_bytes(asdict(value))


def serialize_union_evaluation(value: SparseDenseUnionEvaluation) -> bytes:
    return content_json_bytes(asdict(value))


__all__ = [
    "SparseDenseDiagnostics",
    "SparseDenseQuestionDiagnostics",
    "SparseDenseUnionEvaluation",
    "build_fixed_rrf_60_rankings",
    "diagnose_sparse_dense",
    "evaluate_sparse_dense_union",
    "serialize_dense_diagnostics",
    "serialize_union_evaluation",
]
