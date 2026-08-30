from __future__ import annotations

import json

from legal_rag.evaluation.dense_discovery import (
    build_fixed_rrf_60_rankings,
    diagnose_sparse_dense,
    evaluate_sparse_dense_union,
    serialize_dense_diagnostics,
    serialize_union_evaluation,
)
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryGroup,
    DiscoveryRanking,
)


def _group(question_id: str, positives: tuple[str, ...]) -> DiscoveryGroup:
    return DiscoveryGroup(
        question_id=question_id,
        question_checksum="sha256:" + question_id[1:] * 64,
        question=f"Question {question_id}",
        source_answer_checksum="sha256:" + "a" * 64,
        gold_answer="Official answer.",
        positive_chunk_ids=positives,
    )


def _ranking(question_id: str, chunk_ids: tuple[str, ...]) -> DiscoveryRanking:
    return DiscoveryRanking(
        question_id,
        tuple(DiscoveryCandidate(chunk_id, f"Text {chunk_id}") for chunk_id in chunk_ids),
    )


def test_sparse_dense_diagnostics_classify_paired_recovery_and_rank_delta() -> None:
    groups = (
        _group("q1", ("a",)),
        _group("q2", ("b",)),
        _group("q3", ("c",)),
        _group("q4", ("d",)),
    )
    sparse = (
        _ranking("q1", ("a", "x")),
        _ranking("q2", ("x",)),
        _ranking("q3", ("x", "c")),
        _ranking("q4", ("x",)),
    )
    dense = (
        _ranking("q1", ("x",)),
        _ranking("q2", ("b", "x")),
        _ranking("q3", ("c", "x")),
        _ranking("q4", ("x",)),
    )

    report = diagnose_sparse_dense(groups, sparse, dense)

    assert report.classification_counts == {
        "SPARSE_ONLY": 1,
        "DENSE_ONLY": 1,
        "BOTH": 1,
        "NEITHER": 1,
    }
    assert report.dense_novel_positive_group_ids == ("q2",)
    assert report.sparse_only_positive_group_ids == ("q1",)
    q3 = next(row for row in report.rows if row.question_id == "q3")
    assert q3.first_positive_rank_delta_dense_minus_sparse == -1
    assert json.loads(serialize_dense_diagnostics(report))["schema_version"] == (
        "evaluation.sparse-dense-diagnostics.v1"
    )


def test_union_recovers_multi_positive_set_and_fixed_rrf_is_deterministic() -> None:
    groups = (_group("q5", ("a", "b")),)
    sparse = (_ranking("q5", ("a", "common", "sparse")),)
    dense = (_ranking("q5", ("b", "common", "dense")),)

    union = evaluate_sparse_dense_union(groups, sparse, dense)
    first = build_fixed_rrf_60_rankings(sparse, dense, limit=5)
    replay = build_fixed_rrf_60_rankings(sparse, dense, limit=5)

    assert union.recall_at[5] == 1.0
    assert union.evidence_set_recall_at[5] == 1.0
    assert union.positive_assignment_recall_at[5] == 1.0
    assert tuple(item.chunk_id for item in first[0].candidates) == (
        "common",
        "a",
        "b",
        "dense",
        "sparse",
    )
    assert first == replay
    assert serialize_union_evaluation(union) == serialize_union_evaluation(union)
