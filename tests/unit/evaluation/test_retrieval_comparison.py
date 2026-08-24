from __future__ import annotations

import json

import pytest

from legal_rag.evaluation.retrieval_comparison import (
    RetrievalComparisonError,
    compare_retrieval_experiments,
)


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in rows
    )


def _labels() -> bytes:
    return _jsonl(
        [
            {
                "question_id": question_id,
                "relevant_evidence": [
                    {"evidence_id": f"{question_id}-relevant", "relevance": "relevant"}
                ],
            }
            for question_id in ("q1", "q2", "q3")
        ]
    )


def _output(*, candidate: bool) -> bytes:
    rows: list[dict[str, object]] = []
    for question_id in ("q1", "q2", "q3"):
        relevant = f"{question_id}-relevant"
        irrelevant = [f"{question_id}-n{index}" for index in range(10)]
        ids = [relevant, *irrelevant] if candidate else [*irrelevant, relevant]
        rows.append(
            {
                "question_id": question_id,
                "candidates": [{"evidence_id": evidence_id} for evidence_id in ids],
            }
        )
    return _jsonl(rows)


def test_retrieval_comparison_promotes_a_fixed_candidate_with_positive_recall_ci() -> None:
    rendered = compare_retrieval_experiments(
        grounding_benchmark_data=_labels(),
        baseline_output_data=_output(candidate=False),
        candidate_output_data=_output(candidate=True),
        baseline_run_id="R0-fixture",
        candidate_run_id="R2-fixture",
    )

    value = json.loads(rendered)
    assert value["fixed_candidate_universe_verified"] is True
    assert value["metrics"]["recall_at_10_mean_delta"] == 1.0
    assert value["metrics"]["recall_at_10_ci95"] == [1.0, 1.0]
    assert value["promotion_state"] == "promoted"
    assert value["promotion_blockers"] == []


def test_retrieval_comparison_rejects_a_changed_candidate_universe() -> None:
    candidate = _output(candidate=True).replace(b"q1-n9", b"q1-new")

    with pytest.raises(RetrievalComparisonError) as caught:
        compare_retrieval_experiments(
            grounding_benchmark_data=_labels(),
            baseline_output_data=_output(candidate=False),
            candidate_output_data=candidate,
            baseline_run_id="R0-fixture",
            candidate_run_id="R2-fixture",
        )

    assert caught.value.code == "RETRIEVAL_COMPARISON_NOT_FIXED"
