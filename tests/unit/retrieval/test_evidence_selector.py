from __future__ import annotations

import pytest

from legal_rag.retrieval.evidence_selector import (
    EvidenceSelectionCandidate,
    select_evidence_set,
)


def _candidate(
    evidence_id: str,
    *,
    path: tuple[str, ...],
    start: int,
    end: int,
    cost: int = 10,
    context: str = "ctx",
    integrity: str = "valid",
    role: str = "ranked",
    admits_complementary_context: bool = False,
    sparse_score: float | None = None,
) -> EvidenceSelectionCandidate:
    return EvidenceSelectionCandidate(
        evidence_id=evidence_id,
        context_id=context,
        hierarchy_path=path,
        canonical_start=start,
        canonical_end=end,
        token_cost=cost,
        integrity_status=integrity,
        selection_role=role,
        admits_complementary_context=admits_complementary_context,
        sparse_score=sparse_score,
    )


def test_selector_keeps_rank_order_and_complementary_sibling() -> None:
    candidates = (
        _candidate("a", path=("Điều 1", "Khoản 1"), start=0, end=100),
        _candidate("b", path=("Điều 1", "Khoản 2"), start=101, end=200),
        _candidate("c", path=("Điều 2",), start=201, end=300),
        _candidate("d", path=("Điều 3",), start=301, end=400),
    )

    result = select_evidence_set(candidates, maximum_input_tokens=30)

    assert result.selected_evidence_ids == ("a", "b", "c")
    assert result.schema_version == "evidence-set-selector.v2"
    assert result.input_token_cost == 30
    assert tuple(decision.reason for decision in result.decisions) == (
        "SELECT_PRIMARY",
        "SELECT_COMPLEMENTARY_COORDINATE",
        "SELECT_COMPLEMENTARY_COORDINATE",
        "SKIP_LOW_INCREMENTAL_COVERAGE",
    )


def test_selector_rejects_duplicate_span_overlap_and_parent_child() -> None:
    candidates = (
        _candidate("a", path=("Điều 1",), start=0, end=100),
        _candidate("span", path=("Điều X",), start=0, end=100),
        _candidate("overlap", path=("Điều 2",), start=10, end=90),
        _candidate("child", path=("Điều 1", "Khoản 1"), start=20, end=80),
        _candidate("good", path=("Điều 3",), start=101, end=150),
    )

    result = select_evidence_set(candidates, maximum_input_tokens=100)

    assert result.selected_evidence_ids == ("a", "good")
    assert tuple(decision.reason for decision in result.decisions) == (
        "SELECT_PRIMARY",
        "SKIP_DUPLICATE_SPAN",
        "SKIP_EXCESSIVE_OVERLAP",
        "SKIP_PARENT_CHILD_REDUNDANCY",
        "SELECT_COMPLEMENTARY_COORDINATE",
    )


def test_selector_rejects_duplicate_id_invalid_and_over_budget_at_exact_boundary() -> None:
    candidates = (
        _candidate("a", path=("Điều 1",), start=0, end=10, cost=10),
        _candidate("a", path=("Điều 2",), start=11, end=20, cost=1),
        _candidate("bad", path=("Điều 3",), start=21, end=30, integrity="quarantined"),
        _candidate("b", path=("Điều 4",), start=31, end=40, cost=10),
        _candidate("c", path=("Điều 5",), start=41, end=50, cost=1),
    )

    result = select_evidence_set(candidates, maximum_input_tokens=20)

    assert result.selected_evidence_ids == ("a", "b")
    assert tuple(decision.reason for decision in result.decisions) == (
        "SELECT_PRIMARY",
        "SKIP_DUPLICATE_ID",
        "SKIP_INVALID_EVIDENCE",
        "SELECT_COMPLEMENTARY_COORDINATE",
        "SKIP_TOKEN_BUDGET",
    )


def test_selector_is_byte_order_stable_for_ties_and_empty_input() -> None:
    candidates = (
        _candidate("z", path=("Điều 1",), start=0, end=10),
        _candidate("a", path=("Điều 2",), start=11, end=20),
    )
    assert select_evidence_set(candidates, maximum_input_tokens=20) == select_evidence_set(
        candidates, maximum_input_tokens=20
    )
    assert select_evidence_set((), maximum_input_tokens=20).selected_evidence_ids == ()


def test_selector_uses_exact_caller_token_cost_including_prompt_overhead() -> None:
    candidates = (
        _candidate("a", path=("Điều 1",), start=0, end=10, cost=1),
        _candidate("b", path=("Điều 2",), start=11, end=20, cost=1),
    )
    costs = {(): 8, ("a",): 12, ("a", "b"): 21}

    result = select_evidence_set(
        candidates,
        maximum_input_tokens=20,
        input_token_cost=lambda selected: costs[
            tuple(candidate.evidence_id for candidate in selected)
        ],
    )

    assert result.selected_evidence_ids == ("a",)
    assert result.input_token_cost == 12
    assert result.decisions[1].reason == "SKIP_TOKEN_BUDGET"


def test_selector_stops_when_a_fragment_adds_no_new_legal_coordinate() -> None:
    candidates = (
        _candidate("a", path=("Điều 1",), start=0, end=10),
        _candidate("same-coordinate", path=("Điều 1",), start=11, end=20),
    )

    result = select_evidence_set(candidates, maximum_input_tokens=100)

    assert result.selected_evidence_ids == ("a",)
    assert result.decisions[1].reason == "SKIP_LOW_INCREMENTAL_COVERAGE"


def test_selector_admits_only_explicit_complementary_parent_and_sibling_context() -> None:
    candidates = (
        _candidate("primary", path=("Điều 1", "Khoản 1"), start=10, end=20),
        _candidate(
            "parent",
            path=("Điều 1",),
            start=0,
            end=30,
            role="parent_context",
            admits_complementary_context=True,
        ),
        _candidate(
            "sibling",
            path=("Điều 1", "Khoản 2"),
            start=21,
            end=30,
            role="sibling_context",
            admits_complementary_context=True,
        ),
    )

    result = select_evidence_set(candidates, maximum_input_tokens=100)

    assert result.selected_evidence_ids == ("primary", "parent", "sibling")
    assert tuple(decision.reason for decision in result.decisions) == (
        "SELECT_PRIMARY",
        "SELECT_PARENT_CONTEXT",
        "SELECT_SIBLING_CONTEXT",
    )


def test_selector_applies_frozen_relative_score_after_preserving_primary() -> None:
    candidates = (
        _candidate("primary", path=("article-1",), start=0, end=10, sparse_score=10.0),
        _candidate("boundary", path=("article-2",), start=11, end=20, sparse_score=6.0),
        _candidate("low", path=("article-3",), start=21, end=30, sparse_score=5.9),
    )

    result = select_evidence_set(
        candidates,
        maximum_input_tokens=100,
        minimum_relative_sparse_score=0.6,
    )

    assert result.selected_evidence_ids == ("primary", "boundary")
    assert tuple(decision.reason for decision in result.decisions) == (
        "SELECT_PRIMARY",
        "SELECT_COMPLEMENTARY_COORDINATE",
        "SKIP_LOW_INCREMENTAL_COVERAGE",
    )


def test_selector_relative_score_treats_missing_primary_score_as_exact_only() -> None:
    candidates = (
        _candidate("exact", path=("article-1",), start=0, end=10),
        _candidate("sparse", path=("article-2",), start=11, end=20, sparse_score=8.0),
    )

    result = select_evidence_set(
        candidates,
        maximum_input_tokens=100,
        minimum_relative_sparse_score=0.5,
    )

    assert result.selected_evidence_ids == ("exact",)
    assert result.decisions[1].reason == "SKIP_LOW_INCREMENTAL_COVERAGE"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_selector_rejects_invalid_relative_score_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        select_evidence_set(
            (_candidate("a", path=("article-1",), start=0, end=10, sparse_score=1.0),),
            maximum_input_tokens=10,
            minimum_relative_sparse_score=threshold,
        )


def test_selector_rejects_invalid_contract_limits_and_candidate_span() -> None:
    with pytest.raises(ValueError):
        select_evidence_set((), maximum_input_tokens=0)
    with pytest.raises(ValueError):
        select_evidence_set((), maximum_input_tokens=1, maximum_overlap_ratio=1.1)
    with pytest.raises(ValueError):
        select_evidence_set(
            (_candidate("bad", path=("Điều 1",), start=1, end=1),),
            maximum_input_tokens=10,
        )
