from __future__ import annotations

from legal_rag.evaluation.oracle_retrieval import (
    OracleCandidate,
    select_bounded_oracle_evidence,
)


def test_oracle_selects_only_positive_labels_in_source_rank_order() -> None:
    candidates = (
        OracleCandidate("noise", "not_relevant", 10),
        OracleCandidate("partial", "partially_relevant", 20),
        OracleCandidate("gold", "relevant", 30),
        OracleCandidate("extra", "relevant", 40),
    )

    result = select_bounded_oracle_evidence(
        candidates,
        maximum_evidence_count=3,
        maximum_input_tokens=100,
        input_token_cost=lambda ids: sum(
            candidate.token_cost for candidate in candidates if candidate.evidence_id in ids
        ),
    )

    assert result.selected_evidence_ids == ("partial", "gold", "extra")
    assert result.excluded_not_positive_ids == ("noise",)
    assert result.excluded_budget_ids == ()
    assert result.input_token_cost == 90


def test_oracle_records_budget_and_count_exclusions_without_reordering() -> None:
    candidates = (
        OracleCandidate("a", "relevant", 40),
        OracleCandidate("b", "relevant", 70),
        OracleCandidate("c", "partially_relevant", 20),
        OracleCandidate("d", "relevant", 10),
    )

    result = select_bounded_oracle_evidence(
        candidates,
        maximum_evidence_count=2,
        maximum_input_tokens=60,
        input_token_cost=lambda ids: sum(
            candidate.token_cost for candidate in candidates if candidate.evidence_id in ids
        ),
    )

    assert result.selected_evidence_ids == ("a", "c")
    assert result.excluded_budget_ids == ("b",)
    assert result.excluded_count_ids == ("d",)
    assert result.input_token_cost == 60


def test_oracle_empty_positive_set_is_explicitly_unresolved() -> None:
    result = select_bounded_oracle_evidence(
        (OracleCandidate("a", "not_relevant", 1),),
        maximum_evidence_count=3,
        maximum_input_tokens=10,
        input_token_cost=lambda _: 0,
    )

    assert result.selected_evidence_ids == ()
    assert result.status == "UNRESOLVED_LABEL_OR_GOLD"
