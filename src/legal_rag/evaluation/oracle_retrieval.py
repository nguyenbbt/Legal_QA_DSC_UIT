"""Bounded, non-promotable evidence-oracle selection contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

OracleRelevance = Literal["relevant", "partially_relevant", "not_relevant"]


@dataclass(frozen=True, slots=True)
class OracleCandidate:
    evidence_id: str
    relevance: OracleRelevance
    token_cost: int


@dataclass(frozen=True, slots=True)
class OracleSelection:
    selected_evidence_ids: tuple[str, ...]
    excluded_not_positive_ids: tuple[str, ...]
    excluded_budget_ids: tuple[str, ...]
    excluded_count_ids: tuple[str, ...]
    input_token_cost: int
    status: Literal["SELECTED", "UNRESOLVED_LABEL_OR_GOLD"]


def select_bounded_oracle_evidence(
    candidates: tuple[OracleCandidate, ...],
    *,
    maximum_evidence_count: int,
    maximum_input_tokens: int,
    input_token_cost: Callable[[tuple[str, ...]], int],
) -> OracleSelection:
    """Select positive evidence in source order under exact caller-supplied token cost."""

    if maximum_evidence_count < 1 or maximum_input_tokens < 1:
        raise ValueError("oracle evidence and token limits must be positive")
    evidence_ids = tuple(candidate.evidence_id for candidate in candidates)
    if any(not evidence_id for evidence_id in evidence_ids) or len(evidence_ids) != len(
        set(evidence_ids)
    ):
        raise ValueError("oracle candidate evidence IDs must be non-empty and unique")
    if any(candidate.token_cost < 0 for candidate in candidates):
        raise ValueError("oracle candidate token costs must be non-negative")
    selected: list[str] = []
    not_positive: list[str] = []
    over_budget: list[str] = []
    over_count: list[str] = []
    for candidate in candidates:
        if candidate.relevance == "not_relevant":
            not_positive.append(candidate.evidence_id)
            continue
        if len(selected) >= maximum_evidence_count:
            over_count.append(candidate.evidence_id)
            continue
        proposed = (*selected, candidate.evidence_id)
        if input_token_cost(proposed) > maximum_input_tokens:
            over_budget.append(candidate.evidence_id)
            continue
        selected.append(candidate.evidence_id)
    selected_ids = tuple(selected)
    return OracleSelection(
        selected_evidence_ids=selected_ids,
        excluded_not_positive_ids=tuple(not_positive),
        excluded_budget_ids=tuple(over_budget),
        excluded_count_ids=tuple(over_count),
        input_token_cost=input_token_cost(selected_ids),
        status="SELECTED" if selected_ids else "UNRESOLVED_LABEL_OR_GOLD",
    )


__all__ = ["OracleCandidate", "OracleSelection", "select_bounded_oracle_evidence"]
