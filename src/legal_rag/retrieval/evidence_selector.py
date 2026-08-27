"""Provider-neutral deterministic selection of canonical evidence sets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

SelectionReason = Literal[
    "SELECT_PRIMARY",
    "SELECT_COMPLEMENTARY_COORDINATE",
    "SELECT_PARENT_CONTEXT",
    "SELECT_SIBLING_CONTEXT",
    "SKIP_DUPLICATE_ID",
    "SKIP_DUPLICATE_SPAN",
    "SKIP_EXCESSIVE_OVERLAP",
    "SKIP_PARENT_CHILD_REDUNDANCY",
    "SKIP_LOW_INCREMENTAL_COVERAGE",
    "SKIP_TOKEN_BUDGET",
    "SKIP_INVALID_EVIDENCE",
]
SelectionRole = Literal["ranked", "parent_context", "sibling_context"]


@dataclass(frozen=True, slots=True)
class EvidenceSelectionCandidate:
    evidence_id: str
    context_id: str
    hierarchy_path: tuple[str, ...]
    canonical_start: int
    canonical_end: int
    token_cost: int
    integrity_status: str = "valid"
    selection_role: SelectionRole = "ranked"
    admits_complementary_context: bool = False
    sparse_score: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSelectionDecision:
    evidence_id: str
    source_rank: int
    action: Literal["selected", "rejected"]
    reason: SelectionReason


@dataclass(frozen=True, slots=True)
class EvidenceSelectionResult:
    schema_version: Literal["evidence-set-selector.v2"]
    selected_evidence_ids: tuple[str, ...]
    input_token_cost: int
    maximum_input_tokens: int
    maximum_evidence_count: int
    maximum_overlap_ratio: float
    minimum_relative_sparse_score: float | None
    decisions: tuple[EvidenceSelectionDecision, ...]


def _overlap(left: EvidenceSelectionCandidate, right: EvidenceSelectionCandidate) -> int:
    if left.context_id != right.context_id:
        return 0
    return max(
        0,
        min(left.canonical_end, right.canonical_end)
        - max(left.canonical_start, right.canonical_start),
    )


def _parent_child(left: EvidenceSelectionCandidate, right: EvidenceSelectionCandidate) -> bool:
    if left.context_id != right.context_id or _overlap(left, right) == 0:
        return False
    if left.hierarchy_path == right.hierarchy_path:
        return False
    shorter, longer = sorted((left.hierarchy_path, right.hierarchy_path), key=len)
    return len(shorter) < len(longer) and longer[: len(shorter)] == shorter


def _overlap_ratio(left: EvidenceSelectionCandidate, right: EvidenceSelectionCandidate) -> float:
    overlap = _overlap(left, right)
    shorter_length = min(
        left.canonical_end - left.canonical_start,
        right.canonical_end - right.canonical_start,
    )
    return float(overlap) / float(shorter_length) if shorter_length else 0.0


def _validate(candidates: tuple[EvidenceSelectionCandidate, ...]) -> None:
    if any(
        not candidate.evidence_id
        or not candidate.context_id
        or not candidate.hierarchy_path
        or candidate.canonical_start < 0
        or candidate.canonical_start >= candidate.canonical_end
        or candidate.token_cost < 0
        or (
            candidate.sparse_score is not None
            and (not math.isfinite(candidate.sparse_score) or candidate.sparse_score < 0.0)
        )
        or candidate.selection_role not in {"ranked", "parent_context", "sibling_context"}
        for candidate in candidates
    ):
        raise ValueError("evidence selection candidate identity, span, hierarchy, or cost invalid")


def select_evidence_set(
    candidates: tuple[EvidenceSelectionCandidate, ...],
    *,
    maximum_input_tokens: int,
    maximum_evidence_count: int = 3,
    maximum_overlap_ratio: float = 0.8,
    minimum_relative_sparse_score: float | None = None,
    input_token_cost: Callable[[tuple[EvidenceSelectionCandidate, ...]], int] | None = None,
) -> EvidenceSelectionResult:
    """Select up to three complementary canonical items in frozen rank order."""

    if maximum_input_tokens < 1 or maximum_evidence_count < 1:
        raise ValueError("selection token and evidence limits must be positive")
    if not 0.0 <= maximum_overlap_ratio <= 1.0:
        raise ValueError("selection overlap ratio must be within [0, 1]")
    if minimum_relative_sparse_score is not None and (
        not math.isfinite(minimum_relative_sparse_score)
        or not 0.0 <= minimum_relative_sparse_score <= 1.0
    ):
        raise ValueError("selection relative sparse-score threshold must be within [0, 1]")
    _validate(candidates)
    cost = input_token_cost or (lambda values: sum(item.token_cost for item in values))
    selected: list[EvidenceSelectionCandidate] = []
    seen_ids: set[str] = set()
    decisions: list[EvidenceSelectionDecision] = []
    token_cost = cost(())
    for rank, candidate in enumerate(candidates, start=1):
        reason: SelectionReason | None = None
        if candidate.evidence_id in seen_ids:
            reason = "SKIP_DUPLICATE_ID"
        seen_ids.add(candidate.evidence_id)
        if reason is None and candidate.integrity_status != "valid":
            reason = "SKIP_INVALID_EVIDENCE"
        if reason is None and any(
            candidate.context_id == item.context_id
            and (candidate.canonical_start, candidate.canonical_end)
            == (item.canonical_start, item.canonical_end)
            for item in selected
        ):
            reason = "SKIP_DUPLICATE_SPAN"
        admits_context = candidate.admits_complementary_context and candidate.selection_role in {
            "parent_context",
            "sibling_context",
        }
        if (
            reason is None
            and not admits_context
            and any(_parent_child(candidate, item) for item in selected)
        ):
            reason = "SKIP_PARENT_CHILD_REDUNDANCY"
        if (
            reason is None
            and any(_overlap_ratio(candidate, item) >= maximum_overlap_ratio for item in selected)
            and not admits_context
        ):
            reason = "SKIP_EXCESSIVE_OVERLAP"
        if reason is None and len(selected) >= maximum_evidence_count:
            reason = "SKIP_LOW_INCREMENTAL_COVERAGE"
        if (
            reason is None
            and not admits_context
            and any(
                candidate.context_id == item.context_id
                and candidate.hierarchy_path == item.hierarchy_path
                for item in selected
            )
        ):
            reason = "SKIP_LOW_INCREMENTAL_COVERAGE"
        if reason is None and selected and minimum_relative_sparse_score is not None:
            primary_score = selected[0].sparse_score
            candidate_score = candidate.sparse_score
            if (
                primary_score is None
                or primary_score <= 0.0
                or candidate_score is None
                or candidate_score / primary_score < minimum_relative_sparse_score
            ):
                reason = "SKIP_LOW_INCREMENTAL_COVERAGE"
        proposed_cost = cost((*selected, candidate))
        if reason is None and proposed_cost > maximum_input_tokens:
            reason = "SKIP_TOKEN_BUDGET"
        if reason is None:
            selected.append(candidate)
            token_cost = proposed_cost
            if len(selected) == 1:
                selected_reason: SelectionReason = "SELECT_PRIMARY"
            elif candidate.selection_role == "parent_context":
                selected_reason = "SELECT_PARENT_CONTEXT"
            elif candidate.selection_role == "sibling_context":
                selected_reason = "SELECT_SIBLING_CONTEXT"
            else:
                selected_reason = "SELECT_COMPLEMENTARY_COORDINATE"
            decisions.append(
                EvidenceSelectionDecision(candidate.evidence_id, rank, "selected", selected_reason)
            )
        else:
            decisions.append(
                EvidenceSelectionDecision(candidate.evidence_id, rank, "rejected", reason)
            )
    return EvidenceSelectionResult(
        schema_version="evidence-set-selector.v2",
        selected_evidence_ids=tuple(candidate.evidence_id for candidate in selected),
        input_token_cost=token_cost,
        maximum_input_tokens=maximum_input_tokens,
        maximum_evidence_count=maximum_evidence_count,
        maximum_overlap_ratio=maximum_overlap_ratio,
        minimum_relative_sparse_score=minimum_relative_sparse_score,
        decisions=tuple(decisions),
    )


__all__ = [
    "EvidenceSelectionCandidate",
    "EvidenceSelectionDecision",
    "EvidenceSelectionResult",
    "SelectionReason",
    "SelectionRole",
    "select_evidence_set",
]
