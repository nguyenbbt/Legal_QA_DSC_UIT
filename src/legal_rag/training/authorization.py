"""Milestone and owner authorization gate checking.

No model loading, GPU, or training occurs in this module. It only validates
that the required approval chain exists before allowing downstream work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class AuthorizationError(Exception):
    """Stable failure at the training authorization boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class MilestoneGate:
    """Typed representation of a project milestone authorization state."""

    milestone: str
    state: Literal[
        "not_started",
        "active",
        "implementation_complete_labels_pending",
        "ready_for_owner_approval",
        "owner_approved",
        "complete",
    ]
    owner_approval: bool
    blocking_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingAuthorization:
    """Authorization record for a specific training action."""

    action: Literal["FT-EMBED", "FT-RERANK", "FT-GENERATOR", "NO_FT"]
    milestone: str
    owner_approved: bool
    model_btc_approved: bool
    parameter_gate_passed: bool
    dataset_provenance_valid: bool
    backend_authorized: bool
    oq003_resolved: bool
    backend: Literal["local", "private-modal"]
    can_proceed: bool
    blocking_codes: tuple[str, ...]


def check_milestone_gate(
    active_milestone: str,
    required_milestone: str,
    owner_approval: bool,
) -> MilestoneGate:
    """Validate that the required milestone is authorized."""
    blocking: list[str] = []

    if active_milestone != required_milestone:
        blocking.append(f"MILESTONE_{required_milestone}_NOT_ACTIVE")

    if not owner_approval:
        blocking.append(f"OWNER_APPROVAL_{required_milestone}_MISSING")

    state: Literal[
        "not_started",
        "active",
        "ready_for_owner_approval",
        "owner_approved",
        "complete",
    ]
    if blocking:
        state = "not_started"
    elif owner_approval:
        state = "owner_approved"
    else:
        state = "active"

    return MilestoneGate(
        milestone=required_milestone,
        state=state,
        owner_approval=owner_approval,
        blocking_codes=tuple(blocking),
    )


def check_training_authorization(
    *,
    action: Literal["FT-EMBED", "FT-RERANK", "FT-GENERATOR", "NO_FT"],
    milestone: str,
    owner_approved: bool,
    model_btc_approved: bool,
    parameter_gate_passed: bool,
    dataset_provenance_valid: bool,
    backend_authorized: bool,
    oq003_resolved: bool,
    backend: Literal["local", "private-modal"] = "private-modal",
) -> TrainingAuthorization:
    """Check whether a training action is fully authorized.

    All gates MUST pass. Missing any single gate blocks the action.
    """
    blocking: list[str] = []

    if action == "NO_FT":
        return TrainingAuthorization(
            action=action,
            milestone=milestone,
            owner_approved=owner_approved,
            model_btc_approved=model_btc_approved,
            parameter_gate_passed=parameter_gate_passed,
            dataset_provenance_valid=dataset_provenance_valid,
            backend_authorized=backend_authorized,
            oq003_resolved=oq003_resolved,
            backend=backend,
            can_proceed=True,
            blocking_codes=(),
        )

    if not owner_approved:
        blocking.append("OWNER_FT_APPROVAL_MISSING")
    # D-056 moves exact-model registration to final promotion/submission. Retain
    # the legacy field in v1 artifacts for compatibility, but do not gate an
    # exploratory or fitting workload on it.
    if not parameter_gate_passed:
        blocking.append("PARAMETER_GATE_FAILED")
    if not dataset_provenance_valid:
        blocking.append("DATASET_PROVENANCE_INVALID")
    if not backend_authorized:
        blocking.append("BACKEND_NOT_AUTHORIZED")
    if backend == "private-modal" and not oq003_resolved:
        blocking.append("OQ003_UNRESOLVED")

    return TrainingAuthorization(
        action=action,
        milestone=milestone,
        owner_approved=owner_approved,
        model_btc_approved=model_btc_approved,
        parameter_gate_passed=parameter_gate_passed,
        dataset_provenance_valid=dataset_provenance_valid,
        backend_authorized=backend_authorized,
        oq003_resolved=oq003_resolved,
        backend=backend,
        can_proceed=len(blocking) == 0,
        blocking_codes=tuple(blocking),
    )
