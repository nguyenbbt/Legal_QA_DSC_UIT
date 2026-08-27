"""Immutable identities for bounded public-generation campaigns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Literal, NoReturn


class PublicCampaignError(Exception):
    """Stable fail-closed campaign configuration error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise PublicCampaignError(code, message)


@dataclass(frozen=True, slots=True)
class PublicCampaign:
    campaign_id: str
    approval_id: str
    job_identity: str
    evidence_relative_path: PurePosixPath
    output_relative_path: PurePosixPath
    response_checkpoint_relative_path: PurePosixPath
    replay_checkpoint_relative_path: PurePosixPath
    selection_evidence_relative_path: PurePosixPath
    run_id: str
    generator_id: str
    profile_state: Literal["diagnostic_dry_run"] = "diagnostic_dry_run"
    public_results_usage: Literal["reporting_only_not_fitting_or_selection"] = (
        "reporting_only_not_fitting_or_selection"
    )
    require_fresh_remote_responses: bool = True

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.campaign_id,
                self.approval_id,
                self.job_identity,
                self.run_id,
                self.generator_id,
            )
        ):
            _fail("PUBLIC_CAMPAIGN_INVALID", "campaign identities must be non-empty")
        for path in self.artifact_paths:
            if path.is_absolute() or ".." in path.parts or not path.parts:
                _fail(
                    "PUBLIC_CAMPAIGN_INVALID",
                    "campaign artifact paths must remain project-relative",
                )
        if self.response_checkpoint_relative_path == self.replay_checkpoint_relative_path:
            _fail(
                "PUBLIC_CAMPAIGN_NAMESPACE_REUSED",
                "remote and replay checkpoint paths must be distinct",
            )
        if not self.require_fresh_remote_responses:
            _fail(
                "PUBLIC_CAMPAIGN_OLD_RESPONSE_FORBIDDEN",
                "campaign must require fresh remote responses",
            )

    @property
    def artifact_paths(self) -> tuple[PurePosixPath, ...]:
        return (
            self.evidence_relative_path,
            self.output_relative_path,
            self.response_checkpoint_relative_path,
            self.replay_checkpoint_relative_path,
            self.selection_evidence_relative_path,
        )

    def with_response_checkpoint_path(self, value: PurePosixPath) -> PublicCampaign:
        """Return a testable immutable variant without mutating campaign constants."""

        return replace(self, response_checkpoint_relative_path=value)


LEGACY_R0_PUBLIC_CAMPAIGN = PublicCampaign(
    campaign_id="legacy-r0",
    approval_id="APPROVE_OQ003_MODAL_A10_PUBLIC_GENERATION_V1",
    job_identity="public-generation-v1",
    evidence_relative_path=PurePosixPath(
        "artifacts/evaluations/public/G1R0A512-public-1000-diagnostic-v1/public.evidence.v1.jsonl"
    ),
    output_relative_path=PurePosixPath(
        "artifacts/evaluations/public/G1R0A512-public-1000-modal-a10-diagnostic-v1"
    ),
    response_checkpoint_relative_path=PurePosixPath(
        ".local/runs/public-r0-g1a512-v1/modal-answer-checkpoints"
    ),
    replay_checkpoint_relative_path=PurePosixPath(
        ".local/runs/public-r0-g1a512-v1/modal-replay-checkpoints"
    ),
    selection_evidence_relative_path=PurePosixPath(
        "artifacts/evaluations/grounding/assessments/"
        "G1R0A512-qwen3-1.7b-prompt-a-btc-approved-v1.grounding.manifest.v1.json"
    ),
    run_id="G1R0A512-public-1000-modal-a10-diagnostic-v1",
    generator_id="qwen3-1.7b-prompt-a-512-modal-a10-v1",
)


D061_PUBLIC_CAMPAIGN = PublicCampaign(
    campaign_id="d061-base-reranker",
    approval_id="APPROVE_D061_BASE_RERANKER_PUBLIC_DIAGNOSTIC",
    job_identity="d061-base-reranker-public-v1",
    evidence_relative_path=PurePosixPath(
        "artifacts/evaluations/public/D061-base-reranker-public-1000-evidence-v1/"
        "public.evidence.v1.jsonl"
    ),
    output_relative_path=PurePosixPath(
        "artifacts/evaluations/public/D061-base-reranker-G1A512-public-1000-modal-a10-v1"
    ),
    response_checkpoint_relative_path=PurePosixPath(
        ".local/runs/d061-base-reranker-public-v1/modal-answer-checkpoints"
    ),
    replay_checkpoint_relative_path=PurePosixPath(
        ".local/runs/d061-base-reranker-public-v1/modal-replay-checkpoints"
    ),
    selection_evidence_relative_path=PurePosixPath(
        "artifacts/evaluations/recovery/R-008/GC0-vs-R0-comparison-v1/comparison.v1.json"
    ),
    run_id="D061-base-reranker-G1A512-public-1000-modal-a10-v1",
    generator_id="qwen3-1.7b-prompt-a-512-modal-a10-d061-v1",
)


def assert_fresh_campaign(candidate: PublicCampaign, *, previous: PublicCampaign) -> None:
    """Reject any reuse of a previous public answer/evidence namespace."""

    candidate_values = {
        candidate.evidence_relative_path,
        candidate.output_relative_path,
        candidate.response_checkpoint_relative_path,
        candidate.replay_checkpoint_relative_path,
    }
    previous_values = {
        previous.evidence_relative_path,
        previous.output_relative_path,
        previous.response_checkpoint_relative_path,
        previous.replay_checkpoint_relative_path,
    }
    if candidate_values & previous_values or candidate.run_id == previous.run_id:
        _fail(
            "PUBLIC_CAMPAIGN_NAMESPACE_REUSED",
            "new public campaign overlaps a previous public artifact namespace",
        )


def public_campaign(campaign_id: str) -> PublicCampaign:
    campaigns = {
        LEGACY_R0_PUBLIC_CAMPAIGN.campaign_id: LEGACY_R0_PUBLIC_CAMPAIGN,
        D061_PUBLIC_CAMPAIGN.campaign_id: D061_PUBLIC_CAMPAIGN,
    }
    try:
        campaign = campaigns[campaign_id]
    except KeyError as error:
        raise PublicCampaignError(
            "PUBLIC_CAMPAIGN_UNKNOWN", "public campaign identity is not approved"
        ) from error
    if campaign is D061_PUBLIC_CAMPAIGN:
        assert_fresh_campaign(campaign, previous=LEGACY_R0_PUBLIC_CAMPAIGN)
    return campaign


__all__ = [
    "D061_PUBLIC_CAMPAIGN",
    "LEGACY_R0_PUBLIC_CAMPAIGN",
    "PublicCampaign",
    "PublicCampaignError",
    "assert_fresh_campaign",
    "public_campaign",
]
