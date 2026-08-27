from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from legal_rag.providers.public_campaign import (
    D061_PUBLIC_CAMPAIGN,
    LEGACY_R0_PUBLIC_CAMPAIGN,
    PublicCampaignError,
    assert_fresh_campaign,
    public_campaign,
)


def test_d061_campaign_is_disjoint_from_legacy_public_answers() -> None:
    assert_fresh_campaign(D061_PUBLIC_CAMPAIGN, previous=LEGACY_R0_PUBLIC_CAMPAIGN)

    assert D061_PUBLIC_CAMPAIGN.evidence_relative_path != (
        LEGACY_R0_PUBLIC_CAMPAIGN.evidence_relative_path
    )
    assert D061_PUBLIC_CAMPAIGN.response_checkpoint_relative_path != (
        LEGACY_R0_PUBLIC_CAMPAIGN.response_checkpoint_relative_path
    )
    assert D061_PUBLIC_CAMPAIGN.replay_checkpoint_relative_path != (
        LEGACY_R0_PUBLIC_CAMPAIGN.replay_checkpoint_relative_path
    )
    assert D061_PUBLIC_CAMPAIGN.output_relative_path != (
        LEGACY_R0_PUBLIC_CAMPAIGN.output_relative_path
    )
    assert D061_PUBLIC_CAMPAIGN.run_id != LEGACY_R0_PUBLIC_CAMPAIGN.run_id


def test_d061_campaign_paths_are_project_relative_and_reporting_only() -> None:
    campaign = public_campaign("d061-base-reranker")

    for path in campaign.artifact_paths:
        assert isinstance(path, PurePosixPath)
        assert not path.is_absolute()
        assert ".." not in path.parts
    assert campaign.profile_state == "diagnostic_dry_run"
    assert campaign.public_results_usage == "reporting_only_not_fitting_or_selection"
    assert campaign.require_fresh_remote_responses is True


def test_fresh_campaign_rejects_any_reused_answer_namespace() -> None:
    duplicate = D061_PUBLIC_CAMPAIGN.with_response_checkpoint_path(
        LEGACY_R0_PUBLIC_CAMPAIGN.response_checkpoint_relative_path
    )

    with pytest.raises(PublicCampaignError) as captured:
        assert_fresh_campaign(duplicate, previous=LEGACY_R0_PUBLIC_CAMPAIGN)

    assert captured.value.code == "PUBLIC_CAMPAIGN_NAMESPACE_REUSED"


def test_unknown_public_campaign_fails_closed() -> None:
    with pytest.raises(PublicCampaignError) as captured:
        public_campaign("leaderboard-experiment")

    assert captured.value.code == "PUBLIC_CAMPAIGN_UNKNOWN"
