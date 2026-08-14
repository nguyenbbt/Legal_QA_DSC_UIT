"""Closed schemas and pure preflight rules for all three execution modes."""

from __future__ import annotations

import nltk
import pytest
from pydantic import TypeAdapter, ValidationError

from legal_rag.domain.execution import (
    ArtifactTransfer,
    ExecutionModeConfig,
    ExecutionModeError,
    LocalOfflineConfig,
    ModalSubmissionState,
    PrepareOnlineConfig,
    PrivateModalConfig,
    preflight_execution,
    redact_sensitive_message,
)

CHECKSUM = "sha256:" + "0" * 64
MODAL_ORIGIN = "https://api.modal.com"


def prepare_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "execution.mode.v1",
        "mode": "prepare-online",
        "accepted_origins": ("https://files.pythonhosted.org", "https://pypi.org"),
        "resource_manifest_updates_required": True,
        "accepted_competition_answers_allowed": False,
    }
    values.update(changes)
    return values


def local_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "execution.mode.v1",
        "mode": "local-offline",
        "resource_manifest_checksum": CHECKSUM,
        "required_resource_ids": ("nltk.omw-1.4", "nltk.wordnet"),
        "outbound_network": "deny-non-loopback",
        "loopback_dependencies": (),
        "evaluation_wall_clock": "forbidden",
    }
    values.update(changes)
    return values


def transfer(
    artifact_id: str = "fixture.questions",
    artifact_class: str = "fixture",
    direction: str = "local-to-modal",
    checksum: str = CHECKSUM,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "direction": direction,
        "checksum": checksum,
    }


def private_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "execution.mode.v1",
        "mode": "private-modal",
        "control_plane_origin": MODAL_ORIGIN,
        "control_plane_origin_allowlist": (MODAL_ORIGIN,),
        "workload_egress_disabled": True,
        "workload_egress_verified": True,
        "private_storage_ids": ("legal-rag-fixtures",),
        "required_resource_ids": ("model.public",),
        "transfer_allowlist": (transfer(),),
        "real_data_approved": False,
        "max_submission_retries": 3,
        "submission_backoff_seconds": (1, 2, 4),
        "declared_job_identity": "fixture-parity-v1",
        "secret_policy": "credential-store-only-redacted",
    }
    values.update(changes)
    return values


@pytest.mark.parametrize(
    ("values", "expected_type"),
    [
        (prepare_values(), PrepareOnlineConfig),
        (local_values(), LocalOfflineConfig),
        (private_values(), PrivateModalConfig),
    ],
)
def test_exactly_one_execution_mode_schema(
    values: dict[str, object], expected_type: type[object]
) -> None:
    parsed = TypeAdapter(ExecutionModeConfig).validate_python(values)

    assert isinstance(parsed, expected_type)


def test_mode_union_rejects_missing_mode_and_cross_mode_fields() -> None:
    missing = prepare_values()
    missing.pop("mode")
    mixed = local_values(accepted_origins=("https://pypi.org",))

    for values in (missing, mixed):
        with pytest.raises(ValidationError):
            TypeAdapter(ExecutionModeConfig).validate_python(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"resource_manifest_updates_required": False},
        {"accepted_competition_answers_allowed": True},
        {"accepted_origins": ()},
    ],
)
def test_prepare_online_requires_manifesting_and_forbids_accepted_answers(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PrepareOnlineConfig.model_validate(prepare_values(**changes))


def test_local_offline_accepts_only_manifested_resources() -> None:
    config = LocalOfflineConfig.model_validate(local_values())

    assert (
        preflight_execution(
            config,
            available_resource_ids=("nltk.wordnet", "nltk.omw-1.4"),
        )
        is config
    )
    with pytest.raises(ExecutionModeError) as captured:
        preflight_execution(config, available_resource_ids=("nltk.wordnet",))
    assert captured.value.code == "OFFLINE_RESOURCE_MISSING"
    assert "nltk.omw-1.4" in captured.value.message


@pytest.mark.parametrize(
    "changes",
    [
        {"outbound_network": "allow"},
        {"evaluation_wall_clock": "allowed"},
        {"required_resource_ids": ("nltk.wordnet", "nltk.wordnet")},
    ],
)
def test_local_offline_schema_rejects_egress_clock_and_duplicate_resources(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        LocalOfflineConfig.model_validate(local_values(**changes))


def test_private_modal_rejects_unverified_no_egress_before_transfer() -> None:
    config = PrivateModalConfig.model_validate(private_values(workload_egress_verified=False))

    with pytest.raises(ExecutionModeError) as captured:
        preflight_execution(config, available_resource_ids=("model.public",))

    assert captured.value.code == "MODAL_EGRESS_UNVERIFIED"


@pytest.mark.parametrize(
    "artifact_class",
    [
        "organizer_question",
        "organizer_context",
        "organizer_answer",
        "derived_chunk",
        "embedding",
        "index",
        "cache",
    ],
)
def test_private_modal_blocks_every_real_or_derived_data_class(artifact_class: str) -> None:
    requested = ArtifactTransfer.model_validate(
        transfer(artifact_class=artifact_class, artifact_id=f"blocked.{artifact_class}")
    )
    config = PrivateModalConfig.model_validate(private_values())

    with pytest.raises(ExecutionModeError) as captured:
        preflight_execution(
            config,
            available_resource_ids=("model.public",),
            requested_transfers=(requested,),
        )

    assert captured.value.code == "MODAL_REAL_DATA_NOT_APPROVED"


def test_private_modal_requires_manifested_resources_and_exact_transfer_allowlist() -> None:
    config = PrivateModalConfig.model_validate(private_values())
    requested = ArtifactTransfer.model_validate(transfer(checksum="sha256:" + "1" * 64))

    with pytest.raises(ExecutionModeError) as missing:
        preflight_execution(config, available_resource_ids=())
    assert missing.value.code == "MODAL_RESOURCE_MISSING"

    with pytest.raises(ExecutionModeError) as disallowed:
        preflight_execution(
            config,
            available_resource_ids=("model.public",),
            requested_transfers=(requested,),
        )
    assert disallowed.value.code == "MODAL_TRANSFER_NOT_ALLOWLISTED"


@pytest.mark.parametrize(
    "changes",
    [
        {"control_plane_origin_allowlist": ("https://other.example",)},
        {"control_plane_origin": "https://user:secret@api.modal.com"},
        {"control_plane_origin": "http://api.modal.com"},
        {"workload_egress_disabled": False},
        {"real_data_approved": True},
        {"max_submission_retries": 4},
        {"submission_backoff_seconds": (1, 2, 8)},
        {"private_storage_ids": ()},
    ],
)
def test_private_modal_schema_rejects_unsafe_or_unapproved_configuration(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PrivateModalConfig.model_validate(private_values(**changes))


def test_modal_submission_state_has_three_retries_then_one_job_identity() -> None:
    state = ModalSubmissionState()
    observed_backoff: list[int] = []
    for expected_attempt in range(1, 4):
        state = state.record_submission_attempt()
        assert state.submission_attempts == expected_attempt
        observed_backoff.append(state.retry_delay_seconds())
    assert observed_backoff == [1, 2, 4]

    state = state.record_submission_attempt().record_job_id("job-fixture-001")
    assert state.job_id == "job-fixture-001"
    with pytest.raises(ExecutionModeError) as captured:
        state.record_submission_attempt()
    assert captured.value.code == "MODAL_JOB_ALREADY_DECLARED"
    with pytest.raises(ExecutionModeError) as second_job:
        state.record_job_id("job-fixture-002")
    assert second_job.value.code == "MODAL_JOB_ALREADY_DECLARED"


def test_modal_submission_state_stops_after_retry_budget() -> None:
    state = ModalSubmissionState()
    for _ in range(4):
        state = state.record_submission_attempt()

    with pytest.raises(ExecutionModeError) as captured:
        state.record_submission_attempt()
    assert captured.value.code == "MODAL_RETRY_LIMIT"
    with pytest.raises(ExecutionModeError) as no_delay:
        state.retry_delay_seconds()
    assert no_delay.value.code == "MODAL_RETRY_LIMIT"


def test_secret_redaction_removes_configured_and_bearer_values() -> None:
    message = "token=private-token Authorization: Bearer bearer-value"

    redacted = redact_sensitive_message(message, secret_values=("private-token",))

    assert "private-token" not in redacted
    assert "bearer-value" not in redacted
    assert redacted == "token=[REDACTED] Authorization: Bearer [REDACTED]"


def test_preflight_is_pure_and_does_not_accept_a_workload_submitter() -> None:
    class WorkloadSpy:
        def __init__(self) -> None:
            self.calls = 0

        def submit(self) -> str:
            self.calls += 1
            return "job-must-not-exist"

    config = PrivateModalConfig.model_validate(private_values())
    spy = WorkloadSpy()

    with pytest.raises(TypeError):
        preflight_execution(
            config,
            available_resource_ids=("model.public",),
            workload_submitter=spy,  # type: ignore[call-arg]
        )

    assert spy.calls == 0


def test_execution_preflight_never_calls_nltk_download(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def download_spy(*args: object, **_kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr(nltk, "download", download_spy)
    config = LocalOfflineConfig.model_validate(local_values())

    preflight_execution(
        config,
        available_resource_ids=("nltk.omw-1.4", "nltk.wordnet"),
    )

    assert calls == []
