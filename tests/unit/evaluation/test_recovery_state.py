from __future__ import annotations

import json

import pytest

from legal_rag.evaluation.recovery_state import (
    RecoveryStageError,
    build_recovery_stage_manifest,
)

SHA = "sha256:" + "a" * 64


def test_recovery_stage_manifest_is_closed_sorted_and_byte_stable() -> None:
    kwargs = {
        "stage_id": "R-005B",
        "state": "not_triggered",
        "reason_code": "DENSE_TRIGGER_NOT_MET",
        "dependency_evidence": {"R-005A": SHA, "R-002A": SHA},
        "artifact_checksums": {"report": SHA},
    }
    first = build_recovery_stage_manifest(**kwargs)
    second = build_recovery_stage_manifest(**kwargs)

    assert first == second
    manifest = json.loads(first)
    assert list(manifest["dependency_evidence"]) == ["R-002A", "R-005A"]
    assert manifest["state"] == "not_triggered"


@pytest.mark.parametrize("state", ("done", "pending", "running"))
def test_recovery_stage_manifest_rejects_ambiguous_states(state: str) -> None:
    with pytest.raises(RecoveryStageError) as caught:
        build_recovery_stage_manifest(
            stage_id="R-007",
            state=state,
            reason_code="MODEL_NOT_APPROVED",
            dependency_evidence={},
            artifact_checksums={},
        )
    assert caught.value.code == "RECOVERY_STAGE_MANIFEST_INVALID"


def test_recovery_stage_manifest_requires_reason_and_typed_checksums() -> None:
    with pytest.raises(RecoveryStageError) as caught:
        build_recovery_stage_manifest(
            stage_id="R-008",
            state="blocked_external",
            reason_code="",
            dependency_evidence={"R-007": "bad"},
            artifact_checksums={},
        )
    assert caught.value.code == "RECOVERY_STAGE_MANIFEST_INVALID"

    with pytest.raises(RecoveryStageError) as caught:
        build_recovery_stage_manifest(
            stage_id="R-008",
            state="blocked_external",
            reason_code="TRAIN_LABELS_MISSING",
            dependency_evidence={"R-007": "bad"},
            artifact_checksums={},
        )
    assert caught.value.code == "RECOVERY_STAGE_MANIFEST_INVALID"
