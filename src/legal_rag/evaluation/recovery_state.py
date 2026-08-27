"""Deterministic terminal-state evidence for every retrieval recovery stage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from legal_rag.domain.checksums import content_json_bytes

RecoveryStageState = Literal[
    "completed",
    "rejected",
    "skipped_by_gate",
    "blocked_external",
    "not_triggered",
]

_STATES = frozenset(
    {"completed", "rejected", "skipped_by_gate", "blocked_external", "not_triggered"}
)
_STAGE_ID = re.compile(r"R-[0-9]{3}[A-Z]?\Z")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RecoveryStageError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _validated_checksums(values: Mapping[str, str]) -> dict[str, str]:
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(checksum, str)
        or _CHECKSUM.fullmatch(checksum) is None
        for name, checksum in values.items()
    ):
        raise RecoveryStageError(
            "RECOVERY_STAGE_MANIFEST_INVALID",
            "recovery evidence names and checksums must be valid",
        )
    return dict(sorted(values.items(), key=lambda item: item[0].encode("utf-8")))


def build_recovery_stage_manifest(
    *,
    stage_id: str,
    state: RecoveryStageState | str,
    reason_code: str,
    dependency_evidence: Mapping[str, str],
    artifact_checksums: Mapping[str, str],
) -> bytes:
    """Serialize an explicit terminal state; ambiguous progress states are forbidden."""

    if (
        _STAGE_ID.fullmatch(stage_id) is None
        or state not in _STATES
        or _REASON_CODE.fullmatch(reason_code) is None
    ):
        raise RecoveryStageError(
            "RECOVERY_STAGE_MANIFEST_INVALID",
            "stage identity, terminal state, or reason code is invalid",
        )
    dependencies = _validated_checksums(dependency_evidence)
    artifacts = _validated_checksums(artifact_checksums)
    return content_json_bytes(
        {
            "schema_version": "retrieval.recovery.stage-state.v1",
            "stage_id": stage_id,
            "state": state,
            "reason_code": reason_code,
            "dependency_evidence": dependencies,
            "artifact_checksums": artifacts,
        }
    )


__all__ = [
    "RecoveryStageError",
    "RecoveryStageState",
    "build_recovery_stage_manifest",
]
