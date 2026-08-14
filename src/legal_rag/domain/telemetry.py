"""Link excluded operational telemetry to deterministic run manifests."""

from __future__ import annotations

from legal_rag.domain.checksums import DeterminismError, canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import OperationalTelemetry, RunManifest


def validate_telemetry_link(
    telemetry: OperationalTelemetry, manifest: RunManifest
) -> OperationalTelemetry:
    """Validate telemetry identity without importing its dynamic fields into run bytes."""

    if telemetry.run_id != manifest.run_id:
        raise DeterminismError(
            "TELEMETRY_RUN_ID_MISMATCH",
            "operational telemetry run_id does not match the run manifest",
        )
    expected = checksum_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    if telemetry.run_manifest_checksum != expected:
        raise DeterminismError(
            "TELEMETRY_MANIFEST_CHECKSUM_MISMATCH",
            "operational telemetry checksum does not match the deterministic run manifest",
        )
    return telemetry
