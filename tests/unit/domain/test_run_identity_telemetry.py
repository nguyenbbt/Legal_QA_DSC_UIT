"""Run fingerprint, completed-output, and operational-linkage contracts."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from legal_rag.domain.checksums import (
    DeterminismError,
    canonical_json_bytes,
    checksum_bytes,
    compute_run_id,
    validate_run_manifest_identity,
    validate_run_output_checksums,
)
from legal_rag.domain.models import OperationalTelemetry, RunManifest
from legal_rag.domain.telemetry import validate_telemetry_link

CHECKSUM = "sha256:" + "0" * 64
GOLDEN_RUN_ID = "run_880a58562ce9f4820eed295d"


def valid_manifest(**changes: object) -> RunManifest:
    value: dict[str, object] = {
        "schema_version": "run.manifest.v1",
        "run_id": GOLDEN_RUN_ID,
        "pipeline_version": "pipeline.v1",
        "code_revision": "tree:" + CHECKSUM,
        "source_tree_checksum": CHECKSUM,
        "scoped_source_paths": (
            "configs/fixture.yaml",
            "src/legal_rag/domain/models.py",
        ),
        "config_checksum": CHECKSUM,
        "question_checksum": CHECKSUM,
        "corpus_checksum": CHECKSUM,
        "index_checksum": None,
        "split_checksum": None,
        "model_id": None,
        "model_revision": None,
        "tokenizer_id": "legal-retrieval-unicode-v1",
        "tokenizer_revision": "unicode-15.0.0",
        "prompt_revision": None,
        "seed": "fixture-v1",
        "execution_mode": "local-offline",
        "competition_policy": "baseline.v1",
        "comparison_type": "baseline",
        "resolved_as_of_date": None,
        "as_of_timezone": None,
        "resource_manifest_checksum": CHECKSUM,
        "evidence_diagnostics_checksum": CHECKSUM,
        "answer_artifact_checksum": CHECKSUM,
    }
    value.update(changes)
    return RunManifest.model_validate(value)


def test_run_id_matches_independently_computed_golden_and_validates() -> None:
    manifest = valid_manifest()

    assert compute_run_id(manifest) == GOLDEN_RUN_ID
    assert validate_run_manifest_identity(manifest) is manifest


def test_material_field_change_changes_run_id() -> None:
    original = valid_manifest()
    changed = valid_manifest(seed="fixture-v2")

    assert compute_run_id(changed) != compute_run_id(original)


def test_output_checksums_do_not_change_run_id() -> None:
    original = valid_manifest()
    changed = valid_manifest(
        evidence_diagnostics_checksum="sha256:" + "1" * 64,
        answer_artifact_checksum="sha256:" + "2" * 64,
    )

    assert compute_run_id(changed) == compute_run_id(original)


def test_manifest_identity_rejects_incorrect_stored_run_id() -> None:
    manifest = valid_manifest(run_id="run_000000000000000000000000")

    with pytest.raises(DeterminismError) as captured:
        validate_run_manifest_identity(manifest)

    assert captured.value.code == "RUN_ID_MISMATCH"


def test_completed_manifest_validates_exact_output_file_checksums(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    answer = tmp_path / "answers.jsonl"
    evidence.write_bytes(b'{"evidence":true}\n')
    answer.write_bytes(b'{"answer":"yes"}\n')
    manifest = valid_manifest(
        evidence_diagnostics_checksum=checksum_bytes(evidence.read_bytes()),
        answer_artifact_checksum=checksum_bytes(answer.read_bytes()),
    )

    assert validate_run_output_checksums(manifest, evidence, answer) is manifest
    answer.write_bytes(b'{"answer":"no"}\n')
    with pytest.raises(DeterminismError) as captured:
        validate_run_output_checksums(manifest, evidence, answer)
    assert captured.value.code == "RUN_OUTPUT_CHECKSUM_MISMATCH"


def test_telemetry_instances_may_differ_but_link_to_identical_manifest_bytes() -> None:
    manifest = valid_manifest()
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    manifest_checksum = checksum_bytes(manifest_bytes)
    first = OperationalTelemetry(
        schema_version="operational.telemetry.v1",
        run_id=manifest.run_id,
        run_instance_id=UUID("2a940d89-1934-41ef-897a-f6ab2d150f26"),
        run_manifest_checksum=manifest_checksum,
    )
    second = OperationalTelemetry(
        schema_version="operational.telemetry.v1",
        run_id=manifest.run_id,
        run_instance_id=UUID("fb2feb82-e01e-4946-8adc-ac81e335a79d"),
        run_manifest_checksum=manifest_checksum,
    )

    assert first.run_instance_id != second.run_instance_id
    assert validate_telemetry_link(first, manifest) is first
    assert validate_telemetry_link(second, manifest) is second
    with pytest.raises(DeterminismError) as captured:
        canonical_json_bytes(first.model_dump(mode="json"))
    assert captured.value.code == "RUN_CANONICAL_FIELD_FORBIDDEN"


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"run_id": "run_000000000000000000000000"}, "TELEMETRY_RUN_ID_MISMATCH"),
        ({"run_manifest_checksum": "sha256:" + "f" * 64}, "TELEMETRY_MANIFEST_CHECKSUM_MISMATCH"),
    ],
)
def test_telemetry_link_rejects_wrong_identity(changes: dict[str, object], code: str) -> None:
    manifest = valid_manifest()
    values: dict[str, object] = {
        "schema_version": "operational.telemetry.v1",
        "run_id": manifest.run_id,
        "run_instance_id": UUID("2a940d89-1934-41ef-897a-f6ab2d150f26"),
        "run_manifest_checksum": checksum_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json"))
        ),
    }
    values.update(changes)
    telemetry = OperationalTelemetry.model_validate(values)

    with pytest.raises(DeterminismError) as captured:
        validate_telemetry_link(telemetry, manifest)

    assert captured.value.code == code
