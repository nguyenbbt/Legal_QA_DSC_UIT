from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.pipeline.fixture import (
    FixturePipelineError,
    FixturePipelineRequest,
    run_fixture_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def request(output_directory: Path) -> FixturePipelineRequest:
    return FixturePipelineRequest(
        project_root=PROJECT_ROOT,
        config_path="configs/fixture.yaml",
        questions_path="data/fixtures/mil-002/questions.json",
        contexts_directory="data/fixtures/mil-002/contexts",
        aliases_path="data/fixtures/mil-002/aliases.jsonl",
        resource_manifest_path="data/fixtures/mil-002/resource-manifest.json",
        output_directory=output_directory,
    )


def test_injected_stage_failure_is_typed_and_never_replaces_submission(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    predictions = output / "predictions.json"
    predictions.write_bytes(b"existing-valid-submission")

    def fail_after_retrieval(stage: str) -> None:
        if stage == "retrieval":
            raise RuntimeError("injected unsafe detail")

    with pytest.raises(FixturePipelineError) as captured:
        run_fixture_pipeline(request(output), stage_hook=fail_after_retrieval)

    assert captured.value.code == "PIPELINE_STAGE_FAILED"
    assert captured.value.stage == "retrieval"
    assert predictions.read_bytes() == b"existing-valid-submission"
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure == {
        "code": "PIPELINE_STAGE_FAILED",
        "schema_version": "pipeline.failure.v1",
        "stage": "retrieval",
    }
    assert "unsafe" not in (output / "failure.json").read_text(encoding="utf-8")


def test_resource_manifest_checksum_mismatch_fails_before_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "resource-manifest.json"
    manifest.write_bytes(
        (PROJECT_ROOT / "data/fixtures/mil-002/resource-manifest.json")
        .read_bytes()
        .replace(b"sha256:1639", b"sha256:0000")
    )
    active = request(tmp_path / "run")
    active = FixturePipelineRequest(
        project_root=active.project_root,
        config_path=active.config_path,
        questions_path=active.questions_path,
        contexts_directory=active.contexts_directory,
        aliases_path=active.aliases_path,
        resource_manifest_path=manifest.relative_to(PROJECT_ROOT).as_posix(),
        output_directory=active.output_directory,
    )

    with pytest.raises(FixturePipelineError) as captured:
        run_fixture_pipeline(active)

    assert captured.value.code == "FIXTURE_RESOURCE_CHECKSUM_MISMATCH"
    assert not (tmp_path / "run" / "predictions.json").exists()


def test_oversized_fixture_manifest_fails_before_reading_or_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "oversized-resource-manifest.json"
    manifest.write_bytes(b" " * 1_048_577)
    active = request(tmp_path / "run")
    active = FixturePipelineRequest(
        project_root=active.project_root,
        config_path=active.config_path,
        questions_path=active.questions_path,
        contexts_directory=active.contexts_directory,
        aliases_path=active.aliases_path,
        resource_manifest_path=manifest.relative_to(PROJECT_ROOT).as_posix(),
        output_directory=active.output_directory,
    )

    with pytest.raises(FixturePipelineError) as captured:
        run_fixture_pipeline(active)

    assert captured.value.code == "FIXTURE_RESOURCE_TOO_LARGE"
    assert not (tmp_path / "run" / "predictions.json").exists()
