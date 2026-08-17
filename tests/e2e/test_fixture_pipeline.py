from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import NoReturn

import pytest

from legal_rag.pipeline.fixture import FixturePipelineRequest, run_fixture_pipeline
from legal_rag.submission.writer import validate_submission

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _request(output_directory: Path) -> FixturePipelineRequest:
    return FixturePipelineRequest(
        project_root=PROJECT_ROOT,
        config_path="configs/fixture.yaml",
        questions_path="data/fixtures/mil-002/questions.json",
        contexts_directory="data/fixtures/mil-002/contexts",
        aliases_path="data/fixtures/mil-002/aliases.jsonl",
        resource_manifest_path="data/fixtures/mil-002/resource-manifest.json",
        output_directory=output_directory,
    )


def _forbidden(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("fixture pipeline attempted network access")


@pytest.mark.integration
def test_fixture_pipeline_is_offline_valid_and_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    first = run_fixture_pipeline(_request(tmp_path / "first"))
    second = run_fixture_pipeline(_request(tmp_path / "second"))

    assert first.run_id == second.run_id
    assert first.artifact_names == second.artifact_names
    for name in first.artifact_names:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()

    predictions = (tmp_path / "first" / "predictions.json").read_bytes()
    validation = validate_submission(
        (PROJECT_ROOT / "data/fixtures/mil-002/questions.json").read_bytes(),
        predictions,
    )
    decoded = json.loads(predictions)
    assert validation.count == 2
    assert decoded["q-exact"]["answer"] == "Người đủ 18 tuổi được cấp thẻ."
    assert decoded["q-missing"]["answer"].startswith("Không đủ căn cứ")
    assert first.predictions_checksum == validation.checksum
