from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from legal_rag.domain.models import RunManifest
from legal_rag.domain.validation import parse_record_json

CHECKSUM = "sha256:" + "0" * 64


def valid_manifest(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "run.manifest.v1",
        "run_id": "run_4f12b96c4b87a168759b3ca0",
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
    return value


def test_run_manifest_exact_round_trip_and_immutability() -> None:
    record = RunManifest.model_validate(valid_manifest())
    expected = valid_manifest(scoped_source_paths=list(valid_manifest()["scoped_source_paths"]))

    assert record.model_dump(mode="json") == expected
    with pytest.raises(ValidationError):
        record.seed = "changed"  # type: ignore[misc]


def test_run_manifest_json_arrays_become_immutable_tuples() -> None:
    raw = (json.dumps(valid_manifest(), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    record = parse_record_json(raw, RunManifest, artifact_path="run.manifest.json")
    assert record.scoped_source_paths == (
        "configs/fixture.yaml",
        "src/legal_rag/domain/models.py",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"run_id": "run_ABC"},
        {"code_revision": "main"},
        {"source_tree_checksum": "0" * 64},
        {"scoped_source_paths": ()},
        {"scoped_source_paths": ("src/z.py", "src/a.py")},
        {"scoped_source_paths": ("src/a.py", "src/a.py")},
        {"scoped_source_paths": ("../secret.py",)},
        {"execution_mode": "online"},
        {"competition_policy": "research.v1"},
        {"comparison_type": "generator"},
        {"model_id": ""},
        {"tokenizer_revision": ""},
        {"seed": "   "},
    ],
)
def test_run_manifest_rejects_invalid_contracts(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(valid_manifest(**changes))


@pytest.mark.parametrize(
    ("resolved_date", "timezone"),
    [
        ("2026-08-14", None),
        (None, "Asia/Ho_Chi_Minh"),
        ("2026-02-30", "Asia/Ho_Chi_Minh"),
        ("2026-08-14", "UTC"),
        ("2026-8-14", "Asia/Ho_Chi_Minh"),
    ],
)
def test_run_manifest_rejects_invalid_resolved_date_pairs(
    resolved_date: str | None, timezone: str | None
) -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(
            valid_manifest(resolved_as_of_date=resolved_date, as_of_timezone=timezone)
        )


def test_run_manifest_accepts_fixed_resolved_date_pair() -> None:
    record = RunManifest.model_validate(
        valid_manifest(resolved_as_of_date="2026-08-14", as_of_timezone="Asia/Ho_Chi_Minh")
    )
    assert record.resolved_as_of_date == "2026-08-14"


@pytest.mark.parametrize(
    "comparison_type",
    [
        "baseline",
        "generator_only",
        "retrieval_only",
        "joint_chunking_retrieval_generation",
        "bug_fix",
    ],
)
def test_run_manifest_accepts_every_comparison_type(comparison_type: str) -> None:
    record = RunManifest.model_validate(valid_manifest(comparison_type=comparison_type))
    assert record.comparison_type == comparison_type


def test_run_manifest_accepts_git_code_revision() -> None:
    record = RunManifest.model_validate(valid_manifest(code_revision="git:" + "a" * 40))
    assert record.code_revision == "git:" + "a" * 40
