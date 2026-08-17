from __future__ import annotations

from pathlib import Path

from legal_rag.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_and_submission_cli_contracts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output = tmp_path / "fixture-run"

    exit_code = main(
        [
            "pipeline",
            "run",
            "--config",
            "configs/fixture.yaml",
            "--questions",
            "data/fixtures/mil-002/questions.json",
            "--contexts",
            "data/fixtures/mil-002/contexts",
            "--aliases",
            "data/fixtures/mil-002/aliases.jsonl",
            "--resource-manifest",
            "data/fixtures/mil-002/resource-manifest.json",
            "--output-directory",
            str(output),
            "--execution-mode",
            "local-offline",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("PIPELINE COMPLETE run_id=run_")
    assert captured.err == ""

    rebuilt = tmp_path / "rebuilt.json"
    exit_code = main(
        [
            "submit",
            "build",
            "--questions",
            "data/fixtures/mil-002/questions.json",
            "--answers",
            str(output / "answers.jsonl"),
            "--output",
            str(rebuilt),
            "--profile",
            "competition",
        ]
    )
    build_output = capsys.readouterr()
    assert exit_code == 0
    assert build_output.out.startswith("VALID SUBMISSION count=2 sha256:")
    assert rebuilt.read_bytes() == (output / "predictions.json").read_bytes()

    exit_code = main(
        [
            "submit",
            "validate",
            "--questions",
            "data/fixtures/mil-002/questions.json",
            "--predictions",
            str(rebuilt),
            "--profile",
            "competition",
        ]
    )
    validate_output = capsys.readouterr()
    assert exit_code == 0
    assert validate_output.out.startswith("VALID SUBMISSION count=2 sha256:")


def test_submit_validation_failure_uses_exit_two(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"{}\n")

    exit_code = main(
        [
            "submit",
            "validate",
            "--questions",
            "data/fixtures/mil-002/questions.json",
            "--predictions",
            str(invalid),
            "--profile",
            "competition",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "SUB_ID_MISMATCH" in captured.err
