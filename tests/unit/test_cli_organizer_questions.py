"""CLI contracts for deterministic organizer question conversion."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.cli import main


def test_organizer_import_questions_writes_jsonl_and_empty_errors(tmp_path: Path, capsys) -> None:
    source = tmp_path / "train.json"
    output = tmp_path / "train.questions.jsonl"
    errors = tmp_path / "train.errors.jsonl"
    source.write_text(
        json.dumps(
            {"001": {"question": "Câu hỏi", "answer": "Câu trả lời"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "organizer",
            "import-questions",
            "--kind",
            "train",
            "--input",
            str(source),
            "--output",
            str(output),
            "--errors",
            str(errors),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("IMPORT COMPLETE kind=train count=1 sha256:")
    assert captured.err == ""
    record = json.loads(output.read_bytes())
    assert record["schema_version"] == "internal.question.v1"
    assert record["question_id"] == "001"
    assert errors.read_bytes() == b""


def test_organizer_import_failure_writes_safe_error_only(tmp_path: Path, capsys) -> None:
    source = tmp_path / "public.json"
    output = tmp_path / "public.questions.jsonl"
    errors = tmp_path / "public.errors.jsonl"
    source.write_bytes(b'{"q":{"question":"valid","answer":"must be null"}}')

    exit_code = main(
        [
            "organizer",
            "import-questions",
            "--kind",
            "public",
            "--input",
            str(source),
            "--output",
            str(output),
            "--errors",
            str(errors),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DATA_ANSWER_TYPE" in captured.err
    assert not output.exists()
    error = json.loads(errors.read_bytes())
    assert error["code"] == "DATA_ANSWER_TYPE"
    assert "must be null" not in errors.read_text(encoding="utf-8")
