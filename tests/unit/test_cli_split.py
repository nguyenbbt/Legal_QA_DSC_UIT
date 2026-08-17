"""CLI contracts for the MIL-003 immutable split builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from legal_rag.cli import main
from legal_rag.ingestion.organizer import OrganizerQuestionReader


def _write_questions(path: Path, value: object, *, kind: Literal["train", "public"]) -> None:
    source = json.dumps(value, ensure_ascii=False).encode()
    imported = OrganizerQuestionReader().read_bytes(source, kind=kind, artifact_path="fixture")
    path.write_bytes(imported.jsonl_bytes())


def test_split_build_writes_manifest_and_reports_summary(tmp_path: Path, capsys) -> None:
    train = tmp_path / "train.questions.jsonl"
    public = tmp_path / "public.questions.jsonl"
    output = tmp_path / "split.v1.json"
    _write_questions(
        train,
        {
            "q5": {"question": "Một câu hỏi", "answer": "Một đáp án"},
            "q0": {"question": "Câu khác", "answer": "Đáp án khác"},
        },
        kind="train",
    )
    _write_questions(public, {"p": {"question": "MỘT CÂU HỎI!", "answer": None}}, kind="public")

    exit_code = main(
        [
            "split",
            "build",
            "--questions",
            str(train),
            "--public-questions",
            str(public),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("SPLIT COMPLETE questions=2 groups=2 sha256:")
    assert captured.err == ""
    manifest = json.loads(output.read_bytes())
    assert manifest["schema_version"] == "split.manifest.v1"
    assert manifest["overlap_report"]["pair_count"] == 1


def test_split_build_does_not_replace_a_different_manifest(tmp_path: Path, capsys) -> None:
    train = tmp_path / "train.questions.jsonl"
    public = tmp_path / "public.questions.jsonl"
    output = tmp_path / "split.v1.json"
    _write_questions(train, {"q": {"question": "One", "answer": "Answer"}}, kind="train")
    _write_questions(public, {"p": {"question": "Public", "answer": None}}, kind="public")
    arguments = [
        "split",
        "build",
        "--questions",
        str(train),
        "--public-questions",
        str(public),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    capsys.readouterr()
    original = output.read_bytes()
    _write_questions(train, {"changed": {"question": "Two", "answer": "Answer"}}, kind="train")

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "SPLIT_MANIFEST_IMMUTABLE" in captured.err
    assert output.read_bytes() == original
