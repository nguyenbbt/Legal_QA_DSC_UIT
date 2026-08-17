"""CLI contract for the MIL-003 public/development plumbing baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.split import SplitQuestion, build_split_manifest
from legal_rag.generation.fixture import FIXED_REFUSAL
from legal_rag.ingestion.organizer import OrganizerQuestionReader


def test_baseline_builds_valid_public_and_development_artifacts(tmp_path: Path, capsys) -> None:
    train_value = {
        f"q{index:04}": {
            "question": f"Train {hashlib.sha256(str(index).encode()).hexdigest()}",
            "answer": f"Gold {index}",
        }
        for index in range(700)
    }
    public_value = {
        f"p{index}": {"question": f"Public question {index}", "answer": None} for index in range(3)
    }
    train_source = json.dumps(train_value).encode()
    public_source = json.dumps(public_value).encode()
    train_path = tmp_path / "train.json"
    public_path = tmp_path / "public.json"
    split_path = tmp_path / "split.v1.json"
    output = tmp_path / "baseline"
    train_path.write_bytes(train_source)
    public_path.write_bytes(public_source)
    imported = OrganizerQuestionReader().read_bytes(
        train_source, kind="train", artifact_path="train-source"
    )
    internal_bytes = imported.jsonl_bytes()
    split = build_split_manifest(
        tuple(SplitQuestion(row.question_id, row.question) for row in imported.records),
        (),
        source_checksum=checksum_bytes(internal_bytes),
        public_source_checksum=checksum_bytes(b""),
    )
    split_path.write_bytes(split.json_bytes())

    arguments = [
        "baseline",
        "build",
        "--train",
        str(train_path),
        "--public",
        str(public_path),
        "--split-manifest",
        str(split_path),
        "--output-directory",
        str(output),
    ]
    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert captured.out.startswith("BASELINE COMPLETE public=3 development=")
    assert captured.err == ""
    public_predictions = json.loads((output / "public/predictions.json").read_bytes())
    assert tuple(public_predictions) == ("p0", "p1", "p2")
    assert {row["answer"] for row in public_predictions.values()} == {FIXED_REFUSAL}
    development_predictions = json.loads((output / "development/predictions.json").read_bytes())
    development_references = json.loads((output / "development/references.json").read_bytes())
    assert tuple(development_predictions) == tuple(development_references)
    first_bytes = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert main(arguments) == 0
    capsys.readouterr()
    assert first_bytes == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
