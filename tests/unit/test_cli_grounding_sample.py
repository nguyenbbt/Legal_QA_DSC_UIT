"""CLI contract for freezing the pre-index grounding sample."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.split import SplitQuestion, build_split_manifest
from legal_rag.ingestion.organizer import OrganizerQuestionReader


def test_grounding_sample_cli_freezes_sixty_development_ids(tmp_path: Path, capsys) -> None:
    organizer = {
        f"q{index:04}": {
            "question": f"{'Điều 1' if index % 2 else 'Quy định'} nội dung {index}",
            "answer": f"Đáp án {index}",
        }
        for index in range(700)
    }
    source = json.dumps(organizer, ensure_ascii=False).encode()
    imported = OrganizerQuestionReader().read_bytes(source, kind="train", artifact_path="fixture")
    question_bytes = imported.jsonl_bytes()
    questions_path = tmp_path / "train.questions.jsonl"
    split_path = tmp_path / "split.v1.json"
    output = tmp_path / "grounding.sample.v1.json"
    questions_path.write_bytes(question_bytes)
    manifest = build_split_manifest(
        tuple(SplitQuestion(row.question_id, row.question) for row in imported.records),
        (),
        source_checksum=checksum_bytes(question_bytes),
        public_source_checksum=checksum_bytes(b""),
    )
    split_path.write_bytes(manifest.json_bytes())

    exit_code = main(
        [
            "grounding",
            "sample",
            "--questions",
            str(questions_path),
            "--split-manifest",
            str(split_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("GROUNDING SAMPLE COMPLETE selected=60 eligible=")
    assert captured.err == ""
    rendered = json.loads(output.read_bytes())
    assert rendered["sample_size"] == 60
    assert len(set(rendered["selected_question_ids"])) == 60
