"""CLI contract for exact development evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.cli import main


def test_evaluate_competition_writes_exact_reports(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions.json"
    references = tmp_path / "references.json"
    per_query = tmp_path / "per-query.jsonl"
    report = tmp_path / "report.json"
    predictions.write_text(json.dumps({"q": {"answer": "one"}}), encoding="utf-8")
    references.write_text(json.dumps({"q": "one"}), encoding="utf-8")

    exit_code = main(
        [
            "evaluate",
            "competition",
            "--predictions",
            str(predictions),
            "--references",
            str(references),
            "--mode",
            "official_exact",
            "--per-query",
            str(per_query),
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("EVALUATION COMPLETE questions=1 meteor=")
    assert captured.err == ""
    assert json.loads(report.read_bytes())["mode"] == "official_exact"
    assert json.loads(per_query.read_bytes())["question_id"] == "q"
