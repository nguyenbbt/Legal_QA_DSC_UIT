"""CLI contracts for deterministic organizer context conversion."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.cli import main
from legal_rag.ingestion.organizer import OrganizerContextReader, OrganizerFile


def _write_context(path: Path, context_id: int, passage: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": context_id,
                "link": f"https://example.invalid/{context_id}",
                "passage": passage,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def test_organizer_import_contexts_writes_ordered_artifacts(tmp_path: Path, capsys) -> None:
    source = tmp_path / "contexts"
    source.mkdir()
    _write_context(source / "context_10.json", 10, "")
    _write_context(source / "context_2.json", 2, "Điều 1. Nội dung.")
    output = tmp_path / "contexts.jsonl"
    manifest = tmp_path / "contexts.manifest.json"
    errors = tmp_path / "contexts.errors.jsonl"

    exit_code = main(
        [
            "organizer",
            "import-contexts",
            "--input-dir",
            str(source),
            "--pattern",
            "context_*.json",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--errors",
            str(errors),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith(
        "IMPORT COMPLETE kind=contexts count=2 indexable=1 quarantined=1 sha256:"
    )
    assert captured.err == ""
    records = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert [record["context_id"] for record in records] == ["2", "10"]
    assert [record["source_position"] for record in records] == [0, 1]
    imported_manifest = json.loads(manifest.read_bytes())
    assert [row["context_id"] for row in imported_manifest["entries"]] == ["2", "10"]
    assert imported_manifest["entries"][1]["quarantine_reason"] == "EMPTY_PASSAGE"
    assert errors.read_bytes() == b""


def test_organizer_import_contexts_failure_does_not_write_outputs(tmp_path: Path, capsys) -> None:
    source = tmp_path / "contexts"
    source.mkdir()
    (source / "context_1.json").write_bytes(b'{"id":1,"link":"relative","passage":"private text"}')
    output = tmp_path / "contexts.jsonl"
    manifest = tmp_path / "contexts.manifest.json"
    errors = tmp_path / "contexts.errors.jsonl"

    exit_code = main(
        [
            "organizer",
            "import-contexts",
            "--input-dir",
            str(source),
            "--pattern",
            "context_*.json",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--errors",
            str(errors),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DATA_CONTEXT_LINK_INVALID" in captured.err
    assert not output.exists()
    assert not manifest.exists()
    error = json.loads(errors.read_bytes())
    assert error["code"] == "DATA_CONTEXT_LINK_INVALID"
    assert "private text" not in errors.read_text(encoding="utf-8")


def test_corpus_report_builds_streaming_chunk_artifacts(tmp_path: Path, capsys) -> None:
    source = ('{"id":1,"link":"https://example.invalid/1","passage":"Điều 1. Nội dung."}').encode()
    imported = OrganizerContextReader().read_files((OrganizerFile("context_1.json", source),))
    contexts = tmp_path / "contexts.jsonl"
    context_manifest = tmp_path / "contexts.manifest.json"
    contexts.write_bytes(imported.jsonl_bytes())
    context_manifest.write_bytes(imported.manifest_bytes())
    chunks = tmp_path / "chunks.jsonl"
    chunk_manifest = tmp_path / "chunks.manifest.json"
    json_report = tmp_path / "corpus.report.json"
    markdown_report = tmp_path / "corpus.report.md"

    exit_code = main(
        [
            "corpus",
            "report",
            "--contexts",
            str(contexts),
            "--context-manifest",
            str(context_manifest),
            "--chunks",
            str(chunks),
            "--chunk-manifest",
            str(chunk_manifest),
            "--json-report",
            str(json_report),
            "--markdown-report",
            str(markdown_report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith(
        "CORPUS COMPLETE contexts=1 indexable=1 quarantined=0 chunks=1 sha256:"
    )
    assert captured.err == ""
    assert chunks.is_file()
    assert chunk_manifest.is_file()
    assert json_report.is_file()
    assert markdown_report.is_file()
