"""Streaming real-corpus chunk artifact and report contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.domain.checksums import checksum_file_set
from legal_rag.ingestion.corpus import (
    CorpusBuildError,
    corpus_checksum_from_import_manifest,
    write_corpus_artifacts,
)
from legal_rag.ingestion.organizer import OrganizerContextReader, OrganizerFile

_CORPUS_CHECKSUM = "sha256:" + "1" * 64
_IMPORT_MANIFEST_CHECKSUM = "sha256:" + "2" * 64


def _context_bytes() -> bytes:
    imported = OrganizerContextReader().read_files(
        (
            OrganizerFile(
                "context_1.json",
                (
                    '{"id":1,"link":"https://example.invalid/1",'
                    '"passage":"Điều 1. Phạm vi\\n1. Nội dung của khoản."}'
                ).encode(),
            ),
            OrganizerFile(
                "context_2.json",
                b'{"id":2,"link":"https://example.invalid/2","passage":""}',
            ),
        )
    )
    return imported.jsonl_bytes()


def test_corpus_builder_streams_chunks_and_reports_quarantine(tmp_path: Path) -> None:
    contexts_path = tmp_path / "contexts.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "chunks.manifest.json"
    report_path = tmp_path / "corpus.report.json"
    markdown_path = tmp_path / "corpus.report.md"
    contexts_path.write_bytes(_context_bytes())

    first = write_corpus_artifacts(
        contexts_path=contexts_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        report_path=report_path,
        markdown_path=markdown_path,
        corpus_checksum=_CORPUS_CHECKSUM,
        context_import_manifest_checksum=_IMPORT_MANIFEST_CHECKSUM,
    )
    second = write_corpus_artifacts(
        contexts_path=contexts_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        report_path=report_path,
        markdown_path=markdown_path,
        corpus_checksum=_CORPUS_CHECKSUM,
        context_import_manifest_checksum=_IMPORT_MANIFEST_CHECKSUM,
    )

    chunks = [json.loads(line) for line in chunks_path.read_bytes().splitlines()]
    report = json.loads(report_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    source_context = json.loads(contexts_path.read_bytes().splitlines()[0])

    assert first == second
    assert first.context_count == 2
    assert first.indexable_context_count == 1
    assert first.quarantined_context_count == 1
    assert first.chunk_count == len(chunks) > 0
    assert {chunk["context_id"] for chunk in chunks} == {"1"}
    for chunk in chunks:
        assert (
            source_context["passage"][chunk["canonical_start"] : chunk["canonical_end"]]
            == chunk["display_text"]
        )
    assert report["context_counts"] == {"indexable": 1, "quarantined": 1, "total": 2}
    assert report["chunk_count"] == len(chunks)
    assert report["token_percentiles"]["method"] == "nearest_rank_integer"
    assert manifest["chunks_artifact_checksum"] == first.chunks_checksum
    assert manifest["corpus_checksum"] == _CORPUS_CHECKSUM
    assert markdown_path.read_text(encoding="utf-8").startswith("# MIL-004 Corpus Report\n")


def test_corpus_builder_rejects_nonconsecutive_source_positions_atomically(
    tmp_path: Path,
) -> None:
    rows = _context_bytes().splitlines(keepends=True)
    contexts_path = tmp_path / "contexts.jsonl"
    contexts_path.write_bytes(b"".join(reversed(rows)))
    chunks_path = tmp_path / "chunks.jsonl"

    with pytest.raises(CorpusBuildError) as captured:
        write_corpus_artifacts(
            contexts_path=contexts_path,
            chunks_path=chunks_path,
            manifest_path=tmp_path / "chunks.manifest.json",
            report_path=tmp_path / "corpus.report.json",
            markdown_path=tmp_path / "corpus.report.md",
            corpus_checksum=_CORPUS_CHECKSUM,
            context_import_manifest_checksum=_IMPORT_MANIFEST_CHECKSUM,
        )

    assert captured.value.code == "CORPUS_SOURCE_ORDER_INVALID"
    assert not chunks_path.exists()


def test_import_manifest_reconstructs_the_raw_file_set_checksum(tmp_path: Path) -> None:
    first = tmp_path / "context_2.json"
    second = tmp_path / "context_10.json"
    first.write_bytes(b'{"id":2,"link":"https://example.invalid/2","passage":"first"}')
    second.write_bytes(b'{"id":10,"link":"https://example.invalid/10","passage":"second"}')
    imported = OrganizerContextReader().read_files(
        (
            OrganizerFile("context_2.json", first.read_bytes()),
            OrganizerFile("context_10.json", second.read_bytes()),
        )
    )

    expected = checksum_file_set(tmp_path, ("context_2.json", "context_10.json"))

    assert corpus_checksum_from_import_manifest(imported.manifest_bytes()) == expected.checksum
