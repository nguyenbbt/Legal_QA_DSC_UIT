from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from legal_rag.cli import main
from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, ChunkRecord, chunk_context
from legal_rag.retrieval.bm25 import APPROVED_BM25_RUNTIME_ID, build_bm25_index
from legal_rag.retrieval.disk_bm25 import (
    DiskBm25Error,
    build_disk_bm25_index,
    open_disk_bm25_index,
)


def chunk_for(context_id: int, text: str) -> ChunkRecord:
    context = ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": str(context_id),
            "original_id": str(context_id),
            "original_id_kind": "json_integer",
            "source_position": context_id,
            "source_artifact": f"fixtures/context_{context_id}.json",
            "source_checksum": checksum_bytes(text.encode()),
            "name": None,
            "source_url": f"https://example.invalid/{context_id}",
            "passage": text,
            "indexable": True,
            "quarantine_reason": None,
        }
    )
    return chunk_context(context, config=ChunkingConfig(minimum_fragment_tokens=1)).chunks[0]


def build_index(chunks: tuple[ChunkRecord, ...]):
    return build_bm25_index(
        chunks,
        corpus_checksum=checksum_bytes(b"fixture-corpus"),
        alias_manifest_checksum=checksum_bytes(b"fixture-aliases"),
        runtime_compatibility_id=APPROVED_BM25_RUNTIME_ID,
    )


def _chunk_bytes(*texts: str) -> bytes:
    rows = []
    for context_id, text in enumerate(texts, start=1):
        chunk = chunk_for(context_id, text)
        rows.append(
            {
                "schema_version": "retrieval.chunk.v1",
                "chunk_id": chunk.chunk_id,
                "context_id": chunk.context_id,
                "source_url": chunk.source_url,
                "hierarchy_path": list(chunk.hierarchy_path),
                "hierarchy_rule_id": chunk.hierarchy_rule_id,
                "hierarchy_kind": chunk.hierarchy_kind,
                "hierarchy_ordinal": chunk.hierarchy_ordinal,
                "canonical_start": chunk.canonical_start,
                "canonical_end": chunk.canonical_end,
                "display_text": chunk.display_text,
                "retrieval_text": chunk.retrieval_text,
                "window_index": chunk.window_index,
                "chunk_checksum": chunk.chunk_checksum,
                "context_checksum": chunk.context_checksum,
            }
        )
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in rows
    )


def _build(tmp_path: Path, name: str, chunks_data: bytes):
    output = tmp_path / name
    output.mkdir()
    chunks_path = output / "chunks.jsonl"
    chunks_path.write_bytes(chunks_data)
    database_path = output / "index.sqlite3"
    manifest_path = output / "index.manifest.json"
    summary = build_disk_bm25_index(
        chunks_path=chunks_path,
        chunks_checksum=checksum_bytes(chunks_data),
        corpus_checksum=checksum_bytes(b"corpus"),
        alias_manifest_checksum=checksum_bytes(b"aliases"),
        database_path=database_path,
        manifest_path=manifest_path,
        runtime_compatibility_id=APPROVED_BM25_RUNTIME_ID,
    )
    return chunks_path, database_path, manifest_path, summary


def test_disk_bm25_matches_the_reference_scores_and_order(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("luật luật mẫu", "luật khác", "điều mẫu")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)
    disk = open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    )
    memory = build_index(
        tuple(
            chunk_for(i, text)
            for i, text in enumerate(("luật luật mẫu", "luật khác", "điều mẫu"), start=1)
        )
    )

    disk_result = disk.retrieve("luật điều")
    memory_result = memory.retrieve("luật điều")

    assert [item.chunk.chunk_id for item in disk_result.candidates] == [
        item.chunk.chunk_id for item in memory_result.candidates
    ]
    assert [item.sparse_score for item in disk_result.candidates] == [
        item.sparse_score for item in memory_result.candidates
    ]
    assert disk_result.index_checksum == disk.index_checksum
    assert tuple(chunk.context_id for chunk in disk.chunks_for_context("1")) == ("1",)
    assert tuple(chunk.context_id for chunk in disk.chunks_for_coordinate("root", None)) == (
        "1",
        "2",
        "3",
    )
    requested_ids = tuple(item.chunk.chunk_id for item in disk_result.candidates[:2])
    assert tuple(chunk.chunk_id for chunk in disk.chunks_by_ids(requested_ids)) == requested_ids
    disk.close()


def test_disk_bm25_database_and_manifest_are_byte_deterministic(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("luật mẫu", "điều luật")
    _chunks, first_database, first_manifest, first = _build(tmp_path, "first", chunks_data)
    _chunks, second_database, second_manifest, second = _build(tmp_path, "second", chunks_data)

    assert first.database_checksum == second.database_checksum
    assert first.index_checksum == second.index_checksum
    assert first_database.read_bytes() == second_database.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_disk_bm25_rejects_changed_chunk_bytes_before_query(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("luật mẫu")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)
    chunks_path.write_bytes(chunks_data + b"\n")

    with pytest.raises(DiskBm25Error, match="chunk artifact checksum") as mismatch:
        open_disk_bm25_index(
            database_path=database_path,
            chunks_path=chunks_path,
            manifest_data=manifest_path.read_bytes(),
        )

    assert mismatch.value.code == "SPARSE_CHUNK_CHECKSUM_MISMATCH"


def test_disk_bm25_returns_the_canonical_nfc_query(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("luật mẫu")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)
    decomposed = unicodedata.normalize("NFD", "luật")

    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    ) as index:
        result = index.retrieve(decomposed)

    assert result.query == "luật"


def test_index_build_cli_consumes_checksum_linked_chunk_and_alias_manifests(
    tmp_path: Path, capsys
) -> None:
    chunks_data = _chunk_bytes("luật mẫu", "điều luật")
    chunks_path = tmp_path / "chunks.v1.jsonl"
    chunks_path.write_bytes(chunks_data)
    aliases_path = tmp_path / "aliases.active.v1.jsonl"
    aliases_path.write_bytes(b"")
    corpus_checksum = checksum_bytes(b"corpus")
    chunk_manifest = tmp_path / "corpus.chunks.v1.json"
    chunk_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "corpus.chunk.manifest.v1",
                "chunking_version": "chunking.v1",
                "tokenizer_id": "legal-retrieval-unicode-v1",
                "tokenizer_revision": "unicode-15.0.0",
                "unicode_version": "15.0.0",
                "corpus_checksum": corpus_checksum,
                "context_import_manifest_checksum": checksum_bytes(b"context-import"),
                "context_artifact_checksum": checksum_bytes(b"contexts"),
                "chunks_artifact_checksum": checksum_bytes(chunks_data),
                "context_count": 2,
                "indexable_context_count": 2,
                "quarantined_context_count": 0,
                "chunk_count": 2,
            }
        )
    )
    alias_manifest = tmp_path / "aliases.active.v1.json"
    alias_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "legal.reference.alias.manifest.v1",
                "document_key_version": "legal-document-number-key.v1",
                "unicode_version": "15.0.0",
                "corpus_checksum": corpus_checksum,
                "ordered_files": [
                    {
                        "path": aliases_path.name,
                        "checksum": checksum_bytes(b""),
                    }
                ],
                "record_count": 0,
                "aggregate_checksum": checksum_bytes(b""),
            }
        )
    )

    exit_code = main(
        [
            "index",
            "build",
            "--chunks",
            str(chunks_path),
            "--chunk-manifest",
            str(chunk_manifest),
            "--aliases",
            str(aliases_path),
            "--alias-manifest",
            str(alias_manifest),
            "--database",
            str(tmp_path / "bm25.v1.sqlite3"),
            "--manifest",
            str(tmp_path / "bm25.v1.manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("INDEX BUILD COMPLETE documents=2 sha256:")
    assert captured.err == ""
