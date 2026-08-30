from __future__ import annotations

import json
from pathlib import Path

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.dense_store import (
    MemmapDenseIndex,
    audit_dense_store,
    build_dense_store,
)
from legal_rag.retrieval.resumable_dense_store import (
    DenseBuildIdentity,
    DenseSourceRow,
    build_resumable_dense_store,
)


class _Backend:
    model_id = "fixture/embed"
    model_revision = "revision-1"
    dimension = 2

    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, float], ...]:
        return tuple((1.0, 0.0) if text == "alpha" else (0.0, 1.0) for text in texts)


def test_dense_store_round_trip_is_deterministic(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_bytes(
        b"".join(
            (json.dumps({"chunk_id": chunk_id, "retrieval_text": text}) + "\n").encode()
            for chunk_id, text in (("b", "beta"), ("a", "alpha"))
        )
    )
    output = tmp_path / "index"

    manifest = build_dense_store(chunks, output, _Backend(), batch_size=1)  # type: ignore[arg-type]
    index = MemmapDenseIndex(output, block_rows=1)
    results = index.retrieve([1.0, 0.0], limit=2)

    assert manifest.chunk_count == 2
    assert tuple(item.chunk_id for item in results) == ("a", "b")


def test_resumable_v2_dense_store_is_audited_and_searchable(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    rows: list[DenseSourceRow] = []
    offset = 0
    source = bytearray()
    for chunk_id, text, length in (("b", "beta", 2), ("a", "alpha", 1)):
        line = content_json_bytes({"chunk_id": chunk_id, "retrieval_text": text})
        rows.append(DenseSourceRow(chunk_id, length, offset, len(line)))
        source.extend(line)
        offset += len(line)
    chunks.write_bytes(source)
    ordered = tuple(sorted(rows, key=lambda row: (row.document_length, row.chunk_id.encode())))
    identity = DenseBuildIdentity(
        namespace="fixture-v2",
        model_id="fixture/embed",
        model_revision="revision-1",
        corpus_checksum=checksum_bytes(bytes(source)),
        model_artifact_checksum="sha256:" + "b" * 64,
        tokenizer_checksum="sha256:" + "c" * 64,
        indexing_config_checksum="sha256:" + "d" * 64,
        dimension=2,
        storage_dtype="float16",
        ordering="document-length-then-chunk-id.v1",
    )
    output = tmp_path / "index-v2"
    build_resumable_dense_store(
        chunks_path=chunks,
        output_directory=output,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=identity,
        source_rows=ordered,
        batch_size=1,
    )

    audit = audit_dense_store(
        output,
        expected_chunk_ids=tuple(row.chunk_id for row in ordered),
        block_rows=1,
    )
    index = MemmapDenseIndex(output, block_rows=1)
    results = index.retrieve([1.0, 0.0], limit=2)

    assert audit.chunk_count == 2
    assert audit.missing_chunk_count == 0
    assert audit.duplicate_chunk_count == 0
    assert audit.nonfinite_vector_count == 0
    assert audit.deterministic_mapping is True
    assert index.chunk_ids == ("a", "b")
    assert tuple(ids for ids, _vectors in index.vector_blocks()) == (("a",), ("b",))
    assert tuple(item.chunk_id for item in results) == ("a", "b")
