from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.dense import DenseRetrievalError
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


def _source(tmp_path: Path) -> tuple[Path, tuple[DenseSourceRow, ...]]:
    path = tmp_path / "chunks.jsonl"
    rows: list[DenseSourceRow] = []
    offset = 0
    values = (("b", "beta", 2), ("a", "alpha", 1), ("c", "beta", 3))
    data = bytearray()
    for chunk_id, text, length in values:
        line = content_json_bytes({"chunk_id": chunk_id, "retrieval_text": text})
        rows.append(DenseSourceRow(chunk_id, length, offset, len(line)))
        data.extend(line)
        offset += len(line)
    path.write_bytes(bytes(data))
    return path, tuple(sorted(rows, key=lambda row: (row.document_length, row.chunk_id.encode())))


def _identity(source: Path, *, config: str | None = None) -> DenseBuildIdentity:
    return DenseBuildIdentity(
        namespace="d066-fixture",
        model_id="fixture/embed",
        model_revision="revision-1",
        corpus_checksum=checksum_bytes(source.read_bytes()),
        model_artifact_checksum="sha256:" + "b" * 64,
        tokenizer_checksum="sha256:" + "c" * 64,
        indexing_config_checksum=config or "sha256:" + "d" * 64,
        dimension=2,
        storage_dtype="float16",
        ordering="document-length-then-chunk-id.v1",
    )


def test_resumable_dense_store_resumes_and_matches_one_shot_bytes(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    resumed_dir = tmp_path / "resumed"

    interrupted = build_resumable_dense_store(
        chunks_path=source,
        output_directory=resumed_dir,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
        maximum_batches=1,
    )

    assert interrupted.status == "INCOMPLETE"
    assert interrupted.completed_count == 1
    assert (resumed_dir / "checkpoint.json").is_file()
    with (resumed_dir / "chunk-ids.jsonl.partial").open("ab") as stream:
        stream.write(content_json_bytes({"row": 1, "chunk_id": "uncheckpointed-tail"}))
    progress: list[tuple[int, int]] = []
    completed = build_resumable_dense_store(
        chunks_path=source,
        output_directory=resumed_dir,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
        progress=lambda completed_count, total_count: progress.append(
            (completed_count, total_count)
        ),
    )
    one_shot_dir = tmp_path / "one-shot"
    one_shot = build_resumable_dense_store(
        chunks_path=source,
        output_directory=one_shot_dir,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
    )

    assert completed.status == "COMPLETE"
    assert completed.completed_count == 3
    assert completed.manifest_checksum == one_shot.manifest_checksum
    assert (resumed_dir / "vectors.f16.npy").read_bytes() == (
        one_shot_dir / "vectors.f16.npy"
    ).read_bytes()
    assert (resumed_dir / "chunk-ids.jsonl").read_bytes() == (
        one_shot_dir / "chunk-ids.jsonl"
    ).read_bytes()
    ids = [
        json.loads(line)["chunk_id"]
        for line in (resumed_dir / "chunk-ids.jsonl").read_bytes().splitlines()
    ]
    assert ids == ["a", "b", "c"]
    assert progress[-1] == (3, 3)


def test_resumable_dense_store_rejects_stale_checkpoint_identity(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    output = tmp_path / "index"
    build_resumable_dense_store(
        chunks_path=source,
        output_directory=output,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
        maximum_batches=1,
    )

    with pytest.raises(DenseRetrievalError) as captured:
        build_resumable_dense_store(
            chunks_path=source,
            output_directory=output,
            backend=_Backend(),  # type: ignore[arg-type]
            identity=_identity(source, config="sha256:" + "e" * 64),
            source_rows=rows,
            batch_size=1,
        )

    assert captured.value.code == "DENSE_CHECKPOINT_FINGERPRINT_MISMATCH"


def test_resumable_dense_store_rejects_tampered_complete_files(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    output = tmp_path / "index"
    build_resumable_dense_store(
        chunks_path=source,
        output_directory=output,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
    )
    with (output / "chunk-ids.jsonl").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(DenseRetrievalError) as captured:
        build_resumable_dense_store(
            chunks_path=source,
            output_directory=output,
            backend=_Backend(),  # type: ignore[arg-type]
            identity=_identity(source),
            source_rows=rows,
            batch_size=1,
        )

    assert captured.value.code == "DENSE_INDEX_CHECKSUM_MISMATCH"


def test_resumable_dense_store_rejects_tampered_vector_prefix(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    output = tmp_path / "index"
    build_resumable_dense_store(
        chunks_path=source,
        output_directory=output,
        backend=_Backend(),  # type: ignore[arg-type]
        identity=_identity(source),
        source_rows=rows,
        batch_size=1,
        maximum_batches=1,
    )
    vectors = np.load(output / "vectors.f16.npy.partial", mmap_mode="r+")
    vectors[0, 0] = np.float16(0.5)
    vectors.flush()
    del vectors

    with pytest.raises(DenseRetrievalError) as captured:
        build_resumable_dense_store(
            chunks_path=source,
            output_directory=output,
            backend=_Backend(),  # type: ignore[arg-type]
            identity=_identity(source),
            source_rows=rows,
            batch_size=1,
        )

    assert captured.value.code == "DENSE_CHECKPOINT_INVALID"
