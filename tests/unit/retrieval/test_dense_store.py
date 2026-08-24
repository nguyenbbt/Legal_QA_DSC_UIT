from __future__ import annotations

import json
from pathlib import Path

from legal_rag.retrieval.dense_store import MemmapDenseIndex, build_dense_store


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
    results = MemmapDenseIndex(output, block_rows=1).retrieve([1.0, 0.0], limit=2)

    assert manifest.chunk_count == 2
    assert tuple(item.chunk_id for item in results) == ("a", "b")
