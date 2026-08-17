from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, chunk_context


def context_record(passage: str) -> ContextRecord:
    return ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": "7",
            "original_id": "7",
            "original_id_kind": "json_integer",
            "source_position": 0,
            "source_artifact": "fixtures/context_7.json",
            "source_checksum": checksum_bytes(passage.encode()),
            "name": "Luật mẫu",
            "source_url": "https://example.invalid/7",
            "passage": passage,
            "indexable": True,
            "quarantine_reason": None,
        }
    )


def test_chunker_uses_leaf_hierarchy_units_and_preserves_headers() -> None:
    context = context_record("Điều 1\n1. Người đủ 18 tuổi được cấp thẻ.\n2. Hồ sơ phải đầy đủ.\n")

    result = chunk_context(context, config=ChunkingConfig(minimum_fragment_tokens=2))

    assert len(result.chunks) == 2
    assert result.chunks[0].hierarchy_path == ("Điều 1", "Khoản 1")
    assert result.chunks[0].display_text.startswith("Điều 1\n1. Người")
    assert result.chunks[1].hierarchy_path == ("Điều 1", "Khoản 2")
    assert all(
        context.passage[chunk.canonical_start : chunk.canonical_end] == chunk.display_text
        for chunk in result.chunks
    )


def test_hierarchy_free_fallback_uses_512_token_windows_with_64_overlap() -> None:
    passage = " ".join(f"t{index}" for index in range(600))
    context = context_record(passage)

    result = chunk_context(context)

    assert len(result.chunks) == 2
    assert len(result.chunks[0].retrieval_text.split()) == 512
    assert (
        result.chunks[0].retrieval_text.split()[-64:]
        == result.chunks[1].retrieval_text.split()[:64]
    )
    assert result.chunks[1].window_index == 1


def test_chunk_ids_and_manifest_are_stable_for_identical_input() -> None:
    context = context_record("Điều 1. Nội dung ổn định.")

    first = chunk_context(context)
    second = chunk_context(context)

    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.manifest_bytes() == second.manifest_bytes()
    assert all(
        chunk.chunk_id.startswith("chunk_") and len(chunk.chunk_id) == 30 for chunk in first.chunks
    )


def test_long_hierarchy_unit_uses_configured_window_and_overlap() -> None:
    passage = "Điều 1. " + " ".join(f"n{index}" for index in range(30))
    context = context_record(passage)
    config = ChunkingConfig(
        hierarchy_max_tokens=10,
        window_tokens=8,
        overlap_tokens=2,
        minimum_fragment_tokens=2,
    )

    result = chunk_context(context, config=config)

    assert len(result.chunks) > 1
    assert (
        result.chunks[0].retrieval_text.split()[-2:] == result.chunks[1].retrieval_text.split()[:2]
    )
    assert [chunk.window_index for chunk in result.chunks] == list(range(len(result.chunks)))


def test_short_first_fragment_attaches_to_following_compatible_unit() -> None:
    context = context_record("Điều 1\n1. Ngắn.\n2. Cũng ngắn.\n")

    result = chunk_context(context)

    assert len(result.chunks) == 1
    assert result.chunks[0].hierarchy_path == ("Điều 1", "Khoản 2")
    assert result.chunks[0].display_text == context.passage


def test_quarantined_context_produces_no_chunks() -> None:
    context = ContextRecord.model_validate(
        {
            **context_record("x").model_dump(mode="json"),
            "passage": "",
            "indexable": False,
            "quarantine_reason": "EMPTY_PASSAGE",
        }
    )

    result = chunk_context(context)

    assert result.chunks == ()


def test_legal_content_that_looks_like_an_absolute_path_remains_chunkable() -> None:
    context = context_record("Điều 1.\n/private/path is quoted legal content.")

    first = chunk_context(context)
    second = chunk_context(context)

    assert first.chunks
    assert [chunk.chunk_checksum for chunk in first.chunks] == [
        chunk.chunk_checksum for chunk in second.chunks
    ]
