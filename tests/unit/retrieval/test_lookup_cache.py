from __future__ import annotations

from legal_rag.retrieval.lookup_cache import CachedExpandedSparseIndex


class _Source:
    index_checksum = "sha256:" + "a" * 64

    def __init__(self) -> None:
        self.context_calls = 0
        self.coordinate_calls = 0
        self.retrieve_calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, candidate_limit: int = 12):  # type: ignore[no-untyped-def]
        self.retrieve_calls.append((query, candidate_limit))
        return (query, candidate_limit)

    def chunks_for_context(self, context_id: str):  # type: ignore[no-untyped-def]
        self.context_calls += 1
        return (f"context:{context_id}",)

    def chunks_for_coordinate(self, kind: str, ordinal: str | None):  # type: ignore[no-untyped-def]
        self.coordinate_calls += 1
        return (f"coordinate:{kind}:{ordinal}",)


def test_lookup_cache_preserves_results_and_avoids_repeated_immutable_reads() -> None:
    source = _Source()
    cached = CachedExpandedSparseIndex(source)  # type: ignore[arg-type]

    assert cached.chunks_for_context("1") == ("context:1",)
    assert cached.chunks_for_context("1") == ("context:1",)
    assert cached.chunks_for_coordinate("article", "1") == ("coordinate:article:1",)
    assert cached.chunks_for_coordinate("article", "1") == ("coordinate:article:1",)
    assert source.context_calls == 1
    assert source.coordinate_calls == 1
    assert cached.index_checksum == source.index_checksum


def test_lookup_cache_does_not_cache_query_rankings_or_cross_instances() -> None:
    source = _Source()
    first = CachedExpandedSparseIndex(source)  # type: ignore[arg-type]
    second = CachedExpandedSparseIndex(source)  # type: ignore[arg-type]

    assert first.retrieve("q", candidate_limit=50) == ("q", 50)
    assert first.retrieve("q", candidate_limit=50) == ("q", 50)
    first.chunks_for_context("1")
    second.chunks_for_context("1")

    assert source.retrieve_calls == [("q", 50), ("q", 50)]
    assert source.context_calls == 2
