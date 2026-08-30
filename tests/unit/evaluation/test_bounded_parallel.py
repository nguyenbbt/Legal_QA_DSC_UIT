from __future__ import annotations

import threading

import pytest

from legal_rag.evaluation.bounded_parallel import ordered_bounded_map


def test_bounded_parallel_map_preserves_input_order_and_uses_bounded_threads() -> None:
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def work(value: int) -> int:
        with lock:
            thread_ids.add(threading.get_ident())
        return value * value

    output = ordered_bounded_map(work, tuple(range(32)), max_workers=4)

    assert output == tuple(value * value for value in range(32))
    assert 1 <= len(thread_ids) <= 4


def test_bounded_parallel_map_rejects_unbounded_worker_counts() -> None:
    with pytest.raises(ValueError, match="within"):
        ordered_bounded_map(str, (1,), max_workers=0)
    with pytest.raises(ValueError, match="within"):
        ordered_bounded_map(str, (1,), max_workers=9)


def test_bounded_parallel_map_reports_completed_ordered_positions() -> None:
    progress: list[tuple[int, int]] = []

    output = ordered_bounded_map(
        str,
        (3, 2, 1),
        max_workers=2,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert output == ("3", "2", "1")
    assert progress == [(1, 3), (2, 3), (3, 3)]
