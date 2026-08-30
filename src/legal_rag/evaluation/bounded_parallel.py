"""Small deterministic boundary for bounded local parallel execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor


def ordered_bounded_map[InputT, OutputT](
    function: Callable[[InputT], OutputT],
    values: Sequence[InputT],
    *,
    max_workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[OutputT, ...]:
    """Execute with at most eight local threads and return exact input order."""

    if max_workers < 1 or max_workers > 8:
        raise ValueError("bounded worker count must be within [1, 8]")
    outputs: list[OutputT] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="legal-rag-local") as pool:
        for position, output in enumerate(pool.map(function, values), start=1):
            outputs.append(output)
            if progress is not None:
                progress(position, len(values))
    return tuple(outputs)


__all__ = ["ordered_bounded_map"]
