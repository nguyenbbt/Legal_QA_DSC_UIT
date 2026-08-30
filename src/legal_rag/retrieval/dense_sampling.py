"""Deterministic length-stratified sampling for dense-index resource preflight."""

from __future__ import annotations


def stratified_positions(*, total_count: int, sample_count: int) -> tuple[int, ...]:
    """Select evenly spaced positions including both endpoints of a frozen ordering."""

    if total_count < 2 or sample_count < 2 or sample_count > total_count:
        raise ValueError("dense preflight sample must contain [2, total_count] rows")
    return tuple(
        position * (total_count - 1) // (sample_count - 1) for position in range(sample_count)
    )


__all__ = ["stratified_positions"]
