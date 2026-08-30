"""Measured bounded-runtime projection for D-066 sparse retrieval arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SparsePreflightInput:
    arm_id: str
    total_question_count: int
    observed_question_count: int
    observed_seconds_millis: int
    maximum_runtime_seconds: int
    worker_count: int

    def __post_init__(self) -> None:
        if (
            not self.arm_id
            or self.total_question_count < 1
            or self.observed_question_count < 1
            or self.observed_question_count > self.total_question_count
            or self.observed_seconds_millis < 1
            or self.maximum_runtime_seconds < 1
            or self.worker_count < 1
            or self.worker_count > 8
        ):
            raise ValueError("sparse preflight values must be bounded and positive")


@dataclass(frozen=True, slots=True)
class SparsePreflightReport:
    schema_version: str
    arm_id: str
    status: Literal["PASS", "BLOCKED"]
    blocker_codes: tuple[str, ...]
    total_question_count: int
    observed_question_count: int
    observed_seconds_millis: int
    projected_runtime_seconds: int
    maximum_runtime_seconds: int
    worker_count: int


def evaluate_sparse_preflight(value: SparsePreflightInput) -> SparsePreflightReport:
    """Project the full immutable population without hiding validation startup time."""

    projected = math.ceil(
        value.total_question_count
        * value.observed_seconds_millis
        / value.observed_question_count
        / 1000.0
    )
    blockers = ("OPS002_RUNTIME_LIMIT",) if projected > value.maximum_runtime_seconds else ()
    return SparsePreflightReport(
        schema_version="retrieval.sparse-preflight.v1",
        arm_id=value.arm_id,
        status="BLOCKED" if blockers else "PASS",
        blocker_codes=blockers,
        total_question_count=value.total_question_count,
        observed_question_count=value.observed_question_count,
        observed_seconds_millis=value.observed_seconds_millis,
        projected_runtime_seconds=projected,
        maximum_runtime_seconds=value.maximum_runtime_seconds,
        worker_count=value.worker_count,
    )


def sample_timeout_seconds(
    *,
    total_question_count: int,
    sample_question_count: int,
    maximum_runtime_seconds: int,
) -> int:
    """Return the first whole sample second whose linear projection must exceed the cap."""

    if (
        total_question_count < 1
        or sample_question_count < 1
        or sample_question_count > total_question_count
        or maximum_runtime_seconds < 1
    ):
        raise ValueError("sparse timeout values must be bounded and positive")
    return math.floor(maximum_runtime_seconds * sample_question_count / total_question_count) + 1


__all__ = [
    "SparsePreflightInput",
    "SparsePreflightReport",
    "evaluate_sparse_preflight",
    "sample_timeout_seconds",
]
