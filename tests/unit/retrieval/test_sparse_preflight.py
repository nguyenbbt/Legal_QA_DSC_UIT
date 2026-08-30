from __future__ import annotations

from legal_rag.retrieval.sparse_preflight import (
    SparsePreflightInput,
    evaluate_sparse_preflight,
    sample_timeout_seconds,
)


def test_sparse_preflight_projects_full_population_from_fixed_sample() -> None:
    report = evaluate_sparse_preflight(
        SparsePreflightInput(
            arm_id="R-DISC-0-BM25",
            total_question_count=2_391,
            observed_question_count=100,
            observed_seconds_millis=600_000,
            maximum_runtime_seconds=21_600,
            worker_count=4,
        )
    )

    assert report.projected_runtime_seconds == 14_346
    assert report.status == "PASS"
    assert report.blocker_codes == ()


def test_sparse_preflight_blocks_projection_over_ops002() -> None:
    report = evaluate_sparse_preflight(
        SparsePreflightInput(
            arm_id="R-DISC-0-BM25",
            total_question_count=2_391,
            observed_question_count=100,
            observed_seconds_millis=1_000_000,
            maximum_runtime_seconds=21_600,
            worker_count=4,
        )
    )

    assert report.status == "BLOCKED"
    assert report.blocker_codes == ("OPS002_RUNTIME_LIMIT",)


def test_sparse_sample_timeout_is_first_whole_second_that_cannot_pass() -> None:
    assert (
        sample_timeout_seconds(
            total_question_count=2_391,
            sample_question_count=100,
            maximum_runtime_seconds=21_600,
        )
        == 904
    )
