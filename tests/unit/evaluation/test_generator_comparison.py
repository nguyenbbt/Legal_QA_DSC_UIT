from __future__ import annotations

import json

import pytest

from legal_rag.evaluation.generator_comparison import (
    GeneratorComparisonError,
    compare_generator_experiments,
)


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join((json.dumps(row) + "\n").encode() for row in rows)


def _manifest(*, run_id: str, baseline_run_id: str | None, adapter: bool) -> bytes:
    value: dict[str, object] = {
        "run_id": run_id,
        "model_id": "fixture/model",
        "model_revision": "revision-1",
        "prompt_checksum": "sha256:" + "a" * 64,
        "retrieval_output_checksum": "sha256:" + "b" * 64,
        "annotation_queue_checksum": "sha256:" + "c" * 64,
        "evidence_limit": 3,
        "decoding": {
            "maximum_input_tokens": 2048,
            "maximum_new_tokens": 512,
            "do_sample": False,
            "enable_thinking": False,
        },
        "comparison": {
            "baseline_run_id": baseline_run_id,
            "changed_axes": ["adapter"] if adapter else [],
        },
    }
    if adapter:
        value["adapter"] = {
            "adapter_id": "fixture-adapter",
            "adapter_checksum": "sha256:" + "d" * 64,
            "adapter_config_checksum": "sha256:" + "e" * 64,
        }
    return json.dumps(value).encode()


def test_comparison_rejects_metric_and_runtime_regression_deterministically() -> None:
    baseline = _jsonl(
        [
            {"question_id": "q1", "meteor": 0.3, "rouge_l": 0.4},
            {"question_id": "q2", "meteor": 0.2, "rouge_l": 0.3},
        ]
    )
    candidate = _jsonl(
        [
            {"question_id": "q1", "meteor": 0.2, "rouge_l": 0.3},
            {"question_id": "q2", "meteor": 0.1, "rouge_l": 0.2},
        ]
    )
    arguments = {
        "baseline_per_query_data": baseline,
        "candidate_per_query_data": candidate,
        "baseline_manifest_data": _manifest(run_id="G1", baseline_run_id=None, adapter=False),
        "candidate_manifest_data": _manifest(run_id="G3", baseline_run_id="G1", adapter=True),
        "baseline_runtime_seconds": 10.0,
        "candidate_runtime_seconds": 15.0,
    }

    first = compare_generator_experiments(**arguments)
    second = compare_generator_experiments(**arguments)

    assert first == second
    value = json.loads(first)
    assert value["numeric_evaluation_gate"] == "failed"
    assert value["resource_gate"] == "failed"
    assert value["promotion_state"] == "rejected_preserved"
    assert set(value["promotion_blockers"]) == {
        "METEOR_DELTA_BELOW_THRESHOLD",
        "METEOR_CI_LOWER_NOT_POSITIVE",
        "ROUGE_L_REGRESSION_EXCEEDS_GUARDRAIL",
        "RUNTIME_REGRESSION_EXCEEDS_25_PERCENT",
    }


def test_comparison_rejects_a_non_fixed_generator_axis() -> None:
    baseline_manifest = json.loads(_manifest(run_id="G1", baseline_run_id=None, adapter=False))
    baseline_manifest["prompt_checksum"] = "sha256:" + "f" * 64

    with pytest.raises(GeneratorComparisonError) as caught:
        compare_generator_experiments(
            baseline_per_query_data=_jsonl([{"question_id": "q1", "meteor": 0.2, "rouge_l": 0.3}]),
            candidate_per_query_data=_jsonl([{"question_id": "q1", "meteor": 0.3, "rouge_l": 0.4}]),
            baseline_manifest_data=json.dumps(baseline_manifest).encode(),
            candidate_manifest_data=_manifest(run_id="G3", baseline_run_id="G1", adapter=True),
            baseline_runtime_seconds=10.0,
            candidate_runtime_seconds=10.0,
        )

    assert caught.value.code == "GENERATOR_COMPARISON_NOT_FIXED"
