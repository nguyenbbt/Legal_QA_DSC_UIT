"""Operational measurement contracts for the fixed-refusal baseline."""

from __future__ import annotations

import json

from legal_rag.evaluation.performance import measure_fixed_refusal
from legal_rag.generation.fixture import FIXED_REFUSAL


def test_measurement_labels_exact_methods_and_unavailable_resources() -> None:
    source = json.dumps(
        {
            "p1": {"question": "Question one", "answer": None},
            "p2": {"question": "Question two", "answer": None},
        }
    ).encode()

    measurement = measure_fixed_refusal(source, run_id="run_" + "a" * 24, warm_samples=3)

    assert measurement["schema_version"] == "fixed.refusal.measurement.v1"
    assert measurement["question_count"] == 2
    assert measurement["cold_sample_count"] == 1
    assert measurement["warm_sample_count"] == 3
    assert measurement["quantile_method"] == "linear"
    for temperature in ("cold", "warm"):
        for stage in (
            "input_parse",
            "fixed_refusal_generation",
            "submission_render_validate",
            "end_to_end",
        ):
            summary = measurement["latency_ms"][temperature][stage]
            assert summary["p50"] >= 0
            assert summary["p95"] >= summary["p50"]
    assert measurement["cpu_memory"]["method"] == "python_tracemalloc_peak"
    assert measurement["cpu_memory"]["peak_bytes"] > 0
    assert measurement["gpu_vram"]["status"] == "not_applicable"
    assert measurement["index_disk"]["bytes"] == 0
    assert measurement["generated_tokens"]["method"] == (
        "legal-retrieval-unicode-v1_proxy_not_model_tokens"
    )
    assert measurement["generated_tokens"]["per_answer"] > 0
    assert measurement["generated_tokens"]["total"] == (
        measurement["generated_tokens"]["per_answer"] * 2
    )
    assert measurement["cost_usd"] == {"actual": 0.0, "estimated": 0.0}
    assert measurement["answer_text"] == FIXED_REFUSAL
