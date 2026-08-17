"""Explicitly labeled local CPU measurements for the fixed-refusal baseline."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import AnswerRecord
from legal_rag.generation.fixture import FIXED_REFUSAL, FixtureExtractiveGenerator
from legal_rag.ingestion.organizer import OrganizerQuestionReader
from legal_rag.retrieval.tokenizer import RETRIEVAL_TOKENIZER_ID, retrieval_tokens
from legal_rag.submission.writer import build_submission, validate_submission


def _quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (ordered[lower] * (1 - weight) + ordered[upper] * weight) / 1_000_000


def _summary(values: list[int]) -> dict[str, float]:
    return {"p50": _quantile(values, 0.5), "p95": _quantile(values, 0.95)}


def _one_run(source: bytes, run_id: str) -> tuple[dict[str, int], bytes]:
    total_start = time.perf_counter_ns()
    stage_start = total_start
    imported = OrganizerQuestionReader().read_bytes(
        source, kind="public", artifact_path="public-source"
    )
    input_parse = time.perf_counter_ns() - stage_start

    stage_start = time.perf_counter_ns()
    generator = FixtureExtractiveGenerator()
    generated = tuple(generator.generate(question, ()) for question in imported.records)
    answers = tuple(
        AnswerRecord.model_validate(
            {
                "schema_version": "internal.answer.v1",
                "question_id": answer.question_id,
                "answer": answer.answer_text,
                "generator_id": answer.generator_id,
                "evidence_ids": answer.used_evidence_ids,
                "run_id": run_id,
            }
        )
        for answer in generated
    )
    generation = time.perf_counter_ns() - stage_start

    stage_start = time.perf_counter_ns()
    predictions = build_submission(source, answers)
    validate_submission(source, predictions)
    submission = time.perf_counter_ns() - stage_start
    return (
        {
            "input_parse": input_parse,
            "fixed_refusal_generation": generation,
            "submission_render_validate": submission,
            "end_to_end": time.perf_counter_ns() - total_start,
        },
        predictions,
    )


def measure_fixed_refusal(source: bytes, *, run_id: str, warm_samples: int = 7) -> dict[str, Any]:
    """Measure one cold and repeated warm local CPU runs without writing artifacts."""

    if not 1 <= warm_samples <= 100:
        raise ValueError("warm_samples must be in [1, 100]")
    cold, predictions = _one_run(source, run_id)
    warm_results = [_one_run(source, run_id)[0] for _ in range(warm_samples)]

    tracemalloc.start()
    try:
        _one_run(source, run_id)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    question_count = len(
        OrganizerQuestionReader()
        .read_bytes(source, kind="public", artifact_path="public-source")
        .records
    )
    stages = tuple(cold)
    latency = {
        "cold": {stage: _summary([cold[stage]]) for stage in stages},
        "warm": {
            stage: _summary([measurement[stage] for measurement in warm_results])
            for stage in stages
        },
    }
    warm_p50 = latency["warm"]["end_to_end"]["p50"]
    tokens_per_answer = len(retrieval_tokens(FIXED_REFUSAL))
    return {
        "schema_version": "fixed.refusal.measurement.v1",
        "run_id": run_id,
        "question_count": question_count,
        "cold_sample_count": 1,
        "warm_sample_count": warm_samples,
        "quantile_method": "linear",
        "latency_ms": latency,
        "throughput_questions_per_second": (question_count * 1000 / warm_p50 if warm_p50 else None),
        "cpu_memory": {
            "method": "python_tracemalloc_peak",
            "scope": "one_end_to_end_run",
            "peak_bytes": peak_bytes,
            "os_rss_measured": False,
        },
        "gpu_vram": {"status": "not_applicable", "reason": "cpu_only_no_gpu_workload"},
        "index_disk": {"bytes": 0, "reason": "no_index_in_mil_003"},
        "generated_tokens": {
            "method": f"{RETRIEVAL_TOKENIZER_ID}_proxy_not_model_tokens",
            "per_answer": tokens_per_answer,
            "total": tokens_per_answer * question_count,
        },
        "cost_usd": {"actual": 0.0, "estimated": 0.0},
        "predictions_checksum": checksum_bytes(predictions),
        "answer_text": FIXED_REFUSAL,
    }
