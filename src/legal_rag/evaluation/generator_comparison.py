"""Deterministic EVAL-005 numeric and runtime comparison for generator runs."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, NoReturn

import numpy as np

from legal_rag.domain.checksums import content_json_bytes

_SEED_STRING = "dsc2026-bootstrap-v1"
_RESAMPLES = 10_000
_FIXED_FIELDS = (
    "model_id",
    "model_revision",
    "prompt_checksum",
    "retrieval_output_checksum",
    "annotation_queue_checksum",
    "references_checksum",
    "evidence_limit",
    "decoding",
)
_RETRIEVAL_FIXED_FIELDS = (
    "model_id",
    "model_revision",
    "prompt_checksum",
    "annotation_queue_checksum",
    "references_checksum",
    "evidence_limit",
    "decoding",
    "adapter",
)
_PROMPT_FIXED_FIELDS = (
    "model_id",
    "model_revision",
    "retrieval_output_checksum",
    "annotation_queue_checksum",
    "references_checksum",
    "evidence_limit",
    "decoding",
    "adapter",
)
_MODEL_FIXED_FIELDS = (
    "prompt_checksum",
    "retrieval_output_checksum",
    "annotation_queue_checksum",
    "references_checksum",
    "evidence_limit",
    "decoding",
    "adapter",
)
_POSTPROCESSOR_FIXED_FIELDS = (
    "model_id",
    "model_revision",
    "prompt_checksum",
    "retrieval_output_checksum",
    "annotation_queue_checksum",
    "references_checksum",
    "evidence_limit",
    "decoding",
    "adapter",
)


class GeneratorComparisonError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise GeneratorComparisonError(code, message)


def _object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GeneratorComparisonError(
            "GENERATOR_COMPARISON_INPUT_INVALID", f"{label} is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        _fail("GENERATOR_COMPARISON_INPUT_INVALID", f"{label} must be a JSON object")
    return value


def _metric_rows(data: bytes, *, label: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    try:
        values = (json.loads(line) for line in data.splitlines())
        for value in values:
            question_id = value["question_id"]
            meteor = float(value["meteor"])
            rouge_l = float(value["rouge_l"])
            if (
                not isinstance(question_id, str)
                or question_id in rows
                or not math.isfinite(meteor)
                or not math.isfinite(rouge_l)
            ):
                raise ValueError
            rows[question_id] = (meteor, rouge_l)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise GeneratorComparisonError(
            "GENERATOR_COMPARISON_INPUT_INVALID", f"{label} metric rows are invalid"
        ) from error
    if not rows:
        _fail("GENERATOR_COMPARISON_INPUT_INVALID", f"{label} has no metric rows")
    return rows


def _interval(deltas: np.ndarray, indices: np.ndarray) -> list[float]:
    means = deltas[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975], method="linear")]


def _compare_metric_rows(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
    schema_version: str,
    classification: str,
    changed_axes: list[str],
    comparison_metadata: dict[str, object] | None = None,
) -> bytes:
    if (
        not math.isfinite(baseline_runtime_seconds)
        or not math.isfinite(candidate_runtime_seconds)
        or baseline_runtime_seconds <= 0
        or candidate_runtime_seconds <= 0
    ):
        _fail("GENERATOR_COMPARISON_INPUT_INVALID", "runtime values must be positive")
    baseline = _metric_rows(baseline_per_query_data, label="baseline")
    candidate = _metric_rows(candidate_per_query_data, label="candidate")
    if set(baseline) != set(candidate):
        _fail("GENERATOR_COMPARISON_ID_MISMATCH", "paired question IDs differ")
    question_ids = sorted(baseline, key=lambda value: value.encode("utf-8"))
    meteor_deltas = np.asarray(
        [candidate[item][0] - baseline[item][0] for item in question_ids], dtype=np.float64
    )
    rouge_deltas = np.asarray(
        [candidate[item][1] - baseline[item][1] for item in question_ids], dtype=np.float64
    )
    seed_uint64 = int.from_bytes(
        hashlib.sha256(_SEED_STRING.encode()).digest()[:8], "big", signed=False
    )
    generator = np.random.Generator(np.random.PCG64(seed_uint64))
    indices = generator.integers(
        0,
        len(question_ids),
        size=(_RESAMPLES, len(question_ids)),
        endpoint=False,
        dtype=np.int64,
    )
    meteor_delta = float(meteor_deltas.mean())
    rouge_delta = float(rouge_deltas.mean())
    meteor_interval = _interval(meteor_deltas, indices)
    runtime_ratio = candidate_runtime_seconds / baseline_runtime_seconds
    blockers: list[str] = []
    if meteor_delta < 0.002:
        blockers.append("METEOR_DELTA_BELOW_THRESHOLD")
    if meteor_interval[0] <= 0:
        blockers.append("METEOR_CI_LOWER_NOT_POSITIVE")
    if rouge_delta < -0.002:
        blockers.append("ROUGE_L_REGRESSION_EXCEEDS_GUARDRAIL")
    numeric_failed = bool(blockers)
    if runtime_ratio > 1.25:
        blockers.append("RUNTIME_REGRESSION_EXCEEDS_25_PERCENT")

    return content_json_bytes(
        {
            "schema_version": schema_version,
            "baseline_run_id": baseline_manifest["run_id"],
            "candidate_run_id": candidate_manifest["run_id"],
            "classification": classification,
            "changed_axes": changed_axes,
            **(comparison_metadata or {}),
            "fixed_inputs_verified": True,
            "question_count": len(question_ids),
            "metrics": {
                "baseline_meteor": float(
                    np.mean([baseline[item][0] for item in question_ids], dtype=np.float64)
                ),
                "candidate_meteor": float(
                    np.mean([candidate[item][0] for item in question_ids], dtype=np.float64)
                ),
                "meteor_mean_delta": meteor_delta,
                "meteor_ci95": meteor_interval,
                "baseline_rouge_l": float(
                    np.mean([baseline[item][1] for item in question_ids], dtype=np.float64)
                ),
                "candidate_rouge_l": float(
                    np.mean([candidate[item][1] for item in question_ids], dtype=np.float64)
                ),
                "rouge_l_mean_delta": rouge_delta,
                "rouge_l_ci95": _interval(rouge_deltas, indices),
            },
            "paired_bootstrap": {
                "schema_version": "paired-bootstrap.v1",
                "seed_string": _SEED_STRING,
                "seed_uint64": seed_uint64,
                "resamples": _RESAMPLES,
                "quantile_method": "linear",
            },
            "runtime": {
                "baseline_seconds": baseline_runtime_seconds,
                "candidate_seconds": candidate_runtime_seconds,
                "candidate_to_baseline_ratio": runtime_ratio,
            },
            "numeric_evaluation_gate": "failed" if numeric_failed else "passed",
            "resource_gate": "failed" if runtime_ratio > 1.25 else "passed",
            "promotion_state": "rejected_preserved" if blockers else "pending_grounding",
            "promotion_blockers": blockers,
            "downstream_grounding_and_reproducibility": (
                "not_run_after_automatic_rejection" if blockers else "required"
            ),
        }
    )


def compare_generator_experiments(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest_data: bytes,
    candidate_manifest_data: bytes,
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
) -> bytes:
    """Compare one adapter-only candidate against a fixed generator baseline."""

    baseline_manifest = _object(baseline_manifest_data, label="baseline manifest")
    candidate_manifest = _object(candidate_manifest_data, label="candidate manifest")
    if any(baseline_manifest.get(key) != candidate_manifest.get(key) for key in _FIXED_FIELDS):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "generator-only comparison changed an input other than the adapter",
        )
    comparison = candidate_manifest.get("comparison")
    if (
        not isinstance(comparison, dict)
        or comparison.get("baseline_run_id") != baseline_manifest.get("run_id")
        or comparison.get("changed_axes") != ["adapter"]
        or not isinstance(candidate_manifest.get("adapter"), dict)
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "candidate must declare the baseline and adapter as its only changed axis",
        )
    return _compare_metric_rows(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=candidate_per_query_data,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_runtime_seconds=baseline_runtime_seconds,
        candidate_runtime_seconds=candidate_runtime_seconds,
        schema_version="generator.adapter.comparison.v1",
        classification="generator_only_single_axis",
        changed_axes=["adapter"],
    )


def compare_retrieval_generation_experiments(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest_data: bytes,
    candidate_manifest_data: bytes,
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
) -> bytes:
    """Apply EVAL-005 to a retrieval-only change with a fixed generator."""

    baseline_manifest = _object(baseline_manifest_data, label="baseline manifest")
    candidate_manifest = _object(candidate_manifest_data, label="candidate manifest")
    if any(
        baseline_manifest.get(key) != candidate_manifest.get(key) for key in _RETRIEVAL_FIXED_FIELDS
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "retrieval-only comparison changed the generator configuration",
        )
    comparison = candidate_manifest.get("comparison")
    declared_baseline = comparison.get("baseline_run_id") if isinstance(comparison, dict) else None
    if (
        not isinstance(comparison, dict)
        or not isinstance(declared_baseline, str)
        or not declared_baseline.strip()
        or comparison.get("changed_axes") != ["retrieval"]
        or baseline_manifest.get("retrieval_output_checksum")
        == candidate_manifest.get("retrieval_output_checksum")
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "candidate must declare retrieval as its only changed axis",
        )
    return _compare_metric_rows(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=candidate_per_query_data,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_runtime_seconds=baseline_runtime_seconds,
        candidate_runtime_seconds=candidate_runtime_seconds,
        schema_version="generator.retrieval.comparison.v1",
        classification="retrieval_only_single_axis",
        changed_axes=["retrieval"],
        comparison_metadata={
            "comparison_pairing": (
                "declared_retrieval_ablation"
                if declared_baseline == baseline_manifest.get("run_id")
                else "posthoc_fixed_retrieval_pair"
            ),
            "candidate_declared_baseline_run_id": declared_baseline,
        },
    )


def compare_prompt_generation_experiments(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest_data: bytes,
    candidate_manifest_data: bytes,
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
) -> bytes:
    """Apply EVAL-005 to a prompt-only change with retrieval and decoding fixed."""

    baseline_manifest = _object(baseline_manifest_data, label="baseline manifest")
    candidate_manifest = _object(candidate_manifest_data, label="candidate manifest")
    if any(
        baseline_manifest.get(key) != candidate_manifest.get(key) for key in _PROMPT_FIXED_FIELDS
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "prompt-only comparison changed a non-prompt generator input",
        )
    comparison = candidate_manifest.get("comparison")
    if (
        not isinstance(comparison, dict)
        or comparison.get("baseline_run_id") != baseline_manifest.get("run_id")
        or comparison.get("changed_axes") != ["prompt"]
        or baseline_manifest.get("prompt_checksum") == candidate_manifest.get("prompt_checksum")
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "candidate must declare prompt as its only changed axis",
        )
    return _compare_metric_rows(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=candidate_per_query_data,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_runtime_seconds=baseline_runtime_seconds,
        candidate_runtime_seconds=candidate_runtime_seconds,
        schema_version="generator.prompt.comparison.v1",
        classification="generator_only_single_axis",
        changed_axes=["prompt"],
    )


def compare_model_generation_experiments(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest_data: bytes,
    candidate_manifest_data: bytes,
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
) -> bytes:
    """Apply EVAL-005 to one model-only change with retrieval and prompting fixed."""

    baseline_manifest = _object(baseline_manifest_data, label="baseline manifest")
    candidate_manifest = _object(candidate_manifest_data, label="candidate manifest")
    if any(
        baseline_manifest.get(key) != candidate_manifest.get(key) for key in _MODEL_FIXED_FIELDS
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "model-only comparison changed retrieval, prompt, decoding, or adapter inputs",
        )
    comparison = candidate_manifest.get("comparison")
    baseline_identity = (
        baseline_manifest.get("model_id"),
        baseline_manifest.get("model_revision"),
    )
    candidate_identity = (
        candidate_manifest.get("model_id"),
        candidate_manifest.get("model_revision"),
    )
    if (
        not isinstance(comparison, dict)
        or comparison.get("baseline_run_id") != baseline_manifest.get("run_id")
        or comparison.get("changed_axes") != ["model"]
        or baseline_identity == candidate_identity
        or not all(isinstance(value, str) and value.strip() for value in candidate_identity)
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "candidate must declare a pinned model as its only changed axis",
        )
    return _compare_metric_rows(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=candidate_per_query_data,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_runtime_seconds=baseline_runtime_seconds,
        candidate_runtime_seconds=candidate_runtime_seconds,
        schema_version="generator.model.comparison.v1",
        classification="generator_only_single_axis",
        changed_axes=["model"],
    )


def compare_postprocessed_generation_experiments(
    *,
    baseline_per_query_data: bytes,
    candidate_per_query_data: bytes,
    baseline_manifest_data: bytes,
    candidate_manifest_data: bytes,
    baseline_runtime_seconds: float,
    candidate_runtime_seconds: float,
) -> bytes:
    """Apply EVAL-005 to one answer postprocessor with generation held fixed."""

    baseline_manifest = _object(baseline_manifest_data, label="baseline manifest")
    candidate_manifest = _object(candidate_manifest_data, label="candidate manifest")
    if any(
        baseline_manifest.get(key) != candidate_manifest.get(key)
        for key in _POSTPROCESSOR_FIXED_FIELDS
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "postprocessor comparison changed generation or retrieval inputs",
        )
    comparison = candidate_manifest.get("comparison")
    postprocessor = candidate_manifest.get("postprocessor")
    if (
        not isinstance(comparison, dict)
        or comparison.get("baseline_run_id") != baseline_manifest.get("run_id")
        or comparison.get("changed_axes") != ["postprocessor"]
        or not isinstance(postprocessor, dict)
        or not isinstance(postprocessor.get("policy_id"), str)
        or not isinstance(postprocessor.get("policy_checksum"), str)
    ):
        _fail(
            "GENERATOR_COMPARISON_NOT_FIXED",
            "candidate must declare one identified postprocessor as its only changed axis",
        )
    return _compare_metric_rows(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=candidate_per_query_data,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_runtime_seconds=baseline_runtime_seconds,
        candidate_runtime_seconds=candidate_runtime_seconds,
        schema_version="generator.postprocessor.comparison.v1",
        classification="generator_output_postprocessor_single_axis",
        changed_axes=["postprocessor"],
    )


__all__ = [
    "GeneratorComparisonError",
    "compare_generator_experiments",
    "compare_model_generation_experiments",
    "compare_postprocessed_generation_experiments",
    "compare_prompt_generation_experiments",
    "compare_retrieval_generation_experiments",
]
