"""Deterministic paired retrieval-only promotion comparison."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np

from legal_rag.domain.checksums import content_json_bytes

_SEED_STRING = "dsc2026-retrieval-bootstrap-v1"
_RESAMPLES = 10_000


class RetrievalComparisonError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _Metrics:
    recall_at_10: float
    mrr_at_10: float
    evidence_set_recall_at_10: float


def _fail(code: str, message: str) -> NoReturn:
    raise RetrievalComparisonError(code, message)


def _rows(data: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if not data or b"\r" in data or not data.endswith(b"\n"):
        _fail("RETRIEVAL_COMPARISON_INPUT_INVALID", f"{label} framing is invalid")
    try:
        values = tuple(json.loads(line) for line in data.splitlines())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalComparisonError(
            "RETRIEVAL_COMPARISON_INPUT_INVALID", f"{label} is not valid JSONL"
        ) from error
    if not all(isinstance(value, dict) for value in values):
        _fail("RETRIEVAL_COMPARISON_INPUT_INVALID", f"{label} rows must be objects")
    return values


def _labels(data: bytes) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    try:
        for row in _rows(data, label="grounding benchmark"):
            question_id = row["question_id"]
            relevant = tuple(
                item["evidence_id"]
                for item in row["relevant_evidence"]
                if item["relevance"] in {"relevant", "partially_relevant"}
            )
            if (
                not isinstance(question_id, str)
                or not question_id
                or question_id in result
                or any(not isinstance(value, str) or not value for value in relevant)
                or len(relevant) != len(set(relevant))
            ):
                raise ValueError
            result[question_id] = relevant
    except (KeyError, TypeError, ValueError) as error:
        raise RetrievalComparisonError(
            "RETRIEVAL_COMPARISON_INPUT_INVALID", "grounding benchmark rows are invalid"
        ) from error
    return result


def _outputs(data: bytes, *, label: str) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    try:
        for row in _rows(data, label=label):
            question_id = row["question_id"]
            evidence_ids = tuple(item["evidence_id"] for item in row["candidates"])
            if (
                not isinstance(question_id, str)
                or not question_id
                or question_id in result
                or any(not isinstance(value, str) or not value for value in evidence_ids)
                or len(evidence_ids) != len(set(evidence_ids))
            ):
                raise ValueError
            result[question_id] = evidence_ids
    except (KeyError, TypeError, ValueError) as error:
        raise RetrievalComparisonError(
            "RETRIEVAL_COMPARISON_INPUT_INVALID", f"{label} rows are invalid"
        ) from error
    return result


def _metrics(relevant: tuple[str, ...], retrieved: tuple[str, ...]) -> _Metrics:
    gold = frozenset(relevant)
    top_10 = retrieved[:10]
    recall = len(gold.intersection(top_10)) / len(gold)
    reciprocal_rank = next(
        (1.0 / rank for rank, evidence_id in enumerate(top_10, start=1) if evidence_id in gold),
        0.0,
    )
    return _Metrics(recall, reciprocal_rank, float(gold.issubset(top_10)))


def _interval(deltas: np.ndarray, indices: np.ndarray) -> list[float]:
    means = deltas[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975], method="linear")]


def compare_retrieval_experiments(
    *,
    grounding_benchmark_data: bytes,
    baseline_output_data: bytes,
    candidate_output_data: bytes,
    baseline_run_id: str,
    candidate_run_id: str,
) -> bytes:
    """Compare a candidate reordering against an identical labeled candidate universe."""

    if not baseline_run_id.strip() or not candidate_run_id.strip():
        _fail("RETRIEVAL_COMPARISON_INPUT_INVALID", "retrieval run IDs must be non-empty")
    labels = _labels(grounding_benchmark_data)
    baseline = _outputs(baseline_output_data, label="baseline output")
    candidate = _outputs(candidate_output_data, label="candidate output")
    if set(labels) != set(baseline) or set(labels) != set(candidate):
        _fail("RETRIEVAL_COMPARISON_ID_MISMATCH", "retrieval comparison IDs differ")
    if any(set(baseline[key]) != set(candidate[key]) for key in labels):
        _fail(
            "RETRIEVAL_COMPARISON_NOT_FIXED",
            "retrieval comparison changed the admitted candidate universe",
        )
    question_ids = tuple(
        sorted((key for key, relevant in labels.items() if relevant), key=lambda key: key.encode())
    )
    if not question_ids:
        _fail("RETRIEVAL_COMPARISON_EMPTY", "retrieval comparison has no evaluable rows")
    baseline_metrics = tuple(_metrics(labels[key], baseline[key]) for key in question_ids)
    candidate_metrics = tuple(_metrics(labels[key], candidate[key]) for key in question_ids)
    recall_deltas = np.asarray(
        [
            candidate_value.recall_at_10 - baseline_value.recall_at_10
            for baseline_value, candidate_value in zip(
                baseline_metrics, candidate_metrics, strict=True
            )
        ],
        dtype=np.float64,
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

    def mean(values: tuple[_Metrics, ...], field: str) -> float:
        result = float(np.mean([getattr(value, field) for value in values], dtype=np.float64))
        if not math.isfinite(result):
            _fail("RETRIEVAL_COMPARISON_INPUT_INVALID", "retrieval metrics are non-finite")
        return result

    baseline_recall = mean(baseline_metrics, "recall_at_10")
    candidate_recall = mean(candidate_metrics, "recall_at_10")
    baseline_mrr = mean(baseline_metrics, "mrr_at_10")
    candidate_mrr = mean(candidate_metrics, "mrr_at_10")
    baseline_evidence = mean(baseline_metrics, "evidence_set_recall_at_10")
    candidate_evidence = mean(candidate_metrics, "evidence_set_recall_at_10")
    recall_delta = float(recall_deltas.mean())
    recall_interval = _interval(recall_deltas, indices)
    mrr_delta = candidate_mrr - baseline_mrr
    evidence_delta = candidate_evidence - baseline_evidence
    blockers: list[str] = []
    if recall_delta <= 0:
        blockers.append("RECALL_AT_10_DELTA_NOT_POSITIVE")
    if recall_interval[0] <= 0:
        blockers.append("RECALL_AT_10_CI_LOWER_NOT_POSITIVE")
    if mrr_delta < 0:
        blockers.append("MRR_AT_10_REGRESSION")
    if evidence_delta < 0:
        blockers.append("EVIDENCE_SET_RECALL_AT_10_REGRESSION")
    return content_json_bytes(
        {
            "schema_version": "retrieval.comparison.v1",
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "classification": "retrieval_only_reordering",
            "fixed_candidate_universe_verified": True,
            "benchmark_question_count": len(labels),
            "retrieval_evaluable_count": len(question_ids),
            "metrics": {
                "baseline_recall_at_10": baseline_recall,
                "candidate_recall_at_10": candidate_recall,
                "recall_at_10_mean_delta": recall_delta,
                "recall_at_10_ci95": recall_interval,
                "baseline_mrr_at_10": baseline_mrr,
                "candidate_mrr_at_10": candidate_mrr,
                "mrr_at_10_delta": mrr_delta,
                "baseline_evidence_set_recall_at_10": baseline_evidence,
                "candidate_evidence_set_recall_at_10": candidate_evidence,
                "evidence_set_recall_at_10_delta": evidence_delta,
            },
            "paired_bootstrap": {
                "schema_version": "retrieval-paired-bootstrap.v1",
                "seed_string": _SEED_STRING,
                "seed_uint64": seed_uint64,
                "resamples": _RESAMPLES,
                "quantile_method": "linear",
            },
            "promotion_state": "rejected_preserved" if blockers else "promoted",
            "promotion_blockers": blockers,
        }
    )


__all__ = ["RetrievalComparisonError", "compare_retrieval_experiments"]
