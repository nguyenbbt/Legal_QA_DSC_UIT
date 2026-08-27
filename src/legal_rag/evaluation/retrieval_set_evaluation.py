"""Deterministic set-aware evaluation for stored R0/R2R retrieval artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.grounding_labels import (
    GroundingLabelError,
    load_approved_grounding_benchmark,
)
from legal_rag.evaluation.retrieval_metrics import (
    RetrievalCandidateMetadata,
    RetrievalEvaluationError,
    RetrievalOutputRow,
    evaluate_retrieval,
    evaluate_retrieval_set,
)


class RetrievalSetArtifactError(Exception):
    """Stable safe failure at the stored set-evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _jsonl(data: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    if not data:
        raise RetrievalSetArtifactError("RETRIEVAL_SET_INPUT_INVALID", f"{name} must not be empty")
    rows: list[dict[str, Any]] = []
    for line in data.splitlines():
        if not line:
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_INPUT_INVALID", f"{name} contains an empty row"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_INPUT_INVALID", f"{name} contains invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_INPUT_INVALID", f"{name} rows must be objects"
            )
        rows.append(value)
    return tuple(rows)


def _by_question_id(rows: tuple[dict[str, Any], ...], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in result:
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_ID_INVALID", f"{name} question IDs must be non-empty and unique"
            )
        result[question_id] = row
    return result


def _metadata(candidate: dict[str, Any]) -> RetrievalCandidateMetadata:
    try:
        evidence_id = candidate["evidence_id"]
        context_id = candidate["context_id"]
        hierarchy_path = candidate["hierarchy_path"]
        canonical_start = candidate["canonical_start"]
        canonical_end = candidate["canonical_end"]
        token_cost = candidate.get("token_cost")
    except KeyError as error:
        raise RetrievalSetArtifactError(
            "RETRIEVAL_SET_METADATA_INVALID", "queue candidate metadata is incomplete"
        ) from error
    if (
        not isinstance(evidence_id, str)
        or not isinstance(context_id, str)
        or not isinstance(hierarchy_path, list)
        or not all(isinstance(item, str) and item for item in hierarchy_path)
        or not isinstance(canonical_start, int)
        or isinstance(canonical_start, bool)
        or not isinstance(canonical_end, int)
        or isinstance(canonical_end, bool)
        or (token_cost is not None and (not isinstance(token_cost, int) or token_cost < 0))
    ):
        raise RetrievalSetArtifactError(
            "RETRIEVAL_SET_METADATA_INVALID", "queue candidate metadata is invalid"
        )
    return RetrievalCandidateMetadata(
        evidence_id=evidence_id,
        context_id=context_id,
        hierarchy_path=tuple(hierarchy_path),
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        token_cost=token_cost,
    )


def _answer_metrics(data: bytes | None) -> dict[str, tuple[float, float]] | None:
    if data is None:
        return None
    result: dict[str, tuple[float, float]] = {}
    for row in _jsonl(data, name="answer per-query metrics"):
        question_id = row.get("question_id")
        meteor = row.get("meteor")
        rouge_l = row.get("rouge_l")
        if (
            not isinstance(question_id, str)
            or question_id in result
            or not isinstance(meteor, (int, float))
            or isinstance(meteor, bool)
            or not isinstance(rouge_l, (int, float))
            or isinstance(rouge_l, bool)
        ):
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_ANSWER_METRIC_INVALID",
                "answer per-query metrics contain an invalid row",
            )
        result[question_id] = (float(meteor), float(rouge_l))
    return result


def evaluate_stored_retrieval_set(
    *,
    retrieval_output_data: bytes,
    annotation_queue_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
    answer_per_query_data: bytes | None,
    run_id: str,
) -> bytes:
    """Join stored rankings to canonical queue metadata and emit a private report."""

    if not run_id:
        raise RetrievalSetArtifactError("RETRIEVAL_SET_RUN_ID_INVALID", "run ID is required")
    try:
        approved = load_approved_grounding_benchmark(benchmark_manifest_data, benchmark_data)
    except GroundingLabelError as error:
        raise RetrievalSetArtifactError(error.code, error.message) from error
    queue_by_id = _by_question_id(
        _jsonl(annotation_queue_data, name="annotation queue"), name="annotation queue"
    )
    ranking_by_id = _by_question_id(
        _jsonl(retrieval_output_data, name="retrieval output"), name="retrieval output"
    )
    expected_ids = set(approved.manifest.ordered_question_ids)
    if set(queue_by_id) != expected_ids or not expected_ids.issubset(ranking_by_id):
        raise RetrievalSetArtifactError(
            "RETRIEVAL_SET_ID_MISMATCH",
            "queue must exactly cover labels and ranking must cover every labeled question ID",
        )
    outputs: list[RetrievalOutputRow] = []
    for label in approved.records:
        queue = queue_by_id[label.question_id]
        ranking = ranking_by_id[label.question_id]
        if (
            queue.get("question_checksum") != label.question_checksum
            or ranking.get("question_checksum") != label.question_checksum
        ):
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_QUESTION_MISMATCH", "question checksums do not match"
            )
        queue_candidates = queue.get("candidates")
        ranked_candidates = ranking.get("candidates")
        if not isinstance(queue_candidates, list) or not isinstance(ranked_candidates, list):
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_INPUT_INVALID", "candidate lists are required"
            )
        metadata_by_id: dict[str, RetrievalCandidateMetadata] = {}
        for candidate in queue_candidates:
            if not isinstance(candidate, dict):
                raise RetrievalSetArtifactError(
                    "RETRIEVAL_SET_METADATA_INVALID", "queue candidate must be an object"
                )
            item = _metadata(candidate)
            if item.evidence_id in metadata_by_id:
                raise RetrievalSetArtifactError(
                    "RETRIEVAL_SET_CANDIDATE_DUPLICATE",
                    "queue candidate evidence IDs must be unique",
                )
            metadata_by_id[item.evidence_id] = item
        ranked_ids: list[str] = []
        for candidate in ranked_candidates:
            evidence_id = candidate.get("evidence_id") if isinstance(candidate, dict) else None
            if not isinstance(evidence_id, str) or evidence_id not in metadata_by_id:
                raise RetrievalSetArtifactError(
                    "RETRIEVAL_SET_CANDIDATE_MISMATCH",
                    "ranking candidate is outside the labeled candidate universe",
                )
            ranked_ids.append(evidence_id)
        if len(ranked_ids) != len(set(ranked_ids)):
            raise RetrievalSetArtifactError(
                "RETRIEVAL_SET_CANDIDATE_DUPLICATE",
                "ranking candidate evidence IDs must be unique",
            )
        outputs.append(
            RetrievalOutputRow(
                question_id=label.question_id,
                retrieved_evidence_ids=tuple(ranked_ids),
                candidate_metadata=tuple(metadata_by_id[item] for item in ranked_ids),
            )
        )
    try:
        historical = evaluate_retrieval(approved.retrieval_labels, tuple(outputs))
        set_metrics = evaluate_retrieval_set(
            approved.retrieval_labels,
            tuple(outputs),
            answer_metrics=_answer_metrics(answer_per_query_data),
            label_scope_establishes_candidate_absence=True,
        )
    except RetrievalEvaluationError as error:
        raise RetrievalSetArtifactError(error.code, error.message) from error
    return content_json_bytes(
        {
            "schema_version": "retrieval.set-evaluation.artifact.v1",
            "run_id": run_id,
            "label_scope": "bounded_labeled_candidate_universe",
            "retrieval_output_checksum": checksum_bytes(retrieval_output_data),
            "annotation_queue_checksum": checksum_bytes(annotation_queue_data),
            "benchmark_manifest_checksum": checksum_bytes(benchmark_manifest_data),
            "benchmark_checksum": checksum_bytes(benchmark_data),
            "answer_per_query_checksum": (
                checksum_bytes(answer_per_query_data) if answer_per_query_data is not None else None
            ),
            "historical_metrics": asdict(historical),
            "metrics": asdict(set_metrics),
        }
    )


__all__ = ["RetrievalSetArtifactError", "evaluate_stored_retrieval_set"]
