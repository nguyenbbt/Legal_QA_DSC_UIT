"""Private model-backed reranking experiment over the approved labeled queue."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.retrieval_metrics import (
    RetrievalLabelRow,
    RetrievalOutputRow,
    evaluate_retrieval,
)
from legal_rag.retrieval.reranker import RerankerBackend


class ModelRetrievalExperimentError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def run_labeled_reranker_experiment(
    *,
    annotation_queue_data: bytes,
    grounding_benchmark_data: bytes,
    backend: RerankerBackend,
    run_id: str,
    candidate_limit: int = 12,
) -> tuple[bytes, bytes]:
    """Rerank the frozen candidate universe and return safe outputs plus metrics."""

    if candidate_limit < 1:
        raise ModelRetrievalExperimentError(
            "RERANK_LIMIT_INVALID", "reranker candidate limit must be positive"
        )
    try:
        work_items = tuple(json.loads(line) for line in annotation_queue_data.splitlines())
        labels = tuple(json.loads(line) for line in grounding_benchmark_data.splitlines())
    except (json.JSONDecodeError, TypeError) as error:
        raise ModelRetrievalExperimentError(
            "RERANK_EXPERIMENT_INPUT_INVALID", "reranker experiment input is invalid"
        ) from error
    labels_by_id = {row.get("question_id"): row for row in labels}
    if not work_items or len(labels_by_id) != len(labels):
        raise ModelRetrievalExperimentError(
            "RERANK_EXPERIMENT_INPUT_INVALID", "reranker experiment IDs are invalid"
        )
    outputs: list[dict[str, object]] = []
    metric_labels: list[RetrievalLabelRow] = []
    metric_outputs: list[RetrievalOutputRow] = []
    started = time.perf_counter()
    for work_item in work_items:
        question_id = work_item.get("question_id")
        question = work_item.get("question")
        candidates = work_item.get("candidates")
        if (
            not isinstance(question_id, str)
            or not isinstance(question, str)
            or not isinstance(candidates, list)
            or question_id not in labels_by_id
            or len(candidates) > candidate_limit
        ):
            raise ModelRetrievalExperimentError(
                "RERANK_EXPERIMENT_INPUT_INVALID", "reranker experiment row is invalid"
            )
        scores = tuple(
            float(value)
            for value in backend.score(
                question, tuple(str(candidate["display_text"]) for candidate in candidates)
            )
        )
        if len(scores) != len(candidates):
            raise ModelRetrievalExperimentError(
                "RERANK_OUTPUT_CARDINALITY", "reranker returned the wrong score count"
            )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], str(item[0]["evidence_id"]).encode("utf-8")),
        )
        evidence_ids = tuple(str(candidate["evidence_id"]) for candidate, _ in ranked)
        label = labels_by_id[question_id]
        relevant = tuple(
            str(item["evidence_id"])
            for item in label.get("relevant_evidence", ())
            if item.get("relevance") in {"relevant", "partially_relevant"}
        )
        metric_labels.append(RetrievalLabelRow(question_id, relevant))
        metric_outputs.append(RetrievalOutputRow(question_id, evidence_ids))
        outputs.append(
            {
                "schema_version": "model.retrieval.output.v1",
                "run_id": run_id,
                "question_id": question_id,
                "question_checksum": work_item.get("question_checksum"),
                "candidates": [
                    {"evidence_id": candidate["evidence_id"], "reranker_score": score}
                    for candidate, score in ranked
                ],
            }
        )
    elapsed = time.perf_counter() - started
    metrics = evaluate_retrieval(tuple(metric_labels), tuple(metric_outputs))
    output_data = b"".join(content_json_bytes(row) for row in outputs)
    report = content_json_bytes(
        {
            "schema_version": "model.retrieval.experiment.report.v1",
            "run_id": run_id,
            "profile_state": "exploratory_non_promotable",
            "model_id": backend.model_id,
            "model_revision": backend.model_revision,
            "question_count": len(outputs),
            "candidate_limit": candidate_limit,
            "elapsed_seconds": elapsed,
            "output_checksum": checksum_bytes(output_data),
            "metrics": asdict(metrics),
        }
    )
    return output_data, report


__all__ = ["ModelRetrievalExperimentError", "run_labeled_reranker_experiment"]
