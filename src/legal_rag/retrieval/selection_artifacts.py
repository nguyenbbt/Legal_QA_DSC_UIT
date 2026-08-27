"""Deterministic artifact adapter for provider-neutral evidence selection."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.evidence_selector import (
    EvidenceSelectionCandidate,
    select_evidence_set,
)

TokenCounter = Callable[[str, tuple[str, ...]], int]
_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)


class EvidenceSelectionArtifactError(Exception):
    """Stable safe failure while adapting stored rankings to the selector."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class EvidenceSelectionArtifacts:
    retrieval_output: bytes
    selection_report: bytes


def _rows(data: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    try:
        rows = tuple(json.loads(line) for line in data.splitlines() if line)
    except json.JSONDecodeError as error:
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_INPUT_INVALID", f"{name} contains invalid JSON"
        ) from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_INPUT_INVALID", f"{name} must contain object rows"
        )
    return rows


def _by_id(rows: tuple[dict[str, Any], ...], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in result:
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_ID_INVALID",
                f"{name} question IDs must be non-empty and unique",
            )
        result[question_id] = row
    return result


def _candidate(
    value: dict[str, Any], *, question: str, token_counter: TokenCounter
) -> tuple[EvidenceSelectionCandidate, str]:
    try:
        evidence_id = value["evidence_id"]
        context_id = value["context_id"]
        hierarchy_path = value["hierarchy_path"]
        canonical_start = value["canonical_start"]
        canonical_end = value["canonical_end"]
        display_text = value["display_text"]
    except KeyError as error:
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_METADATA_INVALID", "queue candidate metadata is incomplete"
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
        or not isinstance(display_text, str)
    ):
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_METADATA_INVALID", "queue candidate metadata is invalid"
        )
    sparse_score = value.get("sparse_score")
    if sparse_score is not None and (
        isinstance(sparse_score, bool)
        or not isinstance(sparse_score, (int, float))
        or not math.isfinite(float(sparse_score))
        or float(sparse_score) < 0.0
    ):
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_METADATA_INVALID", "queue sparse score is invalid"
        )
    return (
        EvidenceSelectionCandidate(
            evidence_id=evidence_id,
            context_id=context_id,
            hierarchy_path=tuple(hierarchy_path),
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            token_cost=token_counter(question, (display_text,)),
            sparse_score=None if sparse_score is None else float(sparse_score),
        ),
        display_text,
    )


def _exact_input_cost(
    selected: tuple[EvidenceSelectionCandidate, ...],
    *,
    question: str,
    lookup: dict[str, tuple[EvidenceSelectionCandidate, str]],
    token_counter: TokenCounter,
) -> int:
    return token_counter(
        question,
        tuple(lookup[item.evidence_id][1] for item in selected),
    )


def build_evidence_selection_artifacts(
    *,
    annotation_queue_data: bytes,
    retrieval_output_data: bytes,
    source_run_id: str,
    selected_run_id: str,
    maximum_input_tokens: int,
    token_counter: TokenCounter,
    minimum_relative_sparse_score: float | None = None,
    calibration_checksum: str | None = None,
) -> EvidenceSelectionArtifacts:
    """Apply evidence-set-selector.v2 to a stored ranking without changing candidates."""

    if not source_run_id or not selected_run_id:
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_RUN_ID_INVALID", "source and selected run IDs are required"
        )
    if (minimum_relative_sparse_score is None) != (calibration_checksum is None):
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_CALIBRATION_INVALID",
            "relative-score selection requires exactly one calibration checksum",
        )
    if calibration_checksum is not None and _CHECKSUM.fullmatch(calibration_checksum) is None:
        raise EvidenceSelectionArtifactError(
            "EVIDENCE_SELECTION_CALIBRATION_INVALID",
            "calibration checksum must be a typed SHA-256 value",
        )
    queue_rows = _rows(annotation_queue_data, name="annotation queue")
    ranking_by_id = _by_id(
        _rows(retrieval_output_data, name="retrieval output"), name="retrieval output"
    )
    output_rows: list[dict[str, Any]] = []
    question_reports: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for queue in queue_rows:
        question_id = queue.get("question_id")
        question = queue.get("question")
        queue_candidates = queue.get("candidates")
        if (
            not isinstance(question_id, str)
            or not isinstance(question, str)
            or not isinstance(queue_candidates, list)
            or question_id not in ranking_by_id
        ):
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_INPUT_INVALID", "queue question/candidates are invalid"
            )
        candidates_by_id: dict[str, tuple[EvidenceSelectionCandidate, str]] = {}
        for value in queue_candidates:
            if not isinstance(value, dict):
                raise EvidenceSelectionArtifactError(
                    "EVIDENCE_SELECTION_METADATA_INVALID", "queue candidate must be an object"
                )
            candidate, text = _candidate(value, question=question, token_counter=token_counter)
            if candidate.evidence_id in candidates_by_id:
                raise EvidenceSelectionArtifactError(
                    "EVIDENCE_SELECTION_CANDIDATE_DUPLICATE", "queue evidence IDs must be unique"
                )
            candidates_by_id[candidate.evidence_id] = (candidate, text)
        ranking_values = ranking_by_id[question_id].get("candidates")
        if not isinstance(ranking_values, list):
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_INPUT_INVALID", "ranking candidates are required"
            )
        ranked: list[EvidenceSelectionCandidate] = []
        for value in ranking_values:
            evidence_id = value.get("evidence_id") if isinstance(value, dict) else None
            if not isinstance(evidence_id, str) or evidence_id not in candidates_by_id:
                raise EvidenceSelectionArtifactError(
                    "EVIDENCE_SELECTION_CANDIDATE_MISMATCH",
                    "ranking evidence is outside the canonical queue universe",
                )
            ranked.append(candidates_by_id[evidence_id][0])
        if len({item.evidence_id for item in ranked}) != len(ranked):
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_CANDIDATE_DUPLICATE", "ranking evidence IDs must be unique"
            )
        try:
            result = select_evidence_set(
                tuple(ranked),
                maximum_input_tokens=maximum_input_tokens,
                minimum_relative_sparse_score=minimum_relative_sparse_score,
                input_token_cost=partial(
                    _exact_input_cost,
                    question=question,
                    lookup=candidates_by_id,
                    token_counter=token_counter,
                ),
            )
        except ValueError as error:
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_POLICY_INVALID", "evidence selection policy is invalid"
            ) from error
        if not result.selected_evidence_ids:
            raise EvidenceSelectionArtifactError(
                "EVIDENCE_SELECTION_EMPTY", "selector produced no generator evidence"
            )
        selected_counts[str(len(result.selected_evidence_ids))] += 1
        reason_counts.update(decision.reason for decision in result.decisions)
        output_rows.append(
            {
                "schema_version": "model.retrieval.output.v1",
                "run_id": selected_run_id,
                "question_id": question_id,
                "question_checksum": queue.get("question_checksum"),
                "candidates": [
                    {"evidence_id": evidence_id, "selector_selected": True}
                    for evidence_id in result.selected_evidence_ids
                ],
            }
        )
        question_reports.append(
            {
                "question_id": question_id,
                "selected_evidence_ids": result.selected_evidence_ids,
                "input_token_cost": result.input_token_cost,
                "decisions": tuple(asdict(decision) for decision in result.decisions),
            }
        )
    output = b"".join(content_json_bytes(row) for row in output_rows)
    report = content_json_bytes(
        {
            "schema_version": "evidence-set-selection.report.v1",
            "selector_version": "evidence-set-selector.v2",
            "source_run_id": source_run_id,
            "selected_run_id": selected_run_id,
            "maximum_input_tokens": maximum_input_tokens,
            "maximum_evidence_count": 3,
            "maximum_overlap_ratio": 0.8,
            "minimum_relative_sparse_score": minimum_relative_sparse_score,
            "calibration_checksum": calibration_checksum,
            "question_count": len(question_reports),
            "selected_count_distribution": dict(sorted(selected_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "annotation_queue_checksum": checksum_bytes(annotation_queue_data),
            "source_retrieval_checksum": checksum_bytes(retrieval_output_data),
            "selected_retrieval_checksum": checksum_bytes(output),
            "questions": question_reports,
        }
    )
    return EvidenceSelectionArtifacts(output, report)


__all__ = [
    "EvidenceSelectionArtifactError",
    "EvidenceSelectionArtifacts",
    "TokenCounter",
    "build_evidence_selection_artifacts",
]
