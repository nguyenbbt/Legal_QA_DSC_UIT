"""Build the deterministic R-002A bounded oracle inputs without generating text."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.grounding_labels import load_approved_grounding_benchmark
from legal_rag.evaluation.oracle_retrieval import (
    OracleCandidate,
    OracleRelevance,
    OracleSelection,
    select_bounded_oracle_evidence,
)


class OracleArtifactError(Exception):
    """Stable safe failure while building bounded-oracle inputs."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class BoundedOracleArtifacts:
    eligible_annotation_queue: bytes
    o2_retrieval_output: bytes
    o3_retrieval_output: bytes
    o4_retrieval_output: bytes
    selection_report: bytes


InputTokenCounter = Callable[[str, tuple[str, ...]], int]


def _rows(data: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    try:
        rows = tuple(json.loads(line) for line in data.splitlines() if line)
    except json.JSONDecodeError as error:
        raise OracleArtifactError(
            "ORACLE_INPUT_INVALID", f"{name} contains invalid JSON"
        ) from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise OracleArtifactError("ORACLE_INPUT_INVALID", f"{name} must contain object rows")
    return rows


def _by_id(rows: tuple[dict[str, Any], ...], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in result:
            raise OracleArtifactError(
                "ORACLE_ID_INVALID", f"{name} question IDs must be non-empty and unique"
            )
        result[question_id] = row
    return result


def _ranking_ids(row: dict[str, Any]) -> tuple[str, ...]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise OracleArtifactError("ORACLE_INPUT_INVALID", "ranking candidates are required")
    values: list[str] = []
    for candidate in candidates:
        evidence_id = candidate.get("evidence_id") if isinstance(candidate, dict) else None
        if not isinstance(evidence_id, str) or not evidence_id:
            raise OracleArtifactError("ORACLE_INPUT_INVALID", "ranking evidence ID is invalid")
        values.append(evidence_id)
    if len(values) != len(set(values)):
        raise OracleArtifactError("ORACLE_CANDIDATE_DUPLICATE", "ranking IDs must be unique")
    return tuple(values)


def _select_variant(
    *,
    ranking: tuple[str, ...],
    relevance_by_id: Mapping[str, OracleRelevance],
    display_by_id: Mapping[str, str],
    question: str,
    maximum_input_tokens: int,
    token_counter: InputTokenCounter,
) -> OracleSelection:
    if any(evidence_id not in display_by_id for evidence_id in ranking):
        raise OracleArtifactError(
            "ORACLE_CANDIDATE_MISMATCH", "ranking references evidence outside the labeled queue"
        )
    labeled_ranking = tuple(
        OracleCandidate(
            evidence_id=evidence_id,
            relevance=relevance_by_id.get(evidence_id, "not_relevant"),
            token_cost=token_counter(question, (display_by_id[evidence_id],)),
        )
        for evidence_id in ranking
    )
    return select_bounded_oracle_evidence(
        labeled_ranking,
        maximum_evidence_count=3,
        maximum_input_tokens=maximum_input_tokens,
        input_token_cost=lambda selected: token_counter(
            question, tuple(display_by_id[evidence_id] for evidence_id in selected)
        ),
    )


def build_bounded_oracle_artifacts(
    *,
    annotation_queue_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
    r0_retrieval_output_data: bytes,
    r2r_retrieval_output_data: bytes,
    maximum_input_tokens: int,
    token_counter: InputTokenCounter,
) -> BoundedOracleArtifacts:
    """Build O2/O3/O4 ranking inputs over the frozen judged candidate universe."""

    if maximum_input_tokens < 1:
        raise OracleArtifactError("ORACLE_TOKEN_LIMIT_INVALID", "token limit must be positive")
    approved = load_approved_grounding_benchmark(benchmark_manifest_data, benchmark_data)
    queue_by_id = _by_id(_rows(annotation_queue_data, name="annotation queue"), name="queue")
    r0_by_id = _by_id(_rows(r0_retrieval_output_data, name="R0 output"), name="R0")
    r2r_by_id = _by_id(_rows(r2r_retrieval_output_data, name="R2R output"), name="R2R")
    expected = set(approved.manifest.ordered_question_ids)
    if (
        set(queue_by_id) != expected
        or not expected.issubset(r0_by_id)
        or set(r2r_by_id) != expected
    ):
        raise OracleArtifactError(
            "ORACLE_ID_MISMATCH", "oracle labels, queue, R0, and R2R IDs do not align"
        )
    o2_rows: list[dict[str, Any]] = []
    o3_rows: list[dict[str, Any]] = []
    o4_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    eligible_queue: list[dict[str, Any]] = []
    records_by_id = {record.question_id: record for record in approved.records}
    for question_id in approved.manifest.ordered_question_ids:
        queue = queue_by_id[question_id]
        candidates = queue.get("candidates")
        question = queue.get("question")
        if not isinstance(candidates, list) or not isinstance(question, str):
            raise OracleArtifactError("ORACLE_INPUT_INVALID", "queue question/candidates invalid")
        display_by_id: dict[str, str] = {}
        for candidate in candidates:
            evidence_id = candidate.get("evidence_id") if isinstance(candidate, dict) else None
            display_text = candidate.get("display_text") if isinstance(candidate, dict) else None
            if (
                not isinstance(evidence_id, str)
                or not isinstance(display_text, str)
                or evidence_id in display_by_id
            ):
                raise OracleArtifactError(
                    "ORACLE_INPUT_INVALID", "queue evidence identity/text is invalid"
                )
            display_by_id[evidence_id] = display_text
        record = records_by_id[question_id]
        relevance_by_id = {
            evidence.evidence_id: evidence.relevance for evidence in record.relevant_evidence
        }
        r0_ranking = _ranking_ids(r0_by_id[question_id])
        r2r_ranking = _ranking_ids(r2r_by_id[question_id])
        selections = {
            "O2": _select_variant(
                ranking=r0_ranking,
                relevance_by_id=relevance_by_id,
                display_by_id=display_by_id,
                question=question,
                maximum_input_tokens=maximum_input_tokens,
                token_counter=token_counter,
            ),
            "O3": _select_variant(
                ranking=r2r_ranking,
                relevance_by_id=relevance_by_id,
                display_by_id=display_by_id,
                question=question,
                maximum_input_tokens=maximum_input_tokens,
                token_counter=token_counter,
            ),
        }
        selections["O4"] = selections["O2"]
        report_rows.append(
            {
                "question_id": question_id,
                "selections": {name: asdict(value) for name, value in selections.items()},
            }
        )
        if selections["O2"].status == "SELECTED":
            eligible_queue.append(queue)
            for name, rows in (("O2", o2_rows), ("O3", o3_rows), ("O4", o4_rows)):
                rows.append(
                    {
                        "schema_version": "model.retrieval.output.v1",
                        "run_id": f"R-002A-{name}-bounded-oracle-v1",
                        "question_id": question_id,
                        "question_checksum": queue.get("question_checksum"),
                        "candidates": [
                            {"evidence_id": evidence_id, "oracle_selected": True}
                            for evidence_id in selections[name].selected_evidence_ids
                        ],
                    }
                )

    def jsonl(rows: list[dict[str, Any]]) -> bytes:
        return b"".join(content_json_bytes(row) for row in rows)

    eligible_data = jsonl(eligible_queue)
    o2_data = jsonl(o2_rows)
    o3_data = jsonl(o3_rows)
    o4_data = jsonl(o4_rows)
    report = content_json_bytes(
        {
            "schema_version": "retrieval.oracle-selection.v1",
            "label_scope": "bounded_labeled_candidate_oracle",
            "promotable": False,
            "maximum_evidence_count": 3,
            "maximum_input_tokens": maximum_input_tokens,
            "question_count": len(report_rows),
            "generation_eligible_count": len(eligible_queue),
            "unresolved_count": len(report_rows) - len(eligible_queue),
            "inputs": {
                "annotation_queue_checksum": checksum_bytes(annotation_queue_data),
                "benchmark_manifest_checksum": checksum_bytes(benchmark_manifest_data),
                "benchmark_checksum": checksum_bytes(benchmark_data),
                "r0_retrieval_checksum": checksum_bytes(r0_retrieval_output_data),
                "r2r_retrieval_checksum": checksum_bytes(r2r_retrieval_output_data),
            },
            "outputs": {
                "eligible_queue_checksum": checksum_bytes(eligible_data),
                "o2_retrieval_checksum": checksum_bytes(o2_data),
                "o3_retrieval_checksum": checksum_bytes(o3_data),
                "o4_retrieval_checksum": checksum_bytes(o4_data),
            },
            "questions": report_rows,
        }
    )
    return BoundedOracleArtifacts(eligible_data, o2_data, o3_data, o4_data, report)


__all__ = [
    "BoundedOracleArtifacts",
    "InputTokenCounter",
    "OracleArtifactError",
    "build_bounded_oracle_artifacts",
]
