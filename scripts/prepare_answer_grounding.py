"""Build and safely prefill a checksum-bound answer-grounding work queue."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.grounding.answer_assessment import (
    build_answer_assessment_queue,
    prefill_answer_assessment_queue,
)

_SAFE_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\Z")


def prepare_answer_grounding_artifacts(
    *,
    artifact_prefix: str,
    evaluated_run_id: str,
    annotation_queue_data: bytes,
    retrieval_output_data: bytes,
    predictions_data: bytes,
    source_queue_data: bytes,
    source_labeled_data: bytes,
    evidence_limit: int,
) -> dict[str, bytes]:
    """Return deterministic pending, prefilled, and prefill-report artifacts."""

    if _SAFE_PREFIX.fullmatch(artifact_prefix) is None:
        raise ValueError("artifact prefix must contain only letters, digits, and hyphens")
    queue_data = build_answer_assessment_queue(
        annotation_queue_data=annotation_queue_data,
        retrieval_output_data=retrieval_output_data,
        predictions_data=predictions_data,
        evaluated_run_id=evaluated_run_id,
        evidence_limit=evidence_limit,
    )
    prefilled_data, report_data = prefill_answer_assessment_queue(
        candidate_queue_data=queue_data,
        source_queue_data=source_queue_data,
        source_labeled_data=source_labeled_data,
    )
    return {
        f"{artifact_prefix}-answer-grounding-work-queue.v1.jsonl": queue_data,
        f"{artifact_prefix}-answer-grounding-work-queue.v1.prefilled.jsonl": prefilled_data,
        f"{artifact_prefix}-answer-grounding-prefill-report.v1.json": report_data,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--annotation-queue", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--source-labeled", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--evidence-limit", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    artifacts = prepare_answer_grounding_artifacts(
        artifact_prefix=arguments.artifact_prefix,
        evaluated_run_id=arguments.run_id,
        annotation_queue_data=arguments.annotation_queue.read_bytes(),
        retrieval_output_data=arguments.retrieval.read_bytes(),
        predictions_data=arguments.predictions.read_bytes(),
        source_queue_data=arguments.source_queue.read_bytes(),
        source_labeled_data=arguments.source_labeled.read_bytes(),
        evidence_limit=arguments.evidence_limit,
    )
    checksums = {
        name: write_immutable_bytes(arguments.output_directory / name, data)
        for name, data in artifacts.items()
    }
    report_name = f"{arguments.artifact_prefix}-answer-grounding-prefill-report.v1.json"
    report = json.loads(artifacts[report_name])
    print(
        json.dumps(
            {
                "run_id": arguments.run_id,
                "question_count": report["question_count"],
                "prefilled_count": report["prefilled_count"],
                "pending_count": report["pending_count"],
                "checksums": checksums,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
