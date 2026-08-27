"""Freeze and prefill the grounding queue for the G2 prompt ablation."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.grounding.answer_assessment import (
    build_answer_assessment_queue,
    prefill_answer_assessment_queue,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUNDING_DIR = PROJECT_ROOT / "artifacts/evaluations/grounding"
RUN_ID = "G2R0A512-qwen3-1.7b-prompt-b-btc-approved-v1"
RUN_DIR = PROJECT_ROOT / "artifacts/evaluations/mil-006" / RUN_ID


def main() -> int:
    queue_data = build_answer_assessment_queue(
        annotation_queue_data=(GROUNDING_DIR / "annotation-work-queue.v1.jsonl").read_bytes(),
        retrieval_output_data=(
            PROJECT_ROOT / "artifacts/evaluations/mil-004/retrieval.v1.jsonl"
        ).read_bytes(),
        predictions_data=(RUN_DIR / "predictions.json").read_bytes(),
        evaluated_run_id=RUN_ID,
        evidence_limit=3,
    )
    prefilled_data, report_data = prefill_answer_assessment_queue(
        candidate_queue_data=queue_data,
        source_queue_data=(
            GROUNDING_DIR / "G1R0A512-answer-grounding-work-queue.v1.jsonl"
        ).read_bytes(),
        source_labeled_data=(
            GROUNDING_DIR / "G1R0A512-answer-grounding-work-queue.v1.labeled.jsonl"
        ).read_bytes(),
    )
    outputs = {
        "G2-answer-grounding-work-queue.v1.jsonl": queue_data,
        "G2-answer-grounding-work-queue.v1.prefilled.jsonl": prefilled_data,
        "G2-answer-grounding-prefill-report.v1.json": report_data,
    }
    checksums = {
        name: write_immutable_bytes(GROUNDING_DIR / name, data) for name, data in outputs.items()
    }
    report = json.loads(report_data)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
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
