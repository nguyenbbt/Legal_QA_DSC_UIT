"""Run the single D-060 train-calibrated answer-compaction recovery."""

from __future__ import annotations

import json
import time
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.competition import evaluate_competition_bytes
from legal_rag.evaluation.generator_comparison import (
    compare_postprocessed_generation_experiments,
)
from legal_rag.generation.answer_compaction import (
    build_deletion_only_grounding_proof,
    compact_answer,
    derive_answer_compaction_policy,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "G1R0A512-qwen3-1.7b-prompt-a-btc-approved-v1"
RUN_ID = "D060-G1R0A512-train-median-prefix-v1"
BASELINE = ROOT / "artifacts/evaluations/mil-006" / BASELINE_ID
OUTPUT = ROOT / "artifacts/evaluations/mil-006" / RUN_ID
BASELINE_MODAL_PUBLIC_SECONDS = 4815.1867675
MAXIMUM_PUBLIC_SECONDS = 6 * 60 * 60


def _object(data: bytes) -> dict[str, object]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("D-060 expected a JSON object")
    return value


def main() -> int:
    train_data = (ROOT / "data/train.json").read_bytes()
    policy, policy_data = derive_answer_compaction_policy(train_data, split="train")
    if policy.source_row_count != 7_000 or policy.maximum_whitespace_tokens != 312:
        raise RuntimeError("D-060 official-train policy identity differs from the frozen contract")

    baseline_predictions_data = (BASELINE / "predictions.json").read_bytes()
    baseline_predictions = _object(baseline_predictions_data)
    started = time.perf_counter()
    candidate_predictions: dict[str, dict[str, str]] = {}
    for question_id, row in baseline_predictions.items():
        if not isinstance(question_id, str) or not isinstance(row, dict):
            raise RuntimeError("D-060 baseline predictions are invalid")
        answer = row.get("answer")
        if not isinstance(answer, str):
            raise RuntimeError("D-060 baseline answer is invalid")
        candidate_predictions[question_id] = {"answer": compact_answer(answer, policy)}
    candidate_predictions_data = content_json_bytes(candidate_predictions)
    compaction_seconds = time.perf_counter() - started
    deletion_proof_data = build_deletion_only_grounding_proof(
        baseline_predictions_data,
        candidate_predictions_data,
    )
    deletion_proof = _object(deletion_proof_data)

    references_data = (BASELINE / "references.json").read_bytes()
    baseline_manifest_data = (BASELINE / "manifest.json").read_bytes()
    candidate_manifest = _object(baseline_manifest_data)
    candidate_manifest.update(
        {
            "run_id": RUN_ID,
            "profile_state": "diagnostic_non_promotable",
            "predictions_checksum": checksum_bytes(candidate_predictions_data),
            "comparison": {
                "baseline_run_id": BASELINE_ID,
                "changed_axes": ["postprocessor"],
            },
            "postprocessor": {
                "policy_id": policy.policy_id,
                "policy_checksum": checksum_bytes(policy_data),
                "source_checksum": policy.source_checksum,
                "maximum_whitespace_tokens": policy.maximum_whitespace_tokens,
            },
        }
    )
    candidate_manifest_data = content_json_bytes(candidate_manifest)
    evaluation = evaluate_competition_bytes(
        candidate_predictions_data,
        references_data,
        scorer_root=ROOT / "Scoring-Program-Task-LegalQA",
        nltk_data_root=ROOT / "resources/nltk_data",
        baseline_kind="d060_train_median_prefix_compaction",
        limitation="single_train_calibrated_postprocessor_development_evaluation",
    )
    baseline_telemetry = _object((BASELINE / "telemetry.json").read_bytes())
    baseline_seconds = float(baseline_telemetry["wall_seconds"])
    candidate_seconds = baseline_seconds + compaction_seconds
    comparison_data = compare_postprocessed_generation_experiments(
        baseline_per_query_data=(BASELINE / "evaluation-per-query.jsonl").read_bytes(),
        candidate_per_query_data=evaluation.per_query_bytes,
        baseline_manifest_data=baseline_manifest_data,
        candidate_manifest_data=candidate_manifest_data,
        baseline_runtime_seconds=baseline_seconds,
        candidate_runtime_seconds=candidate_seconds,
    )
    comparison = _object(comparison_data)
    projected_public_seconds = BASELINE_MODAL_PUBLIC_SECONDS + compaction_seconds * (1000 / 60)
    hard_resource_gate = (
        "passed" if projected_public_seconds <= MAXIMUM_PUBLIC_SECONDS else "failed"
    )
    public_gate = (
        comparison["numeric_evaluation_gate"] == "passed"
        and comparison["resource_gate"] == "passed"
        and deletion_proof["proof_state"] == "passed"
        and hard_resource_gate == "passed"
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "answer.compaction.telemetry.v1",
            "run_id": RUN_ID,
            "execution_mode": "local-offline-model-free",
            "paid_service_used": False,
            "question_count": evaluation.question_count,
            "compaction_seconds": compaction_seconds,
            "candidate_pipeline_seconds": candidate_seconds,
            "projected_1000_a10_seconds": projected_public_seconds,
            "hard_six_hour_resource_gate": hard_resource_gate,
            "macro_meteor": evaluation.macro_meteor,
            "macro_rouge_l": evaluation.macro_rouge_l,
        }
    )
    state_data = content_json_bytes(
        {
            "schema_version": "d060.development.state.v1",
            "run_id": RUN_ID,
            "public_generation_gate": "passed" if public_gate else "failed",
            "numeric_evaluation_gate": comparison["numeric_evaluation_gate"],
            "resource_gate": comparison["resource_gate"],
            "deletion_only_grounding_gate": deletion_proof["proof_state"],
            "hard_six_hour_resource_gate": hard_resource_gate,
        }
    )
    outputs = {
        "policy.v1.json": policy_data,
        "predictions.json": candidate_predictions_data,
        "references.json": references_data,
        "manifest.json": candidate_manifest_data,
        "evaluation-per-query.jsonl": evaluation.per_query_bytes,
        "evaluation-report.json": evaluation.report_bytes,
        "comparison-vs-g1r0a512.json": comparison_data,
        "deletion-only-grounding-proof.v1.json": deletion_proof_data,
        "telemetry.json": telemetry_data,
        "development-state.v1.json": state_data,
    }
    checksums = {name: write_immutable_bytes(OUTPUT / name, data) for name, data in outputs.items()}
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "maximum_whitespace_tokens": policy.maximum_whitespace_tokens,
                "changed_answer_count": deletion_proof["changed_answer_count"],
                "macro_meteor": evaluation.macro_meteor,
                "macro_rouge_l": evaluation.macro_rouge_l,
                "public_generation_gate": "passed" if public_gate else "failed",
                "checksums": checksums,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
