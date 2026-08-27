"""Run the approved prompt-B generator ablation on frozen development evidence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.competition import evaluate_competition_bytes
from legal_rag.evaluation.generator_comparison import compare_prompt_generation_experiments
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_B
from legal_rag.models.huggingface_local import Qwen3GeneratorBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "G2R0A512-qwen3-1.7b-prompt-b-btc-approved-v1"
BASELINE_RUN_ID = "G1R0A512-qwen3-1.7b-prompt-a-btc-approved-v1"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

ANNOTATION_QUEUE = PROJECT_ROOT / "artifacts/evaluations/grounding/annotation-work-queue.v1.jsonl"
RETRIEVAL_OUTPUT = PROJECT_ROOT / "artifacts/evaluations/mil-004/retrieval.v1.jsonl"
BASELINE_DIR = PROJECT_ROOT / "artifacts/evaluations/mil-006" / BASELINE_RUN_ID
OUTPUT_DIR = PROJECT_ROOT / "artifacts/evaluations/mil-006" / RUN_ID
MODEL_CHECKPOINT = PROJECT_ROOT / ".local/models/qwen3-1.7b" / MODEL_REVISION


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _number(value: object, *, field: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"expected numeric field: {field}")
    return float(value)


def main() -> int:
    annotation_data = ANNOTATION_QUEUE.read_bytes()
    retrieval_data = RETRIEVAL_OUTPUT.read_bytes()
    baseline_manifest_data = (BASELINE_DIR / "manifest.json").read_bytes()
    baseline_per_query_data = (BASELINE_DIR / "evaluation-per-query.jsonl").read_bytes()
    baseline_telemetry = _read_json(BASELINE_DIR / "telemetry.json")

    backend = Qwen3GeneratorBackend(
        MODEL_CHECKPOINT,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device="cuda",
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
    )
    torch.cuda.reset_peak_memory_stats()
    predictions_data, references_data, manifest_data = run_grounded_generation_experiment(
        annotation_queue_data=annotation_data,
        retrieval_output_data=retrieval_data,
        backend=backend,
        system_prompt=PROMPT_B,
        run_id=RUN_ID,
        evidence_limit=3,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id=BASELINE_RUN_ID,
        changed_axes=("prompt",),
        profile_state="btc_approved_local_ablation",
    )
    evaluation = evaluate_competition_bytes(
        predictions_data,
        references_data,
        scorer_root=PROJECT_ROOT / "Scoring-Program-Task-LegalQA",
        nltk_data_root=PROJECT_ROOT / "resources/nltk_data",
        baseline_kind="generator_prompt_ablation",
        limitation="development_only_prompt_b_single_axis",
    )
    manifest = _read_json_bytes(manifest_data)
    candidate_runtime = _number(manifest["elapsed_seconds"], field="elapsed_seconds")
    comparison_data = compare_prompt_generation_experiments(
        baseline_per_query_data=baseline_per_query_data,
        candidate_per_query_data=evaluation.per_query_bytes,
        baseline_manifest_data=baseline_manifest_data,
        candidate_manifest_data=manifest_data,
        baseline_runtime_seconds=_number(baseline_telemetry["wall_seconds"], field="wall_seconds"),
        candidate_runtime_seconds=candidate_runtime,
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "model.generation.telemetry.v1",
            "run_id": RUN_ID,
            "execution_mode": "local-offline",
            "paid_service_used": False,
            "question_count": evaluation.question_count,
            "macro_rouge_l": evaluation.macro_rouge_l,
            "macro_meteor": evaluation.macro_meteor,
            "wall_seconds": candidate_runtime,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    outputs = {
        "predictions.json": predictions_data,
        "references.json": references_data,
        "manifest.json": manifest_data,
        "evaluation-per-query.jsonl": evaluation.per_query_bytes,
        "evaluation-report.json": evaluation.report_bytes,
        "telemetry.json": telemetry_data,
        "comparison-vs-r0.json": comparison_data,
    }
    checksums = {
        name: write_immutable_bytes(OUTPUT_DIR / name, data) for name, data in outputs.items()
    }
    comparison = _read_json_bytes(comparison_data)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "macro_meteor": evaluation.macro_meteor,
                "macro_rouge_l": evaluation.macro_rouge_l,
                "promotion_state": comparison["promotion_state"],
                "checksums": checksums,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_json_bytes(data: bytes) -> dict[str, object]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object bytes")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
