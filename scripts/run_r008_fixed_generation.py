"""Run fixed G1A512 over one frozen R-008 retrieval output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.competition import evaluate_competition_bytes
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.huggingface_local import Qwen3GeneratorBackend

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scorer-root", required=True, type=Path)
    parser.add_argument("--nltk-data", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    queue_data = arguments.queue.read_bytes()
    retrieval_data = arguments.retrieval.read_bytes()
    backend = Qwen3GeneratorBackend(
        arguments.checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device="cuda",
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
    )
    torch.cuda.reset_peak_memory_stats()
    predictions_data, references_data, manifest_data = run_grounded_generation_experiment(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        backend=backend,
        system_prompt=PROMPT_A,
        run_id=arguments.run_id,
        evidence_limit=3,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        profile_state="diagnostic_non_promotable",
    )
    evaluation = evaluate_competition_bytes(
        predictions_data,
        references_data,
        scorer_root=arguments.scorer_root,
        nltk_data_root=arguments.nltk_data,
        baseline_kind="r008_fixed_generator",
        limitation="development_only_retrieval_single_axis",
    )
    manifest = json.loads(manifest_data)
    telemetry_data = content_json_bytes(
        {
            "schema_version": "model.generation.telemetry.v1",
            "run_id": arguments.run_id,
            "execution_mode": "local-offline",
            "paid_service_used": False,
            "question_count": evaluation.question_count,
            "macro_rouge_l": evaluation.macro_rouge_l,
            "macro_meteor": evaluation.macro_meteor,
            "wall_seconds": manifest["elapsed_seconds"],
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
    }
    checksums = {
        name: write_immutable_bytes(arguments.output_directory / name, data)
        for name, data in outputs.items()
    }
    print(
        json.dumps(
            {
                "run_id": arguments.run_id,
                "macro_meteor": evaluation.macro_meteor,
                "macro_rouge_l": evaluation.macro_rouge_l,
                "checksums": checksums,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
