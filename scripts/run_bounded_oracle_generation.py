"""Run the fixed local G1A512 generator over R-002A O2/O3 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.competition import (
    evaluate_competition_bytes,
    write_competition_evaluation,
)
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.huggingface_local import Qwen3GeneratorBackend

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--o2-retrieval", required=True, type=Path)
    parser.add_argument("--o3-retrieval", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scorer-root", required=True, type=Path)
    parser.add_argument("--nltk-data", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def _run_variant(
    *,
    name: str,
    queue: bytes,
    retrieval: bytes,
    backend: Qwen3GeneratorBackend,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    output = arguments.output_directory / name
    predictions, references, manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue,
        retrieval_output_data=retrieval,
        backend=backend,
        system_prompt=PROMPT_A,
        run_id=f"R-002A-{name}-G1A512-v1",
        evidence_limit=3,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        profile_state="diagnostic_non_promotable",
    )
    write_immutable_bytes(output / "predictions.json", predictions)
    write_immutable_bytes(output / "references.json", references)
    write_immutable_bytes(output / "manifest.json", manifest)
    evaluation = evaluate_competition_bytes(
        predictions,
        references,
        scorer_root=arguments.scorer_root,
        nltk_data_root=arguments.nltk_data,
    )
    write_competition_evaluation(
        evaluation,
        per_query_path=output / "evaluation-per-query.jsonl",
        report_path=output / "evaluation-report.json",
    )
    manifest_value = json.loads(manifest)
    telemetry = {
        "schema_version": "retrieval.oracle-generation.telemetry.v1",
        "variant": name,
        "question_count": evaluation.question_count,
        "wall_seconds": manifest_value["elapsed_seconds"],
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "macro_meteor": evaluation.macro_meteor,
        "macro_rouge_l": evaluation.macro_rouge_l,
        "paid_service_used": False,
        "execution_mode": "local-offline",
    }
    write_immutable_bytes(output / "telemetry.json", content_json_bytes(telemetry))
    return telemetry


def main() -> int:
    arguments = _parser().parse_args()
    queue = arguments.queue.read_bytes()
    backend = Qwen3GeneratorBackend(
        arguments.checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device="cuda",
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
    )
    results: list[dict[str, object]] = []
    for name, path in (("O2", arguments.o2_retrieval), ("O3", arguments.o3_retrieval)):
        torch.cuda.reset_peak_memory_stats()
        results.append(
            _run_variant(
                name=name,
                queue=queue,
                retrieval=path.read_bytes(),
                backend=backend,
                arguments=arguments,
            )
        )
    write_immutable_bytes(
        arguments.output_directory / "oracle-generation-summary.v1.json",
        content_json_bytes(
            {
                "schema_version": "retrieval.oracle-generation.summary.v1",
                "label_scope": "bounded_labeled_candidate_oracle",
                "promotable": False,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "variants": results,
                "o4_execution": "SKIPPED_BYTE_EQUIVALENT_SELECTION_TO_O2",
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
