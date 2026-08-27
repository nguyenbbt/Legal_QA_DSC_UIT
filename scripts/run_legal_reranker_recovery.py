"""Run R-005A exact-approved Qwen reranking over the R-003 sparse top 50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, checksum_file, content_json_bytes
from legal_rag.evaluation.legal_reranker_contract import (
    LEGAL_EVIDENCE_INSTRUCTION,
    LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
)
from legal_rag.evaluation.model_retrieval import run_labeled_reranker_experiment
from legal_rag.models.huggingface_local import Qwen3RerankerBackend

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--annotation-queue", required=True, type=Path)
    parser.add_argument("--grounding-benchmark", required=True, type=Path)
    parser.add_argument("--grounding-manifest", required=True, type=Path)
    parser.add_argument("--parameter-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()

    queue_rows = tuple(
        json.loads(line) for line in arguments.annotation_queue.read_bytes().splitlines()
    )
    top_50_queue = b"".join(
        content_json_bytes({**row, "candidates": row["candidates"][:50]}) for row in queue_rows
    )
    benchmark_data = arguments.grounding_benchmark.read_bytes()
    backend = Qwen3RerankerBackend(
        arguments.checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        instruction=LEGAL_EVIDENCE_INSTRUCTION,
        device="cuda",
        batch_size=2,
        maximum_length=8192,
    )
    torch.cuda.reset_peak_memory_stats()
    run_id = "X1-qwen3-legal-reranker-top50-v1"
    output_data, report_data = run_labeled_reranker_experiment(
        annotation_queue_data=top_50_queue,
        grounding_benchmark_data=benchmark_data,
        backend=backend,
        run_id=run_id,
        candidate_limit=50,
    )
    report = json.loads(report_data)
    manifest = content_json_bytes(
        {
            "schema_version": "model.retrieval.experiment.manifest.v1",
            "run_id": run_id,
            "profile_state": "btc_approved_legal_instruction_candidate",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_checkpoint_checksum": checksum_file(arguments.checkpoint / "model.safetensors"),
            "parameter_manifest_checksum": checksum_bytes(
                arguments.parameter_manifest.read_bytes()
            ),
            "instruction": LEGAL_EVIDENCE_INSTRUCTION,
            "instruction_checksum": LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
            "system_prompt_checksum": checksum_bytes(backend._SYSTEM.encode("utf-8")),
            "dtype": "float16",
            "device": "cuda",
            "batch_size": 2,
            "maximum_length": 8192,
            "candidate_limit": 50,
            "source_candidate_limit": 100,
            "source_annotation_queue_checksum": checksum_bytes(
                arguments.annotation_queue.read_bytes()
            ),
            "top_50_queue_checksum": checksum_bytes(top_50_queue),
            "grounding_benchmark_checksum": checksum_bytes(benchmark_data),
            "grounding_benchmark_manifest_checksum": checksum_bytes(
                arguments.grounding_manifest.read_bytes()
            ),
            "output_checksum": checksum_bytes(output_data),
            "report_checksum": checksum_bytes(report_data),
            "question_count": report["question_count"],
            "elapsed_seconds": report["elapsed_seconds"],
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "public_results_usage": "not_applicable_development_selection_only",
        }
    )
    write_immutable_bytes(
        arguments.output_directory / "candidate-queue.top50.v1.jsonl", top_50_queue
    )
    write_immutable_bytes(arguments.output_directory / "retrieval.v1.jsonl", output_data)
    write_immutable_bytes(arguments.output_directory / "report.v1.json", report_data)
    write_immutable_bytes(arguments.output_directory / "manifest.v1.json", manifest)
    print(report_data.decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
