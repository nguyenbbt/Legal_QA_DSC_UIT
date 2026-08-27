"""Rebaseline the R-008 LoRA against its fixed base-model control."""

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
from legal_rag.models.huggingface_local import (
    Qwen3AdapterRerankerBackend,
    Qwen3RerankerBackend,
)
from legal_rag.retrieval.qwen3_reranker_prompt import QWEN3_RERANKER_SYSTEM
from legal_rag.training.reranker_lora import directory_checksum

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
ADAPTER_ID = "R008-qwen3-reranker-0.6b-central-v1"
MAXIMUM_LENGTH = 1536
CANDIDATE_LIMIT = 50


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--candidate-queue", required=True, type=Path)
    parser.add_argument("--grounding-benchmark", required=True, type=Path)
    parser.add_argument("--grounding-manifest", required=True, type=Path)
    parser.add_argument("--parameter-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-id", default=ADAPTER_ID)
    parser.add_argument("--adapter-audit", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.mode == "adapter" and (
        arguments.adapter is None or arguments.adapter_audit is None
    ):
        raise ValueError("adapter mode requires the adapter and its exact audit")
    if arguments.mode == "base" and (
        arguments.adapter is not None or arguments.adapter_audit is not None
    ):
        raise ValueError("base control cannot load an adapter")

    queue_data = arguments.candidate_queue.read_bytes()
    benchmark_data = arguments.grounding_benchmark.read_bytes()
    common = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instruction": LEGAL_EVIDENCE_INSTRUCTION,
        "device": "cuda",
        "batch_size": 2,
        "maximum_length": MAXIMUM_LENGTH,
    }
    if arguments.mode == "adapter":
        backend = Qwen3AdapterRerankerBackend(
            arguments.checkpoint,
            arguments.adapter,
            adapter_id=arguments.adapter_id,
            **common,
        )
        run_id = arguments.run_id or "R008-C1-qwen3-reranker-lora-top50-v1"
        adapter_checksum = directory_checksum(arguments.adapter)
        adapter_audit_checksum = checksum_bytes(arguments.adapter_audit.read_bytes())
    else:
        backend = Qwen3RerankerBackend(arguments.checkpoint, **common)
        run_id = arguments.run_id or "R008-C0-qwen3-reranker-base-top50-v1"
        adapter_checksum = None
        adapter_audit_checksum = None

    torch.cuda.reset_peak_memory_stats()
    output_data, report_data = run_labeled_reranker_experiment(
        annotation_queue_data=queue_data,
        grounding_benchmark_data=benchmark_data,
        backend=backend,
        run_id=run_id,
        candidate_limit=CANDIDATE_LIMIT,
    )
    report = json.loads(report_data)
    manifest = content_json_bytes(
        {
            "schema_version": "model.retrieval.experiment.manifest.v1",
            "run_id": run_id,
            "profile_state": "r008_fixed_candidate_rebaseline",
            "comparison_role": arguments.mode,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_checkpoint_checksum": checksum_file(arguments.checkpoint / "model.safetensors"),
            "adapter_id": arguments.adapter_id if arguments.mode == "adapter" else None,
            "adapter_checksum": adapter_checksum,
            "adapter_audit_checksum": adapter_audit_checksum,
            "parameter_manifest_checksum": checksum_bytes(
                arguments.parameter_manifest.read_bytes()
            ),
            "instruction_checksum": LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
            "system_prompt_checksum": checksum_bytes(QWEN3_RERANKER_SYSTEM.encode()),
            "dtype": "float16",
            "device": "cuda",
            "batch_size": 2,
            "maximum_length": MAXIMUM_LENGTH,
            "candidate_limit": CANDIDATE_LIMIT,
            "candidate_queue_checksum": checksum_bytes(queue_data),
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
    write_immutable_bytes(arguments.output_directory / "retrieval.v1.jsonl", output_data)
    write_immutable_bytes(arguments.output_directory / "report.v1.json", report_data)
    write_immutable_bytes(arguments.output_directory / "manifest.v1.json", manifest)
    print(report_data.decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
