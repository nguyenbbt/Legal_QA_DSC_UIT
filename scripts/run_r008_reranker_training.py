from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.training.reranker_lora import (
    RerankerLoraRunConfig,
    train_qwen3_reranker_lora,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded R-008 Qwen3 reranker LoRA")
    parser.add_argument("--mode", choices=("smoke", "central", "corrective"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument(
        "--model-revision",
        default="e61197ed45024b0ed8a2d74b80b4d909f1255473",
    )
    parser.add_argument("--whole-system-base-parameters", type=int, default=3_223_292_928)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    groups_data = arguments.groups.read_bytes()
    result = train_qwen3_reranker_lora(
        checkpoint=arguments.checkpoint,
        groups_data=groups_data,
        output_directory=arguments.output_directory,
        model_id=arguments.model_id,
        model_revision=arguments.model_revision,
        config=RerankerLoraRunConfig(mode=arguments.mode),
        device=arguments.device,
    )
    whole_system_parameters = (
        arguments.whole_system_base_parameters + result.adapter_parameter_count
    )
    if whole_system_parameters >= 4_000_000_000:
        raise RuntimeError("MODEL_PARAMETER_LIMIT")
    report = {
        "schema_version": "r008.reranker-lora.run-report.v1",
        "mode": result.mode,
        "model_id": arguments.model_id,
        "model_revision": arguments.model_revision,
        "groups_checksum": checksum_bytes(groups_data),
        "recipe_checksum": result.recipe_checksum,
        "train_pair_count": result.train_pair_count,
        "validation_pair_count": result.validation_pair_count,
        "optimizer_step_count": result.optimizer_step_count,
        "mean_train_loss": result.mean_train_loss,
        "validation_loss": result.validation_loss,
        "validation_pair_accuracy": result.validation_pair_accuracy,
        "adapter_parameter_count": result.adapter_parameter_count,
        "adapter_checksum": result.adapter_checksum,
        "whole_system_parameter_count": whole_system_parameters,
        "passes_parameter_gate": True,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_cuda_bytes": result.peak_cuda_bytes,
    }
    report_checksum = write_immutable_bytes(
        arguments.output_directory / "run-report.v1.json",
        content_json_bytes(report),
    )
    print(
        f"R008 {result.mode.upper()} pairs={result.train_pair_count} "
        f"val_accuracy={result.validation_pair_accuracy:.6f} "
        f"adapter={result.adapter_checksum} report={report_checksum}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
