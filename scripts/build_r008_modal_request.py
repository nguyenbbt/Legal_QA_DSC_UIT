from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.providers.modal_reranker_training import (
    R008TrainingLifecycle,
    validate_r008_training_payload,
)
from legal_rag.training.recipes import FT_RERANK_CENTRAL, recipe_checksum


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the closed R-008 Modal request"
    )
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--run-id", default="R008-qwen3-reranker-0.6b-central-v1")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    groups_data = (arguments.dataset_directory / "training-groups.v1.jsonl").read_bytes()
    provenance_data = (arguments.dataset_directory / "training-examples.v1.jsonl").read_bytes()
    dataset_manifest_data = (arguments.dataset_directory / "training-manifest.v1.json").read_bytes()
    manifest = json.loads(dataset_manifest_data)
    model_manifest_data = arguments.model_manifest.read_bytes()
    model_manifest = json.loads(model_manifest_data)
    reranker = next(model for model in model_manifest["models"] if model["role"] == "reranker")
    request_data = content_json_bytes(
        {
            "schema_version": "modal.r008.training-request.v1",
            "run_id": arguments.run_id,
            "model_id": reranker["model_id"],
            "model_revision": reranker["model_revision"],
            "tokenizer_revision": reranker["tokenizer_revision"],
            "model_manifest_checksum": checksum_bytes(model_manifest_data),
            "recipe_checksum": recipe_checksum(FT_RERANK_CENTRAL),
            "dataset_manifest_checksum": checksum_bytes(dataset_manifest_data),
            "groups_checksum": checksum_bytes(groups_data),
            "provenance_checksum": checksum_bytes(provenance_data),
            "group_count": manifest["group_count"],
            "pair_count": manifest["pair_count"],
            "maximum_length": 1536,
            "seed": FT_RERANK_CENTRAL.seed,
            "base_parameter_count": reranker["exact_parameter_count"],
            "whole_system_base_parameter_count": model_manifest["system_parameter_count"],
        }
    )
    payload = validate_r008_training_payload(
        request_data=request_data,
        groups_data=groups_data,
        provenance_data=provenance_data,
        dataset_manifest_data=dataset_manifest_data,
    )
    lifecycle = R008TrainingLifecycle().record_preflight()
    request_checksum = write_immutable_bytes(
        arguments.dataset_directory / "modal.training-request.v1.json",
        request_data,
    )
    preflight_data = content_json_bytes(
        {
            "schema_version": "modal.r008.preflight-report.v1",
            "run_id": payload.run_id,
            "state": lifecycle.state,
            "request_checksum": request_checksum,
            "dataset_manifest_checksum": payload.dataset_manifest_checksum,
            "groups_checksum": payload.groups_checksum,
            "provenance_checksum": payload.provenance_checksum,
            "group_count": payload.group_count,
            "pair_count": payload.pair_count,
            "only_official_train": True,
            "contains_generated_text": False,
            "contains_train_answers": False,
            "passes_parameter_gate": True,
        }
    )
    preflight_checksum = write_immutable_bytes(
        arguments.dataset_directory / "modal.preflight-report.v1.json",
        preflight_data,
    )
    print(
        f"R008 PREFLIGHT groups={payload.group_count} pairs={payload.pair_count} "
        f"request={request_checksum} report={preflight_checksum}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
