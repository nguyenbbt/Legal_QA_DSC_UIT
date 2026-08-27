from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "dsc2026-legalqa-r008-reranker-v1"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_VOLUME_NAME = "dsc2026-qwen3-reranker-0-6b-r008-model-v1"
RUN_VOLUME_NAME = "dsc2026-r008-reranker-central-v1"
MODEL_ROOT = "/models"
RUN_ROOT = "/runs"
MODEL_PATH = f"{MODEL_ROOT}/{MODEL_ID.replace('/', '--')}/{MODEL_REVISION}"
RUN_ID = "R008-qwen3-reranker-0.6b-central-v1"
MAXIMUM_REMOTE_WALL_SECONDS = 4 * 60 * 60
WHOLE_SYSTEM_BASE_PARAMETER_COUNT = 3_223_292_928

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==1.14.0",
        "huggingface-hub==1.7.1",
        "peft==0.20.0",
        "pydantic==2.13.4",
        "safetensors==0.8.0",
        "torch==2.8.0",
        "transformers==5.15.1",
    )
    .add_local_python_source("legal_rag")
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    serialized=True,
    include_source=False,
    volumes={MODEL_ROOT: model_volume},
)
def download_pinned_reranker() -> dict[str, str]:
    from pathlib import Path as ContainerPath

    from huggingface_hub import snapshot_download

    ContainerPath(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_PATH)
    model_volume.commit()
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}


@app.function(
    gpu="A10",
    cpu=4,
    memory=16384,
    max_containers=1,
    timeout=MAXIMUM_REMOTE_WALL_SECONDS,
    serialized=True,
    include_source=False,
    block_network=True,
    restrict_modal_access=True,
    volumes={
        MODEL_ROOT: model_volume.with_mount_options(read_only=True),
        RUN_ROOT: run_volume,
    },
)
def train_central_reranker(
    request_data: bytes,
    groups_data: bytes,
    provenance_data: bytes,
    dataset_manifest_data: bytes,
) -> bytes:
    from pathlib import Path as ContainerPath

    import torch

    from legal_rag.domain.checksums import content_json_bytes
    from legal_rag.providers.modal_reranker_training import (
        validate_r008_training_payload,
        validate_r008_training_response,
    )
    from legal_rag.training.reranker_lora import (
        RerankerLoraRunConfig,
        train_qwen3_reranker_lora,
    )

    payload = validate_r008_training_payload(
        request_data=request_data,
        groups_data=groups_data,
        provenance_data=provenance_data,
        dataset_manifest_data=dataset_manifest_data,
    )
    if (
        payload.run_id != RUN_ID
        or payload.model_id != MODEL_ID
        or payload.model_revision != MODEL_REVISION
        or payload.whole_system_base_parameter_count != WHOLE_SYSTEM_BASE_PARAMETER_COUNT
    ):
        raise RuntimeError("Modal R-008 request does not match the closed central run")
    checkpoint = ContainerPath(MODEL_PATH)
    if not checkpoint.is_dir():
        raise RuntimeError("pinned reranker checkpoint is absent")
    output_directory = ContainerPath(RUN_ROOT) / payload.run_id
    result = train_qwen3_reranker_lora(
        checkpoint=checkpoint,
        groups_data=groups_data,
        output_directory=output_directory,
        model_id=payload.model_id,
        model_revision=payload.model_revision,
        config=RerankerLoraRunConfig(mode="central"),
        device="cuda",
    )
    whole_system_parameters = (
        payload.whole_system_base_parameter_count + result.adapter_parameter_count
    )
    response_data = content_json_bytes(
        {
            "schema_version": "modal.r008.training-response.v1",
            "run_id": payload.run_id,
            "adapter_path": f"{payload.run_id}/adapter",
            "adapter_checksum": result.adapter_checksum,
            "adapter_parameter_count": result.adapter_parameter_count,
            "whole_system_parameter_count": whole_system_parameters,
            "training_metrics": {
                "epochs": 2,
                "mean_train_loss": result.mean_train_loss,
                "optimizer_steps": result.optimizer_step_count,
                "train_pairs": result.train_pair_count,
                "validation_loss": result.validation_loss,
                "validation_pair_accuracy": result.validation_pair_accuracy,
                "validation_pairs": result.validation_pair_count,
            },
            "telemetry": {
                "elapsed_seconds": result.elapsed_seconds,
                "gpu_name": torch.cuda.get_device_name(0),
                "peak_cuda_bytes": result.peak_cuda_bytes,
                "torch_version": torch.__version__,
            },
        }
    )
    validate_r008_training_response(request_data=request_data, response_data=response_data)
    (output_directory / "modal.training-response.v1.json").write_bytes(response_data)
    run_volume.commit()
    print(
        f"completed run_id={payload.run_id} pairs={result.train_pair_count} "
        f"adapter_parameters={result.adapter_parameter_count}",
        flush=True,
    )
    return response_data


@app.function(
    cpu=1,
    memory=2048,
    timeout=10 * 60,
    serialized=True,
    include_source=False,
    block_network=True,
    restrict_modal_access=True,
    volumes={RUN_ROOT: run_volume.with_mount_options(read_only=True)},
)
def inspect_remote_adapter() -> dict[str, object]:
    import hashlib
    from pathlib import Path as ContainerPath

    from legal_rag.training.reranker_lora import directory_checksum

    adapter = ContainerPath(RUN_ROOT) / RUN_ID / "adapter"
    files = tuple(sorted(path for path in adapter.rglob("*") if path.is_file()))
    return {
        "adapter_checksum": directory_checksum(adapter),
        "files": [
            {
                "name": path.relative_to(adapter).as_posix(),
                "bytes": path.stat().st_size,
                "checksum": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            }
            for path in files
        ],
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@app.local_entrypoint()
def main(stage: str = "download", dataset_directory: str = "") -> None:
    if stage == "download":
        identity = download_pinned_reranker.remote()
        print(
            f"downloaded model_id={identity['model_id']} revision={identity['model_revision']}",
            flush=True,
        )
        return
    if stage == "inspect":
        print(json.dumps(inspect_remote_adapter.remote(), sort_keys=True), flush=True)
        return
    if stage != "train":
        raise ValueError("stage must be download, inspect, or train")

    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.providers.modal_reranker_training import (
        R008TrainingLifecycle,
        validate_r008_training_payload,
        validate_r008_training_response,
    )

    root = _project_root()
    dataset = (
        Path(dataset_directory)
        if dataset_directory
        else root / "artifacts" / "training" / "reranker" / "R008-qwen3-reranker-0.6b-central-v1"
    )
    request_data = (dataset / "modal.training-request.v1.json").read_bytes()
    groups_data = (dataset / "training-groups.v1.jsonl").read_bytes()
    provenance_data = (dataset / "training-examples.v1.jsonl").read_bytes()
    dataset_manifest_data = (dataset / "training-manifest.v1.json").read_bytes()
    payload = validate_r008_training_payload(
        request_data=request_data,
        groups_data=groups_data,
        provenance_data=provenance_data,
        dataset_manifest_data=dataset_manifest_data,
    )
    lifecycle = R008TrainingLifecycle().record_preflight()
    print(
        f"preflight state={lifecycle.state} groups={payload.group_count} "
        f"pairs={payload.pair_count}",
        flush=True,
    )
    response_data = train_central_reranker.remote(
        request_data,
        groups_data,
        provenance_data,
        dataset_manifest_data,
    )
    response = validate_r008_training_response(
        request_data=request_data,
        response_data=response_data,
    )
    lifecycle = lifecycle.record_remote_run(response.run_id)
    response_checksum = write_immutable_bytes(
        dataset / "modal.training-response.v1.json",
        response_data,
    )
    response_value = json.loads(response_data)
    print(
        f"remote_complete state={lifecycle.state} adapter={response.adapter_checksum} "
        f"response={response_checksum} metrics={json.dumps(response_value['training_metrics'])}",
        flush=True,
    )
