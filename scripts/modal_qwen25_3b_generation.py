"""Run the single D-059 Qwen2.5-3B development/public generator campaign."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import unicodedata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dsc2026-legalqa-qwen25-3b-d059-v1"
CAMPAIGN_ID = "D059-qwen25-3b-r0-prompt-a-v1"
APPROVAL_ID = "D-059"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_LICENSE = "qwen-research"
EXPECTED_PARAMETER_COUNT = 3_085_938_688
MODEL_VOLUME_NAME = "dsc2026-qwen25-3b-d059-model-v1"
MODEL_ROOT = "/models"
MODEL_PATH = f"{MODEL_ROOT}/{MODEL_ID.replace('/', '--')}/{MODEL_REVISION}"
PROMPT_A_CHECKSUM = "86ef397543903db7eb12b61409c0b0809628278367c6debda6f53e0c36f822af"
ALLOWED_REQUEST_FIELDS = frozenset({"question_id", "question", "evidence", "system_prompt"})
BATCH_SIZE = 8
MAXIMUM_INPUT_TOKENS = 2048
MAXIMUM_NEW_TOKENS = 512
MAX_REMOTE_WALL_SECONDS = 6 * 60 * 60
BASELINE_GENERATOR_PARAMETER_COUNT = 2_031_739_904
BASELINE_MODAL_PUBLIC_SECONDS = 4815.1867675
FIXED_REFUSAL = "Không đủ căn cứ trong dữ liệu được cung cấp để trả lời câu hỏi này."

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "accelerate==1.14.0",
    "huggingface-hub==1.7.1",
    "safetensors==0.8.0",
    "torch==2.8.0",
    "transformers==5.15.1",
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _inventory_checksum(root: Path, names: tuple[str, ...]) -> str:
    entries: list[tuple[bytes, bytes]] = []
    for name in names:
        relative = name.encode("utf-8")
        digest = hashlib.sha256((root / name).read_bytes()).digest()
        entries.append((relative, struct.pack(">Q", len(relative)) + relative + digest))
    entries.sort(key=lambda item: item[0])
    return "sha256:" + hashlib.sha256(b"".join(entry for _, entry in entries)).hexdigest()


@app.function(
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    serialized=False,
    include_source=True,
    volumes={MODEL_ROOT: model_volume},
)
def download_candidate_model() -> dict[str, object]:
    from pathlib import Path as ContainerPath

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = ContainerPath(MODEL_PATH)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=root)
    weight_names = tuple(path.name for path in sorted(root.glob("*.safetensors")))
    if not weight_names:
        raise RuntimeError("pinned snapshot has no Safetensors weights")
    tensors: list[dict[str, object]] = []
    for weight_name in weight_names:
        with safe_open(root / weight_name, framework="pt", device="cpu") as stream:
            for name in stream.keys():  # noqa: SIM118 - safe_open is not iterable
                tensors.append(
                    {
                        "name": name,
                        "shape": tuple(int(value) for value in stream.get_slice(name).get_shape()),
                    }
                )
    exact_count = sum(_numel(tuple(int(value) for value in tensor["shape"])) for tensor in tensors)
    if exact_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("downloaded snapshot parameter count differs from D-059")
    model_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        *weight_names,
    )
    tokenizer_names = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
    if any(not (root / name).is_file() for name in (*model_names, *tokenizer_names, "LICENSE")):
        raise RuntimeError("pinned snapshot inventory is incomplete")
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "license_checksum": "sha256:" + hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest(),
        "exact_parameter_count": exact_count,
        "model_hash": _inventory_checksum(root, model_names),
        "tokenizer_hash": _inventory_checksum(root, tokenizer_names),
        "tensors": tensors,
    }
    model_volume.commit()
    return result


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


@app.cls(
    gpu="A10",
    max_containers=1,
    scaledown_window=10 * 60,
    timeout=20 * 60,
    serialized=False,
    include_source=True,
    block_network=True,
    restrict_modal_access=True,
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
)
class CandidateGenerator:
    @modal.enter()
    def load(self) -> None:
        from pathlib import Path as ContainerPath

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not ContainerPath(MODEL_PATH).is_dir():
            raise RuntimeError("pinned D-059 model snapshot is absent")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
        ).to("cuda")
        self.model.eval()
        exact_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        if exact_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("loaded model parameter count differs from D-059")
        print(
            f"loaded model_id={MODEL_ID} revision={MODEL_REVISION} "
            f"numel={exact_count} gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def _validate_request(self, value: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
        if set(value) != ALLOWED_REQUEST_FIELDS:
            raise ValueError("request fields exceed the D-059 allowlist")
        question_id = value["question_id"]
        question = value["question"]
        evidence_value = value["evidence"]
        system_prompt = value["system_prompt"]
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(system_prompt, str)
            or not isinstance(evidence_value, (list, tuple))
            or not 1 <= len(evidence_value) <= 3
            or any(not isinstance(item, str) or not item.strip() for item in evidence_value)
            or unicodedata.normalize("NFC", question_id) != question_id
            or unicodedata.normalize("NFC", question) != question
            or hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() != PROMPT_A_CHECKSUM
        ):
            raise ValueError("request value exceeds the D-059 allowlist")
        return question_id, question, tuple(evidence_value), system_prompt

    @modal.method()
    def generate_batch(self, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(values) <= BATCH_SIZE:
            raise ValueError("remote batch size exceeds the D-059 bound")
        outputs: list[dict[str, Any]] = []
        for value in values:
            question_id, question, evidence, system_prompt = self._validate_request(value)
            evidence_text = "\n\n".join(
                f"[EVIDENCE {index}]\n{text}" for index, text in enumerate(evidence, start=1)
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Câu hỏi:\n{question}\n\nCăn cứ được cung cấp:\n{evidence_text}",
                },
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=MAXIMUM_INPUT_TOKENS,
            ).to("cuda")
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=MAXIMUM_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.torch.cuda.synchronize()
            continuation = generated[0, inputs["input_ids"].shape[1] :]
            answer = str(self.tokenizer.decode(continuation, skip_special_tokens=True)).strip()
            answer = unicodedata.normalize("NFC", answer or FIXED_REFUSAL)
            outputs.append(
                {
                    "question_id": question_id,
                    "answer": answer,
                    "elapsed_seconds": time.perf_counter() - started,
                    "input_tokens": int(inputs["attention_mask"].sum().item()),
                    "output_tokens": int(continuation.shape[0]),
                    "peak_cuda_bytes": int(self.torch.cuda.max_memory_allocated()),
                }
            )
        print(f"completed d059_rows={len(outputs)}", flush=True)
        return outputs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path.name}")
    return value


def _write_governance(identity: dict[str, object]) -> dict[str, str]:
    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.domain.checksums import content_json_bytes
    from legal_rag.models.approval import validate_experiment_profile
    from legal_rag.models.manifest import ModelParameterManifest
    from legal_rag.models.parameter_audit import ParameterTensor, audit_parameters

    tensors_value = identity.get("tensors")
    if not isinstance(tensors_value, list):
        raise RuntimeError("remote parameter inventory is invalid")
    report = audit_parameters(
        tuple(
            ParameterTensor(
                name=str(tensor["name"]),
                shape=tuple(int(value) for value in tensor["shape"]),
                category="base",
                trainable=False,
            )
            for tensor in tensors_value
        )
    )
    if report.exact_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("local exact audit differs from D-059")
    manifest = ModelParameterManifest.model_validate(
        {
            "schema_version": "model.parameter_manifest.v1",
            "models": (
                {
                    "role": "generator",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "tokenizer_id": MODEL_ID,
                    "tokenizer_revision": MODEL_REVISION,
                    "license": MODEL_LICENSE,
                    "exact_parameter_count": report.exact_parameter_count,
                    "trainable_parameter_count": 0,
                    "adapter_parameter_count": 0,
                    "quantization": None,
                    "parameter_audit_checksum": report.parameter_audit_checksum,
                    "btc_approval_state": "pending",
                    "btc_approval_evidence": None,
                    "local_model_hash": identity["model_hash"],
                    "local_tokenizer_hash": identity["tokenizer_hash"],
                },
            ),
            "system_parameter_count": report.exact_parameter_count,
            "competition_limit_exclusive": 4_000_000_000,
            "passes_parameter_gate": True,
        }
    )
    validate_experiment_profile(manifest)
    root = _project_root()
    output = root / "artifacts/models/d059-qwen25-3b"
    audit_data = content_json_bytes(
        {
            "schema_version": "model.parameter_audits.v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "license_checksum": identity["license_checksum"],
            "tensor_count": len(report.tensors),
            "exact_parameter_count": report.exact_parameter_count,
            "adapter_parameter_count": report.adapter_parameter_count,
            "trainable_parameter_count": report.trainable_parameter_count,
            "parameter_audit_checksum": report.parameter_audit_checksum,
        }
    )
    manifest_data = content_json_bytes(manifest.model_dump(mode="json"))
    return {
        "parameter_audit": write_immutable_bytes(output / "parameter-audit.v1.json", audit_data),
        "parameter_manifest": write_immutable_bytes(
            output / "parameter-manifest.v1.json", manifest_data
        ),
    }


def _require_manifest() -> tuple[bytes, dict[str, object]]:
    from legal_rag.models.approval import validate_experiment_profile
    from legal_rag.models.manifest import ModelParameterManifest

    path = _project_root() / "artifacts/models/d059-qwen25-3b/parameter-manifest.v1.json"
    data = path.read_bytes()
    manifest = ModelParameterManifest.model_validate_json(data)
    validate_experiment_profile(manifest)
    if (
        manifest.system_parameter_count != EXPECTED_PARAMETER_COUNT
        or len(manifest.models) != 1
        or manifest.models[0].model_id != MODEL_ID
        or manifest.models[0].model_revision != MODEL_REVISION
    ):
        raise RuntimeError("D-059 parameter manifest identity mismatch")
    return data, manifest.model_dump(mode="json")


def _local_model_directory() -> Path:
    return (
        _project_root()
        / ".local/models/qwen25-3b/verified-aa8e72537993ba99e69dfaafa59ed015b17504d1"
        / MODEL_ID.replace("/", "--")
        / MODEL_REVISION
    )


def _verify_local_checkpoint(checkpoint: Path, parameter_manifest: dict[str, object]) -> None:
    from legal_rag.models.torch_audit import audit_safetensors_directory

    models = parameter_manifest.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise RuntimeError("D-059 parameter manifest model inventory is invalid")
    identity = models[0]
    weight_names = tuple(path.name for path in sorted(checkpoint.glob("*.safetensors")))
    model_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        *weight_names,
    )
    tokenizer_names = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
    if any(not (checkpoint / name).is_file() for name in (*model_names, *tokenizer_names)):
        raise RuntimeError("local D-059 checkpoint inventory is incomplete")
    audit = audit_safetensors_directory(checkpoint)
    if (
        audit.exact_parameter_count != EXPECTED_PARAMETER_COUNT
        or _inventory_checksum(checkpoint, model_names) != identity.get("local_model_hash")
        or _inventory_checksum(checkpoint, tokenizer_names) != identity.get("local_tokenizer_hash")
    ):
        raise RuntimeError("local D-059 checkpoint differs from the audited Modal snapshot")


def _projected_a10_public_seconds() -> float:
    return BASELINE_MODAL_PUBLIC_SECONDS * (
        EXPECTED_PARAMETER_COUNT / BASELINE_GENERATOR_PARAMETER_COUNT
    )


def _invoke_public_remote(
    *,
    requests: tuple[Any, ...],
    source_checksum: str,
) -> tuple[tuple[Any, ...], dict[str, bytes], dict[str, float | int | str]]:
    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
    from legal_rag.domain.execution import ArtifactTransfer, PrivateModalConfig, preflight_execution
    from legal_rag.providers.modal_public_generation import (
        ModalPublicGenerationResponse,
        validate_modal_public_responses,
    )

    request_data = b"".join(
        content_json_bytes(request.model_dump(mode="json")) for request in requests
    )
    phase = "public"
    request_transfer = ArtifactTransfer(
        artifact_id="d059.public.generation.request",
        artifact_class="d059_public_generation_request",
        direction="local-to-modal",
        checksum=checksum_bytes(request_data),
    )
    config = PrivateModalConfig(
        schema_version="execution.mode.v1",
        mode="private-modal",
        control_plane_origin="https://api.modal.com",
        control_plane_origin_allowlist=("https://api.modal.com",),
        workload_egress_disabled=True,
        workload_egress_verified=True,
        private_storage_ids=("d059-qwen25-3b-model",),
        required_resource_ids=("model.d059.generator",),
        transfer_allowlist=(request_transfer,),
        real_data_approved=True,
        approval_id=APPROVAL_ID,
        modal_function_io_retention_days_maximum=7,
        gpu="A10",
        maximum_gpu_containers=1,
        maximum_account_cost_usd=30,
        private_storage_access="read-only",
        max_submission_retries=3,
        submission_backoff_seconds=(1, 2, 4),
        declared_job_identity=f"{CAMPAIGN_ID}-{phase}",
        secret_policy="credential-store-only-redacted",
    )
    preflight_execution(
        config,
        available_resource_ids=("model.d059.generator",),
        requested_transfers=(request_transfer,),
    )
    config_data = content_json_bytes(config.model_dump(mode="json"))
    scope_data = content_json_bytes(
        {
            "schema_version": "modal.d059.approval-scope.v1",
            "approval_id": APPROVAL_ID,
            "phase": phase,
            "allowed_request_fields": sorted(ALLOWED_REQUEST_FIELDS),
            "allowed_evidence_count": 3,
            "gpu": "A10",
            "maximum_cost_usd": 30,
            "modal_io_retention_days_maximum": 7,
            "request_set_checksum": checksum_bytes(request_data),
        }
    )
    fingerprint = checksum_bytes(
        content_json_bytes(
            {
                "schema_version": "modal.d059.response-fingerprint.v1",
                "campaign_id": CAMPAIGN_ID,
                "phase": phase,
                "source_checksum": source_checksum,
                "request_checksum": checksum_bytes(request_data),
                "approval_scope_checksum": checksum_bytes(scope_data),
                "execution_preflight_checksum": checksum_bytes(config_data),
                "modal_app_source_checksum": checksum_bytes(Path(__file__).read_bytes()),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_checksum": "sha256:" + PROMPT_A_CHECKSUM,
                "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
                "maximum_new_tokens": MAXIMUM_NEW_TOKENS,
                "batch_size": BATCH_SIZE,
                "gpu": "A10",
                "do_sample": False,
                "enable_thinking": False,
            }
        )
    )
    checkpoint_root = (
        _project_root()
        / ".local/runs/d059-qwen25-3b-r0-prompt-a-v1"
        / f"{phase}-answer-checkpoints"
        / fingerprint.removeprefix("sha256:")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    responses_by_id: dict[str, ModalPublicGenerationResponse] = {}
    resumed_count = 0
    generated_count = 0
    started = time.perf_counter()
    generator = CandidateGenerator()
    for offset in range(0, len(requests), BATCH_SIZE):
        if time.perf_counter() - started >= MAX_REMOTE_WALL_SECONDS:
            raise RuntimeError("D-059 Modal generation exceeded the six-hour ceiling")
        batch = requests[offset : offset + BATCH_SIZE]
        missing = []
        for request in batch:
            checkpoint = checkpoint_root / (
                hashlib.sha256(request.question_id.encode("utf-8")).hexdigest() + ".json"
            )
            if checkpoint.exists():
                response = ModalPublicGenerationResponse.model_validate_json(
                    checkpoint.read_bytes()
                )
                if response.question_id != request.question_id:
                    raise RuntimeError("D-059 checkpoint identity mismatch")
                responses_by_id[response.question_id] = response
                resumed_count += 1
            else:
                missing.append(request)
        if not missing:
            continue
        raw = generator.generate_batch.remote(
            [request.model_dump(mode="json") for request in missing]
        )
        validated = validate_modal_public_responses(
            tuple(raw),
            expected_question_ids=tuple(request.question_id for request in missing),
        )
        for response in validated:
            checkpoint = checkpoint_root / (
                hashlib.sha256(response.question_id.encode("utf-8")).hexdigest() + ".json"
            )
            write_immutable_bytes(checkpoint, content_json_bytes(response.model_dump(mode="json")))
            responses_by_id[response.question_id] = response
            generated_count += 1
        print(f"d059_{phase}_progress={len(responses_by_id)}/{len(requests)}", flush=True)
    wall_seconds = time.perf_counter() - started
    expected_ids = tuple(request.question_id for request in requests)
    responses = validate_modal_public_responses(
        tuple(
            responses_by_id[question_id].model_dump(mode="python") for question_id in expected_ids
        ),
        expected_question_ids=expected_ids,
    )
    response_data = b"".join(
        content_json_bytes(response.model_dump(mode="json")) for response in responses
    )
    response_transfer = ArtifactTransfer(
        artifact_id=f"d059.{phase}.generation.response",
        artifact_class=f"d059_{phase}_generation_response",
        direction="modal-to-local",
        checksum=checksum_bytes(response_data),
    )
    response_config = PrivateModalConfig.model_validate(
        {**config.model_dump(mode="python"), "transfer_allowlist": (response_transfer,)}
    )
    preflight_execution(
        response_config,
        available_resource_ids=("model.d059.generator",),
        requested_transfers=(response_transfer,),
    )
    artifacts = {
        "requests": request_data,
        "responses": response_data,
        "approval_scope": scope_data,
        "request_preflight": config_data,
        "response_preflight": content_json_bytes(response_config.model_dump(mode="json")),
    }
    telemetry: dict[str, float | int | str] = {
        "response_fingerprint": fingerprint,
        "question_count": len(responses),
        "generated_question_count": generated_count,
        "resumed_question_count": resumed_count,
        "invocation_elapsed_seconds": wall_seconds,
        "checkpoint_elapsed_seconds": sum(response.elapsed_seconds for response in responses),
        "input_tokens": sum(response.input_tokens for response in responses),
        "output_tokens": sum(response.output_tokens for response in responses),
        "peak_cuda_bytes": max(response.peak_cuda_bytes for response in responses),
    }
    return responses, artifacts, telemetry


def _replay_backend(requests: tuple[Any, ...], responses: tuple[Any, ...]) -> Any:
    from legal_rag.generation.qwen3 import PROMPT_A

    class ReplayBackend:
        model_id = MODEL_ID
        model_revision = MODEL_REVISION

        def __init__(self) -> None:
            self.answers = {
                (request.question, request.evidence): response.answer
                for request, response in zip(requests, responses, strict=True)
            }

        def generate(
            self,
            *,
            system_prompt: str,
            question: str,
            evidence: tuple[str, ...],
        ) -> str:
            if system_prompt != PROMPT_A:
                raise RuntimeError("D-059 replay prompt mismatch")
            return self.answers[(question, evidence)]

    return ReplayBackend()


def _run_development() -> dict[str, object]:
    import torch

    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
    from legal_rag.evaluation.competition import evaluate_competition_bytes
    from legal_rag.evaluation.generator_comparison import compare_model_generation_experiments
    from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
    from legal_rag.generation.qwen3 import PROMPT_A
    from legal_rag.grounding.answer_assessment import build_answer_assessment_queue
    from legal_rag.models.huggingface_local import Qwen25GeneratorBackend

    root = _project_root()
    run_id = "G4R0A512-qwen25-3b-prompt-a-local-d059-v1"
    baseline_id = "G1R0A512-qwen3-1.7b-prompt-a-btc-approved-v1"
    output = root / "artifacts/evaluations/mil-006" / run_id
    baseline = root / "artifacts/evaluations/mil-006" / baseline_id
    queue_data = (
        root / "artifacts/evaluations/grounding/annotation-work-queue.v1.jsonl"
    ).read_bytes()
    retrieval_data = (root / "artifacts/evaluations/mil-004/retrieval.v1.jsonl").read_bytes()
    parameter_data, parameter_manifest = _require_manifest()
    checkpoint = _local_model_directory()
    _verify_local_checkpoint(checkpoint, parameter_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("D-059 local development requires the available CUDA GPU")
    backend = Qwen25GeneratorBackend(
        checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "6GiB"},
        offload_folder=root / ".local/offload/d059-qwen25-3b-development",
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
    )
    checkpoint_fingerprint = checksum_bytes(
        content_json_bytes(
            {
                "schema_version": "d059.local-development-fingerprint.v1",
                "annotation_queue_checksum": checksum_bytes(queue_data),
                "retrieval_output_checksum": checksum_bytes(retrieval_data),
                "parameter_manifest_checksum": checksum_bytes(parameter_data),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_checksum": checksum_bytes(PROMPT_A.encode("utf-8")),
                "evidence_limit": 3,
                "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
                "maximum_new_tokens": MAXIMUM_NEW_TOKENS,
                "do_sample": False,
                "enable_thinking": False,
            }
        )
    )
    checkpoint_root = (
        root
        / ".local/runs/d059-qwen25-3b-r0-prompt-a-v1/development-local-answer-checkpoints"
        / checkpoint_fingerprint.removeprefix("sha256:")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    question_count = len(queue_data.splitlines())

    class CheckpointedBackend:
        model_id = MODEL_ID
        model_revision = MODEL_REVISION

        def __init__(self) -> None:
            self.completed_count = 0
            self.generated_count = 0
            self.resumed_count = 0
            self.total_inference_seconds = 0.0

        def generate(
            self,
            *,
            system_prompt: str,
            question: str,
            evidence: tuple[str, ...],
        ) -> str:
            input_checksum = checksum_bytes(
                content_json_bytes(
                    {
                        "system_prompt": system_prompt,
                        "question": question,
                        "evidence": evidence,
                    }
                )
            )
            checkpoint = checkpoint_root / (input_checksum.removeprefix("sha256:") + ".json")
            if checkpoint.exists():
                value = _load_json(checkpoint)
                if (
                    set(value)
                    != {
                        "schema_version",
                        "run_fingerprint",
                        "input_checksum",
                        "model_id",
                        "model_revision",
                        "answer",
                        "elapsed_seconds",
                    }
                    or value.get("schema_version") != "d059.local-answer-checkpoint.v1"
                    or value.get("run_fingerprint") != checkpoint_fingerprint
                    or value.get("input_checksum") != input_checksum
                    or value.get("model_id") != MODEL_ID
                    or value.get("model_revision") != MODEL_REVISION
                    or not isinstance(value.get("answer"), str)
                    or not str(value["answer"]).strip()
                    or not isinstance(value.get("elapsed_seconds"), int | float)
                    or not math.isfinite(float(value["elapsed_seconds"]))
                    or float(value["elapsed_seconds"]) < 0
                ):
                    raise RuntimeError("D-059 local development checkpoint mismatch")
                answer = str(value["answer"])
                elapsed_seconds = float(value["elapsed_seconds"])
                self.resumed_count += 1
            else:
                started = time.perf_counter()
                answer = backend.generate(
                    system_prompt=system_prompt,
                    question=question,
                    evidence=evidence,
                ).strip()
                elapsed_seconds = time.perf_counter() - started
                if not answer:
                    answer = FIXED_REFUSAL
                value = {
                    "schema_version": "d059.local-answer-checkpoint.v1",
                    "run_fingerprint": checkpoint_fingerprint,
                    "input_checksum": input_checksum,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "answer": unicodedata.normalize("NFC", answer),
                    "elapsed_seconds": elapsed_seconds,
                }
                write_immutable_bytes(checkpoint, content_json_bytes(value))
                answer = str(value["answer"])
                self.generated_count += 1
            self.completed_count += 1
            self.total_inference_seconds += elapsed_seconds
            print(
                f"d059_local_development_progress={self.completed_count}/{question_count}",
                flush=True,
            )
            return answer

    checkpointed_backend = CheckpointedBackend()
    torch.cuda.reset_peak_memory_stats()
    predictions, references, manifest_data = run_grounded_generation_experiment(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        backend=checkpointed_backend,
        system_prompt=PROMPT_A,
        run_id=run_id,
        evidence_limit=3,
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id=baseline_id,
        changed_axes=("model",),
        profile_state="exploratory_non_promotable",
    )
    manifest = json.loads(manifest_data)
    invocation_runtime = float(manifest["elapsed_seconds"])
    candidate_runtime = checkpointed_backend.total_inference_seconds
    manifest["elapsed_seconds"] = candidate_runtime
    manifest["parameter_manifest_checksum"] = checksum_bytes(parameter_data)
    manifest["local_checkpoint_verified"] = True
    manifest_data = content_json_bytes(manifest)
    evaluation = evaluate_competition_bytes(
        predictions,
        references,
        scorer_root=root / "Scoring-Program-Task-LegalQA",
        nltk_data_root=root / "resources/nltk_data",
        baseline_kind="d059_generator_model_ablation",
        limitation="development_only_model_single_axis",
    )
    baseline_telemetry = _load_json(baseline / "telemetry.json")
    comparison_data = compare_model_generation_experiments(
        baseline_per_query_data=(baseline / "evaluation-per-query.jsonl").read_bytes(),
        candidate_per_query_data=evaluation.per_query_bytes,
        baseline_manifest_data=(baseline / "manifest.json").read_bytes(),
        candidate_manifest_data=manifest_data,
        baseline_runtime_seconds=float(baseline_telemetry["wall_seconds"]),
        candidate_runtime_seconds=candidate_runtime,
    )
    projected_public_seconds = _projected_a10_public_seconds()
    hard_resource_gate = (
        "passed" if projected_public_seconds <= MAX_REMOTE_WALL_SECONDS else "failed"
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "model.generation.telemetry.v1",
            "run_id": run_id,
            "execution_mode": "local-offline-fp16-cpu-offload",
            "paid_service_used": False,
            "question_count": evaluation.question_count,
            "macro_meteor": evaluation.macro_meteor,
            "macro_rouge_l": evaluation.macro_rouge_l,
            "wall_seconds": candidate_runtime,
            "current_invocation_seconds": invocation_runtime,
            "generated_question_count": checkpointed_backend.generated_count,
            "resumed_question_count": checkpointed_backend.resumed_count,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "projected_1000_wall_seconds": projected_public_seconds,
            "projection_method": "measured_qwen3_a10_seconds_scaled_by_exact_parameter_ratio",
            "projection_baseline_seconds": BASELINE_MODAL_PUBLIC_SECONDS,
            "projection_baseline_parameter_count": BASELINE_GENERATOR_PARAMETER_COUNT,
            "projection_candidate_parameter_count": EXPECTED_PARAMETER_COUNT,
            "hard_six_hour_resource_gate": hard_resource_gate,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "max_memory": {"cuda:0": "5GiB", "cpu": "6GiB"},
        }
    )
    grounding_queue = build_answer_assessment_queue(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        predictions_data=predictions,
        evaluated_run_id=run_id,
        evidence_limit=3,
    )
    outputs = {
        "predictions.json": predictions,
        "references.json": references,
        "manifest.json": manifest_data,
        "evaluation-per-query.jsonl": evaluation.per_query_bytes,
        "evaluation-report.json": evaluation.report_bytes,
        "comparison-vs-g1r0a512.json": comparison_data,
        "telemetry.json": telemetry_data,
        "answer-grounding-work-queue.v1.jsonl": grounding_queue,
    }
    checksums = {name: write_immutable_bytes(output / name, data) for name, data in outputs.items()}
    comparison = json.loads(comparison_data)
    public_gate = (
        comparison["numeric_evaluation_gate"] == "passed" and hard_resource_gate == "passed"
    )
    state_data = content_json_bytes(
        {
            "schema_version": "d059.development.state.v1",
            "run_id": run_id,
            "public_generation_gate": "passed" if public_gate else "failed",
            "comparison_checksum": checksums["comparison-vs-g1r0a512.json"],
            "parameter_manifest_checksum": checksum_bytes(parameter_data),
            "hard_resource_gate": hard_resource_gate,
        }
    )
    checksums["development-state.v1.json"] = write_immutable_bytes(
        output / "development-state.v1.json", state_data
    )
    return {
        "run_id": run_id,
        "macro_meteor": evaluation.macro_meteor,
        "macro_rouge_l": evaluation.macro_rouge_l,
        "public_generation_gate": "passed" if public_gate else "failed",
        "checksums": checksums,
    }


def _run_public() -> dict[str, object]:
    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
    from legal_rag.evaluation.public_dry_run import run_checkpointed_public_generation
    from legal_rag.generation.qwen3 import PROMPT_A
    from legal_rag.ingestion.organizer import OrganizerQuestionReader
    from legal_rag.providers.modal_public_generation import build_modal_public_requests
    from legal_rag.submission.writer import build_submission_zip, validate_submission

    root = _project_root()
    development_id = "G4R0A512-qwen25-3b-prompt-a-local-d059-v1"
    development = root / "artifacts/evaluations/mil-006" / development_id
    development_state = _load_json(development / "development-state.v1.json")
    if development_state.get("public_generation_gate") != "passed":
        raise RuntimeError("D-059 development hard gates did not open public generation")
    parameter_data, parameter_manifest = _require_manifest()
    public_data = (root / "data/public-official.json").read_bytes()
    evidence_data = (
        root
        / "artifacts/evaluations/public/G1R0A512-public-1000-diagnostic-v1/public.evidence.v1.jsonl"
    ).read_bytes()
    expected_ids = tuple(
        record.question_id
        for record in OrganizerQuestionReader()
        .read_bytes(public_data, kind="public", artifact_path="data/public-official.json")
        .records
    )
    requests = build_modal_public_requests(
        evidence_data,
        public_source_data=public_data,
        expected_question_ids=expected_ids,
        system_prompt=PROMPT_A,
    )
    responses, remote_artifacts, remote_telemetry = _invoke_public_remote(
        requests=requests,
        source_checksum=checksum_bytes(public_data),
    )
    backend = _replay_backend(requests, responses)
    run_id = "G4R0A512-public-1000-qwen25-3b-modal-a10-d059-v1"
    output = root / "artifacts/evaluations/public" / run_id
    replay_checkpoint = (
        root
        / ".local/runs/d059-qwen25-3b-r0-prompt-a-v1/public-replay-checkpoints"
        / str(remote_telemetry["response_fingerprint"]).removeprefix("sha256:")
    )
    artifacts = run_checkpointed_public_generation(
        public_source_data=public_data,
        evidence_queue_data=evidence_data,
        backend=backend,
        system_prompt=PROMPT_A,
        run_id=run_id,
        generator_id="qwen25-3b-prompt-a-512-modal-a10-d059-v1",
        checkpoint_directory=replay_checkpoint,
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
        profile_state="diagnostic_dry_run",
        frozen_inputs={
            "development_state": checksum_bytes(
                (development / "development-state.v1.json").read_bytes()
            ),
            "modal_app_source": checksum_bytes(Path(__file__).read_bytes()),
            "modal_response_set": checksum_bytes(remote_artifacts["responses"]),
            "model_checkpoint": str(parameter_manifest["models"][0]["local_model_hash"]),
            "parameter_manifest": checksum_bytes(parameter_data),
            "prompt": checksum_bytes(PROMPT_A.encode("utf-8")),
        },
    )
    answer_rows = tuple(json.loads(line) for line in artifacts.answers_data.splitlines())
    if tuple(row["answer"] for row in answer_rows) != tuple(
        response.answer for response in responses
    ):
        raise RuntimeError("D-059 organizer answers differ from validated Modal responses")
    submission_validation = validate_submission(public_data, artifacts.predictions_data)
    submission_zip = build_submission_zip(artifacts.predictions_data)
    telemetry_data = content_json_bytes(
        {
            "schema_version": "public.dry-run.telemetry.v1",
            "run_id": run_id,
            "run_fingerprint": json.loads(artifacts.manifest_data)["run_fingerprint"],
            "question_count": len(responses),
            "paid_service_used": True,
            "execution_mode": "modal-a10-owner-approved-d059",
            "actual_cost_usd": None,
            "account_cost_stop_usd": 30,
            **remote_telemetry,
        }
    )
    outputs = {
        "answers.v1.jsonl": artifacts.answers_data,
        "submission.json": artifacts.predictions_data,
        "public.dry-run.manifest.v1.json": artifacts.manifest_data,
        "public.dry-run.telemetry.v1.json": telemetry_data,
        "modal.requests.v1.jsonl": remote_artifacts["requests"],
        "modal.responses.v1.jsonl": remote_artifacts["responses"],
        "modal.approval-scope.v1.json": remote_artifacts["approval_scope"],
        "modal.request-preflight.v1.json": remote_artifacts["request_preflight"],
        "modal.response-preflight.v1.json": remote_artifacts["response_preflight"],
        "public.dry-run.local-replay.telemetry.v1.json": artifacts.telemetry_data,
    }
    checksums = {name: write_immutable_bytes(output / name, data) for name, data in outputs.items()}
    submission_root = root / "artifacts/submissions" / run_id
    checksums["submission-package-json"] = write_immutable_bytes(
        submission_root / "submission.json", artifacts.predictions_data
    )
    checksums["submission-package-zip"] = write_immutable_bytes(
        submission_root / "submission.zip", submission_zip
    )
    return {
        "run_id": run_id,
        "submission_count": submission_validation.count,
        "submission_checksum": submission_validation.checksum,
        "checksums": checksums,
    }


@app.local_entrypoint()
def main(stage: str = "download") -> None:
    if stage == "download":
        identity = download_candidate_model.remote()
        result: dict[str, object] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "exact_parameter_count": identity["exact_parameter_count"],
            "checksums": _write_governance(identity),
        }
    elif stage == "development":
        result = _run_development()
    elif stage == "public":
        result = _run_public()
    else:
        raise ValueError("stage must be download, development, or public")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
