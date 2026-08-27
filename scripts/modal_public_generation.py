from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dsc2026-legalqa-public-generation-v1"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_VOLUME_NAME = "dsc2026-qwen3-1-7b-public-model-v1"
MODEL_ROOT = "/models"
MODEL_PATH = f"{MODEL_ROOT}/{MODEL_ID.replace('/', '--')}/{MODEL_REVISION}"
PROMPT_A_CHECKSUM = "86ef397543903db7eb12b61409c0b0809628278367c6debda6f53e0c36f822af"
FIXED_REFUSAL = "Không đủ căn cứ trong dữ liệu được cung cấp để trả lời câu hỏi này."
ALLOWED_REQUEST_FIELDS = frozenset({"question_id", "question", "evidence", "system_prompt"})
BATCH_SIZE = 8
MAX_REMOTE_WALL_SECONDS = 6 * 60 * 60

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "accelerate==1.14.0",
    "huggingface-hub==1.7.1",
    "safetensors==0.8.0",
    "torch==2.8.0",
    "transformers==5.15.1",
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    serialized=True,
    include_source=False,
    volumes={MODEL_ROOT: model_volume},
)
def download_public_model() -> dict[str, str]:
    from pathlib import Path as ContainerPath

    from huggingface_hub import snapshot_download

    ContainerPath(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_PATH,
    )
    model_volume.commit()
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}


@app.cls(
    gpu="A10",
    max_containers=1,
    scaledown_window=10 * 60,
    timeout=20 * 60,
    serialized=True,
    include_source=False,
    block_network=True,
    restrict_modal_access=True,
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
)
class LegalGenerator:
    @modal.enter()
    def load(self) -> None:
        from pathlib import Path as ContainerPath

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not ContainerPath(MODEL_PATH).is_dir():
            raise RuntimeError("pinned public model snapshot is absent")
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
        print(
            f"loaded model_id={MODEL_ID} revision={MODEL_REVISION} "
            f"gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def _validate_request(self, value: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
        if set(value) != ALLOWED_REQUEST_FIELDS:
            raise ValueError("request fields exceed the OQ-003 allowlist")
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
            or len(evidence_value) > 3
            or any(not isinstance(item, str) or not item.strip() for item in evidence_value)
            or unicodedata.normalize("NFC", question_id) != question_id
            or unicodedata.normalize("NFC", question) != question
            or hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() != PROMPT_A_CHECKSUM
        ):
            raise ValueError("request value exceeds the OQ-003 allowlist")
        return question_id, question, tuple(evidence_value), system_prompt

    @modal.method()
    def generate_batch(self, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(values) <= BATCH_SIZE:
            raise ValueError("remote batch size exceeds the approved bound")
        outputs: list[dict[str, Any]] = []
        for value in values:
            question_id, question, evidence, system_prompt = self._validate_request(value)
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            if evidence:
                evidence_text = "\n\n".join(
                    f"[EVIDENCE {index}]\n{text}" for index, text in enumerate(evidence, start=1)
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Câu hỏi:\n{question}\n\nCăn cứ được cung cấp:\n{evidence_text}"
                        ),
                    },
                ]
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to("cuda")
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=512,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                continuation = generated[0, inputs["input_ids"].shape[1] :]
                answer = str(self.tokenizer.decode(continuation, skip_special_tokens=True)).strip()
                input_tokens = int(inputs["input_ids"].shape[1])
                output_tokens = int(continuation.shape[0])
            else:
                answer = FIXED_REFUSAL
                input_tokens = 0
                output_tokens = 0
            self.torch.cuda.synchronize()
            answer = unicodedata.normalize("NFC", answer)
            if not answer:
                raise RuntimeError("generator returned an empty answer")
            outputs.append(
                {
                    "question_id": question_id,
                    "answer": answer,
                    "elapsed_seconds": time.perf_counter() - started,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "peak_cuda_bytes": int(self.torch.cuda.max_memory_allocated()),
                }
            )
        print(f"completed approved_public_rows={len(outputs)}", flush=True)
        return outputs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_remote_response(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(stage: str = "download", campaign: str = "legacy-r0") -> None:
    if stage == "download":
        identity = download_public_model.remote()
        print(
            f"downloaded model_id={identity['model_id']} revision={identity['model_revision']}",
            flush=True,
        )
        return
    if stage != "generation":
        raise ValueError("stage must be download or generation")

    from legal_rag.domain.artifacts import write_immutable_bytes
    from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
    from legal_rag.domain.execution import (
        ArtifactTransfer,
        PrivateModalConfig,
        preflight_execution,
    )
    from legal_rag.evaluation.public_dry_run import run_checkpointed_public_generation
    from legal_rag.generation.qwen3 import PROMPT_A
    from legal_rag.ingestion.organizer import OrganizerQuestionReader
    from legal_rag.providers.modal_public_generation import (
        ModalPublicGenerationResponse,
        build_modal_public_requests,
        validate_modal_public_responses,
    )
    from legal_rag.providers.public_campaign import public_campaign
    from legal_rag.submission.writer import build_submission_zip, validate_submission

    root = _project_root()
    campaign_config = public_campaign(campaign)
    public_path = root / "data" / "public-official.json"
    evidence_path = root.joinpath(*campaign_config.evidence_relative_path.parts)
    output_root = root.joinpath(*campaign_config.output_relative_path.parts)
    public_data = public_path.read_bytes()
    evidence_data = evidence_path.read_bytes()
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
    request_data = b"".join(
        content_json_bytes(request.model_dump(mode="json")) for request in requests
    )
    request_transfer = ArtifactTransfer(
        artifact_id="public.generation.request",
        artifact_class="public_generation_request",
        direction="local-to-modal",
        checksum=checksum_bytes(request_data),
    )
    modal_preflight = PrivateModalConfig(
        schema_version="execution.mode.v1",
        mode="private-modal",
        control_plane_origin="https://api.modal.com",
        control_plane_origin_allowlist=("https://api.modal.com",),
        workload_egress_disabled=True,
        workload_egress_verified=True,
        private_storage_ids=("qwen3-public-model",),
        required_resource_ids=("model.public",),
        transfer_allowlist=(request_transfer,),
        real_data_approved=True,
        approval_id=campaign_config.approval_id,
        modal_function_io_retention_days_maximum=7,
        gpu="A10",
        maximum_gpu_containers=1,
        maximum_account_cost_usd=30,
        private_storage_access="read-only",
        max_submission_retries=3,
        submission_backoff_seconds=(1, 2, 4),
        declared_job_identity=campaign_config.job_identity,
        secret_policy="credential-store-only-redacted",
    )
    preflight_execution(
        modal_preflight,
        available_resource_ids=("model.public",),
        requested_transfers=(request_transfer,),
    )
    modal_preflight_data = content_json_bytes(modal_preflight.model_dump(mode="json"))
    approval_scope = content_json_bytes(
        {
            "approval_id": campaign_config.approval_id,
            "campaign_id": campaign_config.campaign_id,
            "allowed_request_fields": sorted(ALLOWED_REQUEST_FIELDS),
            "allowed_evidence_count": 3,
            "gpu": "A10",
            "maximum_cost_usd": 30,
            "modal_io_retention_days_maximum": 7,
            "request_set_checksum": checksum_bytes(request_data),
        }
    )
    response_fingerprint = checksum_bytes(
        content_json_bytes(
            {
                "schema_version": "modal.public-response-fingerprint.v1",
                "approval_scope_checksum": checksum_bytes(approval_scope),
                "execution_preflight_checksum": checksum_bytes(modal_preflight_data),
                "evidence_queue_checksum": checksum_bytes(evidence_data),
                "modal_app_source_checksum": checksum_bytes(Path(__file__).read_bytes()),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt_checksum": checksum_bytes(PROMPT_A.encode("utf-8")),
                "maximum_input_tokens": 2048,
                "maximum_new_tokens": 512,
                "batch_size": BATCH_SIZE,
                "gpu": "A10",
                "maximum_gpu_containers": 1,
                "maximum_remote_wall_seconds": MAX_REMOTE_WALL_SECONDS,
                "workload_network_egress": "blocked",
                "modal_sdk_version": "1.5.2",
                "torch_version": "2.8.0",
                "transformers_version": "5.15.1",
                "do_sample": False,
                "enable_thinking": False,
            }
        )
    )
    checkpoint_root = root.joinpath(
        *campaign_config.response_checkpoint_relative_path.parts,
        response_fingerprint.removeprefix("sha256:"),
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    responses_by_id: dict[str, ModalPublicGenerationResponse] = {}
    resumed_count = 0
    generated_count = 0
    remote_started = time.perf_counter()
    generator = LegalGenerator()
    for start in range(0, len(requests), BATCH_SIZE):
        if time.perf_counter() - remote_started >= MAX_REMOTE_WALL_SECONDS:
            raise RuntimeError("Modal public generation exceeded the six-hour run ceiling")
        batch = requests[start : start + BATCH_SIZE]
        missing = []
        for request in batch:
            digest = hashlib.sha256(request.question_id.encode("utf-8")).hexdigest()
            checkpoint = checkpoint_root / f"{digest}.json"
            if checkpoint.exists():
                response = ModalPublicGenerationResponse.model_validate(
                    _load_remote_response(checkpoint)
                )
                if response.question_id != request.question_id:
                    raise RuntimeError("local Modal checkpoint ID mismatch")
                responses_by_id[response.question_id] = response
                resumed_count += 1
            else:
                missing.append(request)
        if not missing:
            continue
        raw_responses = generator.generate_batch.remote(
            [request.model_dump(mode="json") for request in missing]
        )
        validated = validate_modal_public_responses(
            tuple(raw_responses),
            expected_question_ids=tuple(request.question_id for request in missing),
        )
        for response in validated:
            digest = hashlib.sha256(response.question_id.encode("utf-8")).hexdigest()
            checkpoint = checkpoint_root / f"{digest}.json"
            write_immutable_bytes(checkpoint, content_json_bytes(response.model_dump(mode="json")))
            responses_by_id[response.question_id] = response
            generated_count += 1
        print(
            f"modal_progress completed={len(responses_by_id)}/{len(requests)}",
            flush=True,
        )
    remote_wall_seconds = time.perf_counter() - remote_started
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
        artifact_id="public.generation.response",
        artifact_class="public_generation_response",
        direction="modal-to-local",
        checksum=checksum_bytes(response_data),
    )
    response_preflight = PrivateModalConfig.model_validate(
        {
            **modal_preflight.model_dump(mode="python"),
            "transfer_allowlist": (response_transfer,),
        }
    )
    preflight_execution(
        response_preflight,
        available_resource_ids=("model.public",),
        requested_transfers=(response_transfer,),
    )
    response_preflight_data = content_json_bytes(response_preflight.model_dump(mode="json"))

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
                raise RuntimeError("replay prompt mismatch")
            return self.answers[(question, evidence)]

    parameter_path = root / "artifacts" / "models" / "qwen3-btc-approved-parameter-manifest.v1.json"
    selection_evidence_path = root.joinpath(*campaign_config.selection_evidence_relative_path.parts)
    parameter_data = parameter_path.read_bytes()
    generator_entry = next(
        item for item in json.loads(parameter_data)["models"] if item["role"] == "generator"
    )
    execution_config = content_json_bytes(
        {
            "schema_version": "modal.public-execution.v1",
            "app_name": APP_NAME,
            "gpu": "A10",
            "max_containers": 1,
            "batch_size": BATCH_SIZE,
            "maximum_remote_wall_seconds": MAX_REMOTE_WALL_SECONDS,
            "modal_sdk_version": "1.5.2",
            "torch_version": "2.8.0",
            "transformers_version": "5.15.1",
        }
    )
    artifacts = run_checkpointed_public_generation(
        public_source_data=public_data,
        evidence_queue_data=evidence_data,
        backend=ReplayBackend(),
        system_prompt=PROMPT_A,
        run_id=campaign_config.run_id,
        generator_id=campaign_config.generator_id,
        checkpoint_directory=root.joinpath(*campaign_config.replay_checkpoint_relative_path.parts),
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        profile_state="diagnostic_dry_run",
        frozen_inputs={
            "approval_scope": checksum_bytes(approval_scope),
            "selection_evidence": checksum_bytes(selection_evidence_path.read_bytes()),
            "modal_app_source": checksum_bytes(Path(__file__).read_bytes()),
            "modal_execution_config": checksum_bytes(execution_config),
            "modal_response_set": checksum_bytes(response_data),
            "model_checkpoint": str(generator_entry["local_model_hash"]),
            "parameter_manifest": checksum_bytes(parameter_data),
            "prompt": checksum_bytes(PROMPT_A.encode("utf-8")),
        },
    )
    answer_rows = tuple(json.loads(line) for line in artifacts.answers_data.splitlines())
    if tuple(row["answer"] for row in answer_rows) != tuple(
        response.answer for response in responses
    ):
        raise RuntimeError("local organizer artifacts differ from validated Modal responses")
    remote_telemetry = content_json_bytes(
        {
            "schema_version": "public.dry-run.telemetry.v1",
            "run_id": campaign_config.run_id,
            "run_fingerprint": json.loads(artifacts.manifest_data)["run_fingerprint"],
            "question_count": len(responses),
            "generated_question_count": generated_count,
            "resumed_question_count": resumed_count,
            "checkpoint_elapsed_seconds": sum(item.elapsed_seconds for item in responses),
            "invocation_elapsed_seconds": remote_wall_seconds,
            "input_tokens": sum(item.input_tokens for item in responses),
            "output_tokens": sum(item.output_tokens for item in responses),
            "peak_cuda_bytes": max(item.peak_cuda_bytes for item in responses),
            "paid_service_used": True,
            "execution_mode": "modal-a10-owner-approved",
            "approval_scope_checksum": checksum_bytes(approval_scope),
            "response_set_checksum": checksum_bytes(response_data),
            "public_results_usage": campaign_config.public_results_usage,
        }
    )
    submission_validation = validate_submission(public_data, artifacts.predictions_data)
    submission_zip = build_submission_zip(artifacts.predictions_data)
    outputs = {
        "answers.v1.jsonl": artifacts.answers_data,
        "predictions.json": artifacts.predictions_data,
        "submission.json": artifacts.predictions_data,
        "submission.zip": submission_zip,
        "public.dry-run.manifest.v1.json": artifacts.manifest_data,
        "public.dry-run.telemetry.v1.json": remote_telemetry,
        "modal.responses.v1.jsonl": response_data,
        "modal.approval-scope.v1.json": approval_scope,
        "modal.execution.v1.json": execution_config,
        "modal.request-preflight.v1.json": modal_preflight_data,
        "modal.response-preflight.v1.json": response_preflight_data,
        "public.dry-run.local-replay.telemetry.v1.json": artifacts.telemetry_data,
    }
    for name, data in outputs.items():
        checksum = write_immutable_bytes(output_root / name, data)
        print(f"frozen {name} {checksum}", flush=True)
    print(
        f"submission count={submission_validation.count} checksum={submission_validation.checksum}",
        flush=True,
    )
