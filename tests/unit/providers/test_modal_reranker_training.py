from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.providers.modal_reranker_training import (
    ModalRerankerTrainingError,
    R008TrainingLifecycle,
    validate_r008_training_payload,
    validate_r008_training_response,
)


def _artifact_bytes() -> tuple[bytes, bytes, bytes, bytes]:
    group = {
        "schema_version": "reranker.training-group.v1",
        "group_id": "group-1",
        "question_id": "q-1",
        "split": "train",
        "question": "Câu hỏi chính thức?",
        "question_checksum": checksum_bytes("Câu hỏi chính thức?".encode()),
        "positives": [
            {
                "evidence_id": "positive",
                "context_id": "ctx",
                "evidence_checksum": checksum_bytes(b"positive"),
                "hierarchy_path": ["Điều 1"],
                "canonical_start": 0,
                "canonical_end": 20,
                "text": "Điều 1. Nội dung đúng.",
            }
        ],
        "negatives": [
            {
                "evidence_id": "negative",
                "context_id": "ctx",
                "evidence_checksum": checksum_bytes(b"negative"),
                "hierarchy_path": ["Điều 2"],
                "canonical_start": 30,
                "canonical_end": 50,
                "text": "Điều 2. Nội dung sai.",
                "negative_type": "SAME_DOCUMENT_WRONG_ARTICLE",
            }
        ],
        "target_checksum": checksum_bytes(b"target"),
        "construction_version": "r008-reranker-groups.v1",
        "contains_generated_text": False,
    }
    groups = content_json_bytes(group)
    provenance = content_json_bytes(
        {
            "schema_version": "training.example.v1",
            "example_id": "example-1",
            "task": "reranking",
            "question_id": "q-1",
            "split": "train",
            "question_source_checksum": checksum_bytes(b"questions"),
            "evidence_ids": ["positive", "negative"],
            "target_source": "deterministic_relevance",
            "target_checksum": group["target_checksum"],
            "contains_generated_text": False,
            "construction_version": "r008-reranker-groups.v1",
        }
    )
    manifest = content_json_bytes(
        {
            "schema_version": "reranker.training-manifest.v1",
            "construction_version": "r008-reranker-groups.v1",
            "group_count": 1,
            "pair_count": 1,
            "rejected_no_negative_count": 0,
            "unique_question_ids": 1,
            "unique_evidence_ids": 2,
            "maximum_negatives": 8,
            "question_source_checksum": checksum_bytes(b"questions"),
            "split_manifest_checksum": checksum_bytes(b"split"),
            "selection_checksum": checksum_bytes(b"selection"),
            "chunks_checksum": checksum_bytes(b"chunks"),
            "index_checksum": checksum_bytes(b"index"),
            "groups_checksum": checksum_bytes(groups),
            "provenance_checksum": checksum_bytes(provenance),
            "contains_generated_text": False,
        }
    )
    request = content_json_bytes(
        {
            "schema_version": "modal.r008.training-request.v1",
            "run_id": "R008-qwen3-reranker-central-v1",
            "model_id": "Qwen/Qwen3-Reranker-0.6B",
            "model_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
            "tokenizer_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
            "model_manifest_checksum": checksum_bytes(b"model-manifest"),
            "recipe_checksum": checksum_bytes(b"recipe"),
            "dataset_manifest_checksum": checksum_bytes(manifest),
            "groups_checksum": checksum_bytes(groups),
            "provenance_checksum": checksum_bytes(provenance),
            "group_count": 1,
            "pair_count": 1,
            "maximum_length": 1536,
            "seed": 42,
            "base_parameter_count": 595_776_512,
            "whole_system_base_parameter_count": 3_223_292_928,
        }
    )
    return request, groups, provenance, manifest


def test_closed_training_payload_accepts_only_official_train_groups() -> None:
    request, groups, provenance, manifest = _artifact_bytes()

    payload = validate_r008_training_payload(
        request_data=request,
        groups_data=groups,
        provenance_data=provenance,
        dataset_manifest_data=manifest,
    )

    assert payload.group_count == 1
    assert payload.pair_count == 1
    assert payload.model_id == "Qwen/Qwen3-Reranker-0.6B"


@pytest.mark.parametrize(
    ("change", "code"),
    (
        ({"split": "development"}, "MODAL_R008_NONTRAIN_REJECTED"),
        ({"answer": "forbidden"}, "MODAL_R008_GROUP_SCHEMA_INVALID"),
        ({"contains_generated_text": True}, "MODAL_R008_GENERATED_TEXT_REJECTED"),
    ),
)
def test_training_payload_rejects_nontrain_generated_or_extra_data(
    change: dict[str, object], code: str
) -> None:
    request, groups, provenance, manifest = _artifact_bytes()
    group = json.loads(groups)
    group.update(change)

    with pytest.raises(ModalRerankerTrainingError) as raised:
        validate_r008_training_payload(
            request_data=request,
            groups_data=content_json_bytes(group),
            provenance_data=provenance,
            dataset_manifest_data=manifest,
        )

    assert raised.value.code == code


def test_training_payload_rejects_checksum_drift_before_transfer() -> None:
    request, groups, provenance, manifest = _artifact_bytes()

    with pytest.raises(ModalRerankerTrainingError) as raised:
        validate_r008_training_payload(
            request_data=request,
            groups_data=groups.removesuffix(b"\n") + b" \n",
            provenance_data=provenance,
            dataset_manifest_data=manifest,
        )

    assert raised.value.code == "MODAL_R008_CHECKSUM_MISMATCH"


def test_training_payload_recomputes_pair_count_from_closed_groups() -> None:
    request_data, groups, provenance, manifest_data = _artifact_bytes()
    request = json.loads(request_data)
    manifest = json.loads(manifest_data)
    manifest["pair_count"] = 2
    changed_manifest = content_json_bytes(manifest)
    request["pair_count"] = 2
    request["dataset_manifest_checksum"] = checksum_bytes(changed_manifest)

    with pytest.raises(ModalRerankerTrainingError) as raised:
        validate_r008_training_payload(
            request_data=content_json_bytes(request),
            groups_data=groups,
            provenance_data=provenance,
            dataset_manifest_data=changed_manifest,
        )

    assert raised.value.code == "MODAL_R008_CARDINALITY_MISMATCH"


def test_response_binds_adapter_and_keeps_whole_system_below_four_billion() -> None:
    request, *_ = _artifact_bytes()
    response = content_json_bytes(
        {
            "schema_version": "modal.r008.training-response.v1",
            "run_id": "R008-qwen3-reranker-central-v1",
            "adapter_path": "R008-qwen3-reranker-central-v1/adapter",
            "adapter_checksum": checksum_bytes(b"adapter"),
            "adapter_parameter_count": 8_388_608,
            "whole_system_parameter_count": 3_231_681_536,
            "training_metrics": {
                "epochs": 2,
                "mean_train_loss": 0.5,
                "optimizer_steps": 1,
                "train_pairs": 1,
                "validation_loss": 0.4,
                "validation_pair_accuracy": 1.0,
                "validation_pairs": 1,
            },
            "telemetry": {
                "elapsed_seconds": 12.5,
                "gpu_name": "NVIDIA A10",
                "peak_cuda_bytes": 1234,
                "torch_version": "2.8.0+cu128",
            },
        }
    )

    validated = validate_r008_training_response(request_data=request, response_data=response)
    assert validated.adapter_parameter_count == 8_388_608

    invalid = json.loads(response)
    invalid["whole_system_parameter_count"] = 4_000_000_000
    with pytest.raises(ModalRerankerTrainingError) as raised:
        validate_r008_training_response(
            request_data=request,
            response_data=content_json_bytes(invalid),
        )
    assert raised.value.code == "MODEL_PARAMETER_LIMIT"


def test_response_rejects_freeform_telemetry_that_could_exfiltrate_training_text() -> None:
    request, *_ = _artifact_bytes()
    response = {
        "schema_version": "modal.r008.training-response.v1",
        "run_id": "R008-qwen3-reranker-central-v1",
        "adapter_path": "R008-qwen3-reranker-central-v1/adapter",
        "adapter_checksum": checksum_bytes(b"adapter"),
        "adapter_parameter_count": 8_388_608,
        "whole_system_parameter_count": 3_231_681_536,
        "training_metrics": {
            "epochs": 2,
            "mean_train_loss": 0.5,
            "optimizer_steps": 1,
            "train_pairs": 1,
            "validation_loss": 0.4,
            "validation_pair_accuracy": 1.0,
            "validation_pairs": 1,
        },
        "telemetry": {
            "elapsed_seconds": 12.5,
            "gpu_name": "NVIDIA A10",
            "peak_cuda_bytes": 1234,
            "torch_version": "2.8.0+cu128",
            "debug_text": "official training passage",
        },
    }

    with pytest.raises(ModalRerankerTrainingError) as raised:
        validate_r008_training_response(
            request_data=request,
            response_data=content_json_bytes(response),
        )

    assert raised.value.code == "MODAL_R008_RESPONSE_SCHEMA_INVALID"


def test_training_lifecycle_requires_verified_download_before_deletion() -> None:
    state = R008TrainingLifecycle().record_preflight().record_remote_run("run-1")
    with pytest.raises(ModalRerankerTrainingError) as raised:
        state.record_volume_deleted()
    assert raised.value.code == "MODAL_R008_LIFECYCLE_INVALID"

    complete = state.record_download(checksum_bytes(b"adapter")).record_volume_deleted()
    assert complete.state == "deleted"
