from __future__ import annotations

import hashlib

import pytest
import torch

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.qwen3_reranker_prompt import build_qwen3_reranker_prompt
from legal_rag.training.reranker_lora import (
    RerankerLoraRunConfig,
    directory_checksum,
    load_reranker_pairs,
    pairwise_logistic_loss,
    split_pairs_by_group,
)


def _group(group_id: str, question_id: str) -> bytes:
    return content_json_bytes(
        {
            "schema_version": "reranker.training-group.v1",
            "group_id": group_id,
            "question_id": question_id,
            "split": "train",
            "question": f"Câu hỏi {question_id}?",
            "question_checksum": checksum_bytes(f"Câu hỏi {question_id}?".encode()),
            "positives": [
                {
                    "evidence_id": f"{question_id}-p",
                    "context_id": "ctx",
                    "evidence_checksum": checksum_bytes(b"p"),
                    "hierarchy_path": ["Điều 1"],
                    "canonical_start": 0,
                    "canonical_end": 10,
                    "text": "Nội dung đúng.",
                }
            ],
            "negatives": [
                {
                    "evidence_id": f"{question_id}-n1",
                    "context_id": "ctx",
                    "evidence_checksum": checksum_bytes(b"n1"),
                    "hierarchy_path": ["Điều 2"],
                    "canonical_start": 20,
                    "canonical_end": 30,
                    "text": "Nội dung sai một.",
                    "negative_type": "SAME_DOCUMENT_WRONG_ARTICLE",
                },
                {
                    "evidence_id": f"{question_id}-n2",
                    "context_id": "ctx",
                    "evidence_checksum": checksum_bytes(b"n2"),
                    "hierarchy_path": ["Điều 3"],
                    "canonical_start": 40,
                    "canonical_end": 50,
                    "text": "Nội dung sai hai.",
                    "negative_type": "SAME_DOCUMENT_WRONG_ARTICLE",
                },
            ],
            "target_checksum": checksum_bytes(b"target"),
            "construction_version": "r008-reranker-groups.v1",
            "contains_generated_text": False,
        }
    )


def test_qwen3_training_prompt_matches_yes_no_inference_contract() -> None:
    prompt = build_qwen3_reranker_prompt(
        instruction="Find direct Vietnamese legal support.",
        query="Ai có thẩm quyền?",
        document="Điều 1. Bộ trưởng có thẩm quyền.",
    )

    assert "answer can only be yes or no" in prompt
    assert "<Instruct>: Find direct Vietnamese legal support." in prompt
    assert "<Query>: Ai có thẩm quyền?" in prompt
    assert "<Document>: Điều 1. Bộ trưởng có thẩm quyền." in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_loader_expands_each_positive_negative_pair_in_stable_order() -> None:
    pairs = load_reranker_pairs(_group("group-b", "q2") + _group("group-a", "q1"))

    assert len(pairs) == 4
    assert [(pair.group_id, pair.negative_id) for pair in pairs] == [
        ("group-a", "q1-n1"),
        ("group-a", "q1-n2"),
        ("group-b", "q2-n1"),
        ("group-b", "q2-n2"),
    ]


def test_internal_validation_split_never_leaks_a_question_group() -> None:
    data = b"".join(_group(f"group-{index:02d}", f"q{index:02d}") for index in range(20))
    train, validation = split_pairs_by_group(load_reranker_pairs(data))

    train_groups = {pair.group_id for pair in train}
    validation_groups = {pair.group_id for pair in validation}
    assert train_groups
    assert validation_groups
    assert train_groups.isdisjoint(validation_groups)


def test_pairwise_loss_rewards_a_positive_score_above_the_negative() -> None:
    preferred = pairwise_logistic_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    reversed_loss = pairwise_logistic_loss(torch.tensor([0.0]), torch.tensor([2.0]))

    assert preferred.item() < reversed_loss.item()
    assert abs(preferred.item() - 0.126928) < 1e-5


def test_central_run_config_is_bounded_and_rejects_a_hidden_sweep() -> None:
    config = RerankerLoraRunConfig(mode="central")
    assert config.epochs == 2
    assert config.maximum_length == 1536
    assert config.pair_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.maximum_train_pairs is None

    with pytest.raises(ValueError):
        RerankerLoraRunConfig(mode="central", epochs=3)


def test_corrective_run_changes_only_the_epoch_bound() -> None:
    corrective = RerankerLoraRunConfig(mode="corrective")
    central = RerankerLoraRunConfig(mode="central")

    assert corrective.epochs == 1
    assert corrective.maximum_length == central.maximum_length
    assert corrective.pair_batch_size == central.pair_batch_size
    assert corrective.gradient_accumulation_steps == central.gradient_accumulation_steps


def test_directory_checksum_binds_sorted_relative_names_and_bytes(tmp_path) -> None:
    (tmp_path / "b.json").write_bytes(b"two")
    (tmp_path / "a.safetensors").write_bytes(b"one")
    first = directory_checksum(tmp_path)
    (tmp_path / "b.json").write_bytes(b"changed")

    assert first == "sha256:9643ffbb76fe8216737546a6e59ede557b152c424f6e85f4253d50ad5ba6bfbd"
    assert directory_checksum(tmp_path) != first


def test_directory_checksum_uses_utf8_byte_order_across_operating_systems(tmp_path) -> None:
    contents = {
        "README.md": b"three",
        "adapter_config.json": b"one",
        "adapter_model.safetensors": b"two",
    }
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
    reference = hashlib.sha256()
    for name in ("README.md", "adapter_config.json", "adapter_model.safetensors"):
        encoded_name = name.encode()
        data = contents[name]
        reference.update(len(encoded_name).to_bytes(4, "big"))
        reference.update(encoded_name)
        reference.update(len(data).to_bytes(8, "big"))
        reference.update(data)

    assert directory_checksum(tmp_path) == f"sha256:{reference.hexdigest()}"
