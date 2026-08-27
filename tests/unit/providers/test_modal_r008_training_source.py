from __future__ import annotations

from pathlib import Path


def test_modal_r008_source_closes_gpu_network_and_container_bounds() -> None:
    source = Path("scripts/modal_r008_reranker_training.py").read_text(encoding="utf-8")

    assert 'gpu="A10"' in source
    assert "max_containers=1" in source
    assert "serialized=True" in source
    assert "block_network=True" in source
    assert "restrict_modal_access=True" in source
    assert "with_mount_options(read_only=True)" in source
    assert "validate_r008_training_payload" in source
    assert "validate_r008_training_response" in source
    assert "train_qwen3_reranker_lora" in source


def test_modal_r008_source_has_no_hidden_training_sweep() -> None:
    source = Path("scripts/modal_r008_reranker_training.py").read_text(encoding="utf-8")

    assert 'RerankerLoraRunConfig(mode="central")' in source
    assert "grid_search" not in source
    assert "sweep" not in source.lower()
    assert "train.answers" not in source
