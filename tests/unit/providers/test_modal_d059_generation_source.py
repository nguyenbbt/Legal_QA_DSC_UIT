from __future__ import annotations

from pathlib import Path


def _source() -> str:
    return Path("scripts/modal_qwen25_3b_generation.py").read_text(encoding="utf-8")


def test_d059_modal_source_pins_model_and_closes_execution_bounds() -> None:
    source = _source()

    assert 'MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"' in source
    assert 'MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"' in source
    assert "EXPECTED_PARAMETER_COUNT = 3_085_938_688" in source
    assert 'gpu="A10"' in source
    assert "max_containers=1" in source
    assert source.count("serialized=False") == 2
    assert "serialized=True" not in source
    assert source.count("include_source=True") == 2
    assert "include_source=False" not in source
    assert "block_network=True" in source
    assert "restrict_modal_access=True" in source
    assert "with_mount_options(read_only=True)" in source
    assert "BATCH_SIZE = 8" in source
    assert "MAXIMUM_INPUT_TOKENS = 2048" in source
    assert "MAXIMUM_NEW_TOKENS = 512" in source


def test_d059_modal_source_uses_new_storage_and_never_reads_old_response_checkpoints() -> None:
    source = _source()

    assert 'MODEL_VOLUME_NAME = "dsc2026-qwen25-3b-d059-model-v1"' in source
    assert 'CAMPAIGN_ID = "D059-qwen25-3b-r0-prompt-a-v1"' in source
    assert "public-r0-g1a512-v1/modal-answer-checkpoints" not in source
    assert "build_modal_development_requests" not in source
    assert "Qwen25GeneratorBackend" in source
    assert '_invoke_remote(phase="development"' not in source
    assert "development-local-answer-checkpoints" in source
    assert "d059_local_development_progress=" in source
    assert "checkpoint_fingerprint" in source
    assert "build_modal_public_requests" in source
    assert "validate_modal_public_responses" in source
    assert "build_submission_zip" in source
    assert "gold_answer" not in source
    assert "grounding_label" not in source


def test_d059_modal_source_has_no_training_or_sweep_path() -> None:
    source = _source().lower()

    assert "fine_tune" not in source
    assert "trainer(" not in source
    assert "grid_search" not in source
    assert "sweep" not in source
