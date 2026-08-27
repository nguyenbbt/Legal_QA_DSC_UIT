from __future__ import annotations

from legal_rag.training.recipes import (
    FT_RERANK_CENTRAL,
    FT_RERANK_CORRECTIVE_EPOCH_1,
    GENERATOR_QLORA_CENTRAL,
    recipe_checksum,
)


def test_locked_central_recipes_match_the_confirmed_contract() -> None:
    assert FT_RERANK_CENTRAL.task == "reranking"
    assert FT_RERANK_CENTRAL.loss_scope == "pairwise"
    assert GENERATOR_QLORA_CENTRAL.quantization_for_training == "nf4"
    assert GENERATOR_QLORA_CENTRAL.lora_rank == 32
    assert GENERATOR_QLORA_CENTRAL.lora_alpha == 64
    assert GENERATOR_QLORA_CENTRAL.lora_dropout == 0.03
    assert GENERATOR_QLORA_CENTRAL.epochs == 1
    assert GENERATOR_QLORA_CENTRAL.learning_rate == 1e-4
    assert GENERATOR_QLORA_CENTRAL.loss_scope == "answer_tokens_only"


def test_recipe_checksum_is_deterministic_and_material() -> None:
    assert recipe_checksum(FT_RERANK_CENTRAL) == recipe_checksum(FT_RERANK_CENTRAL)
    assert recipe_checksum(FT_RERANK_CENTRAL) != recipe_checksum(GENERATOR_QLORA_CENTRAL)


def test_reranker_corrective_changes_only_the_overfit_epoch_axis() -> None:
    central = FT_RERANK_CENTRAL.model_dump()
    corrective = FT_RERANK_CORRECTIVE_EPOCH_1.model_dump()
    changed = {key for key in central if central[key] != corrective[key]}

    assert changed == {"epochs"}
    assert corrective["epochs"] == 1
    assert recipe_checksum(FT_RERANK_CENTRAL) != recipe_checksum(FT_RERANK_CORRECTIVE_EPOCH_1)
