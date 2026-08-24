"""Closed bounded LoRA/QLoRA recipes and lazy PEFT application."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import FrozenStrictModel, NonEmptyString


class LoraRecipe(FrozenStrictModel, frozen=True):
    schema_version: Literal["training.recipe.v1"]
    task: Literal["embedding", "reranking", "generation"]
    method: Literal["lora", "qlora"]
    objective: NonEmptyString
    seed: Literal[42]
    epochs: int = Field(ge=1, le=3)
    learning_rate: float = Field(gt=0.0, le=0.001)
    lora_rank: int = Field(ge=1, le=64)
    lora_alpha: int = Field(ge=1, le=128)
    lora_dropout: float = Field(ge=0.0, le=0.2)
    target_modules: tuple[NonEmptyString, ...]
    quantization_for_training: Literal["nf4"] | None
    compute_dtype: Literal["bf16", "fp16"]
    loss_scope: Literal["contrastive", "pairwise", "answer_tokens_only"]


FT_RERANK_CENTRAL = LoraRecipe.model_validate(
    {
        "schema_version": "training.recipe.v1",
        "task": "reranking",
        "method": "lora",
        "objective": "pairwise_relevance",
        "seed": 42,
        "epochs": 2,
        "learning_rate": 1e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "quantization_for_training": None,
        "compute_dtype": "bf16",
        "loss_scope": "pairwise",
    }
)

GENERATOR_QLORA_CENTRAL = LoraRecipe.model_validate(
    {
        "schema_version": "training.recipe.v1",
        "task": "generation",
        "method": "qlora",
        "objective": "rag_supervised_fine_tuning",
        "seed": 42,
        "epochs": 1,
        "learning_rate": 1e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.03,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "quantization_for_training": "nf4",
        "compute_dtype": "bf16",
        "loss_scope": "answer_tokens_only",
    }
)


def recipe_checksum(recipe: LoraRecipe) -> str:
    return checksum_bytes(content_json_bytes(recipe.model_dump(mode="json")))


def apply_lora(model: Any, recipe: LoraRecipe) -> Any:
    """Apply one manifested PEFT adapter; no optimizer or hidden sweep is created."""

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("locked PEFT dependency is unavailable") from error
    configuration = LoraConfig(
        r=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        target_modules=list(recipe.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, configuration)


__all__ = [
    "FT_RERANK_CENTRAL",
    "GENERATOR_QLORA_CENTRAL",
    "LoraRecipe",
    "apply_lora",
    "recipe_checksum",
]
