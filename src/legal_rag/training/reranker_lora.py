"""Deterministic pair expansion and loss helpers for R-008 LoRA."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

from legal_rag.retrieval.qwen3_reranker_prompt import build_qwen3_reranker_prompt
from legal_rag.training.recipes import (
    FT_RERANK_CENTRAL,
    FT_RERANK_CORRECTIVE_EPOCH_1,
    apply_lora,
    recipe_checksum,
)

RERANKER_TRAINING_INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the document directly provides "
    "the governing legal rule, actor, condition, exception, or numeric value needed to answer it."
)


class RerankerLoraError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class RerankerPair:
    group_id: str
    question_id: str
    question: str
    positive_id: str
    positive_text: str
    negative_id: str
    negative_text: str
    negative_type: str


@dataclass(frozen=True, slots=True)
class RerankerLoraRunConfig:
    mode: Literal["smoke", "central", "corrective"]
    epochs: int | None = None
    maximum_length: int | None = None
    pair_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    maximum_train_pairs: int | None = None
    maximum_validation_pairs: int | None = None

    def __post_init__(self) -> None:
        expected = {
            "smoke": (1, 512, 1, 1, 4, 4),
            "central": (2, 1536, 2, 8, None, 128),
            "corrective": (1, 1536, 2, 8, None, 128),
        }[self.mode]
        values = (
            self.epochs,
            self.maximum_length,
            self.pair_batch_size,
            self.gradient_accumulation_steps,
            self.maximum_train_pairs,
            self.maximum_validation_pairs,
        )
        names = (
            "epochs",
            "maximum_length",
            "pair_batch_size",
            "gradient_accumulation_steps",
            "maximum_train_pairs",
            "maximum_validation_pairs",
        )
        for name, value, default in zip(names, values, expected, strict=True):
            if value is not None and value != default:
                raise ValueError(f"{self.mode} {name} is fixed by the bounded recipe")
            object.__setattr__(self, name, default)


@dataclass(frozen=True, slots=True)
class RerankerLoraRunResult:
    mode: Literal["smoke", "central", "corrective"]
    train_pair_count: int
    validation_pair_count: int
    optimizer_step_count: int
    mean_train_loss: float
    validation_loss: float
    validation_pair_accuracy: float
    adapter_parameter_count: int
    adapter_checksum: str
    recipe_checksum: str
    elapsed_seconds: float
    peak_cuda_bytes: int


def _fail(code: str, message: str) -> NoReturn:
    raise RerankerLoraError(code, message)


def _nonempty_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("RERANKER_PAIR_DATA_INVALID", "training passage field is invalid")
    return value


def _configured_int(value: int | None) -> int:
    if value is None:
        raise RuntimeError("bounded R-008 configuration was not resolved")
    return value


def _parse_group_rows(groups_data: bytes) -> list[dict[str, object]]:
    if not groups_data or not groups_data.endswith(b"\n"):
        _fail("RERANKER_PAIR_DATA_INVALID", "training groups are not newline-framed JSONL")
    groups: list[dict[str, object]] = []
    for line in groups_data.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RerankerLoraError(
                "RERANKER_PAIR_DATA_INVALID", "training group JSON is invalid"
            ) from error
        if not isinstance(value, dict):
            _fail("RERANKER_PAIR_DATA_INVALID", "training group must be an object")
        groups.append(value)
    return groups


def _expand_group(group: dict[str, object], seen_groups: set[str]) -> tuple[RerankerPair, ...]:
    group_id = group.get("group_id")
    question_id = group.get("question_id")
    question = group.get("question")
    positives = group.get("positives")
    negatives = group.get("negatives")
    if (
        not isinstance(group_id, str)
        or not group_id
        or group_id in seen_groups
        or not isinstance(question_id, str)
        or not question_id
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(positives, list)
        or not positives
        or not isinstance(negatives, list)
        or not negatives
    ):
        _fail("RERANKER_PAIR_DATA_INVALID", "training group value is invalid")
    seen_groups.add(group_id)
    ordered_positives = sorted(positives, key=lambda item: str(item.get("evidence_id", "")))
    ordered_negatives = sorted(negatives, key=lambda item: str(item.get("evidence_id", "")))
    pairs: list[RerankerPair] = []
    for positive in ordered_positives:
        for negative in ordered_negatives:
            if not isinstance(positive, dict) or not isinstance(negative, dict):
                _fail("RERANKER_PAIR_DATA_INVALID", "training passage is invalid")
            pairs.append(
                RerankerPair(
                    group_id=group_id,
                    question_id=question_id,
                    question=question,
                    positive_id=_nonempty_text(positive.get("evidence_id")),
                    positive_text=_nonempty_text(positive.get("text")),
                    negative_id=_nonempty_text(negative.get("evidence_id")),
                    negative_text=_nonempty_text(negative.get("text")),
                    negative_type=_nonempty_text(negative.get("negative_type")),
                )
            )
    return tuple(pairs)


def load_reranker_pairs(groups_data: bytes) -> tuple[RerankerPair, ...]:
    """Expand each official group into deterministic positive/negative pairs."""

    groups = _parse_group_rows(groups_data)
    seen_groups: set[str] = set()
    pairs = tuple(
        pair
        for group in sorted(groups, key=lambda item: str(item.get("group_id", "")).encode())
        for pair in _expand_group(group, seen_groups)
    )
    if not pairs:
        _fail("RERANKER_PAIR_DATA_EMPTY", "training groups produced no pairs")
    return pairs


def split_pairs_by_group(
    pairs: tuple[RerankerPair, ...],
) -> tuple[tuple[RerankerPair, ...], tuple[RerankerPair, ...]]:
    """Create a deterministic internal train/validation split without group leakage."""

    if not pairs:
        _fail("RERANKER_PAIR_DATA_EMPTY", "training pairs are empty")
    group_ids = tuple(sorted({pair.group_id for pair in pairs}, key=str.encode))
    if len(group_ids) < 2:
        _fail("RERANKER_VALIDATION_SPLIT_EMPTY", "at least two groups are required")
    validation_groups = {
        group_id for group_id in group_ids if hashlib.sha256(group_id.encode()).digest()[0] < 26
    }
    if not validation_groups:
        validation_groups = {group_ids[-1]}
    if len(validation_groups) == len(group_ids):
        validation_groups.remove(group_ids[0])
    train = tuple(pair for pair in pairs if pair.group_id not in validation_groups)
    validation = tuple(pair for pair in pairs if pair.group_id in validation_groups)
    if not train or not validation:
        _fail("RERANKER_VALIDATION_SPLIT_EMPTY", "internal training split is empty")
    return train, validation


def pairwise_logistic_loss(positive_scores: Any, negative_scores: Any) -> Any:
    """Return mean ``softplus(-(positive-negative))`` for one pair batch."""

    import torch

    if positive_scores.shape != negative_scores.shape or positive_scores.numel() == 0:
        raise ValueError("pairwise score tensors must have one matching non-empty shape")
    return torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()


def directory_checksum(directory: Path) -> str:
    """Hash sorted relative names and bytes for a small adapter directory."""

    if not directory.is_dir():
        raise ValueError("adapter directory does not exist")
    digest = hashlib.sha256()
    files = tuple(
        sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(directory).as_posix().encode(),
        )
    )
    if not files:
        raise ValueError("adapter directory is empty")
    for path in files:
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def train_qwen3_reranker_lora(
    *,
    checkpoint: Path,
    groups_data: bytes,
    output_directory: Path,
    model_id: str,
    model_revision: str,
    config: RerankerLoraRunConfig,
    device: str = "cuda",
) -> RerankerLoraRunResult:
    """Run the single bounded pairwise LoRA recipe from an offline checkpoint."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not checkpoint.is_dir():
        raise ValueError("pinned reranker checkpoint is absent")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("adapter output directory must be new or empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    all_pairs = load_reranker_pairs(groups_data)
    train_pairs, validation_pairs = split_pairs_by_group(all_pairs)
    if config.maximum_train_pairs is not None:
        train_pairs = train_pairs[: config.maximum_train_pairs]
    if config.maximum_validation_pairs is not None:
        validation_pairs = validation_pairs[: config.maximum_validation_pairs]

    recipe = FT_RERANK_CORRECTIVE_EPOCH_1 if config.mode == "corrective" else FT_RERANK_CENTRAL
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for R-008 training")
        torch.cuda.manual_seed_all(recipe.seed)
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    )
    base = base.to(device)
    base.config.use_cache = False
    model = apply_lora(base, recipe)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError("pinned tokenizer does not provide single yes/no label tokens")
    yes_id, no_id = int(yes_ids[0]), int(no_ids[0])
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=recipe.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    def scores(pairs: tuple[RerankerPair, ...], *, gradient: bool) -> tuple[Any, Any]:
        positive_prompts = [
            build_qwen3_reranker_prompt(
                instruction=RERANKER_TRAINING_INSTRUCTION,
                query=pair.question,
                document=pair.positive_text,
            )
            for pair in pairs
        ]
        negative_prompts = [
            build_qwen3_reranker_prompt(
                instruction=RERANKER_TRAINING_INSTRUCTION,
                query=pair.question,
                document=pair.negative_text,
            )
            for pair in pairs
        ]
        encoded = tokenizer(
            positive_prompts + negative_prompts,
            padding=True,
            truncation=True,
            max_length=config.maximum_length,
            return_tensors="pt",
        ).to(device)
        with torch.set_grad_enabled(gradient):
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, [no_id, yes_id]].float()
            relevance = logits[:, 1] - logits[:, 0]
        size = len(pairs)
        return relevance[:size], relevance[size:]

    started = time.perf_counter()
    raw_losses: list[float] = []
    optimizer_steps = 0
    batch_size = _configured_int(config.pair_batch_size)
    accumulation = _configured_int(config.gradient_accumulation_steps)
    for epoch in range(_configured_int(config.epochs)):
        indices = list(range(len(train_pairs)))
        random.Random(recipe.seed + epoch).shuffle(indices)
        ordered = tuple(train_pairs[index] for index in indices)
        batch_count = (len(ordered) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
            batch = ordered[start : start + batch_size]
            positive_scores, negative_scores = scores(batch, gradient=True)
            raw_loss = pairwise_logistic_loss(positive_scores, negative_scores)
            (raw_loss / accumulation).backward()
            raw_losses.append(float(raw_loss.detach().cpu()))
            is_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count
            if is_boundary:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    model.eval()
    validation_losses: list[float] = []
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(validation_pairs), batch_size):
            batch = validation_pairs[start : start + batch_size]
            positive_scores, negative_scores = scores(batch, gradient=False)
            validation_losses.append(
                float(pairwise_logistic_loss(positive_scores, negative_scores).cpu())
            )
            correct += int((positive_scores > negative_scores).sum().item())

    adapter_directory = output_directory / "adapter"
    peft_config = model.peft_config.get("default")
    if peft_config is not None:
        peft_config.base_model_name_or_path = model_id
        peft_config.revision = model_revision
    model.save_pretrained(adapter_directory, safe_serialization=True)
    adapter_count = sum(parameter.numel() for parameter in trainable)
    elapsed = time.perf_counter() - started
    peak_cuda = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return RerankerLoraRunResult(
        mode=config.mode,
        train_pair_count=len(train_pairs),
        validation_pair_count=len(validation_pairs),
        optimizer_step_count=optimizer_steps,
        mean_train_loss=sum(raw_losses) / len(raw_losses),
        validation_loss=sum(validation_losses) / len(validation_losses),
        validation_pair_accuracy=correct / len(validation_pairs),
        adapter_parameter_count=adapter_count,
        adapter_checksum=directory_checksum(adapter_directory),
        recipe_checksum=recipe_checksum(recipe),
        elapsed_seconds=elapsed,
        peak_cuda_bytes=peak_cuda,
    )


__all__ = [
    "RerankerLoraError",
    "RerankerLoraRunConfig",
    "RerankerLoraRunResult",
    "RerankerPair",
    "directory_checksum",
    "load_reranker_pairs",
    "pairwise_logistic_loss",
    "split_pairs_by_group",
    "train_qwen3_reranker_lora",
]
