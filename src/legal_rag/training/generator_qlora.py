"""Guarded local QLoRA runner over checksum-linked official RAG-SFT rows."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes, content_json_bytes
from legal_rag.training.dataset_policy import validate_training_dataset
from legal_rag.training.provenance import TrainingExample, parse_training_example
from legal_rag.training.recipes import GENERATOR_QLORA_CENTRAL, apply_lora, recipe_checksum

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam. Chỉ trả lời từ căn cứ được cung cấp; "
    "không suy đoán hoặc bổ sung nguồn ngoài."
)
APPROVED_SYSTEM_PARAMETER_COUNT = 3_223_292_928
COMPETITION_PARAMETER_LIMIT = 4_000_000_000


class GeneratorDatasetError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class GeneratorTrainingRow:
    example_id: str
    question_id: str
    question: str
    evidence_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    target: str


@dataclass(frozen=True, slots=True)
class GeneratorTrainingConfig:
    run_mode: Literal["smoke", "full"]
    checkpoint: Path
    output_directory: Path
    maximum_length: int = 1024
    gradient_accumulation_steps: int = 8
    smoke_rows: int = 8
    smoke_steps: int = 2

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_length,
                self.gradient_accumulation_steps,
                self.smoke_rows,
                self.smoke_steps,
            )
            < 1
        ):
            raise ValueError("generator QLoRA limits must be positive")


def _load_manifest(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
        if not isinstance(value, dict) or canonical_json_bytes(value) != data:
            raise ValueError
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise GeneratorDatasetError(
            "GENERATOR_DATASET_MANIFEST_INVALID", "RAG-SFT manifest is invalid"
        ) from error
    return value


def _load_provenance(data: bytes) -> tuple[TrainingExample, ...]:
    try:
        rows = tuple(parse_training_example(json.loads(line)) for line in data.splitlines())
        validate_training_dataset(rows)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise GeneratorDatasetError(
            "GENERATOR_PROVENANCE_INVALID", "generator provenance artifact is invalid"
        ) from error
    if any(row.task != "generation" for row in rows):
        raise GeneratorDatasetError(
            "GENERATOR_PROVENANCE_INVALID", "generator provenance contains another task"
        )
    return rows


def load_generator_rows(
    provenance_data: bytes, material_data: bytes, manifest_data: bytes
) -> tuple[GeneratorTrainingRow, ...]:
    """Join private material to validated public-safe provenance by immutable identity."""

    manifest = _load_manifest(manifest_data)
    if (
        manifest.get("schema_version") != "training.rag_sft.manifest.v1"
        or manifest.get("provenance_checksum") != checksum_bytes(provenance_data)
        or manifest.get("material_checksum") != checksum_bytes(material_data)
    ):
        raise GeneratorDatasetError(
            "GENERATOR_DATASET_CHECKSUM_MISMATCH", "RAG-SFT artifact checksum changed"
        )
    provenance = _load_provenance(provenance_data)
    by_id = {row.example_id: row for row in provenance}
    rows: list[GeneratorTrainingRow] = []
    for line in material_data.splitlines(keepends=True):
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or content_json_bytes(value) != line:
                raise ValueError
            evidence = value["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError
            example_id = value["example_id"]
            provenance_row = by_id.get(example_id)
            evidence_ids = tuple(item["evidence_id"] for item in evidence)
            evidence_text = tuple(item["display_text"] for item in evidence)
            target = value["target"]
            question = value["question"]
            question_id = value["question_id"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise GeneratorDatasetError(
                "GENERATOR_MATERIAL_INVALID", "generator material row is invalid"
            ) from error
        if (
            provenance_row is None
            or question_id != provenance_row.question_id
            or evidence_ids != provenance_row.evidence_ids
            or not all(isinstance(text, str) and text for text in evidence_text)
            or not isinstance(question, str)
            or not isinstance(target, str)
            or unicodedata.normalize("NFC", question) != question
            or unicodedata.normalize("NFC", target) != target
        ):
            raise GeneratorDatasetError(
                "GENERATOR_MATERIAL_PROVENANCE_MISMATCH",
                "generator material does not match provenance",
            )
        if checksum_bytes(target.encode()) != provenance_row.target_checksum:
            raise GeneratorDatasetError(
                "GENERATOR_TARGET_CHECKSUM_MISMATCH", "official target text changed"
            )
        rows.append(
            GeneratorTrainingRow(
                example_id,
                question_id,
                question,
                evidence_ids,
                evidence_text,
                target,
            )
        )
    if len(rows) != len(provenance) or manifest.get("accepted_rows") != len(rows):
        raise GeneratorDatasetError(
            "GENERATOR_DATASET_CARDINALITY_MISMATCH", "RAG-SFT row cardinality changed"
        )
    return tuple(rows)


def _training_text(row: GeneratorTrainingRow) -> tuple[list[dict[str, str]], str]:
    evidence = "\n\n".join(
        f"[EVIDENCE {index}]\n{text}" for index, text in enumerate(row.evidence, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Câu hỏi:\n{row.question}\n\nCăn cứ:\n{evidence}"},
    ]
    return messages, row.target


def tokenize_generator_row(
    tokenizer: Any, row: GeneratorTrainingRow, *, maximum_length: int
) -> dict[str, list[int]]:
    """Tokenize one row with loss masked over every non-answer token."""

    messages, target = _training_text(row)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(
        tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    )
    if len(target_ids) >= maximum_length:
        raise GeneratorDatasetError(
            "GENERATOR_TARGET_TOO_LONG", "official target exceeds the training sequence limit"
        )
    prompt_ids = prompt_ids[-(maximum_length - len(target_ids)) :]
    input_ids = [*prompt_ids, *target_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def run_generator_qlora(
    rows: tuple[GeneratorTrainingRow, ...], config: GeneratorTrainingConfig
) -> dict[str, object]:
    """Execute one bounded local smoke or central QLoRA run and save only the adapter."""

    try:
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("locked model-training dependencies are unavailable") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("local QLoRA requires a CUDA GPU with BF16 support")
    set_seed(GENERATOR_QLORA_CENTRAL.seed)
    local = config.checkpoint.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = rows[: config.smoke_rows] if config.run_mode == "smoke" else rows
    features = tuple(
        tokenize_generator_row(tokenizer, row, maximum_length=config.maximum_length)
        for row in selected
    )

    class _Dataset:
        def __len__(self) -> int:
            return len(features)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return features[index]

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)
        return {
            key: torch.tensor(
                [
                    item[key]
                    + [(-100 if key == "labels" else tokenizer.pad_token_id)]
                    * (width - len(item[key]))
                    for item in batch
                ],
                dtype=torch.long,
            )
            for key in ("input_ids", "attention_mask", "labels")
        }

    bnb_config_type: Any = BitsAndBytesConfig
    quantization = bnb_config_type(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(local),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    prepare_kbit: Any = prepare_model_for_kbit_training
    model = prepare_kbit(model, use_gradient_checkpointing=True)
    model = apply_lora(model, GENERATOR_QLORA_CENTRAL)
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    system_parameters = APPROVED_SYSTEM_PARAMETER_COUNT + adapter_parameters
    if system_parameters >= COMPETITION_PARAMETER_LIMIT:
        raise RuntimeError("QLoRA adapter would violate the whole-system parameter gate")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(config.output_directory),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=GENERATOR_QLORA_CENTRAL.epochs,
        learning_rate=GENERATOR_QLORA_CENTRAL.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=GENERATOR_QLORA_CENTRAL.seed,
        max_steps=config.smoke_steps if config.run_mode == "smoke" else -1,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=_Dataset(), data_collator=collate)
    result = trainer.train()
    model.save_pretrained(config.output_directory / "adapter")
    tokenizer.save_pretrained(config.output_directory / "adapter")
    report = {
        "schema_version": "generator.qlora.run-report.v1",
        "run_mode": config.run_mode,
        "training_rows": len(selected),
        "recipe_checksum": recipe_checksum(GENERATOR_QLORA_CENTRAL),
        "adapter_parameter_count": adapter_parameters,
        "whole_system_parameter_count": system_parameters,
        "competition_parameter_limit_exclusive": COMPETITION_PARAMETER_LIMIT,
        "train_loss": float(result.training_loss),
        "global_steps": int(result.global_step),
        "contains_generated_training_text": False,
    }
    if not math.isfinite(report["train_loss"]):
        raise RuntimeError("QLoRA produced a non-finite training loss")
    (config.output_directory / "run-report.json").write_bytes(content_json_bytes(report))
    return report


__all__ = [
    "GeneratorDatasetError",
    "GeneratorTrainingConfig",
    "GeneratorTrainingRow",
    "load_generator_rows",
    "run_generator_qlora",
    "tokenize_generator_row",
]
