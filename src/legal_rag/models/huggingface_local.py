"""Local-only Hugging Face adapters for the approved model roles.

The module never downloads assets. Callers must acquire a pinned snapshot during
``prepare-online`` and then pass its project-local directory here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_rag.domain.checksums import checksum_file
from legal_rag.models.manifest import ModelManifestError


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ModelManifestError(
            "MODEL_DEPENDENCY_MISSING", "install the locked model optional dependencies"
        ) from error
    return torch, AutoModel, AutoModelForCausalLM, AutoTokenizer


def _local_directory(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or path.is_symlink():
        raise ModelManifestError(
            "MODEL_ARTIFACT_UNSUPPORTED", "checkpoint must be a regular local directory"
        )
    required = ("config.json", "tokenizer_config.json", "tokenizer.json")
    has_weights = (resolved / "model.safetensors").is_file() or (
        resolved / "model.safetensors.index.json"
    ).is_file()
    if not all((resolved / name).is_file() for name in required) or not has_weights:
        raise ModelManifestError(
            "MODEL_ARTIFACT_UNSUPPORTED",
            "checkpoint directory is incomplete or points above the pinned revision",
        )
    return str(resolved)


def _local_adapter_directory(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_dir()
        or path.is_symlink()
        or not (resolved / "adapter_config.json").is_file()
        or not (resolved / "adapter_model.safetensors").is_file()
    ):
        raise ModelManifestError(
            "MODEL_ADAPTER_ARTIFACT_UNSUPPORTED",
            "adapter checkpoint must contain a local config and safetensors weights",
        )
    return str(resolved)


def _last_token_pool(hidden_state: Any, attention_mask: Any) -> Any:
    """Pool the final non-padding token exactly as the Qwen embedding card specifies."""

    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_state.shape[0]
    return hidden_state[range(batch_size), sequence_lengths]


class Qwen3EmbeddingBackend:
    """Normalized local Qwen3 embeddings using an immutable snapshot."""

    dimension = 1024

    def __init__(
        self,
        checkpoint: Path,
        *,
        model_id: str,
        model_revision: str,
        device: str = "cuda",
        batch_size: int = 8,
        maximum_length: int = 8192,
        query_instruction: str,
    ) -> None:
        torch, auto_model, _, auto_tokenizer = _dependencies()
        if batch_size < 1 or maximum_length < 1 or not query_instruction.strip():
            raise ValueError("embedding batch, length, and instruction must be non-empty")
        local = _local_directory(checkpoint)
        self.model_id = model_id
        self.model_revision = model_revision
        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._maximum_length = maximum_length
        self._query_instruction = query_instruction
        self._tokenizer = auto_tokenizer.from_pretrained(
            local, local_files_only=True, trust_remote_code=False, padding_side="left"
        )
        self._model = auto_model.from_pretrained(
            local,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self._model.eval()

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        values = tuple(texts)
        output: list[list[float]] = []
        for start in range(0, len(values), self._batch_size):
            batch = values[start : start + self._batch_size]
            inputs = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self._maximum_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
                pooled = _last_token_pool(hidden, inputs["attention_mask"])
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            output.extend(normalized.float().cpu().tolist())
        return output

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(
            tuple(f"Instruct: {self._query_instruction}\nQuery: {text}" for text in texts)
        )

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)


class Qwen3RerankerBackend:
    """Local yes/no Qwen3 causal reranker over a caller-bounded candidate pool."""

    _SYSTEM = (
        "Judge whether the Document meets the requirements based on the Query and "
        "the Instruct provided. Note that the answer can only be yes or no."
    )

    def __init__(
        self,
        checkpoint: Path,
        *,
        model_id: str,
        model_revision: str,
        instruction: str,
        device: str = "cuda",
        batch_size: int = 2,
        maximum_length: int = 8192,
    ) -> None:
        torch, _, auto_causal, auto_tokenizer = _dependencies()
        local = _local_directory(checkpoint)
        self.model_id = model_id
        self.model_revision = model_revision
        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._maximum_length = maximum_length
        self._instruction = instruction
        self._tokenizer = auto_tokenizer.from_pretrained(
            local, local_files_only=True, trust_remote_code=False, padding_side="left"
        )
        self._model = auto_causal.from_pretrained(
            local,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self._model.eval()
        self._yes_id = self._single_token("yes")
        self._no_id = self._single_token("no")

    def _single_token(self, value: str) -> int:
        ids = self._tokenizer.encode(value, add_special_tokens=False)
        if len(ids) != 1:
            raise ModelManifestError(
                "MODEL_TOKENIZER_INCOMPATIBLE", "reranker label is not one tokenizer token"
            )
        return int(ids[0])

    def _prompt(self, query: str, document: str) -> str:
        user = f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {document}"
        return (
            f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        prompts = tuple(self._prompt(query, document) for document in documents)
        scores: list[float] = []
        for start in range(0, len(prompts), self._batch_size):
            batch = prompts[start : start + self._batch_size]
            inputs = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self._maximum_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                logits = self._model(**inputs, logits_to_keep=1).logits[
                    :, -1, [self._no_id, self._yes_id]
                ]
                probabilities = self._torch.softmax(logits.float(), dim=-1)[:, 1]
            scores.extend(probabilities.cpu().tolist())
        return scores


class Qwen3GeneratorBackend:
    """Deterministic non-thinking local Qwen3 answer generator."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        model_id: str,
        model_revision: str,
        device: str = "cuda",
        maximum_input_tokens: int = 16_384,
        maximum_new_tokens: int = 512,
    ) -> None:
        torch, _, auto_causal, auto_tokenizer = _dependencies()
        local = _local_directory(checkpoint)
        self.model_id = model_id
        self.model_revision = model_revision
        self._torch = torch
        self._device = device
        self._maximum_input_tokens = maximum_input_tokens
        self._maximum_new_tokens = maximum_new_tokens
        self._tokenizer = auto_tokenizer.from_pretrained(
            local, local_files_only=True, trust_remote_code=False
        )
        self._model = auto_causal.from_pretrained(
            local,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self._model.eval()

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str:
        if not evidence:
            raise ValueError("generator requires at least one frozen evidence passage")
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
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self._maximum_input_tokens,
        ).to(self._device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self._maximum_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(continuation, skip_special_tokens=True)).strip()


class Qwen3AdapterGeneratorBackend(Qwen3GeneratorBackend):
    """Deterministic Qwen3 generation with one immutable local PEFT adapter."""

    def __init__(
        self,
        checkpoint: Path,
        adapter_checkpoint: Path,
        *,
        model_id: str,
        model_revision: str,
        adapter_id: str,
        device: str = "cuda",
        maximum_input_tokens: int = 16_384,
        maximum_new_tokens: int = 512,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise ModelManifestError(
                "MODEL_DEPENDENCY_MISSING",
                "install the locked model-training optional dependencies",
            ) from error
        local = _local_directory(checkpoint)
        adapter = _local_adapter_directory(adapter_checkpoint)
        adapter_path = Path(adapter)
        self.model_id = model_id
        self.model_revision = model_revision
        self.adapter_id = adapter_id
        self.adapter_checksum = checksum_file(adapter_path / "adapter_model.safetensors")
        self.adapter_config_checksum = checksum_file(adapter_path / "adapter_config.json")
        self._torch = torch
        self._device = device
        self._maximum_input_tokens = maximum_input_tokens
        self._maximum_new_tokens = maximum_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(
            local, local_files_only=True, trust_remote_code=False
        )
        bits_config_type: Any = BitsAndBytesConfig
        quantization = bits_config_type(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            local,
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map={"": 0},
        )
        self._model = PeftModel.from_pretrained(
            base,
            adapter,
            is_trainable=False,
            local_files_only=True,
        )
        self._model.eval()


__all__ = [
    "Qwen3AdapterGeneratorBackend",
    "Qwen3EmbeddingBackend",
    "Qwen3GeneratorBackend",
    "Qwen3RerankerBackend",
    "_last_token_pool",
    "_local_adapter_directory",
]
