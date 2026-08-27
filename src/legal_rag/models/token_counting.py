"""Local tokenizer-only input accounting for frozen Qwen3 generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_rag.models.huggingface_local import _local_directory


class Qwen3InputTokenCounter:
    """Count the exact non-thinking chat input tokens without loading model weights."""

    def __init__(self, tokenizer: Any, *, system_prompt: str) -> None:
        self._tokenizer = tokenizer
        self._system_prompt = system_prompt

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, *, system_prompt: str) -> Qwen3InputTokenCounter:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("locked transformers dependency is unavailable") from error
        tokenizer = AutoTokenizer.from_pretrained(
            _local_directory(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
        )
        return cls(tokenizer, system_prompt=system_prompt)

    def __call__(self, question: str, evidence: tuple[str, ...]) -> int:
        evidence_text = "\n\n".join(
            f"[EVIDENCE {index}]\n{text}" for index, text in enumerate(evidence, start=1)
        )
        messages: Sequence[dict[str, str]] = (
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Câu hỏi:\n{question}\n\nCăn cứ được cung cấp:\n{evidence_text}",
            },
        )
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = self._tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return len(token_ids)


__all__ = ["Qwen3InputTokenCounter"]
