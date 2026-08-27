"""Shared exact Qwen3 reranker prompt used by fitting and inference."""

from __future__ import annotations

QWEN3_RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be yes or no."
)


def build_qwen3_reranker_prompt(*, instruction: str, query: str, document: str) -> str:
    """Render the pinned non-thinking yes/no prompt without a label token."""

    if not instruction.strip() or not query.strip() or not document.strip():
        raise ValueError("reranker prompt fields must be non-empty")
    user = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return (
        f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


__all__ = ["QWEN3_RERANKER_SYSTEM", "build_qwen3_reranker_prompt"]
