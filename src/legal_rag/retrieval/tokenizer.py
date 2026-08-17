"""Generator-independent Unicode tokenizer used by chunking and BM25."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

RETRIEVAL_TOKENIZER_ID = "legal-retrieval-unicode-v1"
RETRIEVAL_TOKENIZER_REVISION = f"unicode-{unicodedata.unidata_version}"


@dataclass(frozen=True, slots=True)
class RetrievalToken:
    value: str
    canonical_start: int
    canonical_end: int


def retrieval_tokens(text: str) -> tuple[RetrievalToken, ...]:
    """Tokenize NFC text while retaining canonical code-point offsets."""

    canonical = unicodedata.normalize("NFC", text)
    tokens: list[RetrievalToken] = []
    run_start: int | None = None

    def finish_run(end: int) -> None:
        nonlocal run_start
        if run_start is not None:
            tokens.append(RetrievalToken(canonical[run_start:end], run_start, end))
            run_start = None

    for index, character in enumerate(canonical):
        category_group = unicodedata.category(character)[0]
        if category_group in {"L", "M", "N"}:
            if run_start is None:
                run_start = index
            continue
        finish_run(index)
        if category_group in {"P", "S"}:
            tokens.append(RetrievalToken(character, index, index + 1))
    finish_run(len(canonical))
    return tuple(tokens)


def retrieval_token_values(text: str) -> tuple[str, ...]:
    return tuple(token.value for token in retrieval_tokens(text))
