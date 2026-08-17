from __future__ import annotations

import unicodedata

from legal_rag.retrieval.tokenizer import (
    RETRIEVAL_TOKENIZER_ID,
    RETRIEVAL_TOKENIZER_REVISION,
    retrieval_token_values,
    retrieval_tokens,
)


def test_retrieval_tokenizer_groups_letters_marks_numbers_and_emits_symbols() -> None:
    text = "Điều 12A: mức-phạt 1.000₫ ✅"

    tokens = retrieval_tokens(text)

    assert [token.value for token in tokens] == [
        "Điều",
        "12A",
        ":",
        "mức",
        "-",
        "phạt",
        "1",
        ".",
        "000",
        "₫",
        "✅",
    ]
    assert [text[token.canonical_start : token.canonical_end] for token in tokens] == [
        token.value for token in tokens
    ]


def test_retrieval_tokenizer_normalizes_nfc_before_tokenization() -> None:
    decomposed = unicodedata.normalize("NFD", "Điều kiện")

    assert retrieval_token_values(decomposed) == ("Điều", "kiện")


def test_retrieval_tokenizer_exposes_pinned_identity() -> None:
    assert RETRIEVAL_TOKENIZER_ID == "legal-retrieval-unicode-v1"
    assert RETRIEVAL_TOKENIZER_REVISION.startswith("unicode-")
