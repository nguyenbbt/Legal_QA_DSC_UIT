"""Bounded aggregate D-064 taxonomy refinement with citation-aware numerics."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, NoReturn

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.retrieval.legal_citations import (
    mask_legal_reference_numbers,
    parse_legal_citations,
)
from legal_rag.training.rag_sft import RagSftBuildError, load_gold_questions

_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_PERCENTAGE = re.compile(r"(?:\d+(?:[.,]\d+)?\s*%|\bphần\s+trăm\b)", re.I)
_DATE = re.compile(
    r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\b)",
    re.I,
)
_CURRENCY = re.compile(
    r"(?:\b\d+(?:[.,]\d+)*\s*(?:đồng|vnđ|vnd|triệu|tỷ)\b|"
    r"\b(?:đồng|vnđ|vnd)\b)",
    re.I,
)
_DURATION = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|giờ|phút|tuần|buổi)\b", re.I)


class TaxonomyRefinementError(Exception):
    """Stable fail-closed D-064 refinement error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise TaxonomyRefinementError(code, message)


def _signals(text: str) -> tuple[str, ...]:
    masked = mask_legal_reference_numbers(text)
    has_reference = bool(parse_legal_citations(text))
    has_original_number = bool(_NUMBER.search(text))
    semantic_number = bool(_NUMBER.search(masked))
    signals: list[str] = []
    if has_reference:
        signals.append("legal_reference")
    if has_reference and has_original_number and not semantic_number:
        signals.append("reference_number_only")
    if semantic_number:
        signals.append("semantic_numeric")
    if _PERCENTAGE.search(masked):
        signals.append("semantic_percentage")
    if _DATE.search(masked):
        signals.append("semantic_date")
    if _CURRENCY.search(masked):
        signals.append("semantic_currency")
    if _DURATION.search(masked):
        signals.append("semantic_duration")
    return tuple(signals)


def _ordered_counts(values: Counter[str]) -> dict[str, int]:
    names = (
        "legal_reference",
        "reference_number_only",
        "semantic_numeric",
        "semantic_percentage",
        "semantic_date",
        "semantic_currency",
        "semantic_duration",
    )
    return {name: values[name] for name in names}


def build_taxonomy_refinement(
    *,
    questions_data: bytes,
    train_question_ids: Sequence[str],
    expected_questions_checksum: str,
) -> dict[str, Any]:
    """Return train-only aggregate signals; never expose row-level labels or text."""

    actual_checksum = checksum_bytes(questions_data)
    if actual_checksum != expected_questions_checksum:
        _fail("D064_REFINEMENT_INPUT_CHECKSUM_MISMATCH", "question checksum is stale")
    try:
        questions = load_gold_questions(questions_data)
    except RagSftBuildError as error:
        raise TaxonomyRefinementError(
            "D064_REFINEMENT_SOURCE_INVALID", "official questions are invalid"
        ) from error
    train_ids = tuple(train_question_ids)
    if len(train_ids) != len(set(train_ids)):
        _fail("D064_REFINEMENT_NON_TRAIN_INPUT", "train identities are not unique")
    by_id = {question.question_id: question for question in questions}
    if any(question_id not in by_id for question_id in train_ids):
        _fail("D064_REFINEMENT_NON_TRAIN_INPUT", "train identity is absent from source")

    question_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for question_id in train_ids:
        record = by_id[question_id]
        question_counts.update(_signals(record.question))
        assert record.answer is not None
        answer_counts.update(_signals(record.answer))
    return {
        "schema_version": "evaluation.d064-taxonomy-refinement.v2",
        "analysis_version": "citation-aware-numeric-taxonomy.v1",
        "source_questions_checksum": actual_checksum,
        "train_fit_count": len(train_ids),
        "excluded_non_train_count": len(questions) - len(train_ids),
        "question_signals": _ordered_counts(question_counts),
        "answer_signals": _ordered_counts(answer_counts),
        "training_labels": False,
        "row_level_output": False,
        "tuning_performed": False,
        "generated_text_used": False,
        "execution_mode": "local-offline",
    }


__all__ = [
    "TaxonomyRefinementError",
    "build_taxonomy_refinement",
]
