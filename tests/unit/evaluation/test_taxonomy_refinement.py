from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.taxonomy_refinement import (
    TaxonomyRefinementError,
    build_taxonomy_refinement,
)


def _question(question_id: str, question: str, answer: str) -> dict[str, object]:
    return {
        "schema_version": "internal.question.v1",
        "question_id": question_id,
        "original_id": question_id,
        "original_id_kind": "object_key_string",
        "source_position": int(question_id[1:]),
        "source_artifact": "fixture.json",
        "source_checksum": "sha256:" + "a" * 64,
        "question": question,
        "answer": answer,
        "answer_state": "gold",
    }


def test_refinement_separates_reference_numbers_from_semantic_numeric_signals() -> None:
    rows = [
        _question("q0", "Điều 50 Nghị định 90/2017/NĐ-CP quy định gì?", "Tham chiếu."),
        _question("q1", "Mức phạt là bao nhiêu phần trăm?", "Mức phạt là 15%."),
        _question("q2", "Thời hạn là bao lâu?", "Thời hạn là 30 ngày."),
    ]
    questions = b"".join(content_json_bytes(row) for row in rows)

    first = build_taxonomy_refinement(
        questions_data=questions,
        train_question_ids=("q0", "q1", "q2"),
        expected_questions_checksum=checksum_bytes(questions),
    )
    second = build_taxonomy_refinement(
        questions_data=questions,
        train_question_ids=("q0", "q1", "q2"),
        expected_questions_checksum=checksum_bytes(questions),
    )

    assert first == second
    assert first["schema_version"] == "evaluation.d064-taxonomy-refinement.v2"
    assert first["training_labels"] is False
    assert first["row_level_output"] is False
    assert first["question_signals"]["reference_number_only"] == 1
    assert first["question_signals"]["semantic_percentage"] == 1
    assert first["answer_signals"]["semantic_duration"] == 1
    assert "question_id" not in json.dumps(first)


def test_refinement_rejects_unknown_or_duplicate_train_identity() -> None:
    questions = content_json_bytes(_question("q0", "Câu hỏi?", "Câu trả lời."))

    with pytest.raises(TaxonomyRefinementError) as captured:
        build_taxonomy_refinement(
            questions_data=questions,
            train_question_ids=("q0", "dev-row"),
            expected_questions_checksum=checksum_bytes(questions),
        )

    assert captured.value.code == "D064_REFINEMENT_NON_TRAIN_INPUT"
