"""Deterministic fixed-refusal baseline contracts for MIL-003."""

from __future__ import annotations

import json

from legal_rag.domain.checksums import FileSetChecksum, checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.baseline import (
    build_development_inputs,
    build_fixed_refusal_run,
)
from legal_rag.generation.fixture import FIXED_REFUSAL


def _question(question_id: str, answer: str | None) -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": question_id,
            "original_id": question_id,
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixture/questions.json",
            "source_checksum": "sha256:" + "a" * 64,
            "question": f"Question {question_id}",
            "answer": answer,
            "answer_state": "gold" if answer is not None else "unlabeled",
        }
    )


def _source_tree() -> FileSetChecksum:
    return FileSetChecksum(
        checksum="sha256:" + "b" * 64,
        paths=("src/legal_rag/evaluation/baseline.py",),
    )


def test_fixed_refusal_run_has_no_evidence_and_is_byte_deterministic() -> None:
    questions = (_question("q1", "Gold one"), _question("q2", "Gold two"))
    question_bytes = b"synthetic-question-artifact\n"

    first = build_fixed_refusal_run(
        questions,
        question_bytes=question_bytes,
        split_checksum="sha256:" + "c" * 64,
        source_tree=_source_tree(),
    )
    second = build_fixed_refusal_run(
        questions,
        question_bytes=question_bytes,
        split_checksum="sha256:" + "c" * 64,
        source_tree=_source_tree(),
    )

    assert first.artifacts == second.artifacts
    assert first.run_id == second.run_id
    assert tuple(answer.answer for answer in first.answers) == (FIXED_REFUSAL, FIXED_REFUSAL)
    assert all(answer.evidence_ids == () for answer in first.answers)
    assert b"Gold one" not in first.artifacts["answers.jsonl"]
    manifest = json.loads(first.artifacts["run.manifest.json"])
    assert manifest["question_checksum"] == checksum_bytes(question_bytes)
    assert manifest["corpus_checksum"] == checksum_bytes(b"")
    assert manifest["index_checksum"] is None
    assert manifest["model_id"] is None


def test_material_question_input_changes_run_identity() -> None:
    questions = (_question("q1", "Gold"),)
    first = build_fixed_refusal_run(
        questions,
        question_bytes=b"first\n",
        split_checksum="sha256:" + "c" * 64,
        source_tree=_source_tree(),
    )
    changed = build_fixed_refusal_run(
        questions,
        question_bytes=b"second\n",
        split_checksum="sha256:" + "c" * 64,
        source_tree=_source_tree(),
    )

    assert first.run_id != changed.run_id


def test_development_inputs_preserve_order_and_hide_question_text() -> None:
    questions = (_question("q2", "Gold two"), _question("q1", "Gold one"))
    run = build_fixed_refusal_run(
        questions,
        question_bytes=b"questions\n",
        split_checksum="sha256:" + "c" * 64,
        source_tree=_source_tree(),
    )

    predictions, references = build_development_inputs(questions, run.answers)

    assert tuple(json.loads(predictions)) == ("q2", "q1")
    assert tuple(json.loads(references)) == ("q2", "q1")
    assert json.loads(predictions)["q2"] == {"answer": FIXED_REFUSAL}
    assert json.loads(references)["q1"] == "Gold one"
    assert b"Question q1" not in predictions
