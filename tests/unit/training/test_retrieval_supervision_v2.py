from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.training.retrieval_supervision import (
    RetrievalSupervisionError,
    build_retrieval_supervision,
)


def _question(question_id: str, answer: str, position: int) -> dict[str, object]:
    return {
        "schema_version": "internal.question.v1",
        "question_id": question_id,
        "original_id": question_id,
        "original_id_kind": "object_key_string",
        "source_position": position,
        "source_artifact": "fixture.json",
        "source_checksum": "sha256:" + "a" * 64,
        "question": f"Câu hỏi {question_id}?",
        "answer": answer,
        "answer_state": "gold",
    }


def _chunk(
    chunk_id: str,
    context_id: str,
    path: list[str],
    kind: str,
    ordinal: str,
) -> dict[str, object]:
    return {
        "schema_version": "retrieval.chunk.v1",
        "chunk_id": chunk_id,
        "context_id": context_id,
        "source_url": f"https://example.test/{context_id}",
        "hierarchy_path": path,
        "hierarchy_rule_id": "fixture.v1",
        "hierarchy_kind": kind,
        "hierarchy_ordinal": ordinal,
        "canonical_start": 0,
        "canonical_end": 10,
        "display_text": "Nội dung luật.",
        "retrieval_text": "Nội dung luật.",
        "window_index": 0,
        "chunk_checksum": checksum_bytes(chunk_id.encode()),
        "context_checksum": checksum_bytes(context_id.encode()),
    }


def _alias(number: str, context_id: str) -> dict[str, object]:
    from legal_rag.retrieval.exact import document_number_key

    return {
        "canonical_end": None,
        "canonical_start": None,
        "context_id": context_id,
        "document_number": number,
        "document_number_key": document_number_key(number),
        "review_state": "approved",
        "schema_version": "legal.reference.alias.v1",
        "source_kind": "owner_override",
    }


def _context(context_id: str, name: str) -> dict[str, object]:
    return {
        "schema_version": "internal.context.v1",
        "context_id": context_id,
        "original_id": context_id,
        "original_id_kind": "json_integer",
        "source_position": int(context_id),
        "source_artifact": f"context_{context_id}.json",
        "source_checksum": checksum_bytes(context_id.encode()),
        "name": name,
        "source_url": f"https://example.test/{context_id}",
        "passage": "Nội dung luật.",
        "indexable": True,
        "quarantine_reason": None,
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(content_json_bytes(row) for row in rows)


def _fixture() -> dict[str, object]:
    question_rows = [
        _question("q0", "Theo điểm a khoản 2 Điều 3 Nghị định 10/2020/NĐ-CP.", 0),
        _question("q1", "Theo khoản 2 Điều 3 Nghị định 10/2020/NĐ-CP.", 1),
        _question("q2", "Theo Điều 3 Nghị định 10/2020/NĐ-CP.", 2),
        _question("q3", "Theo Nghị định 11/2020/NĐ-CP.", 3),
        _question("q4", "Theo Điều 3 Nghị định 12/2020/NĐ-CP.", 4),
        _question("q5", "Không có căn cứ được nhận diện.", 5),
        _question("dev", "Theo Điều 3 Nghị định 10/2020/NĐ-CP.", 6),
    ]
    chunks = [
        _chunk("c-point", "1", ["Điều 3", "Khoản 2", "Điểm a"], "point", "a"),
        _chunk("c-point-b", "1", ["Điều 3", "Khoản 2", "Điểm b"], "point", "b"),
        _chunk("c-article", "1", ["Điều 4"], "article", "4"),
        _chunk("c-doc", "2", ["Điều 1"], "article", "1"),
    ]
    aliases = [
        _alias("10/2020/NĐ-CP", "1"),
        _alias("11/2020/NĐ-CP", "2"),
        _alias("12/2020/NĐ-CP", "1"),
        _alias("12/2020/NĐ-CP", "2"),
    ]
    historical = [
        {
            "evidence_checksums": [checksum_bytes(b"c-point")],
            "evidence_ids": ["c-point"],
            "question_checksum": checksum_bytes("Câu hỏi q0?".encode()),
            "question_id": "q0",
            "schema_version": "training.evidence.selection.v1",
            "support_policy_version": "historical.v1",
            "support_score": 0.99,
        }
    ]
    return {
        "questions_data": _jsonl(question_rows),
        "train_question_ids": tuple(f"q{i}" for i in range(6)),
        "chunks_data": _jsonl(chunks),
        "contexts_data": _jsonl(
            [
                _context("1", "Nghi-dinh-10-2020-ND-CP"),
                _context("2", "Nghi-dinh-11-2020-ND-CP"),
            ]
        ),
        "aliases_data": _jsonl(aliases),
        "historical_data": _jsonl(historical),
    }


def _build(**overrides: object):
    fixture = _fixture()
    fixture.update(overrides)
    checksums = {
        name.removesuffix("_data"): checksum_bytes(value)
        for name, value in fixture.items()
        if name.endswith("_data") and isinstance(value, bytes)
    }
    return build_retrieval_supervision(
        **fixture,
        expected_input_checksums=checksums,
        expected_train_count=6,
        expected_chunk_count=4,
    )


def test_supervision_classes_multichunk_ambiguity_and_unresolved_fail_closed() -> None:
    artifacts = _build()
    groups = [json.loads(line) for line in artifacts.groups_data.splitlines()]
    by_id = {row["question_id"]: row for row in groups}

    assert [row["question_id"] for row in groups] == sorted(
        (row["question_id"] for row in groups), key=lambda value: value.encode()
    )
    assert by_id["q0"]["mapping_class"] == "EXACT_DOC_ARTICLE_POINT"
    assert by_id["q0"]["canonical_chunk_ids"] == ["c-point"]
    assert by_id["q0"]["historical_v1_overlap"] == {
        "mapping_present": True,
        "positive_set_relation": "EXACT",
        "reproducibly_identifiable": True,
    }
    assert by_id["q1"]["mapping_class"] == "SAME_COORDINATE_MULTICHUNK"
    assert by_id["q1"]["canonical_chunk_ids"] == ["c-point", "c-point-b"]
    assert by_id["q2"]["mapping_class"] == "SAME_COORDINATE_MULTICHUNK"
    assert by_id["q3"]["mapping_class"] == "DOCUMENT_ONLY"
    assert by_id["q3"]["canonical_chunk_ids"] == []
    assert by_id["q4"]["mapping_class"] == "AMBIGUOUS"
    assert by_id["q4"]["canonical_chunk_ids"] == []
    assert by_id["q5"]["mapping_class"] == "UNRESOLVED"
    assert by_id["q5"]["canonical_chunk_ids"] == []


def test_supervision_is_replay_identical_and_historical_rows_are_identifiable() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert checksum_bytes(first.groups_data) == first.report["artifact_checksums"]["groups"]
    assert first.report["historical_v1"]["mapping_count"] == 1
    assert first.report["historical_v1"]["reproducibly_identifiable"] == 1
    assert first.report["historical_v1"]["exact_positive_set_overlap"] == 1
    assert first.report["eligibility_policy"]["minimum_reranker_score"] is None
    assert first.report["eligibility_policy"]["minimum_answer_token_coverage"] is None


def test_supervision_rejects_non_train_identity_and_count_drift() -> None:
    fixture = _fixture()
    with pytest.raises(RetrievalSupervisionError) as captured:
        _build(train_question_ids=(*fixture["train_question_ids"], "dev"))

    assert captured.value.code == "D065_TRAIN_PARTITION_INVALID"


def test_supervision_rejects_stale_input_checksum() -> None:
    fixture = _fixture()
    checksums = {
        name.removesuffix("_data"): checksum_bytes(value)
        for name, value in fixture.items()
        if name.endswith("_data") and isinstance(value, bytes)
    }
    checksums["chunks"] = "sha256:" + "0" * 64

    with pytest.raises(RetrievalSupervisionError) as captured:
        build_retrieval_supervision(
            **fixture,
            expected_input_checksums=checksums,
            expected_train_count=6,
            expected_chunk_count=4,
        )

    assert captured.value.code == "D065_INPUT_CHECKSUM_MISMATCH"
