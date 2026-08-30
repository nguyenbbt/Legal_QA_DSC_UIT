from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import SplitQuestion, build_split_manifest
from legal_rag.evaluation.train_forensics import (
    TrainForensicsError,
    analyze_train_forensics,
    analyze_train_forensics_paths,
)


def _question(question_id: str, question: str, answer: str, position: int) -> dict[str, object]:
    return {
        "schema_version": "internal.question.v1",
        "question_id": question_id,
        "original_id": question_id,
        "original_id_kind": "object_key_string",
        "source_position": position,
        "source_artifact": "train-source",
        "source_checksum": "sha256:" + "a" * 64,
        "question": question,
        "answer": answer,
        "answer_state": "gold",
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(content_json_bytes(row) for row in rows)


def _chunk(chunk_id: str, text: str, checksum: str) -> dict[str, object]:
    return {
        "canonical_end": len(text),
        "canonical_start": 0,
        "chunk_checksum": checksum,
        "chunk_id": chunk_id,
        "context_checksum": "sha256:" + "d" * 64,
        "context_id": "1",
        "display_text": text,
        "hierarchy_kind": "article",
        "hierarchy_ordinal": "1",
        "hierarchy_path": ["Điều 1"],
        "hierarchy_rule_id": "fixture.v1",
        "retrieval_text": text,
        "schema_version": "retrieval.chunk.v1",
        "source_url": "https://example.test/law",
        "window_index": 0,
    }


def _selection(
    question_id: str, question: str, chunk_id: str, chunk_checksum: str
) -> dict[str, object]:
    return {
        "evidence_checksums": [chunk_checksum],
        "evidence_ids": [chunk_id],
        "question_checksum": checksum_bytes(question.encode()),
        "question_id": question_id,
        "schema_version": "training.evidence.selection.v1",
        "support_policy_version": "fixture.v1",
        "support_score": 1.0,
    }


def _fixture() -> tuple[bytes, bytes, bytes, bytes, tuple[str, ...]]:
    rows = [
        _question(
            "q1",
            "Điều kiện và thời hạn nộp hồ sơ là bao lâu?",
            "Căn cứ Điều 1, thời hạn là 30 ngày.",
            0,
        ),
        _question(
            "q2",
            "Cơ quan nào có thẩm quyền và phải làm gì?",
            "1. Bộ A quyết định.\n2. Cơ quan B thực hiện.",
            1,
        ),
        _question("q3", "Ngoại lệ áp dụng khi nào?", "Trừ trường hợp đặc biệt.", 2),
    ]
    question_data = _jsonl(rows)
    questions = tuple(SplitQuestion(str(row["question_id"]), str(row["question"])) for row in rows)
    manifest = build_split_manifest(
        questions,
        (),
        source_checksum=checksum_bytes(question_data),
        public_source_checksum="sha256:" + "b" * 64,
    )
    train_ids = tuple(row.question_id for row in manifest.rows if row.split == "train")
    assert train_ids
    chunk_checksum = "sha256:" + "c" * 64
    chunk_data = _jsonl([_chunk("chunk-1", "Căn cứ Điều 1, thời hạn là 30 ngày.", chunk_checksum)])
    question_by_id = {str(row["question_id"]): str(row["question"]) for row in rows}
    selection_data = _jsonl(
        [_selection(train_ids[0], question_by_id[train_ids[0]], "chunk-1", chunk_checksum)]
    )
    return question_data, manifest.json_bytes(), chunk_data, selection_data, train_ids


def test_train_forensics_is_train_only_aggregate_and_deterministic() -> None:
    question_data, split_data, chunk_data, selection_data, train_ids = _fixture()

    first = analyze_train_forensics(
        questions_data=question_data,
        split_data=split_data,
        chunks_data=chunk_data,
        selections_data=selection_data,
        expected_input_checksums={
            "questions": checksum_bytes(question_data),
            "split": checksum_bytes(split_data),
            "chunks": checksum_bytes(chunk_data),
            "selections": checksum_bytes(selection_data),
        },
    )
    second = analyze_train_forensics(
        questions_data=question_data,
        split_data=split_data,
        chunks_data=chunk_data,
        selections_data=selection_data,
        expected_input_checksums=first["input_checksums"],
    )

    assert first == second
    assert first["train_fit_count"] == len(train_ids)
    assert first["excluded_non_train_count"] == 3 - len(train_ids)
    assert "question_ids" not in first
    assert json.dumps(first, ensure_ascii=False).find("Bộ A quyết định") == -1
    assert sum(first["question_primary_types"].values()) == len(train_ids)
    assert first["tuning_performed"] is False
    assert first["generated_text_used"] is False


def test_train_forensics_characterizes_legal_answer_patterns_and_overlap() -> None:
    question_data, split_data, chunk_data, selection_data, train_ids = _fixture()

    report = analyze_train_forensics(
        questions_data=question_data,
        split_data=split_data,
        chunks_data=chunk_data,
        selections_data=selection_data,
        expected_input_checksums={
            "questions": checksum_bytes(question_data),
            "split": checksum_bytes(split_data),
            "chunks": checksum_bytes(chunk_data),
            "selections": checksum_bytes(selection_data),
        },
    )

    assert report["answer_patterns"]["citation_or_legal_coordinate"] >= 0
    assert report["answer_patterns"]["number"] >= 0
    assert report["answer_patterns"]["list_or_enumeration"] >= 0
    assert report["answer_token_lengths"]["minimum"] >= 1
    assert report["answer_sentence_counts"]["minimum"] >= 1
    assert report["mapped_evidence"]["selection_count"] == 1
    assert sum(report["potential_answer_classes"].values()) == len(train_ids)


def test_train_forensics_rejects_a_stale_input_checksum() -> None:
    question_data, split_data, chunk_data, selection_data, _ = _fixture()

    with pytest.raises(TrainForensicsError) as captured:
        analyze_train_forensics(
            questions_data=question_data,
            split_data=split_data,
            chunks_data=chunk_data,
            selections_data=selection_data,
            expected_input_checksums={
                "questions": "sha256:" + "0" * 64,
                "split": checksum_bytes(split_data),
                "chunks": checksum_bytes(chunk_data),
                "selections": checksum_bytes(selection_data),
            },
        )

    assert captured.value.code == "D064_INPUT_CHECKSUM_MISMATCH"


def test_train_forensics_rejects_non_train_mapped_evidence() -> None:
    question_data, split_data, chunk_data, selection_data, train_ids = _fixture()
    non_train_id = next(
        question_id for question_id in ("q1", "q2", "q3") if question_id not in train_ids
    )
    selection = json.loads(selection_data)
    selection["question_id"] = non_train_id
    invalid_selection_data = content_json_bytes(selection)

    with pytest.raises(TrainForensicsError) as captured:
        analyze_train_forensics(
            questions_data=question_data,
            split_data=split_data,
            chunks_data=chunk_data,
            selections_data=invalid_selection_data,
            expected_input_checksums={
                "questions": checksum_bytes(question_data),
                "split": checksum_bytes(split_data),
                "chunks": checksum_bytes(chunk_data),
                "selections": checksum_bytes(invalid_selection_data),
            },
        )

    assert captured.value.code == "D064_NON_TRAIN_SELECTION"


def test_streaming_path_analysis_matches_in_memory_analysis(tmp_path) -> None:
    question_data, split_data, chunk_data, selection_data, _ = _fixture()
    paths = {}
    for name, data in {
        "questions": question_data,
        "split": split_data,
        "chunks": chunk_data,
        "selections": selection_data,
    }.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(data)
        paths[name] = path
    expected = {name: checksum_bytes(path.read_bytes()) for name, path in paths.items()}

    streamed = analyze_train_forensics_paths(
        questions_path=paths["questions"],
        split_path=paths["split"],
        chunks_path=paths["chunks"],
        selections_path=paths["selections"],
        expected_input_checksums=expected,
    )
    in_memory = analyze_train_forensics(
        questions_data=question_data,
        split_data=split_data,
        chunks_data=chunk_data,
        selections_data=selection_data,
        expected_input_checksums=expected,
    )

    assert streamed == in_memory


def test_train_forensics_rejects_a_stale_selection_question_checksum() -> None:
    question_data, split_data, chunk_data, selection_data, _ = _fixture()
    selection = json.loads(selection_data)
    selection["question_checksum"] = "sha256:" + "0" * 64
    invalid_selection_data = content_json_bytes(selection)

    with pytest.raises(TrainForensicsError) as captured:
        analyze_train_forensics(
            questions_data=question_data,
            split_data=split_data,
            chunks_data=chunk_data,
            selections_data=invalid_selection_data,
            expected_input_checksums={
                "questions": checksum_bytes(question_data),
                "split": checksum_bytes(split_data),
                "chunks": checksum_bytes(chunk_data),
                "selections": checksum_bytes(invalid_selection_data),
            },
        )

    assert captured.value.code == "D064_SELECTION_QUESTION_MISMATCH"
