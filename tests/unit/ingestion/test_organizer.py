from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.ingestion.organizer import (
    OrganizerContextReader,
    OrganizerDataError,
    OrganizerFile,
    OrganizerQuestionReader,
    discover_context_files,
)


def test_train_questions_preserve_order_ids_checksums_and_normalize_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Điều kiện")
    source = json.dumps(
        {
            "0007": {"question": decomposed, "answer": "Đủ điều kiện."},
            "9": {"question": "Ai được cấp thẻ?", "answer": "Người đủ 18 tuổi."},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    result = OrganizerQuestionReader().read_bytes(
        source,
        kind="train",
        artifact_path="fixtures/train.json",
    )

    assert [record.question_id for record in result.records] == ["0007", "9"]
    assert [record.source_position for record in result.records] == [0, 1]
    assert result.records[0].question == "Điều kiện"
    assert result.records[0].source_checksum == checksum_bytes(source)
    assert result.records[0].source_artifact == "fixtures/train.json"
    assert result.records[0].answer_state == "gold"
    assert result.warnings == ()
    assert result.jsonl_bytes().endswith(b"\n")
    assert not result.jsonl_bytes().endswith(b"\n\n")


def test_public_questions_accept_bom_with_warning_and_require_null_answer() -> None:
    source = b"\xef\xbb\xbf" + '{"01":{"question":"Câu hỏi?","answer":null}}'.encode()

    result = OrganizerQuestionReader().read_bytes(
        source,
        kind="public",
        artifact_path="fixtures/public.json",
    )

    assert result.records[0].answer is None
    assert result.records[0].answer_state == "unlabeled"
    assert [warning.code for warning in result.warnings] == ["DATA_UTF8_BOM_ACCEPTED"]
    assert result.records[0].source_checksum == checksum_bytes(source)


@pytest.mark.parametrize(
    ("source", "kind", "code", "json_path"),
    [
        (
            b'{"1":{"question":"a","answer":"b"},"1":{"question":"c","answer":"d"}}',
            "train",
            "DATA_DUPLICATE_KEY",
            '$["1"]',
        ),
        (
            b'{"1":{"question":"a","question":"b","answer":"c"}}',
            "train",
            "DATA_DUPLICATE_KEY",
            '$["1"].question',
        ),
        (b"[]", "train", "DATA_ROOT_TYPE", "$"),
        (b'{"1":{"question":"a"}}', "public", "DATA_FIELD_MISSING", '$["1"].answer'),
        (b'{"1":{"question":4,"answer":null}}', "public", "DATA_QUESTION_TYPE", '$["1"].question'),
        (
            b'{"1":{"question":"a","answer":null,"extra":1}}',
            "public",
            "DATA_FIELD_UNKNOWN",
            '$["1"].extra',
        ),
        (b'{"1":{"question":"a","answer":null}}', "train", "DATA_ANSWER_TYPE", '$["1"].answer'),
        (b'{"1":{"question":"a","answer":"b"}}', "public", "DATA_ANSWER_TYPE", '$["1"].answer'),
    ],
)
def test_question_reader_rejects_invalid_input_atomically(
    source: bytes,
    kind: str,
    code: str,
    json_path: str,
) -> None:
    with pytest.raises(OrganizerDataError) as captured:
        OrganizerQuestionReader().read_bytes(
            source,
            kind=kind,  # type: ignore[arg-type]
            artifact_path="fixtures/questions.json",
        )

    assert captured.value.code == code
    assert captured.value.artifact_path == "fixtures/questions.json"
    assert captured.value.json_path == json_path


def test_question_reader_rejects_invalid_utf8() -> None:
    with pytest.raises(OrganizerDataError, match="strict UTF-8") as captured:
        OrganizerQuestionReader().read_bytes(
            b'{"1":{"question":"\xff","answer":null}}',
            kind="public",
            artifact_path="fixtures/public.json",
        )

    assert captured.value.code == "DATA_UTF8_INVALID"


def test_context_reader_sorts_numeric_ids_and_quarantines_empty_passage() -> None:
    files = (
        OrganizerFile(
            "contexts/context_10.json",
            b'{"id":10,"link":"https://example.invalid/10","passage":""}',
        ),
        OrganizerFile(
            "contexts/context_2.json",
            '{"id":2,"name":"Luật mẫu","link":"https://example.invalid/2",'
            '"passage":"Điều 1. Nội dung."}'.encode(),
        ),
    )

    result = OrganizerContextReader().read_files(tuple(reversed(files)))

    assert [record.context_id for record in result.records] == ["2", "10"]
    assert [record.source_position for record in result.records] == [0, 1]
    assert result.records[0].indexable is True
    assert result.records[0].quarantine_reason is None
    assert result.records[1].indexable is False
    assert result.records[1].quarantine_reason == "EMPTY_PASSAGE"
    manifest = json.loads(result.manifest_bytes())
    assert [entry["context_id"] for entry in manifest["entries"]] == ["2", "10"]
    assert manifest["entries"][1]["quarantine_reason"] == "EMPTY_PASSAGE"


def test_context_reader_is_independent_of_enumeration_order() -> None:
    files = (
        OrganizerFile(
            "contexts/context_11.json",
            '{"id":11,"link":"https://example.invalid/11","passage":"Điều 2."}'.encode(),
        ),
        OrganizerFile(
            "contexts/context_3.json",
            '{"id":3,"link":"https://example.invalid/3","passage":"Điều 1."}'.encode(),
        ),
    )

    first = OrganizerContextReader().read_files(files)
    second = OrganizerContextReader().read_files(tuple(reversed(files)))

    assert first.jsonl_bytes() == second.jsonl_bytes()
    assert first.manifest_bytes() == second.manifest_bytes()


def test_context_reader_rejects_duplicate_ids_atomically() -> None:
    files = (
        OrganizerFile(
            "contexts/context_01.json",
            b'{"id":1,"link":"https://example.invalid/a","passage":"A"}',
        ),
        OrganizerFile(
            "contexts/context_1.json",
            b'{"id":1,"link":"https://example.invalid/b","passage":"B"}',
        ),
    )

    with pytest.raises(OrganizerDataError) as captured:
        OrganizerContextReader().read_files(files)

    assert captured.value.code == "DATA_CONTEXT_ID_DUPLICATE"
    assert captured.value.raw_id == "1"


@pytest.mark.parametrize(
    ("source", "code", "json_path"),
    [
        (
            b'{"id":"2","link":"https://example.invalid/2","passage":"x"}',
            "DATA_CONTEXT_ID_TYPE",
            "$.id",
        ),
        (b'{"id":2,"passage":"x"}', "DATA_FIELD_MISSING", "$.link"),
        (b'{"id":2,"link":"relative","passage":"x"}', "DATA_CONTEXT_LINK_INVALID", "$.link"),
        (
            b'{"id":2,"link":"https://example.invalid/2","passage":null}',
            "DATA_CONTEXT_PASSAGE_TYPE",
            "$.passage",
        ),
        (
            b'{"id":2,"link":"https://example.invalid/2","passage":"x","extra":1}',
            "DATA_FIELD_UNKNOWN",
            "$.extra",
        ),
    ],
)
def test_context_reader_rejects_malformed_records(
    source: bytes,
    code: str,
    json_path: str,
) -> None:
    with pytest.raises(OrganizerDataError) as captured:
        OrganizerContextReader().read_files((OrganizerFile("contexts/context_2.json", source),))

    assert captured.value.code == code
    assert captured.value.json_path == json_path


def test_context_discovery_accepts_only_direct_exact_regular_files(tmp_path: Path) -> None:
    (tmp_path / "context_10.json").write_bytes(b"ten")
    (tmp_path / "context_2.json").write_bytes(b"two")
    (tmp_path / "context_bad.json").write_bytes(b"bad")
    (tmp_path / "notes.json").write_bytes(b"notes")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "context_3.json").write_bytes(b"nested")

    files = discover_context_files(tmp_path, pattern="context_*.json")

    assert [(file.relative_path, file.data) for file in files] == [
        ("context_10.json", b"ten"),
        ("context_2.json", b"two"),
        ("context_bad.json", b"bad"),
    ]


def test_context_discovery_rejects_unapproved_pattern(tmp_path: Path) -> None:
    with pytest.raises(OrganizerDataError) as captured:
        discover_context_files(tmp_path, pattern="*.json")

    assert captured.value.code == "DATA_CONTEXT_PATTERN_INVALID"
    assert captured.value.artifact_path == "context-source"


def test_context_discovery_rejects_matching_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "context_1.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this Windows account cannot create symlinks")

    with pytest.raises(OrganizerDataError) as captured:
        discover_context_files(tmp_path, pattern="context_*.json")

    assert captured.value.code == "DATA_CONTEXT_FILE_UNSUPPORTED"
    assert captured.value.artifact_path == "context_1.json"
