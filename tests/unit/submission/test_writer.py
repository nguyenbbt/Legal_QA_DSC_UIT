from __future__ import annotations

import json

import pytest

from legal_rag.domain.models import AnswerRecord
from legal_rag.submission.writer import (
    SubmissionError,
    build_submission,
    build_submission_zip,
    validate_submission,
    write_submission,
)

SOURCE = (
    '{"01":{"question":"Câu hỏi một?","answer":null},"2":{"question":"Câu hỏi hai?","answer":null}}'
).encode()


def answer(question_id: str, text: str) -> AnswerRecord:
    return AnswerRecord.model_validate(
        {
            "schema_version": "internal.answer.v1",
            "question_id": question_id,
            "answer": text,
            "generator_id": "fixture-extractive-v1",
            "evidence_ids": (),
            "run_id": "run_aaaaaaaaaaaaaaaaaaaaaaaa",
        }
    )


def test_submission_preserves_id_order_and_emits_only_answer() -> None:
    data = build_submission(SOURCE, (answer("01", "Trả lời một."), answer("2", "Trả lời hai.")))

    decoded = json.loads(data)
    assert list(decoded) == ["01", "2"]
    assert decoded["01"] == {"answer": "Trả lời một."}
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert b'\n  "01": {' in data


@pytest.mark.parametrize(
    ("answers", "code"),
    [
        ((answer("01", "Một."),), "SUB_ID_MISMATCH"),
        ((answer("2", "Hai."), answer("01", "Một.")), "SUB_ID_MISMATCH"),
        (
            (answer("01", "Một."), answer("2", "Hai."), answer("3", "Ba.")),
            "SUB_ID_MISMATCH",
        ),
    ],
)
def test_submission_build_rejects_missing_extra_or_reordered_ids(
    answers: tuple[AnswerRecord, ...],
    code: str,
) -> None:
    with pytest.raises(SubmissionError) as captured:
        build_submission(SOURCE, answers)

    assert captured.value.code == code


@pytest.mark.parametrize(
    ("predictions", "code"),
    [
        (b'{"01":{"answer":" "}}\n', "SUB_EMPTY_ANSWER"),
        (
            b'{"01":{"answer":"A","extra":1}}\n',
            "SUB_SCHEMA_INVALID",
        ),
        ('{"01":{"question":"khác","answer":"A"}}\n'.encode(), "SUB_SCHEMA_INVALID"),
        (
            b'{"01":{"answer":"A"},"01":{"answer":"B"}}\n',
            "SUB_DUPLICATE_KEY",
        ),
        (b"\xef\xbb\xbf{}\n", "SUB_ENCODING_INVALID"),
        (b"{}\r\n", "SUB_ENCODING_INVALID"),
    ],
)
def test_submission_validator_rejects_invalid_artifacts(predictions: bytes, code: str) -> None:
    with pytest.raises(SubmissionError) as captured:
        validate_submission(SOURCE, predictions)

    assert captured.value.code == code


def test_submission_validation_printable_summary_is_stable() -> None:
    data = build_submission(SOURCE, (answer("01", "Một."), answer("2", "Hai.")))

    result = validate_submission(SOURCE, data)

    assert result.count == 2
    assert result.summary.startswith("VALID SUBMISSION count=2 sha256:")


def test_failed_build_does_not_replace_existing_submission(tmp_path) -> None:
    destination = tmp_path / "predictions.json"
    destination.write_bytes(b"existing-valid-bytes")

    with pytest.raises(SubmissionError):
        write_submission(destination, SOURCE, (answer("01", "missing second"),))

    assert destination.read_bytes() == b"existing-valid-bytes"


def test_submission_preserves_raw_unicode_id_but_emits_only_answer() -> None:
    source = '{"e\u0301":{"answer":null,"question":"Cafe\u0301?"}}'.encode()

    data = build_submission(source, (answer("é", "Đúng."),))

    assert list(json.loads(data)) == ["e\u0301"]
    assert list(json.loads(data)["e\u0301"]) == ["answer"]
    assert validate_submission(source, data).count == 1


def test_submission_zip_is_deterministic_and_contains_only_root_submission_json() -> None:
    import io
    import zipfile

    submission = build_submission(
        SOURCE,
        (answer("01", "Một."), answer("2", "Hai.")),
    )

    first = build_submission_zip(submission)
    second = build_submission_zip(submission)

    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["submission.json"]
        assert archive.read("submission.json") == submission
