"""Golden tests for pre-index ``grounding.v1`` sample freezing."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from legal_rag.evaluation.grounding import (
    GroundingError,
    GroundingQuestion,
    build_grounding_sample,
    has_exact_reference_syntax,
    write_grounding_sample,
)

SPLIT_CHECKSUM = "sha256:" + "a" * 64


def _question(index: int, *, exact: bool, padding: int) -> GroundingQuestion:
    prefix = "Theo Điều 1, " if exact else "Quy định chung "
    return GroundingQuestion(f"q{index:03}", prefix + "x" * padding)


def test_exact_reference_branch_is_syntactic_only() -> None:
    assert has_exact_reference_syntax("Theo khoản 2 Điều 3 Luật số 12/2020/QH14")
    assert has_exact_reference_syntax("Điều 3 quy định thế nào?")
    assert not has_exact_reference_syntax("Điều 3 và Điều 4 quy định thế nào?")
    assert not has_exact_reference_syntax("Khoản 2 quy định thế nào?")
    assert not has_exact_reference_syntax("Thủ tục được thực hiện ra sao?")


def test_balanced_sample_selects_ten_per_branch_tercile() -> None:
    questions = tuple(
        _question(index, exact=index >= 36, padding=(index % 36) + 1) for index in range(72)
    )

    manifest = build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)

    selected = [row for row in manifest.rows if row.selected]
    assert len(selected) == 60
    assert Counter((row.exact_reference_syntax, row.tercile) for row in selected) == {
        (False, 0): 10,
        (False, 1): 10,
        (False, 2): 10,
        (True, 0): 10,
        (True, 1): 10,
        (True, 2): 10,
    }
    assert {row.fill_reason for row in selected} == {"stratum_primary"}
    assert tuple(manifest.selected_question_ids) == tuple(
        row.question_id
        for row in sorted(
            selected,
            key=lambda row: (bytes.fromhex(row.sample_digest), row.question_id.encode()),
        )
    )


def test_terciles_follow_normalized_length_then_raw_id_rank() -> None:
    questions = tuple(_question(index, exact=False, padding=index // 2) for index in range(60))

    manifest = build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)
    false_rows = sorted(
        (row for row in manifest.rows if not row.exact_reference_syntax),
        key=lambda row: (row.normalized_length, row.question_id.encode()),
    )

    assert [row.tercile for row in false_rows] == [0] * 20 + [1] * 20 + [2] * 20


def test_underfill_uses_same_tercile_then_global_without_duplicates() -> None:
    questions = tuple(
        [_question(index, exact=True, padding=index) for index in range(2)]
        + [_question(index + 2, exact=False, padding=index) for index in range(58)]
    )

    manifest = build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)
    selected = [row for row in manifest.rows if row.selected]

    assert len(selected) == 60
    assert len({row.question_id for row in selected}) == 60
    assert "same_tercile_other_branch" in {row.fill_reason for row in selected}
    assert "global_underfill" in {row.fill_reason for row in selected}


def test_fewer_than_sixty_unique_ids_fails_closed() -> None:
    questions = tuple(_question(index, exact=False, padding=index) for index in range(59))

    with pytest.raises(GroundingError) as captured:
        build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)

    assert captured.value.code == "GROUNDING_SAMPLE_UNDERFILLED"


def test_manifest_records_every_candidate_and_full_digest() -> None:
    questions = tuple(_question(index, exact=index % 2 == 0, padding=index) for index in range(66))

    manifest = build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)
    rows = {row.question_id: row for row in manifest.rows}

    assert len(rows) == 66
    expected = hashlib.sha256(b"dsc2026-grounding-sample-v1\nq000").hexdigest()
    assert rows["q000"].sample_digest == expected
    assert len(manifest.json_bytes()) > 0
    assert b"Theo \xc4\x90i\xe1\xbb\x81u" not in manifest.json_bytes()


def test_grounding_manifest_is_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "grounding.sample.v1.json"
    questions = tuple(_question(index, exact=index % 2 == 0, padding=index) for index in range(60))
    manifest = build_grounding_sample(questions, split_checksum=SPLIT_CHECKSUM)

    checksum = write_grounding_sample(destination, manifest)

    assert write_grounding_sample(destination, manifest) == checksum
    original = destination.read_bytes()
    changed = build_grounding_sample(questions, split_checksum="sha256:" + "b" * 64)
    with pytest.raises(GroundingError) as captured:
        write_grounding_sample(destination, changed)
    assert captured.value.code == "GROUNDING_SAMPLE_IMMUTABLE"
    assert destination.read_bytes() == original
