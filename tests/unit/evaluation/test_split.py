"""Exact fixtures for the immutable ``split.v1`` grouping contract."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from legal_rag.evaluation.split import (
    SPLIT_SEED,
    UNICODE_DATABASE_VERSION,
    SplitError,
    SplitManifest,
    SplitQuestion,
    are_near_duplicates,
    build_split_groups,
    build_split_manifest,
    character_5grams,
    load_split_manifest_rows,
    normalize_split_text,
    write_split_manifest,
)


def test_split_normalization_preserves_non_punctuation_unicode() -> None:
    decomposed = "  ĐIỀU\u00a01:\tPhí + 20%—đúng?!  "

    normalized = normalize_split_text(decomposed)

    assert normalized == "điều 1 phí + 20 đúng"
    assert unicodedata.is_normalized("NFC", normalized)
    assert unicodedata.unidata_version == UNICODE_DATABASE_VERSION
    assert SPLIT_SEED == "dsc2026-split-v1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", frozenset()),
        ("luật", frozenset({"luật"})),
        ("abcdef", frozenset({"abcde", "bcdef"})),
        ("aaaaaa", frozenset({"aaaaa"})),
    ],
)
def test_character_5grams_are_sets_with_short_and_empty_rules(
    text: str, expected: frozenset[str]
) -> None:
    assert character_5grams(text) == expected


def test_near_duplicate_uses_inclusive_length_and_jaccard_boundaries() -> None:
    assert are_near_duplicates("abcdefghij", "abcdefghij")
    assert are_near_duplicates("aaaaaaaaaa", "aaaaaaaaaaa")
    assert not are_near_duplicates("aaaaaaaaaa", "aaaaaaaaaaaa")
    assert not are_near_duplicates("", "")
    assert not are_near_duplicates("abcde", "abcdx")


def test_components_are_transitive_and_input_permutation_invariant() -> None:
    questions = (
        SplitQuestion("10", "abcdefghijklmnopqrst"),
        SplitQuestion("02", "abcdefghijklmnopqrstu"),
        SplitQuestion("01", "abcdefghijklmnopqrstuv"),
        SplitQuestion("z", "unrelated"),
        SplitQuestion("é", "UNRELATED!"),
    )

    forward = build_split_groups(questions)
    reverse = build_split_groups(tuple(reversed(questions)))

    assert forward == reverse
    assert tuple(group.question_ids for group in forward) == (("01", "02", "10"), ("z", "é"))
    assert all(len(group.assignment_digest) == 64 for group in forward)


def test_raw_utf8_id_order_is_not_numeric_order() -> None:
    groups = build_split_groups(
        (
            SplitQuestion("2", "same"),
            SplitQuestion("10", "same"),
            SplitQuestion("01", "same"),
        )
    )

    assert groups[0].question_ids == ("01", "10", "2")
    assert groups[0].smallest_question_id == "01"


def test_duplicate_question_id_fails_closed() -> None:
    with pytest.raises(SplitError) as captured:
        build_split_groups((SplitQuestion("q", "one"), SplitQuestion("q", "two")))

    assert captured.value.code == "SPLIT_QUESTION_ID_DUPLICATE"


def test_prefix_candidate_search_matches_exhaustive_components() -> None:
    questions = tuple(
        SplitQuestion(f"q{index:02}", text)
        for index, text in enumerate(
            (
                "abcdefghijklmnopqrst",
                "abcdefghijklmnopqrstu",
                "abcdefghijklmnopqrstuv",
                "ABCDEFGHIJ-KLMNOPQRST",
                "completely unrelated question",
                "completely unrelated question!",
                "aaaaaaaaaa",
                "aaaaaaaaaaa",
                "aaaaaaaaaaaa",
                "?!",
                "…",
                "+",
            )
        )
    )
    expected = _exhaustive_components(questions)

    actual = tuple(group.question_ids for group in build_split_groups(questions))

    assert actual == expected


def _exhaustive_components(questions: tuple[SplitQuestion, ...]) -> tuple[tuple[str, ...], ...]:
    normalized = tuple(normalize_split_text(item.question) for item in questions)
    neighbors = {index: {index} for index in range(len(questions))}
    for left in range(len(questions)):
        for right in range(left + 1, len(questions)):
            if normalized[left] == normalized[right] or are_near_duplicates(
                normalized[left], normalized[right]
            ):
                neighbors[left].add(right)
                neighbors[right].add(left)
    unseen = set(range(len(questions)))
    components: list[tuple[str, ...]] = []
    while unseen:
        pending = [min(unseen)]
        reached: set[int] = set()
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(neighbors[current] - reached)
        unseen -= reached
        components.append(
            tuple(sorted((questions[index].question_id for index in reached), key=str.encode))
        )
    return tuple(sorted(components, key=lambda members: members[0].encode()))


def test_empty_normalized_questions_share_one_group() -> None:
    groups = build_split_groups(
        (SplitQuestion("p1", "?!"), SplitQuestion("p2", "…"), SplitQuestion("x", "+"))
    )

    assert tuple(group.question_ids for group in groups) == (("p1", "p2"), ("x",))


def test_assignment_digest_and_split_have_golden_values() -> None:
    groups = build_split_groups(
        (
            SplitQuestion("q11", "alpha"),
            SplitQuestion("q5", "different statute"),
            SplitQuestion("q0", "third unrelated legal issue"),
        )
    )
    by_id = {group.question_ids[0]: group for group in groups}

    assert by_id["q0"].assignment_digest == (
        "5f3d46073229dffd3bdd7d4dff0f5069c62bb77aaf8a4123465b62927cb328e3"
    )
    assert by_id["q0"].split == "train"
    assert by_id["q5"].split == "development"
    assert by_id["q11"].split == "local_test"


def test_manifest_is_permutation_invariant_and_reports_cross_dataset_overlap() -> None:
    train = (
        SplitQuestion("q11", "alpha"),
        SplitQuestion("q5", "abcdefghijklmnopqrst"),
        SplitQuestion("q0", "completely unrelated"),
    )
    public = (
        SplitQuestion("public-z", "ALPHA!"),
        SplitQuestion("public-a", "abcdefghijklmnopqrstu"),
        SplitQuestion("public-x", "nothing alike"),
    )

    manifest = build_split_manifest(
        train,
        public,
        source_checksum="sha256:" + "a" * 64,
        public_source_checksum="sha256:" + "b" * 64,
    )
    shuffled = build_split_manifest(
        tuple(reversed(train)),
        tuple(reversed(public)),
        source_checksum="sha256:" + "a" * 64,
        public_source_checksum="sha256:" + "b" * 64,
    )

    assert isinstance(manifest, SplitManifest)
    assert manifest.json_bytes() == shuffled.json_bytes()
    assert tuple(row.question_id for row in manifest.rows) == ("q0", "q11", "q5")
    assert tuple(
        (row.train_question_id, row.public_question_id, row.match_type)
        for row in manifest.overlap_rows
    ) == (
        ("q11", "public-z", "exact_normalized"),
        ("q5", "public-a", "near_duplicate"),
    )
    assert b"completely unrelated" not in manifest.json_bytes()
    assert b"nothing alike" not in manifest.json_bytes()
    assert manifest.as_dict()["public_question_count"] == 3


def test_manifest_rejects_invalid_typed_checksum() -> None:
    with pytest.raises(SplitError) as captured:
        build_split_manifest(
            (SplitQuestion("q", "question"),),
            (),
            source_checksum="not-a-checksum",
            public_source_checksum="sha256:" + "b" * 64,
        )

    assert captured.value.code == "SPLIT_SOURCE_CHECKSUM_INVALID"


def test_manifest_rejects_duplicate_public_id() -> None:
    with pytest.raises(SplitError) as captured:
        build_split_manifest(
            (SplitQuestion("train", "question"),),
            (SplitQuestion("public", "one"), SplitQuestion("public", "two")),
            source_checksum="sha256:" + "a" * 64,
            public_source_checksum="sha256:" + "b" * 64,
        )

    assert captured.value.code == "SPLIT_QUESTION_ID_DUPLICATE"


def test_manifest_writer_is_idempotent_but_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "split.json"
    manifest = build_split_manifest(
        (SplitQuestion("q", "question"),),
        (),
        source_checksum="sha256:" + "a" * 64,
        public_source_checksum="sha256:" + "b" * 64,
    )

    checksum = write_split_manifest(destination, manifest)
    assert write_split_manifest(destination, manifest) == checksum
    original = destination.read_bytes()

    changed = build_split_manifest(
        (SplitQuestion("different", "question"),),
        (),
        source_checksum="sha256:" + "a" * 64,
        public_source_checksum="sha256:" + "b" * 64,
    )
    with pytest.raises(SplitError) as captured:
        write_split_manifest(destination, changed)

    assert captured.value.code == "SPLIT_MANIFEST_IMMUTABLE"
    assert destination.read_bytes() == original


def test_manifest_rows_reload_only_with_exact_source_and_ids() -> None:
    questions = (SplitQuestion("q5", "one"), SplitQuestion("q0", "two"))
    checksum = "sha256:" + "a" * 64
    manifest = build_split_manifest(
        questions,
        (),
        source_checksum=checksum,
        public_source_checksum="sha256:" + "b" * 64,
    )

    rows = load_split_manifest_rows(
        manifest.json_bytes(),
        expected_source_checksum=checksum,
        expected_question_ids=("q5", "q0"),
    )

    assert tuple(row.question_id for row in rows) == ("q0", "q5")


def test_manifest_loader_rejects_noncanonical_or_wrong_source() -> None:
    manifest = build_split_manifest(
        (SplitQuestion("q", "question"),),
        (),
        source_checksum="sha256:" + "a" * 64,
        public_source_checksum="sha256:" + "b" * 64,
    )

    with pytest.raises(SplitError) as captured:
        load_split_manifest_rows(
            manifest.json_bytes().replace(b'"schema_version"', b' "schema_version"', 1),
            expected_source_checksum="sha256:" + "a" * 64,
            expected_question_ids=("q",),
        )
    assert captured.value.code == "SPLIT_MANIFEST_INVALID"

    with pytest.raises(SplitError) as captured:
        load_split_manifest_rows(
            manifest.json_bytes(),
            expected_source_checksum="sha256:" + "c" * 64,
            expected_question_ids=("q",),
        )
    assert captured.value.code == "SPLIT_MANIFEST_SOURCE_MISMATCH"
