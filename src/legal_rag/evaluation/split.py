"""Pure deterministic contracts for the immutable ``split.v1`` dataset split."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from legal_rag.domain.artifacts import ImmutableArtifactError, write_immutable_bytes
from legal_rag.domain.checksums import DeterminismError, canonical_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json

SPLIT_ALGORITHM_VERSION = "split.v1"
SPLIT_NORMALIZATION_VERSION = "split-normalization.v1"
SPLIT_SEED = "dsc2026-split-v1"
UNICODE_DATABASE_VERSION = unicodedata.unidata_version

SplitName = Literal["train", "development", "local_test"]
OverlapMatchType = Literal["exact_normalized", "near_duplicate"]
_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)


class SplitError(Exception):
    """Stable safe failure at a split contract boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SplitQuestion:
    question_id: str
    question: str


@dataclass(frozen=True, slots=True)
class SplitGroup:
    group_id: str
    smallest_question_id: str
    question_ids: tuple[str, ...]
    assignment_digest: str
    split: SplitName


@dataclass(frozen=True, slots=True)
class SplitManifestRow:
    question_id: str
    group_id: str
    assignment_digest: str
    split: SplitName


@dataclass(frozen=True, slots=True)
class SplitOverlapRow:
    train_question_id: str
    public_question_id: str
    match_type: OverlapMatchType


@dataclass(frozen=True, slots=True)
class SplitManifest:
    source_checksum: str
    public_source_checksum: str
    rows: tuple[SplitManifestRow, ...]
    overlap_rows: tuple[SplitOverlapRow, ...]
    group_count: int
    public_question_count: int

    def as_dict(self) -> dict[str, object]:
        split_counts = {
            split: sum(row.split == split for row in self.rows)
            for split in ("train", "development", "local_test")
        }
        overlapping_train = {row.train_question_id for row in self.overlap_rows}
        overlapping_public = {row.public_question_id for row in self.overlap_rows}
        return {
            "schema_version": "split.manifest.v1",
            "algorithm_version": SPLIT_ALGORITHM_VERSION,
            "normalization_version": SPLIT_NORMALIZATION_VERSION,
            "seed": SPLIT_SEED,
            "unicode_database_version": UNICODE_DATABASE_VERSION,
            "source_checksum": self.source_checksum,
            "public_source_checksum": self.public_source_checksum,
            "question_count": len(self.rows),
            "public_question_count": self.public_question_count,
            "group_count": self.group_count,
            "split_counts": split_counts,
            "rows": [
                {
                    "question_id": row.question_id,
                    "group_id": row.group_id,
                    "assignment_digest": row.assignment_digest,
                    "split": row.split,
                }
                for row in self.rows
            ],
            "overlap_report": {
                "pair_count": len(self.overlap_rows),
                "overlapping_train_question_count": len(overlapping_train),
                "overlapping_public_question_count": len(overlapping_public),
                "rows": [
                    {
                        "train_question_id": row.train_question_id,
                        "public_question_id": row.public_question_id,
                        "match_type": row.match_type,
                    }
                    for row in self.overlap_rows
                ],
            },
        }

    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


def _fail(code: str, message: str) -> NoReturn:
    raise SplitError(code, message)


def _id_key(question_id: str) -> bytes:
    return question_id.encode("utf-8")


def _ordered_questions(questions: tuple[SplitQuestion, ...]) -> tuple[SplitQuestion, ...]:
    ordered = tuple(sorted(questions, key=lambda item: _id_key(item.question_id)))
    if any(not item.question_id for item in ordered):
        _fail("SPLIT_QUESTION_ID_EMPTY", "question IDs must be non-empty")
    if len({item.question_id for item in ordered}) != len(ordered):
        _fail("SPLIT_QUESTION_ID_DUPLICATE", "question IDs must be unique")
    return ordered


def normalize_split_text(text: str) -> str:
    """Apply the exact ordered ``split-normalization.v1`` transformations."""

    folded = unicodedata.normalize("NFC", text).casefold()
    punctuation_spaced = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in folded
    )
    return " ".join(punctuation_spaced.split())


def character_5grams(text: str) -> frozenset[str]:
    """Return set-based character 5-grams for one already-normalized string."""

    if not text:
        return frozenset()
    if len(text) < 5:
        return frozenset((text,))
    return frozenset(text[index : index + 5] for index in range(len(text) - 4))


def are_near_duplicates(left: str, right: str) -> bool:
    """Compare two non-empty normalized strings using exact rational thresholds."""

    if not left or not right:
        return False
    longest = max(len(left), len(right))
    if abs(len(left) - len(right)) * 10 > longest:
        return False
    left_grams = character_5grams(left)
    right_grams = character_5grams(right)
    intersection_size = len(left_grams & right_grams)
    union_size = len(left_grams | right_grams)
    return intersection_size * 10 >= union_size * 9


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self._parent[item]
        while parent != item:
            self._parent[item] = self._parent[parent]
            item = self._parent[item]
            parent = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _assignment(smallest_question_id: str) -> tuple[str, SplitName]:
    digest = hashlib.sha256(f"{SPLIT_SEED}\n{smallest_question_id}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    scale = 1 << 64
    if value * 5 < scale * 4:
        split: SplitName = "train"
    elif value * 10 < scale * 9:
        split = "development"
    else:
        split = "local_test"
    return digest.hex(), split


def _candidate_pairs(normalized: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Find all possible 0.90-Jaccard pairs with an exact global prefix filter."""

    grams = tuple(character_5grams(text) for text in normalized)
    frequency = Counter(gram for gram_set in grams for gram in gram_set)
    prefixes: list[tuple[str, ...]] = []
    for gram_set in grams:
        ordered = sorted(gram_set, key=lambda gram: (frequency[gram], gram.encode("utf-8")))
        minimum_overlap = (9 * len(ordered) + 9) // 10
        prefix_length = len(ordered) - minimum_overlap + 1
        prefixes.append(tuple(ordered[:prefix_length]))

    inverted: dict[str, list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for right_index, prefix in enumerate(prefixes):
        candidates: set[int] = set()
        for gram in prefix:
            candidates.update(inverted.get(gram, ()))
        for left_index in candidates:
            left_size = len(grams[left_index])
            right_size = len(grams[right_index])
            if min(left_size, right_size) * 10 < max(left_size, right_size) * 9:
                continue
            if are_near_duplicates(normalized[left_index], normalized[right_index]):
                pairs.add((left_index, right_index))
        for gram in prefix:
            inverted.setdefault(gram, []).append(right_index)
    return tuple(sorted(pairs))


def _cross_dataset_overlaps(
    train: tuple[SplitQuestion, ...], public: tuple[SplitQuestion, ...]
) -> tuple[SplitOverlapRow, ...]:
    train_by_text = _questions_by_normalized_text(train)
    public_by_text = _questions_by_normalized_text(public)

    unique_text = tuple(sorted(set(train_by_text) | set(public_by_text), key=str.encode))
    rows: set[tuple[str, str, OverlapMatchType]] = set()
    for text in set(train_by_text) & set(public_by_text):
        _add_overlap_rows(rows, train_by_text[text], public_by_text[text], "exact_normalized")

    for left_index, right_index in _candidate_pairs(unique_text):
        left = unique_text[left_index]
        right = unique_text[right_index]
        _add_overlap_rows(
            rows, train_by_text.get(left, ()), public_by_text.get(right, ()), "near_duplicate"
        )
        _add_overlap_rows(
            rows, train_by_text.get(right, ()), public_by_text.get(left, ()), "near_duplicate"
        )

    return tuple(
        SplitOverlapRow(train_id, public_id, match_type)
        for train_id, public_id, match_type in sorted(
            rows, key=lambda row: (_id_key(row[0]), _id_key(row[1]), row[2])
        )
    )


def _questions_by_normalized_text(
    questions: tuple[SplitQuestion, ...],
) -> dict[str, list[str]]:
    members: dict[str, list[str]] = {}
    for item in questions:
        members.setdefault(normalize_split_text(item.question), []).append(item.question_id)
    return members


def _add_overlap_rows(
    rows: set[tuple[str, str, OverlapMatchType]],
    train_ids: list[str] | tuple[str, ...],
    public_ids: list[str] | tuple[str, ...],
    match_type: OverlapMatchType,
) -> None:
    rows.update(
        (train_id, public_id, match_type) for train_id in train_ids for public_id in public_ids
    )


def build_split_groups(questions: tuple[SplitQuestion, ...]) -> tuple[SplitGroup, ...]:
    """Build exhaustive-equivalent connected components in raw UTF-8 ID order."""

    ordered = _ordered_questions(questions)

    normalized = tuple(normalize_split_text(item.question) for item in ordered)
    components = _DisjointSet(len(ordered))
    representative_by_text: dict[str, int] = {}
    unique_indices: list[int] = []
    for index, text in enumerate(normalized):
        representative = representative_by_text.setdefault(text, index)
        components.union(representative, index)
        if representative == index:
            unique_indices.append(index)

    unique_text = tuple(normalized[index] for index in unique_indices)
    for left_unique, right_unique in _candidate_pairs(unique_text):
        components.union(unique_indices[left_unique], unique_indices[right_unique])

    members_by_root: dict[int, list[str]] = {}
    for index, item in enumerate(ordered):
        members_by_root.setdefault(components.find(index), []).append(item.question_id)

    groups: list[SplitGroup] = []
    for members in members_by_root.values():
        question_ids = tuple(sorted(members, key=_id_key))
        smallest = question_ids[0]
        digest, split = _assignment(smallest)
        groups.append(
            SplitGroup(
                group_id=f"split_group_{digest[:24]}",
                smallest_question_id=smallest,
                question_ids=question_ids,
                assignment_digest=digest,
                split=split,
            )
        )
    groups.sort(key=lambda group: _id_key(group.smallest_question_id))
    return tuple(groups)


def build_split_manifest(
    train_questions: tuple[SplitQuestion, ...],
    public_questions: tuple[SplitQuestion, ...],
    *,
    source_checksum: str,
    public_source_checksum: str,
) -> SplitManifest:
    """Build the complete non-secret immutable split and overlap manifest."""

    if (
        _CHECKSUM.fullmatch(source_checksum) is None
        or _CHECKSUM.fullmatch(public_source_checksum) is None
    ):
        _fail("SPLIT_SOURCE_CHECKSUM_INVALID", "split sources require typed SHA-256 checksums")

    groups = build_split_groups(train_questions)
    ordered_public = _ordered_questions(public_questions)
    rows = tuple(
        sorted(
            (
                SplitManifestRow(
                    question_id=question_id,
                    group_id=group.group_id,
                    assignment_digest=group.assignment_digest,
                    split=group.split,
                )
                for group in groups
                for question_id in group.question_ids
            ),
            key=lambda row: _id_key(row.question_id),
        )
    )
    return SplitManifest(
        source_checksum=source_checksum,
        public_source_checksum=public_source_checksum,
        rows=rows,
        overlap_rows=_cross_dataset_overlaps(train_questions, ordered_public),
        group_count=len(groups),
        public_question_count=len(ordered_public),
    )


def load_split_questions_jsonl(
    data: bytes, *, expected_answer_state: Literal["gold", "unlabeled"]
) -> tuple[SplitQuestion, ...]:
    """Load a complete strict ``internal.question.v1`` JSONL artifact."""

    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("SPLIT_QUESTION_ARTIFACT_INVALID", "question JSONL byte framing is invalid")
    records: list[SplitQuestion] = []
    for line in data.splitlines(keepends=True):
        if line == b"\n":
            _fail("SPLIT_QUESTION_ARTIFACT_INVALID", "question JSONL contains an empty row")
        try:
            record = parse_record_json(line, QuestionRecord, artifact_path="questions-jsonl")
        except RecordValidationError as error:
            raise SplitError("SPLIT_QUESTION_ARTIFACT_INVALID", error.issues[0].message) from error
        if record.answer_state != expected_answer_state:
            _fail("SPLIT_QUESTION_ROLE_INVALID", "question JSONL has the wrong dataset role")
        records.append(SplitQuestion(record.question_id, record.question))
    return _ordered_questions(tuple(records))


_MANIFEST_FIELDS = {
    "schema_version",
    "algorithm_version",
    "normalization_version",
    "seed",
    "unicode_database_version",
    "source_checksum",
    "public_source_checksum",
    "question_count",
    "public_question_count",
    "group_count",
    "split_counts",
    "rows",
    "overlap_report",
}


def _load_manifest_value(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
        if not isinstance(value, dict) or canonical_json_bytes(value) != data:
            _fail("SPLIT_MANIFEST_INVALID", "split manifest is not canonical JSON")
    except (UnicodeError, json.JSONDecodeError, DeterminismError) as error:
        raise SplitError("SPLIT_MANIFEST_INVALID", "split manifest JSON is invalid") from error
    return value


def _validate_manifest_identity(value: dict[str, Any], expected_source_checksum: str) -> None:
    identity = (
        value.get("schema_version"),
        value.get("algorithm_version"),
        value.get("normalization_version"),
        value.get("seed"),
        value.get("unicode_database_version"),
    )
    expected = (
        "split.manifest.v1",
        SPLIT_ALGORITHM_VERSION,
        SPLIT_NORMALIZATION_VERSION,
        SPLIT_SEED,
        UNICODE_DATABASE_VERSION,
    )
    if set(value) != _MANIFEST_FIELDS or identity != expected:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest contract identity is invalid")
    if value["source_checksum"] != expected_source_checksum:
        _fail("SPLIT_MANIFEST_SOURCE_MISMATCH", "split manifest source checksum is stale")
    if _CHECKSUM.fullmatch(value.get("public_source_checksum", "")) is None:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest public checksum is invalid")
    if type(value.get("public_question_count")) is not int or value["public_question_count"] < 0:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest public count is invalid")


def _parse_manifest_row(raw_row: object) -> SplitManifestRow:
    if not isinstance(raw_row, dict) or set(raw_row) != {
        "question_id",
        "group_id",
        "assignment_digest",
        "split",
    }:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest row shape is invalid")
    question_id = raw_row["question_id"]
    group_id = raw_row["group_id"]
    digest = raw_row["assignment_digest"]
    split = raw_row["split"]
    valid = (
        type(question_id) is str
        and type(group_id) is str
        and type(digest) is str
        and type(split) is str
        and re.fullmatch(r"split_group_[0-9a-f]{24}", group_id, re.ASCII) is not None
        and re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII) is not None
        and split in {"train", "development", "local_test"}
    )
    if not valid:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest row value is invalid")
    return SplitManifestRow(question_id, group_id, digest, cast(SplitName, split))


def _validate_manifest_ids_and_counts(
    value: dict[str, Any],
    rows: tuple[SplitManifestRow, ...],
    expected_question_ids: tuple[str, ...],
) -> None:
    row_ids = tuple(row.question_id for row in rows)
    if row_ids != tuple(sorted(expected_question_ids, key=_id_key)) or len(set(row_ids)) != len(
        row_ids
    ):
        _fail("SPLIT_MANIFEST_ID_MISMATCH", "split manifest IDs do not match questions")
    if type(value["question_count"]) is not int or value["question_count"] != len(rows):
        _fail("SPLIT_MANIFEST_INVALID", "split manifest question count is invalid")
    split_counts = {
        split: sum(row.split == split for row in rows)
        for split in ("train", "development", "local_test")
    }
    if value["split_counts"] != split_counts:
        _fail("SPLIT_MANIFEST_INVALID", "split manifest split counts are invalid")


def _validate_manifest_groups(value: dict[str, Any], rows: tuple[SplitManifestRow, ...]) -> None:
    groups: dict[str, list[SplitManifestRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    for group_rows in groups.values():
        smallest = min((row.question_id for row in group_rows), key=_id_key)
        expected_digest, expected_split = _assignment(smallest)
        if any(
            row.assignment_digest != expected_digest
            or row.group_id != f"split_group_{expected_digest[:24]}"
            or row.split != expected_split
            for row in group_rows
        ):
            _fail("SPLIT_MANIFEST_INVALID", "split manifest group assignment is invalid")
    if type(value["group_count"]) is not int or value["group_count"] != len(groups):
        _fail("SPLIT_MANIFEST_INVALID", "split manifest group count is invalid")


def _validate_overlap_report(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "pair_count",
        "overlapping_train_question_count",
        "overlapping_public_question_count",
        "rows",
    }:
        _fail("SPLIT_MANIFEST_INVALID", "split overlap report shape is invalid")
    raw_rows = value["rows"]
    if not isinstance(raw_rows, list):
        _fail("SPLIT_MANIFEST_INVALID", "split overlap rows must be a list")
    rows: list[tuple[str, str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or set(row) != {
            "train_question_id",
            "public_question_id",
            "match_type",
        }:
            _fail("SPLIT_MANIFEST_INVALID", "split overlap row shape is invalid")
        raw_record = (row["train_question_id"], row["public_question_id"], row["match_type"])
        if not all(type(member) is str for member in raw_record) or raw_record[2] not in {
            "exact_normalized",
            "near_duplicate",
        }:
            _fail("SPLIT_MANIFEST_INVALID", "split overlap row value is invalid")
        rows.append(cast(tuple[str, str, str], raw_record))
    ordered = sorted(rows, key=lambda row: (_id_key(row[0]), _id_key(row[1]), row[2]))
    train_ids = {row[0] for row in rows}
    public_ids = {row[1] for row in rows}
    expected_counts = (len(rows), len(train_ids), len(public_ids))
    actual_counts = (
        value["pair_count"],
        value["overlapping_train_question_count"],
        value["overlapping_public_question_count"],
    )
    if rows != ordered or len(set(rows)) != len(rows) or actual_counts != expected_counts:
        _fail("SPLIT_MANIFEST_INVALID", "split overlap ordering or counts are invalid")


def load_split_manifest_rows(
    data: bytes,
    *,
    expected_source_checksum: str,
    expected_question_ids: tuple[str, ...],
) -> tuple[SplitManifestRow, ...]:
    """Validate canonical split bytes and return their ordered typed assignments."""

    value = _load_manifest_value(data)
    _validate_manifest_identity(value, expected_source_checksum)
    raw_rows = value["rows"]
    if not isinstance(raw_rows, list):
        _fail("SPLIT_MANIFEST_INVALID", "split manifest rows must be a list")
    rows = tuple(_parse_manifest_row(row) for row in raw_rows)
    _validate_manifest_ids_and_counts(value, rows, expected_question_ids)
    _validate_manifest_groups(value, rows)
    _validate_overlap_report(value["overlap_report"])
    return rows


def write_split_manifest(destination: Path, manifest: SplitManifest) -> str:
    """Create an immutable manifest atomically, accepting an identical retry."""

    try:
        return write_immutable_bytes(destination, manifest.json_bytes())
    except ImmutableArtifactError as error:
        code = (
            "SPLIT_MANIFEST_IMMUTABLE"
            if error.code == "ARTIFACT_IMMUTABLE"
            else "SPLIT_MANIFEST_WRITE_FAILED"
        )
        raise SplitError(code, error.message) from error
