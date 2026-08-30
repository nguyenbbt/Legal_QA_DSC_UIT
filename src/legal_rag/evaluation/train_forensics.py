"""Deterministic aggregate-only forensic analysis of official train rows."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, NoReturn, cast

from legal_rag.domain.checksums import checksum_bytes, checksum_file, content_json_bytes
from legal_rag.evaluation.split import SplitError, load_split_manifest_rows
from legal_rag.retrieval.tokenizer import retrieval_token_values
from legal_rag.training.rag_sft import RagSftBuildError, load_gold_questions

_QUESTION_TYPES = (
    "DEFINITION",
    "CONDITION",
    "PROCEDURE",
    "AUTHORITY",
    "RIGHT",
    "OBLIGATION",
    "ELIGIBILITY",
    "DEADLINE",
    "PENALTY",
    "EXCEPTION",
    "NUMERIC",
    "MULTI_CONDITION",
    "MULTI_COORDINATE",
    "OTHER",
)
_QUESTION_CUES: dict[str, tuple[str, ...]] = {
    "DEFINITION": ("là gì", "được hiểu", "khái niệm", "định nghĩa", "thế nào"),
    "CONDITION": ("điều kiện", "khi nào", "trường hợp nào"),
    "PROCEDURE": ("thủ tục", "trình tự", "hồ sơ", "thực hiện như thế nào"),
    "AUTHORITY": ("thẩm quyền", "cơ quan nào", "ai quyết định", "ai có quyền"),
    "RIGHT": ("quyền gì", "được quyền", "có được"),
    "OBLIGATION": ("nghĩa vụ", "trách nhiệm", "phải làm gì"),
    "ELIGIBILITY": ("đối tượng", "ai được", "được hưởng"),
    "DEADLINE": ("thời hạn", "bao lâu", "khi nào"),
    "PENALTY": ("xử phạt", "mức phạt", "phạt bao nhiêu"),
    "EXCEPTION": ("ngoại lệ", "trừ trường hợp", "không áp dụng"),
    "NUMERIC": ("bao nhiêu", "mức", "tỷ lệ", "phần trăm"),
}
_SENTENCE_BOUNDARY = re.compile(r"(?:[.!?]+(?=\s|$)|\n+)")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_DATE = re.compile(
    r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\b)",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?:\d+(?:[.,]\d+)?\s*%|\bphần\s+trăm\b)", re.IGNORECASE)
_LEGAL_COORDINATE = re.compile(r"\b(?:điều|khoản|điểm)\s+[0-9a-zđ]+", re.IGNORECASE)
_DOCUMENT_CITATION = re.compile(
    r"\b(?:luật|nghị\s+định|thông\s+tư|quyết\s+định|pháp\s+lệnh)\b", re.IGNORECASE
)
_LIST_LINE = re.compile(r"(?m)^\s*(?:[-+*]|\d+[.)]|[a-zđ][.)])\s+")
_CONDITION = re.compile(r"\b(?:nếu|khi|trường hợp|với điều kiện|đáp ứng)\b", re.IGNORECASE)
_EXCEPTION = re.compile(r"\b(?:trừ|ngoại trừ|ngoại lệ|không áp dụng)\b", re.IGNORECASE)
_OPENINGS = {
    "CAN_CU": ("căn cứ",),
    "THEO_QUY_DINH": ("theo quy định", "theo"),
    "DIRECT": (),
}
_CLOSINGS = {
    "THEO_DO": ("theo đó",),
    "DO_DO": ("do đó",),
    "VI_VAY": ("vì vậy",),
    "NHU_VAY": ("như vậy",),
    "OTHER": (),
}


class TrainForensicsError(Exception):
    """Stable safe failure at the D-064 analysis boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk_id: str
    chunk_checksum: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Selection:
    question_id: str
    question_checksum: str
    evidence_ids: tuple[str, ...]
    evidence_checksums: tuple[str, ...]


def _fail(code: str, message: str) -> NoReturn:
    raise TrainForensicsError(code, message)


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in retrieval_token_values(text))


def _jsonl_values(
    data: bytes, label: str, *, require_canonical: bool = True
) -> tuple[dict[str, Any], ...]:
    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("D064_INPUT_INVALID", f"{label} JSONL framing is invalid")
    values: list[dict[str, Any]] = []
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TrainForensicsError(
                "D064_INPUT_INVALID", f"{label} contains invalid JSON"
            ) from error
        if not isinstance(value, dict) or (require_canonical and content_json_bytes(value) != line):
            _fail("D064_INPUT_INVALID", f"{label} contains a non-canonical row")
        values.append(cast(dict[str, Any], value))
    return tuple(values)


def _validate_input_checksums(
    inputs: Mapping[str, bytes], expected: Mapping[str, object]
) -> dict[str, str]:
    if set(inputs) != {"questions", "split", "chunks", "selections"} or set(expected) != set(
        inputs
    ):
        _fail("D064_INPUT_CHECKSUM_MISMATCH", "D-064 checksum bindings are incomplete")
    actual = {name: checksum_bytes(data) for name, data in inputs.items()}
    if any(expected[name] != checksum for name, checksum in actual.items()):
        _fail("D064_INPUT_CHECKSUM_MISMATCH", "a D-064 input checksum is stale")
    return actual


def _load_chunks(data: bytes) -> tuple[_Chunk, ...]:
    chunks = tuple(
        _chunk_from_value(value) for value in _jsonl_values(data, "chunks", require_canonical=False)
    )
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        _fail("D064_CHUNK_INVALID", "chunk IDs are not unique")
    if not chunks:
        _fail("D064_CHUNK_INVALID", "chunk artifact is empty")
    return chunks


def _chunk_from_value(value: Mapping[str, Any]) -> _Chunk:
    chunk_id = value.get("chunk_id")
    checksum = value.get("chunk_checksum")
    text = value.get("display_text")
    if (
        value.get("schema_version") != "retrieval.chunk.v1"
        or not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(checksum, str)
        or not checksum.startswith("sha256:")
        or not isinstance(text, str)
        or not text.strip()
    ):
        _fail("D064_CHUNK_INVALID", "chunk identity or text is invalid")
    return _Chunk(chunk_id, checksum, text, _tokens(text))


def _iter_chunks(path: Path) -> Iterator[_Chunk]:
    seen: set[str] = set()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if b"\r" in line or not line.endswith(b"\n"):
                _fail("D064_INPUT_INVALID", "chunks JSONL framing is invalid")
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise TrainForensicsError(
                    "D064_INPUT_INVALID", "chunks contains invalid JSON"
                ) from error
            if not isinstance(value, dict):
                _fail("D064_INPUT_INVALID", "chunks row must be an object")
            chunk = _chunk_from_value(cast(dict[str, Any], value))
            if chunk.chunk_id in seen:
                _fail("D064_CHUNK_INVALID", "chunk IDs are not unique")
            seen.add(chunk.chunk_id)
            count += 1
            yield chunk
    if count == 0:
        _fail("D064_CHUNK_INVALID", "chunk artifact is empty")


def _load_selections(data: bytes) -> tuple[_Selection, ...]:
    selections: list[_Selection] = []
    seen: set[str] = set()
    for value in _jsonl_values(data, "selections"):
        question_id = value.get("question_id")
        question_checksum = value.get("question_checksum")
        raw_ids = value.get("evidence_ids")
        raw_checksums = value.get("evidence_checksums")
        if (
            value.get("schema_version") != "training.evidence.selection.v1"
            or not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(question_checksum, str)
            or not question_checksum.startswith("sha256:")
            or not isinstance(raw_ids, list)
            or not raw_ids
            or not all(isinstance(item, str) and item for item in raw_ids)
            or len(raw_ids) != len(set(raw_ids))
            or not isinstance(raw_checksums, list)
            or len(raw_checksums) != len(raw_ids)
            or not all(
                isinstance(item, str) and item.startswith("sha256:") for item in raw_checksums
            )
        ):
            _fail("D064_SELECTION_INVALID", "mapped-evidence selection is invalid")
        seen.add(question_id)
        selections.append(
            _Selection(
                question_id,
                question_checksum,
                tuple(cast(list[str], raw_ids)),
                tuple(cast(list[str], raw_checksums)),
            )
        )
    return tuple(selections)


def _question_signals(question: str) -> tuple[str, ...]:
    folded = _fold(question)
    signals = [name for name, cues in _QUESTION_CUES.items() if any(cue in folded for cue in cues)]
    coordinate_count = len(_LEGAL_COORDINATE.findall(folded))
    if len(signals) >= 2:
        signals.append("MULTI_CONDITION")
    if coordinate_count >= 2:
        signals.append("MULTI_COORDINATE")
    return tuple(dict.fromkeys(signals))


def _primary_question_type(signals: Sequence[str]) -> str:
    priorities = (
        "MULTI_COORDINATE",
        "MULTI_CONDITION",
        "PENALTY",
        "DEADLINE",
        "AUTHORITY",
        "PROCEDURE",
        "EXCEPTION",
        "CONDITION",
        "ELIGIBILITY",
        "OBLIGATION",
        "RIGHT",
        "DEFINITION",
        "NUMERIC",
    )
    return next((name for name in priorities if name in signals), "OTHER")


def _sentence_count(text: str) -> int:
    return max(1, len([part for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]))


def _nearest_rank(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {
            "minimum": 0,
            "mean": 0.0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "maximum": 0,
        }
    return {
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "p50": _nearest_rank(values, 50),
        "p75": _nearest_rank(values, 75),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "maximum": max(values),
    }


def _named_pattern(text: str, patterns: Mapping[str, tuple[str, ...]]) -> str:
    folded = _fold(text)
    for name, cues in patterns.items():
        if cues and any(folded.startswith(cue) for cue in cues):
            return name
    return next(name for name, cues in patterns.items() if not cues)


def _closing_pattern(text: str) -> str:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return _named_pattern(parts[-1] if parts else text, _CLOSINGS)


def _contains_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    start = needle[0]
    return any(
        tuple(haystack[index : index + len(needle)]) == tuple(needle)
        for index, value in enumerate(haystack[: len(haystack) - len(needle) + 1])
        if value == start
    )


def _anchored_patterns(
    patterns: Sequence[tuple[str, ...]],
) -> dict[str, tuple[int, ...]]:
    frequencies = Counter(token for pattern in patterns for token in set(pattern))
    by_anchor: dict[str, list[int]] = defaultdict(list)
    for index, pattern in enumerate(patterns):
        if pattern:
            anchor = min(
                set(pattern),
                key=lambda token: (frequencies[token], token.encode("utf-8")),
            )
            by_anchor[anchor].append(index)
    return {token: tuple(indices) for token, indices in by_anchor.items()}


def _scan_corpus(
    chunks: Iterable[_Chunk],
    answer_patterns: Sequence[tuple[str, ...]],
    sentence_patterns: Sequence[tuple[int, tuple[str, ...]]],
    selected_chunk_ids: set[str],
) -> tuple[list[bool], list[bool], dict[str, _Chunk], int]:
    answer_matches = [False] * len(answer_patterns)
    sentence_matches = [False] * len(answer_patterns)
    answer_anchors = _anchored_patterns(answer_patterns)
    answer_needles = ["\x00" + "\x00".join(pattern) + "\x00" for pattern in answer_patterns]
    sentence_tokens = [pattern for _, pattern in sentence_patterns]
    sentence_anchors = _anchored_patterns(sentence_tokens)
    sentence_needles = ["\x00" + "\x00".join(pattern) + "\x00" for pattern in sentence_tokens]
    resolved: dict[str, _Chunk] = {}
    chunk_count = 0
    for chunk in chunks:
        chunk_count += 1
        if chunk.chunk_id in selected_chunk_ids:
            resolved[chunk.chunk_id] = chunk
        unique_tokens = set(chunk.tokens)
        searchable = "\x00" + "\x00".join(chunk.tokens) + "\x00"
        answer_candidates = {
            index for token in unique_tokens for index in answer_anchors.get(token, ())
        }
        for index in answer_candidates:
            if not answer_matches[index] and answer_needles[index] in searchable:
                answer_matches[index] = True
        sentence_candidates = {
            index for token in unique_tokens for index in sentence_anchors.get(token, ())
        }
        for pattern_index in sentence_candidates:
            answer_index, _ = sentence_patterns[pattern_index]
            if not sentence_matches[answer_index] and sentence_needles[pattern_index] in searchable:
                sentence_matches[answer_index] = True
    return answer_matches, sentence_matches, resolved, chunk_count


def _mapped_overlap(
    answer_tokens: tuple[str, ...], evidence: Sequence[_Chunk]
) -> tuple[float, int]:
    evidence_tokens = {token for chunk in evidence for token in chunk.tokens}
    answer_unique = set(answer_tokens)
    coverage = len(answer_unique & evidence_tokens) / len(answer_unique) if answer_unique else 0.0
    longest = max(
        (
            SequenceMatcher(None, answer_tokens, chunk.tokens, autojunk=False)
            .find_longest_match()
            .size
            for chunk in evidence
        ),
        default=0,
    )
    return coverage, longest


def _pattern_counts(answers: Sequence[str]) -> dict[str, int]:
    return {
        "citation_or_legal_coordinate": sum(
            bool(_LEGAL_COORDINATE.search(answer) or _DOCUMENT_CITATION.search(answer))
            for answer in answers
        ),
        "article": sum(bool(re.search(r"\bđiều\s+[0-9a-zđ]+", answer, re.I)) for answer in answers),
        "clause": sum(bool(re.search(r"\bkhoản\s+[0-9a-zđ]+", answer, re.I)) for answer in answers),
        "point": sum(bool(re.search(r"\bđiểm\s+[0-9a-zđ]+", answer, re.I)) for answer in answers),
        "number": sum(bool(_NUMBER.search(answer)) for answer in answers),
        "date": sum(bool(_DATE.search(answer)) for answer in answers),
        "percentage": sum(bool(_PERCENT.search(answer)) for answer in answers),
        "list_or_enumeration": sum(bool(_LIST_LINE.search(answer)) for answer in answers),
        "condition": sum(bool(_CONDITION.search(answer)) for answer in answers),
        "exception": sum(bool(_EXCEPTION.search(answer)) for answer in answers),
    }


def _analyze(
    *,
    questions_data: bytes,
    split_data: bytes,
    selections_data: bytes,
    chunks: Iterable[_Chunk],
    checksums: dict[str, str],
) -> dict[str, Any]:
    try:
        questions = load_gold_questions(questions_data)
        split_rows = load_split_manifest_rows(
            split_data,
            expected_source_checksum=checksums["questions"],
            expected_question_ids=tuple(question.question_id for question in questions),
        )
    except (RagSftBuildError, SplitError) as error:
        raise TrainForensicsError(
            "D064_SPLIT_OR_SOURCE_INVALID", "official questions or active split are invalid"
        ) from error

    question_by_id = {question.question_id: question for question in questions}
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    train_id_set = set(train_ids)
    train_questions = tuple(question_by_id[question_id] for question_id in train_ids)
    selections = _load_selections(selections_data)
    for selection in selections:
        if selection.question_id not in train_id_set:
            _fail("D064_NON_TRAIN_SELECTION", "mapped evidence references a non-train row")
        question = question_by_id[selection.question_id]
        if selection.question_checksum != checksum_bytes(question.question.encode("utf-8")):
            _fail(
                "D064_SELECTION_QUESTION_MISMATCH",
                "mapped evidence references stale question text",
            )

    answers = [cast(str, question.answer) for question in train_questions]
    answer_tokens = [_tokens(answer) for answer in answers]
    question_signals = [_question_signals(question.question) for question in train_questions]
    primary_types = Counter(_primary_question_type(signals) for signals in question_signals)
    signal_counts = Counter(signal for signals in question_signals for signal in signals)
    sentence_patterns = [
        (answer_index, _tokens(part.strip()))
        for answer_index, answer in enumerate(answers)
        for part in _SENTENCE_BOUNDARY.split(answer)
        if part.strip()
    ]
    selected_chunk_ids = {
        chunk_id for selection in selections for chunk_id in selection.evidence_ids
    }
    exact_match_flags, sentence_match_flags, chunk_by_id, chunk_count = _scan_corpus(
        chunks, answer_tokens, sentence_patterns, selected_chunk_ids
    )
    selection_by_id: dict[str, tuple[_Chunk, ...]] = {}
    for selection in selections:
        evidence: list[_Chunk] = []
        for chunk_id, expected_checksum in zip(
            selection.evidence_ids, selection.evidence_checksums, strict=True
        ):
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None or chunk.chunk_checksum != expected_checksum:
                _fail("D064_EVIDENCE_MISMATCH", "mapped evidence does not resolve exactly")
            evidence.append(chunk)
        selection_by_id[selection.question_id] = tuple(evidence)

    exact_matches = sum(exact_match_flags)
    sentence_matches = sum(sentence_match_flags)
    class_counts: Counter[str] = Counter()
    mapped_coverages: list[float] = []
    mapped_longest_spans: list[int] = []
    for index, (question, tokens) in enumerate(zip(train_questions, answer_tokens, strict=True)):
        exact = exact_match_flags[index]
        mapped_evidence = selection_by_id.get(question.question_id)
        if exact:
            class_counts["EXTRACTIVE_EXACT_CORPUS"] += 1
        elif mapped_evidence is None:
            class_counts["UNRESOLVED_NO_MAPPING"] += 1
        else:
            coverage, longest = _mapped_overlap(tokens, mapped_evidence)
            mapped_coverages.append(coverage)
            mapped_longest_spans.append(longest)
            class_counts[
                "EXTRACTIVE_COMPOSITE_MAPPED" if coverage == 1.0 else "ABSTRACTIVE_MAPPED"
            ] += 1
        if mapped_evidence is not None and exact:
            coverage, longest = _mapped_overlap(tokens, mapped_evidence)
            mapped_coverages.append(coverage)
            mapped_longest_spans.append(longest)

    opening_counts = Counter(_named_pattern(answer, _OPENINGS) for answer in answers)
    closing_counts = Counter(_closing_pattern(answer) for answer in answers)
    token_lengths = [len(tokens) for tokens in answer_tokens]
    sentence_counts = [_sentence_count(answer) for answer in answers]
    return {
        "schema_version": "training.forensics.v1",
        "analysis_version": "d064-official-train-forensics.v1",
        "input_checksums": checksums,
        "source_question_count": len(questions),
        "train_fit_count": len(train_questions),
        "excluded_non_train_count": len(questions) - len(train_questions),
        "mapped_train_selection_count": len(selections),
        "question_primary_types": {name: primary_types[name] for name in _QUESTION_TYPES},
        "question_signal_counts": {name: signal_counts[name] for name in _QUESTION_TYPES},
        "answer_token_lengths": _summary(token_lengths),
        "answer_sentence_counts": _summary(sentence_counts),
        "answer_patterns": _pattern_counts(answers),
        "answer_opening_patterns": {name: opening_counts[name] for name in _OPENINGS},
        "answer_closing_patterns": {name: closing_counts[name] for name in _CLOSINGS},
        "corpus_overlap": {
            "exact_full_answer_count": exact_matches,
            "exact_answer_sentence_count": sentence_matches,
            "corpus_chunk_count": chunk_count,
        },
        "mapped_evidence": {
            "selection_count": len(selections),
            "answer_token_coverage": {
                "minimum": min(mapped_coverages, default=0.0),
                "mean": sum(mapped_coverages) / len(mapped_coverages) if mapped_coverages else 0.0,
                "maximum": max(mapped_coverages, default=0.0),
            },
            "longest_contiguous_token_span": _summary(mapped_longest_spans),
        },
        "potential_answer_classes": {
            name: class_counts[name]
            for name in (
                "EXTRACTIVE_EXACT_CORPUS",
                "EXTRACTIVE_COMPOSITE_MAPPED",
                "ABSTRACTIVE_MAPPED",
                "UNRESOLVED_NO_MAPPING",
            )
        },
        "generated_text_used": False,
        "tuning_performed": False,
        "model_inference_runs": 0,
    }


def analyze_train_forensics(
    *,
    questions_data: bytes,
    split_data: bytes,
    chunks_data: bytes,
    selections_data: bytes,
    expected_input_checksums: Mapping[str, object],
) -> dict[str, Any]:
    """Build one deterministic D-064 aggregate report from in-memory inputs."""

    input_data = {
        "questions": questions_data,
        "split": split_data,
        "chunks": chunks_data,
        "selections": selections_data,
    }
    checksums = _validate_input_checksums(input_data, expected_input_checksums)
    return _analyze(
        questions_data=questions_data,
        split_data=split_data,
        selections_data=selections_data,
        chunks=_load_chunks(chunks_data),
        checksums=checksums,
    )


def analyze_train_forensics_paths(
    *,
    questions_path: Path,
    split_path: Path,
    chunks_path: Path,
    selections_path: Path,
    expected_input_checksums: Mapping[str, object],
) -> dict[str, Any]:
    """Stream the potentially large corpus while preserving the exact D-064 contract."""

    paths = {
        "questions": questions_path,
        "split": split_path,
        "chunks": chunks_path,
        "selections": selections_path,
    }
    checksums = {name: checksum_file(path) for name, path in paths.items()}
    if set(expected_input_checksums) != set(paths) or any(
        expected_input_checksums[name] != checksum for name, checksum in checksums.items()
    ):
        _fail("D064_INPUT_CHECKSUM_MISMATCH", "a D-064 input checksum is stale")
    return _analyze(
        questions_data=questions_path.read_bytes(),
        split_data=split_path.read_bytes(),
        selections_data=selections_path.read_bytes(),
        chunks=_iter_chunks(chunks_path),
        checksums=checksums,
    )


__all__ = [
    "TrainForensicsError",
    "analyze_train_forensics",
    "analyze_train_forensics_paths",
]
