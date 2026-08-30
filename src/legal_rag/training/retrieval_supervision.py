"""Deterministic train-only retrieval supervision v2 construction."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.retrieval.exact import document_number_key
from legal_rag.retrieval.legal_citations import ParsedLegalCitation, parse_legal_citations
from legal_rag.training.rag_sft import RagSftBuildError, load_gold_questions

MappingClass = Literal[
    "EXACT_DOC_ARTICLE_POINT",
    "EXACT_DOC_ARTICLE_CLAUSE",
    "EXACT_DOC_ARTICLE",
    "SAME_COORDINATE_MULTICHUNK",
    "DOCUMENT_ONLY",
    "AMBIGUOUS",
    "UNRESOLVED",
]

MAPPING_CLASSES: tuple[MappingClass, ...] = (
    "EXACT_DOC_ARTICLE_POINT",
    "EXACT_DOC_ARTICLE_CLAUSE",
    "EXACT_DOC_ARTICLE",
    "SAME_COORDINATE_MULTICHUNK",
    "DOCUMENT_ONLY",
    "AMBIGUOUS",
    "UNRESOLVED",
)


class RetrievalSupervisionError(Exception):
    """Stable fail-closed error at the retrieval-supervision.v2 boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SupervisionArtifacts:
    groups_data: bytes
    report_data: bytes
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk_id: str
    context_id: str
    hierarchy_path: tuple[str, ...]
    chunk_checksum: str


@dataclass(frozen=True, slots=True)
class _Resolution:
    state: Literal["resolved", "document_only", "ambiguous", "unresolved"]
    context_ids: tuple[str, ...]
    chunks: tuple[_Chunk, ...]
    mapping_class: MappingClass
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class _Historical:
    question_id: str
    question_checksum: str
    evidence_ids: tuple[str, ...]
    evidence_checksums: tuple[str, ...]


def _fail(code: str, message: str) -> NoReturn:
    raise RetrievalSupervisionError(code, message)


def _jsonl_values(data: bytes, label: str) -> tuple[dict[str, Any], ...]:
    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("D065_INPUT_INVALID", f"{label} JSONL framing is invalid")
    values: list[dict[str, Any]] = []
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RetrievalSupervisionError(
                "D065_INPUT_INVALID", f"{label} contains invalid JSON"
            ) from error
        if not isinstance(value, dict):
            _fail("D065_INPUT_INVALID", f"{label} row must be an object")
        values.append(cast(dict[str, Any], value))
    return tuple(values)


def _iter_jsonl(path: Path, label: str) -> Iterable[dict[str, Any]]:
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                _fail("D065_INPUT_INVALID", f"{label} JSONL framing is invalid")
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise RetrievalSupervisionError(
                    "D065_INPUT_INVALID", f"{label} contains invalid JSON"
                ) from error
            if not isinstance(value, dict):
                _fail("D065_INPUT_INVALID", f"{label} row must be an object")
            yield cast(dict[str, Any], value)


def _validate_checksums(
    inputs: Mapping[str, bytes], expected: Mapping[str, object]
) -> dict[str, str]:
    required = {"questions", "chunks", "contexts", "aliases", "historical"}
    if set(inputs) != required or set(expected) != required:
        _fail("D065_INPUT_CHECKSUM_MISMATCH", "D-065 checksum bindings are incomplete")
    actual = {name: checksum_bytes(data) for name, data in inputs.items()}
    if any(expected[name] != checksum for name, checksum in actual.items()):
        _fail("D065_INPUT_CHECKSUM_MISMATCH", "a D-065 input checksum is stale")
    return actual


def _validate_path_checksums(
    paths: Mapping[str, Path], expected: Mapping[str, object]
) -> dict[str, str]:
    required = {"questions", "chunks", "contexts", "aliases", "historical"}
    if set(paths) != required or set(expected) != required:
        _fail("D065_INPUT_CHECKSUM_MISMATCH", "D-065 checksum bindings are incomplete")
    actual = {name: _checksum_path(path) for name, path in paths.items()}
    if any(expected[name] != checksum for name, checksum in actual.items()):
        _fail("D065_INPUT_CHECKSUM_MISMATCH", "a D-065 input checksum is stale")
    return actual


def _checksum_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _chunk(value: Mapping[str, Any]) -> _Chunk:
    chunk_id = value.get("chunk_id")
    context_id = value.get("context_id")
    raw_path = value.get("hierarchy_path")
    checksum = value.get("chunk_checksum")
    if (
        value.get("schema_version") != "retrieval.chunk.v1"
        or not isinstance(chunk_id, str)
        or not chunk_id
        or not isinstance(context_id, str)
        or not context_id
        or not isinstance(raw_path, list)
        or not raw_path
        or not all(isinstance(item, str) and item for item in raw_path)
        or not isinstance(checksum, str)
        or not checksum.startswith("sha256:")
    ):
        _fail("D065_CHUNK_INVALID", "canonical chunk identity is invalid")
    return _Chunk(chunk_id, context_id, tuple(cast(list[str], raw_path)), checksum)


def _load_chunks(values: Iterable[Mapping[str, Any]], expected_count: int) -> tuple[_Chunk, ...]:
    chunks = tuple(_chunk(value) for value in values)
    if len(chunks) != expected_count or len({item.chunk_id for item in chunks}) != len(chunks):
        _fail("D065_CHUNK_COUNT_INVALID", "canonical chunk count or identity drifted")
    return chunks


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", value).casefold())
    unmarked = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", unmarked.replace("đ", "d")).split())


def _law_key_from_context_name(name: str | None) -> str | None:
    if not name:
        return None
    folded = _fold(name)
    match = re.match(
        r"(?P<kind>bo luat|luat) (?P<title>.+?) (?P<year>(?:19|20)[0-9]{2})(?: |$)", folded
    )
    if match is None:
        return None
    return f"{match.group('kind')} {match.group('title')} {match.group('year')}"


def _load_law_index(values: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for value in values:
        context_id = value.get("context_id")
        name = value.get("name")
        if (
            value.get("schema_version") != "internal.context.v1"
            or not isinstance(context_id, str)
            or not context_id
            or context_id in seen
            or (name is not None and not isinstance(name, str))
        ):
            _fail("D065_CONTEXT_INVALID", "canonical context identity is invalid")
        seen.add(context_id)
        key = _law_key_from_context_name(name)
        if key is not None:
            index[key].append(context_id)
    return {key: tuple(dict.fromkeys(ids)) for key, ids in index.items()}


def _load_alias_index(values: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = defaultdict(list)
    for value in values:
        number = value.get("document_number")
        key = value.get("document_number_key")
        context_id = value.get("context_id")
        if (
            value.get("schema_version") != "legal.reference.alias.v1"
            or value.get("review_state") != "approved"
            or not isinstance(number, str)
            or not isinstance(key, str)
            or key != document_number_key(number)
            or not isinstance(context_id, str)
            or not context_id
        ):
            _fail("D065_ALIAS_INVALID", "frozen alias identity is invalid")
        index[key].append(context_id)
    return {key: tuple(dict.fromkeys(ids)) for key, ids in index.items()}


def _path_coordinate(member: str) -> tuple[str, str] | None:
    folded = " ".join(unicodedata.normalize("NFC", member).casefold().split())
    labels = (
        ("tiểu mục", "subsection"),
        ("điều", "article"),
        ("khoản", "clause"),
        ("điểm", "point"),
        ("phần", "part"),
        ("chương", "chapter"),
        ("mục", "section"),
    )
    for label, kind in labels:
        prefix = f"{label} "
        if folded.startswith(prefix):
            return kind, folded[len(prefix) :]
    return None


def _matches(chunk: _Chunk, citation: ParsedLegalCitation) -> bool:
    path = dict(
        item for item in (_path_coordinate(member) for member in chunk.hierarchy_path) if item
    )
    if citation.article is not None and path.get("article") != citation.article:
        return False
    if citation.clause is not None and path.get("clause") != citation.clause:
        return False
    if citation.point is not None and path.get("point") != citation.point:
        return False
    return all(path.get(kind) == ordinal for kind, ordinal in citation.other_coordinates)


def _document_contexts(
    citation: ParsedLegalCitation,
    aliases: Mapping[str, tuple[str, ...]],
    laws: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    candidates: list[set[str]] = []
    if citation.document_number is not None:
        candidates.append(set(aliases.get(document_number_key(citation.document_number), ())))
    if citation.law_identity is not None:
        candidates.append(set(laws.get(citation.law_identity, ())))
    if not candidates or any(not value for value in candidates):
        return ()
    intersection = set.intersection(*candidates)
    return tuple(sorted(intersection, key=lambda value: value.encode()))


def _resolve_one(
    citation: ParsedLegalCitation,
    *,
    aliases: Mapping[str, tuple[str, ...]],
    laws: Mapping[str, tuple[str, ...]],
    chunks_by_context: Mapping[str, tuple[_Chunk, ...]],
) -> _Resolution:
    contexts = _document_contexts(citation, aliases, laws)
    identity_present = citation.document_number is not None or citation.law_identity is not None
    if not identity_present or not contexts:
        return _Resolution("unresolved", (), (), "UNRESOLVED", 0.0, "document-identity-unresolved")
    if len(contexts) != 1:
        return _Resolution(
            "ambiguous", contexts, (), "AMBIGUOUS", 0.0, "document-identity-ambiguous"
        )
    if citation.article is None and not citation.other_coordinates:
        return _Resolution(
            "document_only", contexts, (), "DOCUMENT_ONLY", 1.0, "exact-document-only"
        )
    matches = tuple(
        sorted(
            (
                chunk
                for chunk in chunks_by_context.get(contexts[0], ())
                if _matches(chunk, citation)
            ),
            key=lambda item: item.chunk_id.encode(),
        )
    )
    if not matches:
        return _Resolution(
            "unresolved", contexts, (), "UNRESOLVED", 0.0, "canonical-coordinate-unresolved"
        )
    confidence = 1.0 if citation.document_number is not None else 0.95
    reason_prefix = (
        "document-number-alias" if citation.document_number is not None else "law-code-identity"
    )
    if len(matches) > 1:
        return _Resolution(
            "resolved",
            contexts,
            matches,
            "SAME_COORDINATE_MULTICHUNK",
            confidence,
            f"{reason_prefix}+canonical-coordinate-multichunk",
        )
    if citation.point is not None:
        mapping_class: MappingClass = "EXACT_DOC_ARTICLE_POINT"
    elif citation.clause is not None:
        mapping_class = "EXACT_DOC_ARTICLE_CLAUSE"
    elif citation.article is not None:
        mapping_class = "EXACT_DOC_ARTICLE"
    else:
        mapping_class = "SAME_COORDINATE_MULTICHUNK"
    return _Resolution(
        "resolved",
        contexts,
        matches,
        mapping_class,
        confidence,
        f"{reason_prefix}+canonical-hierarchy",
    )


def _load_historical(values: Iterable[Mapping[str, Any]]) -> tuple[_Historical, ...]:
    rows: list[_Historical] = []
    seen: set[str] = set()
    for value in values:
        question_id = value.get("question_id")
        question_checksum = value.get("question_checksum")
        evidence_ids = value.get("evidence_ids")
        evidence_checksums = value.get("evidence_checksums")
        if (
            value.get("schema_version") != "training.evidence.selection.v1"
            or not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(question_checksum, str)
            or not isinstance(evidence_ids, list)
            or not all(isinstance(item, str) for item in evidence_ids)
            or not isinstance(evidence_checksums, list)
            or len(evidence_checksums) != len(evidence_ids)
            or not all(isinstance(item, str) for item in evidence_checksums)
        ):
            _fail("D065_HISTORICAL_INVALID", "historical v1 mapping is invalid")
        seen.add(question_id)
        rows.append(
            _Historical(
                question_id,
                question_checksum,
                tuple(cast(list[str], evidence_ids)),
                tuple(cast(list[str], evidence_checksums)),
            )
        )
    return tuple(rows)


def _aggregate_class(resolutions: Sequence[_Resolution]) -> tuple[MappingClass, float, str]:
    if not resolutions:
        return "UNRESOLVED", 0.0, "no-supported-legal-citation"
    if any(item.state == "ambiguous" for item in resolutions):
        return "AMBIGUOUS", 0.0, "at-least-one-citation-is-ambiguous"
    if any(item.state == "unresolved" for item in resolutions):
        return "UNRESOLVED", 0.0, "at-least-one-citation-is-unresolved"
    resolved = [item for item in resolutions if item.state == "resolved"]
    if not resolved:
        return (
            "DOCUMENT_ONLY",
            min(item.confidence for item in resolutions),
            "document-identity-only",
        )
    if any(item.mapping_class == "SAME_COORDINATE_MULTICHUNK" for item in resolved):
        mapping_class: MappingClass = "SAME_COORDINATE_MULTICHUNK"
    else:
        priority = {
            "EXACT_DOC_ARTICLE_POINT": 3,
            "EXACT_DOC_ARTICLE_CLAUSE": 2,
            "EXACT_DOC_ARTICLE": 1,
        }
        mapping_class = max(
            (item.mapping_class for item in resolved), key=lambda value: priority[value]
        )
    return (
        mapping_class,
        min(item.confidence for item in resolved),
        "+".join(dict.fromkeys(item.reason for item in resolved)),
    )


def _build(
    *,
    questions_data: bytes,
    train_question_ids: Sequence[str],
    chunks: tuple[_Chunk, ...],
    context_values: Iterable[Mapping[str, Any]],
    alias_values: Iterable[Mapping[str, Any]],
    historical_values: Iterable[Mapping[str, Any]],
    input_checksums: dict[str, str],
    split_manifest_checksum: str | None,
    expected_train_count: int,
) -> SupervisionArtifacts:
    try:
        questions = load_gold_questions(questions_data)
    except RagSftBuildError as error:
        raise RetrievalSupervisionError(
            "D065_QUESTION_SOURCE_INVALID", "official train source is invalid"
        ) from error
    train_ids = tuple(train_question_ids)
    question_by_id = {item.question_id: item for item in questions}
    if (
        len(train_ids) != expected_train_count
        or len(train_ids) != len(set(train_ids))
        or any(question_id not in question_by_id for question_id in train_ids)
    ):
        _fail("D065_TRAIN_PARTITION_INVALID", "active train partition identity drifted")

    laws = _load_law_index(context_values)
    aliases = _load_alias_index(alias_values)
    historical = _load_historical(historical_values)
    historical_by_id = {item.question_id: item for item in historical}
    chunks_by_context_lists: dict[str, list[_Chunk]] = defaultdict(list)
    chunk_by_id: dict[str, _Chunk] = {}
    for chunk in chunks:
        chunks_by_context_lists[chunk.context_id].append(chunk)
        chunk_by_id[chunk.chunk_id] = chunk
    chunks_by_context = {
        context_id: tuple(sorted(values, key=lambda item: item.chunk_id.encode()))
        for context_id, values in chunks_by_context_lists.items()
    }

    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    citation_bearing = 0
    exact_document = 0
    exact_article = 0
    exact_clause = 0
    exact_point = 0
    unique_resolved = 0
    positive_assignments = 0
    distinct_positive_ids: set[str] = set()
    for question_id in sorted(train_ids, key=lambda value: value.encode()):
        question = question_by_id[question_id]
        assert question.answer is not None
        citations = parse_legal_citations(question.answer)
        if citations:
            citation_bearing += 1
        resolutions = tuple(
            _resolve_one(
                citation,
                aliases=aliases,
                laws=laws,
                chunks_by_context=chunks_by_context,
            )
            for citation in citations
        )
        mapping_class, confidence, reason = _aggregate_class(resolutions)
        if mapping_class in {"AMBIGUOUS", "UNRESOLVED", "DOCUMENT_ONLY"}:
            positives: tuple[_Chunk, ...] = ()
        else:
            positives = tuple(
                sorted(
                    {
                        chunk.chunk_id: chunk for item in resolutions for chunk in item.chunks
                    }.values(),
                    key=lambda item: item.chunk_id.encode(),
                )
            )
        context_ids = tuple(
            sorted(
                {context_id for item in resolutions for context_id in item.context_ids},
                key=lambda value: value.encode(),
            )
        )
        class_counts[mapping_class] += 1
        if len(context_ids) == 1:
            exact_document += 1
        if positives:
            unique_resolved += 1
            exact_article += int(any(item.article is not None for item in citations))
            exact_clause += int(any(item.clause is not None for item in citations))
            exact_point += int(any(item.point is not None for item in citations))
        positive_assignments += len(positives)
        distinct_positive_ids.update(item.chunk_id for item in positives)
        historical_item = historical_by_id.get(question_id)
        historical_identifiable = False
        historical_relation = "NOT_PRESENT"
        if historical_item is not None:
            historical_identifiable = historical_item.question_checksum == checksum_bytes(
                question.question.encode("utf-8")
            ) and all(
                chunk_by_id.get(chunk_id) is not None
                and chunk_by_id[chunk_id].chunk_checksum == chunk_checksum
                for chunk_id, chunk_checksum in zip(
                    historical_item.evidence_ids,
                    historical_item.evidence_checksums,
                    strict=True,
                )
            )
            if not historical_identifiable:
                historical_relation = "INVALID"
            else:
                v2_ids = {item.chunk_id for item in positives}
                v1_ids = set(historical_item.evidence_ids)
                if v2_ids == v1_ids:
                    historical_relation = "EXACT"
                elif v2_ids & v1_ids:
                    historical_relation = "PARTIAL"
                else:
                    historical_relation = "NONE"
        rows.append(
            {
                "schema_version": "training.retrieval-supervision.group.v2",
                "group_id": f"retrieval_supervision_v2:{question_id}",
                "question_id": question_id,
                "question_checksum": checksum_bytes(question.question.encode("utf-8")),
                "source_answer_checksum": checksum_bytes(question.answer.encode("utf-8")),
                "parsed_legal_citations": [item.as_dict() for item in citations],
                "canonical_document_ids": list(context_ids),
                "canonical_chunk_ids": [item.chunk_id for item in positives],
                "canonical_chunk_checksums": [item.chunk_checksum for item in positives],
                "mapping_class": mapping_class,
                "mapping_confidence": confidence,
                "mapping_reason": reason,
                "ambiguity_state": ("FAIL_CLOSED" if mapping_class == "AMBIGUOUS" else "NONE"),
                "historical_v1_overlap": {
                    "mapping_present": historical_item is not None,
                    "reproducibly_identifiable": historical_identifiable,
                    "positive_set_relation": historical_relation,
                },
                "source_provenance": {
                    "question_source_artifact": question.source_artifact,
                    "question_source_checksum": question.source_checksum,
                    "questions_artifact_checksum": input_checksums["questions"],
                    "split_manifest_checksum": split_manifest_checksum,
                    "chunks_artifact_checksum": input_checksums["chunks"],
                    "contexts_artifact_checksum": input_checksums["contexts"],
                    "alias_artifact_checksum": input_checksums["aliases"],
                    "construction_version": "retrieval-supervision.v2",
                    "contains_generated_text": False,
                },
            }
        )

    groups_data = b"".join(content_json_bytes(row) for row in rows)
    row_by_id = {row["question_id"]: row for row in rows}
    identifiable = exact_overlap = partial_overlap = missing_overlap = 0
    resolved_historical = historical_ambiguous = historical_unresolved = 0
    for item in historical:
        historical_question = question_by_id.get(item.question_id)
        chunks_resolve = all(
            chunk_by_id.get(chunk_id) is not None
            and chunk_by_id[chunk_id].chunk_checksum == checksum
            for chunk_id, checksum in zip(item.evidence_ids, item.evidence_checksums, strict=True)
        )
        if (
            historical_question is None
            or item.question_id not in set(train_ids)
            or item.question_checksum
            != checksum_bytes(historical_question.question.encode("utf-8"))
            or not chunks_resolve
        ):
            continue
        identifiable += 1
        v2_ids = set(cast(list[str], row_by_id[item.question_id]["canonical_chunk_ids"]))
        v1_ids = set(item.evidence_ids)
        if v2_ids:
            resolved_historical += 1
        if v2_ids == v1_ids:
            exact_overlap += 1
        elif v2_ids & v1_ids:
            partial_overlap += 1
        else:
            missing_overlap += 1
        historical_mapping_class = row_by_id[item.question_id]["mapping_class"]
        historical_ambiguous += historical_mapping_class == "AMBIGUOUS"
        historical_unresolved += historical_mapping_class == "UNRESOLVED"

    report: dict[str, Any] = {
        "schema_version": "training.retrieval-supervision.report.v2",
        "construction_version": "retrieval-supervision.v2",
        "input_checksums": {
            **input_checksums,
            "split": split_manifest_checksum,
        },
        "artifact_checksums": {"groups": checksum_bytes(groups_data)},
        "total_train_fit_rows": len(train_ids),
        "canonical_chunk_count": len(chunks),
        "citation_bearing_rows": citation_bearing,
        "exact_document_mappings": exact_document,
        "exact_article_mappings": exact_article,
        "exact_clause_mappings": exact_clause,
        "exact_point_mappings": exact_point,
        "uniquely_resolved_groups": unique_resolved,
        "multi_positive_groups": sum(
            len(cast(list[str], row["canonical_chunk_ids"])) > 1 for row in rows
        ),
        "document_only_groups": class_counts["DOCUMENT_ONLY"],
        "ambiguous_groups": class_counts["AMBIGUOUS"],
        "unresolved_groups": class_counts["UNRESOLVED"],
        "mapping_class_counts": {name: class_counts[name] for name in MAPPING_CLASSES},
        "total_positive_chunks": positive_assignments,
        "distinct_positive_chunks": len(distinct_positive_ids),
        "coverage_relative_to_train_fit": unique_resolved / len(train_ids),
        "historical_v1": {
            "mapping_count": len(historical),
            "reproducibly_identifiable": identifiable,
            "resolved_by_v2": resolved_historical,
            "exact_positive_set_overlap": exact_overlap,
            "partial_positive_set_overlap": partial_overlap,
            "missing_positive_overlap": missing_overlap,
            "any_positive_overlap": exact_overlap + partial_overlap,
            "ambiguous_by_v2": historical_ambiguous,
            "unresolved_by_v2": historical_unresolved,
            "used_for_eligibility": False,
        },
        "eligibility_policy": {
            "minimum_reranker_score": None,
            "minimum_answer_token_coverage": None,
            "ambiguity_fail_closed": True,
        },
        "train_only": True,
        "contains_generated_text": False,
        "model_inference_performed": False,
        "gpu_or_modal_used": False,
        "execution_mode": "local-offline",
    }
    report_data = content_json_bytes(report)
    return SupervisionArtifacts(groups_data, report_data, report)


def build_retrieval_supervision(
    *,
    questions_data: bytes,
    train_question_ids: Sequence[str],
    chunks_data: bytes,
    contexts_data: bytes,
    aliases_data: bytes,
    historical_data: bytes,
    expected_input_checksums: Mapping[str, object],
    expected_train_count: int,
    expected_chunk_count: int,
    split_manifest_checksum: str | None = None,
) -> SupervisionArtifacts:
    """Build supervision from in-memory fixtures with exact input bindings."""

    inputs = {
        "questions": questions_data,
        "chunks": chunks_data,
        "contexts": contexts_data,
        "aliases": aliases_data,
        "historical": historical_data,
    }
    checksums = _validate_checksums(inputs, expected_input_checksums)
    return _build(
        questions_data=questions_data,
        train_question_ids=train_question_ids,
        chunks=_load_chunks(_jsonl_values(chunks_data, "chunks"), expected_chunk_count),
        context_values=_jsonl_values(contexts_data, "contexts"),
        alias_values=_jsonl_values(aliases_data, "aliases"),
        historical_values=_jsonl_values(historical_data, "historical"),
        input_checksums=checksums,
        split_manifest_checksum=split_manifest_checksum,
        expected_train_count=expected_train_count,
    )


def build_retrieval_supervision_paths(
    *,
    questions_path: Path,
    train_question_ids: Sequence[str],
    chunks_path: Path,
    contexts_path: Path,
    aliases_path: Path,
    historical_path: Path,
    expected_input_checksums: Mapping[str, object],
    expected_train_count: int,
    expected_chunk_count: int,
    split_manifest_checksum: str,
) -> SupervisionArtifacts:
    """Stream large canonical inputs while retaining only mapping identities."""

    paths = {
        "questions": questions_path,
        "chunks": chunks_path,
        "contexts": contexts_path,
        "aliases": aliases_path,
        "historical": historical_path,
    }
    checksums = _validate_path_checksums(paths, expected_input_checksums)
    return _build(
        questions_data=questions_path.read_bytes(),
        train_question_ids=train_question_ids,
        chunks=_load_chunks(_iter_jsonl(chunks_path, "chunks"), expected_chunk_count),
        context_values=_iter_jsonl(contexts_path, "contexts"),
        alias_values=_iter_jsonl(aliases_path, "aliases"),
        historical_values=_iter_jsonl(historical_path, "historical"),
        input_checksums=checksums,
        split_manifest_checksum=split_manifest_checksum,
        expected_train_count=expected_train_count,
    )


__all__ = [
    "MAPPING_CLASSES",
    "RetrievalSupervisionError",
    "SupervisionArtifacts",
    "build_retrieval_supervision",
    "build_retrieval_supervision_paths",
]
