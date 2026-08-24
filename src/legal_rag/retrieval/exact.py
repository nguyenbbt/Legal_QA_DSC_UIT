"""Fail-closed legal-reference parsing and approved alias resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import (
    CanonicalIntegerString,
    ContextRecord,
    FrozenStrictModel,
    NfcString,
    NonNegativeInt,
    SafeRelativePath,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic

REFERENCE_PARSER_VERSION = "legal-reference-parser.v1"
DOCUMENT_KEY_VERSION = "legal-document-number-key.v1"

_H = r"[^\S\r\n]"
_FLAGS = re.IGNORECASE | re.UNICODE | re.VERBOSE
_COORDINATE = re.compile(
    rf"""
    (?<!\w)
    (?:điểm {_H}+ (?P<point>[a-zđ]) {_H}* (?:[,;:] {_H}*)?)?
    (?:khoản {_H}+ (?P<clause>[0-9]+[a-zđ]?) {_H}* (?:[,;:] {_H}*)?)?
    điều {_H}+ (?P<article>[0-9]+[a-zđ]?)
    (?!\w)
    """,
    _FLAGS,
)
_DOCUMENT_NUMBER = re.compile(
    rf"""
    (?<!\w)(?:số {_H}*)?
    (?P<document_number>
      [0-9]+/
      (?:[0-9]{{4}}/)?
      [0-9a-zđ]+(?:-[0-9a-zđ]+)*
    )
    (?!\w)
    """,
    _FLAGS,
)
_REFERENCE_TOKEN = re.compile(
    rf"(?<!\w)(?:điểm{_H}+[a-zđ]|khoản{_H}+[0-9]+[a-zđ]?|điều{_H}+[0-9]+[a-zđ]?)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


class AliasArtifactError(Exception):
    def __init__(self, code: str, message: str, *, artifact_path: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.artifact_path = artifact_path


def document_number_key(value: str) -> str:
    canonical = unicodedata.normalize("NFC", value)
    folded = canonical.casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    unmarked = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    mapped = unmarked.replace("đ", "d")
    return unicodedata.normalize("NFC", mapped)


class LegalReferenceAlias(FrozenStrictModel, frozen=True):
    schema_version: Literal["legal.reference.alias.v1"]
    document_number: NfcString
    document_number_key: NfcString
    context_id: CanonicalIntegerString
    source_kind: Literal["passage_header", "organizer_name", "owner_override"]
    canonical_start: NonNegativeInt | None
    canonical_end: NonNegativeInt | None
    review_state: Literal["approved"]

    @model_validator(mode="after")
    def _validate_key_and_offsets(self) -> Self:
        if self.document_number_key != document_number_key(self.document_number):
            raise ValueError("document_number_key does not match its canonical identity")
        paired = self.canonical_start is not None and self.canonical_end is not None
        if self.source_kind == "passage_header":
            if not paired or self.canonical_start >= self.canonical_end:  # type: ignore[operator]
                raise ValueError("passage_header requires a non-empty canonical offset pair")
        elif self.canonical_start is not None or self.canonical_end is not None:
            raise ValueError("non-passage aliases require null canonical offsets")
        return self


@dataclass(frozen=True, slots=True)
class AliasIndex:
    records: tuple[LegalReferenceAlias, ...]
    corpus_checksum: str
    artifact_path: str
    artifact_checksum: str

    def context_ids_for(self, key: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                record.context_id for record in self.records if record.document_number_key == key
            )
        )

    def manifest_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "legal.reference.alias.manifest.v1",
                "document_key_version": DOCUMENT_KEY_VERSION,
                "unicode_version": unicodedata.unidata_version,
                "corpus_checksum": self.corpus_checksum,
                "ordered_files": [{"path": self.artifact_path, "checksum": self.artifact_checksum}],
                "record_count": len(self.records),
                "aggregate_checksum": self.artifact_checksum,
            }
        )

    @property
    def manifest_checksum(self) -> str:
        return checksum_bytes(self.manifest_bytes())


class AliasManifestFile(FrozenStrictModel, frozen=True):
    path: SafeRelativePath
    checksum: Sha256


class AliasManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["legal.reference.alias.manifest.v1"]
    document_key_version: Literal["legal-document-number-key.v1"]
    unicode_version: NfcString
    corpus_checksum: Sha256
    ordered_files: tuple[AliasManifestFile, ...]
    record_count: NonNegativeInt
    aggregate_checksum: Sha256


def _alias_error(code: str, message: str, artifact_path: str) -> AliasArtifactError:
    return AliasArtifactError(code, message, artifact_path=artifact_path)


def _validate_alias_context(
    record: LegalReferenceAlias,
    contexts_by_id: dict[str, ContextRecord],
    artifact_path: str,
) -> None:
    context = contexts_by_id.get(record.context_id)
    if context is None or not context.indexable:
        raise _alias_error(
            "ALIAS_CONTEXT_INVALID",
            "alias context must resolve to one integrity-valid indexable context",
            artifact_path,
        )
    if record.source_kind != "passage_header":
        return
    assert record.canonical_start is not None
    assert record.canonical_end is not None
    if (
        record.canonical_end > len(context.passage)
        or context.passage[record.canonical_start : record.canonical_end] != record.document_number
    ):
        raise _alias_error(
            "ALIAS_PROVENANCE_INVALID",
            "alias canonical offsets do not reconstruct the document number",
            artifact_path,
        )


def _parse_alias_records(data: bytes, artifact_path: str) -> tuple[LegalReferenceAlias, ...]:
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        raise _alias_error(
            "ALIAS_ENCODING_INVALID",
            "alias JSONL must be UTF-8 without BOM and end with LF",
            artifact_path,
        )
    records: list[LegalReferenceAlias] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise _alias_error(
                "ALIAS_SCHEMA_INVALID",
                "alias JSONL contains an empty record",
                artifact_path,
            )
        try:
            record = parse_record_json(
                line + b"\n",
                LegalReferenceAlias,
                artifact_path=artifact_path,
                record_identity=str(line_number),
            )
        except RecordValidationError as error:
            message = error.issues[0].message if error.issues else "alias schema is invalid"
            raise _alias_error("ALIAS_SCHEMA_INVALID", message, artifact_path) from error
        records.append(record)
    order = lambda record: (  # noqa: E731 - local exact contract key
        record.document_number_key.encode("utf-8"),
        int(record.context_id),
        record.document_number.encode("utf-8"),
    )
    if records != sorted(records, key=order):
        raise _alias_error(
            "ALIAS_ORDER_INVALID",
            "alias records are not in canonical byte/numeric order",
            artifact_path,
        )
    if len(records) != len(set(records)):
        raise _alias_error(
            "ALIAS_RECORD_DUPLICATE",
            "alias artifact contains an identical duplicate record",
            artifact_path,
        )
    return tuple(records)


def load_alias_artifact(
    data: bytes,
    *,
    contexts: tuple[ContextRecord, ...],
    corpus_checksum: str,
    artifact_path: str,
) -> AliasIndex:
    """Validate an approved alias JSONL artifact without deriving live aliases."""

    records = _parse_alias_records(data, artifact_path)
    contexts_by_id: dict[str, ContextRecord] = {}
    for context in contexts:
        if context.context_id in contexts_by_id:
            raise _alias_error(
                "ALIAS_CONTEXT_DUPLICATE",
                "active corpus contains a duplicate context ID",
                artifact_path,
            )
        contexts_by_id[context.context_id] = context
    for record in records:
        _validate_alias_context(record, contexts_by_id, artifact_path)
    return AliasIndex(
        records=records,
        corpus_checksum=corpus_checksum,
        artifact_path=artifact_path,
        artifact_checksum=checksum_bytes(data),
    )


def load_frozen_alias_artifact(
    data: bytes,
    *,
    manifest_data: bytes,
    corpus_checksum: str,
    artifact_path: str,
) -> AliasIndex:
    """Load a provenance-validated alias freeze through its immutable manifest."""

    records = _parse_alias_records(data, artifact_path)
    try:
        manifest = parse_record_json(
            manifest_data,
            AliasManifest,
            artifact_path="aliases.active.manifest.json",
            record_identity="manifest",
        )
    except RecordValidationError as error:
        message = error.issues[0].message if error.issues else "alias manifest is invalid"
        raise _alias_error("ALIAS_MANIFEST_INVALID", message, artifact_path) from error
    artifact_checksum = checksum_bytes(data)
    if (
        manifest.corpus_checksum != corpus_checksum
        or manifest.document_key_version != DOCUMENT_KEY_VERSION
        or manifest.unicode_version != unicodedata.unidata_version
        or len(manifest.ordered_files) != 1
        or manifest.ordered_files[0].path != artifact_path
        or manifest.ordered_files[0].checksum != artifact_checksum
        or manifest.aggregate_checksum != artifact_checksum
        or manifest.record_count != len(records)
    ):
        raise _alias_error(
            "ALIAS_MANIFEST_MISMATCH",
            "alias checksum or identity differs from its active manifest",
            artifact_path,
        )
    return AliasIndex(
        records=records,
        corpus_checksum=corpus_checksum,
        artifact_path=artifact_path,
        artifact_checksum=artifact_checksum,
    )


@dataclass(frozen=True, slots=True)
class LegalReference:
    point: str | None
    clause: str | None
    article: str
    document_number: str | None


@dataclass(frozen=True, slots=True)
class LegalReferenceParseResult:
    reference: LegalReference | None
    diagnostics: tuple[RetrievalDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class DocumentNumberMatch:
    document_number: str
    canonical_start: int
    canonical_end: int


def find_document_numbers(text: str) -> tuple[DocumentNumberMatch, ...]:
    """Return grammar-valid document numbers with offsets into canonical NFC text."""

    canonical = unicodedata.normalize("NFC", text)
    return tuple(
        DocumentNumberMatch(
            document_number=match.group("document_number"),
            canonical_start=match.start("document_number"),
            canonical_end=match.end("document_number"),
        )
        for match in _DOCUMENT_NUMBER.finditer(canonical)
    )


def _parse_failure(code: str, message: str, count: int = 0) -> LegalReferenceParseResult:
    return LegalReferenceParseResult(
        reference=None,
        diagnostics=(RetrievalDiagnostic(code=code, message=message, candidate_count=count),),
    )


def parse_legal_reference(question: str) -> LegalReferenceParseResult:
    view = unicodedata.normalize("NFC", question).casefold()
    coordinate_matches = tuple(_COORDINATE.finditer(view))
    if not coordinate_matches:
        code = (
            "EXACT_REFERENCE_MALFORMED"
            if _REFERENCE_TOKEN.search(view)
            else "EXACT_COORDINATE_ABSENT"
        )
        return _parse_failure(code, "question does not contain one valid legal coordinate")
    if len(coordinate_matches) != 1:
        return _parse_failure(
            "EXACT_COORDINATE_AMBIGUOUS",
            "question contains more than one legal coordinate",
            len(coordinate_matches),
        )
    coordinate = coordinate_matches[0]
    if any(
        not (coordinate.start() <= token.start() and token.end() <= coordinate.end())
        for token in _REFERENCE_TOKEN.finditer(view)
    ):
        return _parse_failure(
            "EXACT_REFERENCE_MALFORMED",
            "question contains an unconsumed or out-of-order hierarchy token",
        )
    point = coordinate.group("point")
    clause = coordinate.group("clause")
    if point is not None and clause is None:
        return _parse_failure(
            "EXACT_REFERENCE_MALFORMED",
            "a point coordinate requires its parent clause",
        )
    document_matches = find_document_numbers(view)
    if len(document_matches) > 1:
        return _parse_failure(
            "EXACT_DOCUMENT_AMBIGUOUS",
            "question contains more than one document number",
            len(document_matches),
        )
    document_number = document_matches[0].document_number if document_matches else None
    return LegalReferenceParseResult(
        reference=LegalReference(
            point=point,
            clause=clause,
            article=coordinate.group("article"),
            document_number=document_number,
        ),
        diagnostics=(),
    )


@dataclass(frozen=True, slots=True)
class ExactResolution:
    candidates: tuple[RetrievalCandidate, ...]
    diagnostics: tuple[RetrievalDiagnostic, ...]


def _path_contains(path: tuple[str, ...], label: str, ordinal: str) -> bool:
    target = f"{label} {ordinal}".casefold()
    return any(member.casefold() == target for member in path)


def _matches_coordinate(chunk: ChunkRecord, reference: LegalReference) -> bool:
    if not _path_contains(chunk.hierarchy_path, "Điều", reference.article):
        return False
    if reference.clause is not None and not _path_contains(
        chunk.hierarchy_path, "Khoản", reference.clause
    ):
        return False
    if reference.point is not None:
        return chunk.hierarchy_kind == "point" and chunk.hierarchy_ordinal == reference.point
    if reference.clause is not None:
        return chunk.hierarchy_kind == "clause" and chunk.hierarchy_ordinal == reference.clause
    return chunk.hierarchy_kind == "article" and chunk.hierarchy_ordinal == reference.article


def _resolution_diagnostic(
    code: str,
    message: str,
    aliases: AliasIndex,
    count: int,
) -> ExactResolution:
    return ExactResolution(
        candidates=(),
        diagnostics=(
            RetrievalDiagnostic(
                code=code,
                message=message,
                alias_manifest_checksum=aliases.manifest_checksum,
                candidate_count=count,
            ),
        ),
    )


def resolve_exact_reference(
    reference: LegalReference,
    *,
    aliases: AliasIndex,
    chunks: tuple[ChunkRecord, ...],
) -> ExactResolution:
    allowed_context_ids: set[str] | None = None
    if reference.document_number is not None:
        resolved_ids = aliases.context_ids_for(document_number_key(reference.document_number))
        if not resolved_ids:
            return _resolution_diagnostic(
                "EXACT_DOCUMENT_UNRESOLVED",
                "document number is absent from the approved alias artifact",
                aliases,
                0,
            )
        if len(resolved_ids) > 1:
            return _resolution_diagnostic(
                "EXACT_DOCUMENT_AMBIGUOUS",
                "document number resolves to more than one context",
                aliases,
                len(resolved_ids),
            )
        allowed_context_ids = set(resolved_ids)
    matching = tuple(
        chunk
        for chunk in chunks
        if (allowed_context_ids is None or chunk.context_id in allowed_context_ids)
        and _matches_coordinate(chunk, reference)
    )
    if not matching:
        return _resolution_diagnostic(
            "EXACT_COORDINATE_UNRESOLVED",
            "legal coordinate does not resolve to an active chunk",
            aliases,
            0,
        )
    if len(matching) > 1:
        return _resolution_diagnostic(
            "EXACT_COORDINATE_MULTI_CHUNK",
            "legal coordinate resolves to more than one active chunk",
            aliases,
            len(matching),
        )
    return ExactResolution(
        candidates=(
            RetrievalCandidate(
                chunk=matching[0],
                exact_reference_match=True,
                sparse_score=None,
            ),
        ),
        diagnostics=(),
    )
