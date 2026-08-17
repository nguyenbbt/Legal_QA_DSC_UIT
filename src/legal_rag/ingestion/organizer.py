"""Duplicate-aware organizer JSON readers for immutable external inputs."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal, NoReturn
from urllib.parse import urlsplit

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import ContextRecord, QuestionRecord

QuestionKind = Literal["train", "public"]

_CONTEXT_FILENAME = re.compile(r"(?:^|/)context_([0-9]+)\.json\Z")
_SIMPLE_MEMBER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONTEXT_PATTERN = "context_*.json"
_MAX_CONTEXT_FILES = 10_000
_MAX_CONTEXT_FILE_BYTES = 16 * 1024 * 1024
_MAX_CONTEXT_TOTAL_BYTES = 1024 * 1024 * 1024


class _RawObject(list[tuple[str, Any]]):
    pass


class _IntegerToken(str):
    pass


class OrganizerDataError(Exception):
    """One stable external-data failure with safe source provenance."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        artifact_path: str,
        json_path: str = "$",
        raw_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.artifact_path = artifact_path
        self.json_path = json_path
        self.raw_id = raw_id


@dataclass(frozen=True, slots=True)
class ImportWarning:
    code: str
    artifact_path: str
    message: str


@dataclass(frozen=True, slots=True)
class OrganizerFile:
    relative_path: str
    data: bytes


def discover_context_files(
    input_directory: Path,
    *,
    pattern: str,
) -> tuple[OrganizerFile, ...]:
    """Read a bounded direct set of regular context files in stable byte order."""

    if pattern != _CONTEXT_PATTERN:
        _fail(
            "DATA_CONTEXT_PATTERN_INVALID",
            "context discovery pattern is not approved",
            artifact_path="context-source",
        )
    if input_directory.is_symlink() or not input_directory.is_dir():
        _fail(
            "DATA_CONTEXT_DIRECTORY_INVALID",
            "context source must be a non-symlink directory",
            artifact_path="context-source",
        )
    try:
        with os.scandir(input_directory) as entries:
            selected = [entry for entry in entries if fnmatchcase(entry.name, pattern)]
    except OSError as error:
        raise OrganizerDataError(
            "DATA_CONTEXT_DIRECTORY_INVALID",
            "context source directory cannot be enumerated",
            artifact_path="context-source",
        ) from error
    selected.sort(key=lambda entry: entry.name.encode("utf-8"))
    if len(selected) > _MAX_CONTEXT_FILES:
        _fail(
            "DATA_CONTEXT_COUNT_LIMIT",
            "context source exceeds the approved file-count limit",
            artifact_path="context-source",
        )

    discovered: list[OrganizerFile] = []
    total_bytes = 0
    for entry in selected:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            _fail(
                "DATA_CONTEXT_FILE_UNSUPPORTED",
                "context source member must be a regular non-symlink file",
                artifact_path=entry.name,
            )
        try:
            size = entry.stat(follow_symlinks=False).st_size
            if size > _MAX_CONTEXT_FILE_BYTES:
                _fail(
                    "DATA_CONTEXT_FILE_SIZE_LIMIT",
                    "context source member exceeds the approved byte limit",
                    artifact_path=entry.name,
                )
            data = Path(entry.path).read_bytes()
        except OrganizerDataError:
            raise
        except OSError as error:
            raise OrganizerDataError(
                "DATA_CONTEXT_FILE_UNSUPPORTED",
                "context source member cannot be read",
                artifact_path=entry.name,
            ) from error
        if len(data) != size:
            _fail(
                "DATA_CONTEXT_FILE_CHANGED",
                "context source member changed during discovery",
                artifact_path=entry.name,
            )
        total_bytes += size
        if total_bytes > _MAX_CONTEXT_TOTAL_BYTES:
            _fail(
                "DATA_CONTEXT_TOTAL_SIZE_LIMIT",
                "context source exceeds the approved aggregate byte limit",
                artifact_path="context-source",
            )
        discovered.append(OrganizerFile(relative_path=entry.name, data=data))
    return tuple(discovered)


def _jsonl_bytes(records: tuple[QuestionRecord, ...] | tuple[ContextRecord, ...]) -> bytes:
    lines = (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class OrganizerQuestionSource:
    """Raw organizer fields retained only for exact submission reconstruction."""

    original_id: str
    question: str
    field_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionImport:
    records: tuple[QuestionRecord, ...]
    source_records: tuple[OrganizerQuestionSource, ...]
    warnings: tuple[ImportWarning, ...]

    def jsonl_bytes(self) -> bytes:
        return _jsonl_bytes(self.records)


@dataclass(frozen=True, slots=True)
class ContextImportEntry:
    source_artifact: str
    context_id: str
    source_checksum: str
    indexable: bool
    quarantine_reason: str | None
    source_position: int


@dataclass(frozen=True, slots=True)
class ContextImport:
    records: tuple[ContextRecord, ...]
    entries: tuple[ContextImportEntry, ...]
    warnings: tuple[ImportWarning, ...]

    def jsonl_bytes(self) -> bytes:
        return _jsonl_bytes(self.records)

    def manifest_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "context.import.v1",
                "entries": [
                    {
                        "source_artifact": entry.source_artifact,
                        "context_id": entry.context_id,
                        "source_checksum": entry.source_checksum,
                        "indexable": entry.indexable,
                        "quarantine_reason": entry.quarantine_reason,
                        "source_position": entry.source_position,
                    }
                    for entry in self.entries
                ],
            }
        )


def _member_path(parent: str, member: str) -> str:
    if _SIMPLE_MEMBER.fullmatch(member) is not None:
        return f"{parent}.{member}"
    return f"{parent}[{json.dumps(member, ensure_ascii=False)}]"


def _fail(
    code: str,
    message: str,
    *,
    artifact_path: str,
    json_path: str = "$",
    raw_id: str | None = None,
) -> NoReturn:
    raise OrganizerDataError(
        code,
        message,
        artifact_path=artifact_path,
        json_path=json_path,
        raw_id=raw_id,
    )


def _reject_duplicate_keys(value: Any, *, artifact_path: str, path: str = "$") -> None:
    if isinstance(value, _RawObject):
        seen: set[str] = set()
        for key, member in value:
            member_path = _member_path(path, key)
            if key in seen:
                _fail(
                    "DATA_DUPLICATE_KEY",
                    "organizer JSON contains a duplicate object member",
                    artifact_path=artifact_path,
                    json_path=member_path,
                    raw_id=key if path == "$" else None,
                )
            seen.add(key)
            _reject_duplicate_keys(member, artifact_path=artifact_path, path=member_path)
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _reject_duplicate_keys(
                member,
                artifact_path=artifact_path,
                path=f"{path}[{index}]",
            )


def _decode_json(
    data: bytes,
    *,
    artifact_path: str,
) -> tuple[Any, tuple[ImportWarning, ...]]:
    warnings: tuple[ImportWarning, ...] = ()
    content = data
    if data.startswith(b"\xef\xbb\xbf"):
        content = data[3:]
        warnings = (
            ImportWarning(
                code="DATA_UTF8_BOM_ACCEPTED",
                artifact_path=artifact_path,
                message="organizer UTF-8 BOM was accepted and removed during parsing",
            ),
        )
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail(
            "DATA_UTF8_INVALID",
            "organizer JSON is not strict UTF-8",
            artifact_path=artifact_path,
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_RawObject,
            parse_int=_IntegerToken,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (json.JSONDecodeError, ValueError):
        _fail("DATA_JSON_INVALID", "organizer JSON is malformed", artifact_path=artifact_path)
    _reject_duplicate_keys(value, artifact_path=artifact_path)
    return value, warnings


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, _RawObject) else None


def _validate_fields(
    members: dict[str, Any],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    artifact_path: str,
    path: str,
    raw_id: str | None,
) -> None:
    for field in required:
        if field not in members:
            _fail(
                "DATA_FIELD_MISSING",
                f"required organizer field is missing: {field}",
                artifact_path=artifact_path,
                json_path=_member_path(path, field),
                raw_id=raw_id,
            )
    allowed = set(required) | set(optional)
    for field in members:
        if field not in allowed:
            _fail(
                "DATA_FIELD_UNKNOWN",
                f"unsupported organizer field: {field}",
                artifact_path=artifact_path,
                json_path=_member_path(path, field),
                raw_id=raw_id,
            )


class OrganizerQuestionReader:
    """Convert one strict organizer question object into ordered internal records."""

    def read_bytes(
        self,
        data: bytes,
        *,
        kind: QuestionKind,
        artifact_path: str,
    ) -> QuestionImport:
        value, warnings = _decode_json(data, artifact_path=artifact_path)
        if not isinstance(value, _RawObject):
            _fail("DATA_ROOT_TYPE", "question root must be an object", artifact_path=artifact_path)
        source_checksum = checksum_bytes(data)
        records: list[QuestionRecord] = []
        source_records: list[OrganizerQuestionSource] = []
        for source_position, (raw_id, raw_record) in enumerate(value):
            record_path = _member_path("$", raw_id)
            if not raw_id:
                _fail(
                    "DATA_ID_EMPTY",
                    "question ID must be non-empty",
                    artifact_path=artifact_path,
                    json_path=record_path,
                )
            members = _as_mapping(raw_record)
            if members is None:
                _fail(
                    "DATA_RECORD_TYPE",
                    "question record must be an object",
                    artifact_path=artifact_path,
                    json_path=record_path,
                    raw_id=raw_id,
                )
            _validate_fields(
                members,
                required=("question", "answer"),
                optional=(),
                artifact_path=artifact_path,
                path=record_path,
                raw_id=raw_id,
            )
            question = members["question"]
            if type(question) is not str or not question.strip():
                _fail(
                    "DATA_QUESTION_TYPE",
                    "question must be a non-empty string",
                    artifact_path=artifact_path,
                    json_path=_member_path(record_path, "question"),
                    raw_id=raw_id,
                )
            answer = members["answer"]
            valid_answer = (
                type(answer) is str and bool(answer.strip()) if kind == "train" else answer is None
            )
            if not valid_answer:
                _fail(
                    "DATA_ANSWER_TYPE",
                    f"{kind} answer has the wrong type or state",
                    artifact_path=artifact_path,
                    json_path=_member_path(record_path, "answer"),
                    raw_id=raw_id,
                )
            records.append(
                QuestionRecord.model_validate(
                    {
                        "schema_version": "internal.question.v1",
                        "question_id": unicodedata.normalize("NFC", raw_id),
                        "original_id": unicodedata.normalize("NFC", raw_id),
                        "original_id_kind": "object_key_string",
                        "source_position": source_position,
                        "source_artifact": unicodedata.normalize("NFC", artifact_path),
                        "source_checksum": source_checksum,
                        "question": unicodedata.normalize("NFC", question),
                        "answer": (
                            unicodedata.normalize("NFC", answer) if type(answer) is str else None
                        ),
                        "answer_state": "gold" if kind == "train" else "unlabeled",
                    }
                )
            )
            source_records.append(
                OrganizerQuestionSource(
                    original_id=raw_id,
                    question=question,
                    field_order=tuple(members),
                )
            )
        return QuestionImport(
            records=tuple(records),
            source_records=tuple(source_records),
            warnings=warnings,
        )


def _valid_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
    )


@dataclass(frozen=True, slots=True)
class _ParsedContext:
    raw_path: str
    record: ContextRecord
    warnings: tuple[ImportWarning, ...]


class OrganizerContextReader:
    """Convert enumerated organizer context files using deterministic ID/path order."""

    def read_files(self, files: tuple[OrganizerFile, ...]) -> ContextImport:
        parsed = [self._parse_file(file) for file in files]
        parsed.sort(key=lambda item: (int(item.record.original_id), item.raw_path.encode("utf-8")))
        seen_ids: set[str] = set()
        records: list[ContextRecord] = []
        entries: list[ContextImportEntry] = []
        warnings: list[ImportWarning] = []
        for source_position, item in enumerate(parsed):
            if item.record.context_id in seen_ids:
                _fail(
                    "DATA_CONTEXT_ID_DUPLICATE",
                    "context ID occurs in more than one organizer file",
                    artifact_path=item.record.source_artifact,
                    raw_id=item.record.original_id,
                )
            seen_ids.add(item.record.context_id)
            record = item.record.model_copy(update={"source_position": source_position})
            records.append(record)
            entries.append(
                ContextImportEntry(
                    source_artifact=record.source_artifact,
                    context_id=record.context_id,
                    source_checksum=record.source_checksum,
                    indexable=record.indexable,
                    quarantine_reason=record.quarantine_reason,
                    source_position=source_position,
                )
            )
            warnings.extend(item.warnings)
        return ContextImport(
            records=tuple(records),
            entries=tuple(entries),
            warnings=tuple(warnings),
        )

    def _parse_file(self, file: OrganizerFile) -> _ParsedContext:
        artifact_path = unicodedata.normalize("NFC", file.relative_path.replace("\\", "/"))
        filename_match = _CONTEXT_FILENAME.search(file.relative_path.replace("\\", "/"))
        if filename_match is None:
            _fail(
                "DATA_CONTEXT_FILENAME",
                "context filename must match context_<integer>.json",
                artifact_path=artifact_path,
            )
        value, warnings = _decode_json(file.data, artifact_path=artifact_path)
        members = _as_mapping(value)
        if members is None:
            _fail("DATA_ROOT_TYPE", "context root must be an object", artifact_path=artifact_path)
        _validate_fields(
            members,
            required=("id", "link", "passage"),
            optional=("name",),
            artifact_path=artifact_path,
            path="$",
            raw_id=None,
        )
        raw_id = members["id"]
        if not isinstance(raw_id, _IntegerToken):
            _fail(
                "DATA_CONTEXT_ID_TYPE",
                "context id must be a JSON integer",
                artifact_path=artifact_path,
                json_path="$.id",
            )
        context_id = str(int(raw_id))
        if int(filename_match.group(1)) != int(raw_id):
            _fail(
                "DATA_CONTEXT_ID_FILENAME_MISMATCH",
                "context id does not match its filename",
                artifact_path=artifact_path,
                json_path="$.id",
                raw_id=str(raw_id),
            )
        name = members.get("name")
        if name is not None and type(name) is not str:
            _fail(
                "DATA_CONTEXT_NAME_TYPE",
                "context name must be a string when present",
                artifact_path=artifact_path,
                json_path="$.name",
                raw_id=str(raw_id),
            )
        link = members["link"]
        if type(link) is not str or not link.strip() or not _valid_http_url(link):
            _fail(
                "DATA_CONTEXT_LINK_INVALID",
                "context link must be a non-empty absolute HTTP(S) URL",
                artifact_path=artifact_path,
                json_path="$.link",
                raw_id=str(raw_id),
            )
        passage = members["passage"]
        if type(passage) is not str:
            _fail(
                "DATA_CONTEXT_PASSAGE_TYPE",
                "context passage must be a string",
                artifact_path=artifact_path,
                json_path="$.passage",
                raw_id=str(raw_id),
            )
        canonical_passage = unicodedata.normalize("NFC", passage)
        indexable = bool(canonical_passage.strip())
        record = ContextRecord.model_validate(
            {
                "schema_version": "internal.context.v1",
                "context_id": context_id,
                "original_id": str(raw_id),
                "original_id_kind": "json_integer",
                "source_position": 0,
                "source_artifact": artifact_path,
                "source_checksum": checksum_bytes(file.data),
                "name": unicodedata.normalize("NFC", name) if name is not None else None,
                "source_url": unicodedata.normalize("NFC", link),
                "passage": canonical_passage,
                "indexable": indexable,
                "quarantine_reason": None if indexable else "EMPTY_PASSAGE",
            }
        )
        return _ParsedContext(raw_path=file.relative_path, record=record, warnings=warnings)
