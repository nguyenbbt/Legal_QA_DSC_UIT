"""Exact deterministic byte, checksum, and run-fingerprint contracts."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from legal_rag.domain.models import RunManifest

_STATIC_KEY = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_UUID_TEXT = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
_TIMESTAMP_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:")
_FORBIDDEN_FIELDS = frozenset(
    {
        "actual_cost",
        "billing_event",
        "cpu_utilization",
        "credentials",
        "duration",
        "gpu_utilization",
        "host_id",
        "hostname",
        "infrastructure_message",
        "job_id",
        "local_absolute_path",
        "observed_peak_memory",
        "process_id",
        "retry_timing",
        "run_instance_id",
        "timestamp",
        "wall_clock_duration",
        "wall_clock_timestamp",
    }
)
_RUN_FINGERPRINT_FIELDS = (
    "schema_version",
    "pipeline_version",
    "code_revision",
    "source_tree_checksum",
    "scoped_source_paths",
    "config_checksum",
    "question_checksum",
    "corpus_checksum",
    "index_checksum",
    "split_checksum",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "prompt_revision",
    "seed",
    "execution_mode",
    "competition_policy",
    "comparison_type",
    "resolved_as_of_date",
    "as_of_timezone",
    "resource_manifest_checksum",
)


class DeterminismError(Exception):
    """Stable safe failure for deterministic artifact operations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class FileSetChecksum:
    """A checksum and the exact canonical path order that produced it."""

    checksum: str
    paths: tuple[str, ...]


def _fail(code: str, message: str) -> NoReturn:
    raise DeterminismError(code, message)


def _looks_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE.match(value) is not None


def _canonical_value(value: object) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        if "\x00" in normalized:
            _fail("RUN_CANONICAL_STRING_UNSUPPORTED", "canonical strings must not contain NUL")
        if _looks_absolute_path(normalized):
            _fail(
                "RUN_CANONICAL_ABSOLUTE_PATH",
                "absolute paths are forbidden in deterministic JSON",
            )
        if _UUID_TEXT.fullmatch(normalized) is not None:
            _fail("RUN_CANONICAL_UUID_FORBIDDEN", "UUIDs are forbidden in deterministic JSON")
        if _TIMESTAMP_TEXT.match(normalized) is not None:
            _fail(
                "RUN_CANONICAL_TIMESTAMP_FORBIDDEN",
                "timestamps are forbidden in deterministic JSON",
            )
        return normalized
    if isinstance(value, Mapping):
        normalized_members: list[tuple[str, Any]] = []
        for member_key, member_value in value.items():
            if not isinstance(member_key, str) or _STATIC_KEY.fullmatch(member_key) is None:
                _fail(
                    "RUN_CANONICAL_KEY_UNSUPPORTED",
                    "deterministic JSON keys must be static ASCII snake-case names",
                )
            if member_key in _FORBIDDEN_FIELDS:
                _fail(
                    "RUN_CANONICAL_FIELD_FORBIDDEN",
                    "operational fields are forbidden in deterministic JSON",
                )
            normalized_members.append((member_key, _canonical_value(member_value)))
        normalized_members.sort(key=lambda item: item[0].encode("utf-8"))
        return dict(normalized_members)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_canonical_value(member) for member in value]
    _fail(
        "RUN_CANONICAL_TYPE_UNSUPPORTED",
        f"type {type(value).__name__} is forbidden in deterministic JSON",
    )


def canonical_json_bytes(value: object) -> bytes:
    """Serialize an allowed value using the exact section 5.7 byte contract."""

    canonical = _canonical_value(value)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + b"\n"


def checksum_bytes(data: bytes) -> str:
    """Return the typed SHA-256 checksum of exact bytes."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def checksum_file(path: Path) -> str:
    """Hash one regular non-symlink file without newline conversion."""

    if path.is_symlink() or not path.is_file():
        _fail("RUN_SOURCE_PATH_UNSUPPORTED", "checksum input must be a regular file")
    return checksum_bytes(path.read_bytes())


def _canonical_relative_path(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\\" in normalized
        or "\x00" in normalized
        or _looks_absolute_path(normalized)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(
            "RUN_SOURCE_PATH_UNSUPPORTED",
            "source paths must be normalized repository-relative POSIX paths",
        )
    return path.as_posix()


def _contains_symlink(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def checksum_file_set(repository_root: Path, paths: Sequence[str]) -> FileSetChecksum:
    """Hash an explicit regular-file set using the length-prefixed tree algorithm."""

    root = repository_root.resolve(strict=True)
    entries: list[tuple[bytes, bytes]] = []
    canonical_paths: list[str] = []
    seen: set[str] = set()
    for supplied_path in paths:
        canonical_path = _canonical_relative_path(supplied_path)
        if canonical_path in seen:
            _fail("RUN_SOURCE_PATH_UNSUPPORTED", "source paths must be unique")
        seen.add(canonical_path)
        lookup_relative = PurePosixPath(supplied_path)
        candidate = root.joinpath(*lookup_relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            _fail(
                "RUN_SOURCE_PATH_UNSUPPORTED",
                "source path must resolve to a file inside the repository root",
            )
        if _contains_symlink(root, lookup_relative) or not resolved.is_file():
            _fail(
                "RUN_SOURCE_PATH_UNSUPPORTED",
                "source path must be a regular non-symlink file",
            )
        path_bytes = canonical_path.encode("utf-8")
        file_digest = hashlib.sha256(resolved.read_bytes()).digest()
        entry = struct.pack(">Q", len(path_bytes)) + path_bytes + file_digest
        entries.append((path_bytes, entry))
        canonical_paths.append(canonical_path)

    entries.sort(key=lambda item: item[0])
    canonical_paths.sort(key=lambda item: item.encode("utf-8"))
    digest = hashlib.sha256(b"".join(entry for _, entry in entries)).hexdigest()
    return FileSetChecksum(checksum=f"sha256:{digest}", paths=tuple(canonical_paths))


def _run_fingerprint_payload(manifest: RunManifest) -> dict[str, object]:
    dumped = manifest.model_dump(mode="json")
    return {field: dumped[field] for field in _RUN_FINGERPRINT_FIELDS}


def compute_run_id(manifest: RunManifest) -> str:
    """Compute the content-addressed ID while excluding completed-output checksums."""

    digest = hashlib.sha256(canonical_json_bytes(_run_fingerprint_payload(manifest))).hexdigest()
    return f"run_{digest[:24]}"


def validate_run_manifest_identity(manifest: RunManifest) -> RunManifest:
    """Prove that a stored run ID equals its deterministic fingerprint."""

    if manifest.run_id != compute_run_id(manifest):
        _fail("RUN_ID_MISMATCH", "stored run_id does not match the manifest fingerprint")
    return manifest


def validate_run_output_checksums(
    manifest: RunManifest,
    evidence_diagnostics_path: Path,
    answer_artifact_path: Path,
) -> RunManifest:
    """Prove both post-identity output checksums in a completed run manifest."""

    actual_evidence = checksum_file(evidence_diagnostics_path)
    actual_answer = checksum_file(answer_artifact_path)
    if (
        manifest.evidence_diagnostics_checksum != actual_evidence
        or manifest.answer_artifact_checksum != actual_answer
    ):
        _fail(
            "RUN_OUTPUT_CHECKSUM_MISMATCH",
            "completed output checksum does not match the run manifest",
        )
    return manifest
