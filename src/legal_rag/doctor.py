"""Deterministic, offline readiness diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from legal_rag.config import AppConfig, safe_relative_path
from legal_rag.errors import CliError

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESOURCE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One stable readiness check."""

    code: str
    status: str = "pass"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "status": self.status}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Public `doctor.report.v1` output contract."""

    execution_mode: str
    checks: tuple[CheckResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": "doctor.report.v1",
            "status": "ready",
            "execution_mode": self.execution_mode,
            "checks": [check.as_dict() for check in self.checks],
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CliError(
            "OFFLINE_RESOURCE_MISSING", "required resource manifest is missing", exit_code=4
        ) from exc
    if size > _MAX_MANIFEST_BYTES:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource manifest exceeds size limit")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource manifest is invalid") from exc
    if not isinstance(value, dict):
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource manifest must be an object")
    return value


def _project_path(relative: PurePosixPath, *, root: Path, field: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root) or candidate.is_symlink():
        raise CliError("RESOURCE_MANIFEST_INVALID", f"{field} escapes the project root")
    return candidate


def _validate_resource_entry(value: object) -> tuple[str, PurePosixPath, str, int]:
    if not isinstance(value, dict):
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource entry must be an object")
    resource_id = value.get("resource_id")
    checksum = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(resource_id, str) or _RESOURCE_ID.fullmatch(resource_id) is None:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource_id is invalid")
    if not isinstance(checksum, str) or _HEX_SHA256.fullmatch(checksum) is None:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource sha256 is invalid")
    if type(size) is not int or size < 0:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource size_bytes is invalid")
    relative_path = safe_relative_path(value.get("relative_path"), field="relative_path")
    return resource_id, relative_path, checksum, size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CliError("RESOURCE_READ_ERROR", "manifested resource cannot be read", 4) from exc
    return digest.hexdigest()


def run_doctor(config: AppConfig, execution_mode: str, *, project_root: Path) -> DoctorReport:
    """Validate local manifested resources without network access or workloads."""

    manifest_path = _project_path(
        config.resource_manifest, root=project_root, field="resource_manifest"
    )
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_id") != "resources.manifest.v1":
        raise CliError("RESOURCE_MANIFEST_INVALID", "resource manifest schema_id is invalid")
    if manifest.get("runtime_download_allowed") is not False:
        raise CliError("RESOURCE_MANIFEST_INVALID", "runtime_download_allowed must be false")
    resource_root = safe_relative_path(manifest.get("resource_root"), field="resource_root")
    root_path = _project_path(resource_root, root=project_root, field="resource_root")
    raw_resources = manifest.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise CliError("RESOURCE_MANIFEST_INVALID", "resources must be a non-empty array")
    resources = [_validate_resource_entry(item) for item in raw_resources]
    ids = [item[0] for item in resources]
    if ids != sorted(ids, key=lambda value: value.encode("utf-8")) or len(ids) != len(set(ids)):
        raise CliError("RESOURCE_MANIFEST_INVALID", "resources must have unique ordered IDs")
    for resource_id, relative_path, checksum, expected_size in resources:
        resource_path = _project_path(relative_path, root=root_path, field="relative_path")
        if not resource_path.is_file():
            raise CliError(
                "OFFLINE_RESOURCE_MISSING",
                f"required manifested resource is missing: {resource_id}",
                exit_code=4,
            )
        if resource_path.stat().st_size != expected_size or _sha256_file(resource_path) != checksum:
            raise CliError(
                "RESOURCE_CHECKSUM_MISMATCH",
                f"manifested resource checksum does not match: {resource_id}",
                exit_code=4,
            )
    return DoctorReport(
        execution_mode=execution_mode,
        checks=(
            CheckResult("CONFIG_VALID"),
            CheckResult("RESOURCE_MANIFEST_VALID"),
            CheckResult("RESOURCES_PRESENT"),
        ),
    )
