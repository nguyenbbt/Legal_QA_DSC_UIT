"""Small, strict YAML configuration boundary for the bootstrap CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from legal_rag.errors import CliError

_CONFIG_SCHEMA_ID = "legal-rag.config.v1"
_MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated configuration used by bootstrap diagnostics."""

    resource_manifest: PurePosixPath


def safe_relative_path(value: object, *, field: str) -> PurePosixPath:
    """Return a normalized project-relative POSIX path or fail closed."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CliError("CONFIG_SCHEMA_INVALID", f"{field} must be a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CliError("CONFIG_SCHEMA_INVALID", f"{field} must be a safe relative path")
    return path


def load_config(path: Path) -> AppConfig:
    """Load a closed bootstrap config using PyYAML's safe loader."""

    if path.suffix.casefold() not in {".yaml", ".yml"}:
        raise CliError("CONFIG_EXTENSION_UNSUPPORTED", "config must use .yaml or .yml")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CliError("CONFIG_NOT_FOUND", "config file is not available") from exc
    if size > _MAX_CONFIG_BYTES:
        raise CliError("CONFIG_TOO_LARGE", "config exceeds the 1048576-byte limit")
    try:
        text = path.read_text(encoding="utf-8")
        value: Any = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CliError("CONFIG_PARSE_ERROR", "config is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CliError("CONFIG_SCHEMA_INVALID", "config must be a string-keyed object")
    expected = {"schema_id", "resource_manifest"}
    if set(value) != expected or value.get("schema_id") != _CONFIG_SCHEMA_ID:
        raise CliError("CONFIG_SCHEMA_INVALID", "config does not match legal-rag.config.v1")
    return AppConfig(
        resource_manifest=safe_relative_path(
            value.get("resource_manifest"), field="resource_manifest"
        )
    )
