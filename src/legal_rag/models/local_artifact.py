"""Immutable project-local model artifact inventory and hashing."""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

from legal_rag.models.manifest import ModelManifestError


def checksum_model_directory(path: Path) -> str:
    """Hash a regular-file tree without following symlinks or recording local paths."""

    root = path.resolve(strict=True)
    if not root.is_dir() or path.is_symlink():
        raise ModelManifestError(
            "MODEL_ARTIFACT_UNSUPPORTED", "model artifact must be a regular local directory"
        )
    entries: list[tuple[bytes, bytes]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in (*directories, *files)):
            raise ModelManifestError(
                "MODEL_ARTIFACT_UNSUPPORTED", "model artifact cannot contain symlinks"
            )
        for name in files:
            file_path = current_path / name
            relative = file_path.relative_to(root).as_posix()
            relative_bytes = relative.encode("utf-8")
            digest = hashlib.sha256()
            with file_path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            entry = struct.pack(">Q", len(relative_bytes)) + relative_bytes + digest.digest()
            entries.append((relative_bytes, entry))
    if not entries:
        raise ModelManifestError("MODEL_ARTIFACT_EMPTY", "model artifact contains no files")
    entries.sort(key=lambda item: item[0])
    return "sha256:" + hashlib.sha256(b"".join(entry for _, entry in entries)).hexdigest()


def checksum_artifact_files(path: Path, relative_names: tuple[str, ...]) -> str:
    """Hash a declared subset such as the tokenizer identity files."""

    root = path.resolve(strict=True)
    if not relative_names or len(relative_names) != len(set(relative_names)):
        raise ModelManifestError(
            "MODEL_ARTIFACT_INVENTORY_INVALID",
            "artifact file inventory must be non-empty and unique",
        )
    entries: list[tuple[bytes, bytes]] = []
    for relative_name in relative_names:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelManifestError(
                "MODEL_ARTIFACT_INVENTORY_INVALID", "artifact file inventory is unsafe"
            )
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ModelManifestError(
                "MODEL_ARTIFACT_INVENTORY_INVALID", "artifact inventory file is unavailable"
            )
        relative_bytes = relative.as_posix().encode("utf-8")
        digest = hashlib.sha256(candidate.read_bytes()).digest()
        entry = struct.pack(">Q", len(relative_bytes)) + relative_bytes + digest
        entries.append((relative_bytes, entry))
    entries.sort(key=lambda item: item[0])
    return "sha256:" + hashlib.sha256(b"".join(entry for _, entry in entries)).hexdigest()


__all__ = ["checksum_artifact_files", "checksum_model_directory"]
