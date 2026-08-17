"""Small atomic writer for immutable deterministic artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path


class ImmutableArtifactError(Exception):
    """Stable failure for an immutable artifact create/retry operation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _files_equal(left: Path, right: Path) -> bool:
    if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while left_chunk := left_stream.read(1024 * 1024):
            if left_chunk != right_stream.read(len(left_chunk)):
                return False
    return True


def write_immutable_chunks(destination: Path, chunks: Iterable[bytes]) -> str:
    """Stream an atomic artifact; accept identical retries and reject replacement."""

    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                for chunk in chunks:
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists():
                if _files_equal(temporary_path, destination):
                    return f"sha256:{digest.hexdigest()}"
                raise ImmutableArtifactError(
                    "ARTIFACT_IMMUTABLE", "an existing immutable artifact cannot be replaced"
                )
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                if _files_equal(temporary_path, destination):
                    return f"sha256:{digest.hexdigest()}"
                raise ImmutableArtifactError(
                    "ARTIFACT_IMMUTABLE",
                    "an existing immutable artifact cannot be replaced",
                ) from None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except ImmutableArtifactError:
        raise
    except OSError as error:
        raise ImmutableArtifactError(
            "ARTIFACT_WRITE_FAILED", "immutable artifact could not be written"
        ) from error
    return f"sha256:{digest.hexdigest()}"


def write_immutable_bytes(destination: Path, data: bytes) -> str:
    """Create bytes atomically; accept identical retries and reject replacement."""

    return write_immutable_chunks(destination, (data,))
