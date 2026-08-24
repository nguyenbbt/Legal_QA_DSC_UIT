"""Explicit prepare-online acquisition of immutable public model snapshots."""

from __future__ import annotations

import re
from pathlib import Path

from legal_rag.models.approval import validate_acquisition_mode
from legal_rag.models.manifest import ModelManifestError

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def acquire_snapshot(
    *,
    model_id: str,
    revision: str,
    local_directory: Path,
) -> Path:
    """Download one exact revision; never resolve mutable ``main`` implicitly."""

    validate_acquisition_mode("prepare-online")
    if _COMMIT.fullmatch(revision) is None:
        raise ModelManifestError(
            "MODEL_REVISION_UNPINNED", "model acquisition revision must be a commit SHA"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelManifestError(
            "MODEL_DEPENDENCY_MISSING", "huggingface-hub is not installed"
        ) from error
    local_directory.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_directory,
    )
    return Path(resolved)


__all__ = ["acquire_snapshot"]
