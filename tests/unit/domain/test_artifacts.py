"""Atomic no-clobber tests for immutable deterministic artifact writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from legal_rag.domain.artifacts import (
    ImmutableArtifactError,
    write_immutable_bytes,
    write_immutable_chunks,
)


def test_concurrent_destination_creation_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "artifact.json"
    original_link = os.link

    def create_racing_destination(source: Path, target: Path) -> None:
        target.write_bytes(b"other-process\n")
        original_link(source, target)

    monkeypatch.setattr(os, "link", create_racing_destination)

    with pytest.raises(ImmutableArtifactError) as captured:
        write_immutable_bytes(destination, b"our-process\n")

    assert captured.value.code == "ARTIFACT_IMMUTABLE"
    assert destination.read_bytes() == b"other-process\n"


def test_chunk_writer_creates_and_accepts_an_identical_retry(tmp_path: Path) -> None:
    destination = tmp_path / "large.jsonl"

    first = write_immutable_chunks(destination, iter((b'{"id":1}\n', b'{"id":2}\n')))
    second = write_immutable_chunks(destination, iter((b'{"id":1}\n', b'{"id":2}\n')))

    assert first == second
    assert destination.read_bytes() == b'{"id":1}\n{"id":2}\n'


def test_chunk_writer_removes_temporary_file_when_generation_fails(tmp_path: Path) -> None:
    destination = tmp_path / "large.jsonl"

    def broken_chunks():
        yield b'{"id":1}\n'
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        write_immutable_chunks(destination, broken_chunks())

    assert not destination.exists()
    assert not tuple(tmp_path.glob("*.tmp"))


def test_chunk_writer_rejects_a_different_retry(tmp_path: Path) -> None:
    destination = tmp_path / "large.jsonl"
    write_immutable_chunks(destination, iter((b"first\n",)))

    with pytest.raises(ImmutableArtifactError) as captured:
        write_immutable_chunks(destination, iter((b"second\n",)))

    assert captured.value.code == "ARTIFACT_IMMUTABLE"
    assert destination.read_bytes() == b"first\n"
