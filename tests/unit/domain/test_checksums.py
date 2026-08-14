"""Golden vectors for deterministic bytes and file-set checksums."""

from __future__ import annotations

import unicodedata
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

import pytest

from legal_rag.domain.checksums import (
    DeterminismError,
    canonical_json_bytes,
    checksum_bytes,
    checksum_file,
    checksum_file_set,
)


def test_canonical_json_has_exact_nfc_compact_utf8_bytes_and_one_lf() -> None:
    decomposed = unicodedata.normalize("NFD", "Nội")

    actual = canonical_json_bytes({"z": None, "a": decomposed, "items": [True, 1]})

    assert actual == b'{"a":"N\xe1\xbb\x99i","items":[true,1],"z":null}\n'
    assert checksum_bytes(actual) == (
        "sha256:3b9d923a8a3d484acf29f9799dda821325b784f2b289cd3c6b8b188cccb22b38"
    )


def test_canonical_json_is_independent_of_mapping_insertion_order() -> None:
    left = canonical_json_bytes({"z": 2, "a": {"second": 2, "first": 1}})
    right = canonical_json_bytes({"a": {"first": 1, "second": 2}, "z": 2})

    assert left == right
    assert left.endswith(b"\n")
    assert not left.endswith(b"\n\n")


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ({"value": 1.5}, "RUN_CANONICAL_TYPE_UNSUPPORTED"),
        ({"value": date(2026, 8, 14)}, "RUN_CANONICAL_TYPE_UNSUPPORTED"),
        ({"value": datetime(2026, 8, 14)}, "RUN_CANONICAL_TYPE_UNSUPPORTED"),
        ({"value": UUID(int=0)}, "RUN_CANONICAL_TYPE_UNSUPPORTED"),
        ({"value": "/private/path"}, "RUN_CANONICAL_ABSOLUTE_PATH"),
        ({"value": "C:/private/path"}, "RUN_CANONICAL_ABSOLUTE_PATH"),
        (
            {"run_instance_id": "2a940d89-1934-41ef-897a-f6ab2d150f26"},
            "RUN_CANONICAL_FIELD_FORBIDDEN",
        ),
        ({"khóa": "value"}, "RUN_CANONICAL_KEY_UNSUPPORTED"),
    ],
)
def test_canonical_json_rejects_nondeterministic_inputs(value: object, code: str) -> None:
    with pytest.raises(DeterminismError) as captured:
        canonical_json_bytes(value)

    assert captured.value.code == code


def test_checksum_file_hashes_exact_raw_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "raw.txt"
    artifact.write_bytes(b"abc")

    assert checksum_file(artifact) == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_file_set_checksum_matches_length_prefixed_utf8_golden(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"A\r\n")
    (tmp_path / "\u0111.txt").write_bytes(b"\x00")

    result = checksum_file_set(tmp_path, ("\u0111.txt", "a.txt"))

    assert result.paths == ("a.txt", "\u0111.txt")
    assert result.checksum == (
        "sha256:64c886d5163d9d7b675c7a2078d4dc4892b666625d2c924dcff9e84c38fafff1"
    )
    assert checksum_file_set(tmp_path, ("a.txt", "\u0111.txt")) == result


def test_file_set_checksum_observes_raw_newline_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "a.txt"
    artifact.write_bytes(b"A\r\n")
    before = checksum_file_set(tmp_path, ("a.txt",)).checksum
    artifact.write_bytes(b"A\n")

    assert checksum_file_set(tmp_path, ("a.txt",)).checksum != before


def test_file_set_looks_up_decomposed_name_but_records_canonical_nfc_path(
    tmp_path: Path,
) -> None:
    decomposed = unicodedata.normalize("NFD", "đề.txt")
    canonical = unicodedata.normalize("NFC", decomposed)
    (tmp_path / decomposed).write_bytes(b"unicode-path")

    result = checksum_file_set(tmp_path, (decomposed,))

    assert result.paths == (canonical,)


@pytest.mark.parametrize("path", ["/absolute.txt", "C:/absolute.txt", "../outside.txt"])
def test_file_set_checksum_rejects_unsupported_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(DeterminismError) as captured:
        checksum_file_set(tmp_path, (path,))

    assert captured.value.code == "RUN_SOURCE_PATH_UNSUPPORTED"


def test_file_set_checksum_rejects_directories_and_duplicate_paths(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "file.txt").write_bytes(b"content")

    for paths in (("directory",), ("file.txt", "file.txt")):
        with pytest.raises(DeterminismError) as captured:
            checksum_file_set(tmp_path, paths)
        assert captured.value.code == "RUN_SOURCE_PATH_UNSUPPORTED"


def test_file_set_checksum_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_bytes(b"content")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(DeterminismError) as captured:
        checksum_file_set(tmp_path, ("link.txt",))

    assert captured.value.code == "RUN_SOURCE_PATH_UNSUPPORTED"


def test_file_set_checksum_always_checks_for_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "simulated-link.txt"
    artifact.write_bytes(b"content")
    original = Path.is_symlink

    def simulated_is_symlink(path: Path) -> bool:
        return path.name == artifact.name or original(path)

    monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)
    with pytest.raises(DeterminismError) as captured:
        checksum_file_set(tmp_path, (artifact.name,))

    assert captured.value.code == "RUN_SOURCE_PATH_UNSUPPORTED"
