from __future__ import annotations

from pathlib import Path

from legal_rag.models.local_artifact import checksum_artifact_files, checksum_model_directory


def test_model_tree_checksum_is_path_independent_and_material(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_bytes(b"{}")
    (second / "config.json").write_bytes(b"{}")

    assert checksum_model_directory(first) == checksum_model_directory(second)
    (second / "config.json").write_bytes(b'{"changed":true}')
    assert checksum_model_directory(first) != checksum_model_directory(second)


def test_declared_file_checksum_is_order_independent(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")

    assert checksum_artifact_files(tmp_path, ("a", "b")) == checksum_artifact_files(
        tmp_path, ("b", "a")
    )
