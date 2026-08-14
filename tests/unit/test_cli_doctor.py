from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import NoReturn

import nltk
import pytest

from legal_rag.cli import main


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_ready_config(root: Path) -> Path:
    resource_root = root / "resources"
    resource_root.mkdir()
    payload = "dữ liệu tổng hợp\n".encode()
    (resource_root / "fixture.txt").write_bytes(payload)
    manifest = {
        "schema_id": "resources.manifest.v1",
        "resource_root": "resources",
        "runtime_download_allowed": False,
        "resources": [
            {
                "resource_id": "synthetic-fixture",
                "relative_path": "fixture.txt",
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        ],
    }
    (root / "resource-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    config = root / "fixture.yaml"
    config.write_text(
        "schema_id: legal-rag.config.v1\nresource_manifest: resource-manifest.json\n",
        encoding="utf-8",
    )
    return config


def _forbidden(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("doctor attempted a forbidden side effect")


def test_doctor_json_is_deterministic_redacted_and_side_effect_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_ready_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(nltk, "download", _forbidden)

    exit_code = main(
        [
            "doctor",
            "--config",
            config.name,
            "--execution-mode",
            "local-offline",
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    expected = {
        "checks": [
            {"code": "CONFIG_VALID", "status": "pass"},
            {"code": "RESOURCE_MANIFEST_VALID", "status": "pass"},
            {"code": "RESOURCES_PRESENT", "status": "pass"},
        ],
        "execution_mode": "local-offline",
        "schema_id": "doctor.report.v1",
        "status": "ready",
    }
    expected_bytes = (
        json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert exit_code == 0
    assert captured.out == expected_bytes
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert "token" not in captured.out.casefold()


def test_doctor_text_contract_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_ready_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["doctor", "--config", config.name, "--execution-mode", "local-offline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        "legal-rag doctor\n"
        "status: ready\n"
        "execution_mode: local-offline\n"
        "CONFIG_VALID: pass\n"
        "RESOURCE_MANIFEST_VALID: pass\n"
        "RESOURCES_PRESENT: pass\n"
    )
    assert captured.err == ""


def test_doctor_rejects_wrong_config_extension_with_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "fixture.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "doctor",
            "--config",
            config.name,
            "--execution-mode",
            "local-offline",
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert captured.err == (
        '{"error":{"code":"CONFIG_EXTENSION_UNSUPPORTED",'
        '"message":"config must use .yaml or .yml"},'
        '"schema_id":"cli.error.v1"}\n'
    )


def test_doctor_rejects_unknown_config_fields_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_ready_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8") + "unexpected: true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["doctor", "--config", config.name, "--execution-mode", "local-offline"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert "CONFIG_SCHEMA_INVALID" in captured.err
    assert str(tmp_path) not in captured.err


def test_doctor_safe_loader_rejects_python_object_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "unsafe.yaml"
    config.write_text("!!python/object/apply:os.system ['echo forbidden']\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["doctor", "--config", config.name, "--execution-mode", "local-offline"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert "CONFIG_PARSE_ERROR" in captured.err
    assert "forbidden" not in captured.out


def test_doctor_missing_resource_uses_offline_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_ready_config(tmp_path)
    (tmp_path / "resources" / "fixture.txt").unlink()
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "doctor",
            "--config",
            config.name,
            "--execution-mode",
            "local-offline",
            "--output-format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert captured.err == (
        '{"error":{"code":"OFFLINE_RESOURCE_MISSING",'
        '"message":"required manifested resource is missing: synthetic-fixture"},'
        '"schema_id":"cli.error.v1"}\n'
    )


def test_cli_disables_abbreviated_long_options() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["doctor", "--conf", "fixture.yaml", "--execution-mode", "local-offline"])
    assert exc_info.value.code == 2
