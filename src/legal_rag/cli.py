"""Public command-line entrypoint for the LegalQA system."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_rag.config import load_config
from legal_rag.doctor import DoctorReport, run_doctor
from legal_rag.errors import CliError


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag",
        description="Deterministic DSC 2026 Vietnamese LegalQA tooling",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="validate offline configuration and manifested resources", allow_abbrev=False
    )
    doctor.add_argument("--config", required=True, type=Path)
    doctor.add_argument(
        "--execution-mode",
        required=True,
        choices=("prepare-online", "local-offline", "private-modal"),
    )
    doctor.add_argument("--output-format", choices=("text", "json"), default="text")
    scorer = commands.add_parser(
        "scorer", help="validate official scorer parity", allow_abbrev=False
    )
    scorer_commands = scorer.add_subparsers(dest="scorer_command", required=True)
    parity = scorer_commands.add_parser(
        "parity", help="run fixed and sampled offline parity", allow_abbrev=False
    )
    parity.add_argument("--official", required=True, type=Path)
    parity.add_argument("--fixtures", required=True, type=Path)
    parity.add_argument("--nltk-data", type=Path, default=Path("resources/nltk_data"))
    parity.add_argument(
        "--json-report",
        type=Path,
        default=Path("artifacts/reports/mil-001/scorer-parity.json"),
    )
    parity.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("artifacts/reports/mil-001/scorer-parity.md"),
    )
    parity.add_argument("--output-format", choices=("text", "json"), default="text")
    return parser


def _render_report(report: DoctorReport | dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        value = report.as_dict() if isinstance(report, DoctorReport) else report
        return _json_line(value)
    if isinstance(report, dict):
        metrics = report["metrics"]
        return (
            "legal-rag scorer parity\n"
            f"status: {report['status']}\n"
            f"execution_mode: {report['execution_mode']}\n"
            f"rouge_l_absolute_difference: {metrics['rouge_l']['absolute_difference']}\n"
            f"meteor_absolute_difference: {metrics['meteor']['absolute_difference']}\n"
            f"per_query_max_absolute_difference: "
            f"{metrics['per_query_max_absolute_difference']}\n"
        )
    lines = [
        "legal-rag doctor",
        "status: ready",
        f"execution_mode: {report.execution_mode}",
        *(f"{check.code}: {check.status}" for check in report.checks),
    ]
    return "\n".join(lines) + "\n"


def _render_error(error: CliError, output_format: str) -> str:
    if output_format == "json":
        return _json_line(
            {
                "schema_id": "cli.error.v1",
                "error": {"code": error.code, "message": error.message},
            }
        )
    return f"ERROR {error.code}: {error.message}\n"


def _run_doctor(arguments: argparse.Namespace) -> DoctorReport:
    config = load_config(arguments.config)
    return run_doctor(config, arguments.execution_mode, project_root=Path.cwd())


def _write_report(path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise CliError("REPORT_WRITE_FAILED", "scorer parity report cannot be written") from exc
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _run_scorer(arguments: argparse.Namespace) -> dict[str, Any]:
    from legal_rag.evaluation.official_exact import ScorerError
    from legal_rag.evaluation.parity import parity_json_bytes, parity_markdown, run_parity_suite

    if arguments.scorer_command != "parity":
        raise CliError("SCORER_COMMAND_INVALID", "scorer command is invalid")
    try:
        report = run_parity_suite(
            scoring_path=arguments.official,
            fixtures_directory=arguments.fixtures,
            nltk_data_root=arguments.nltk_data,
        )
    except ScorerError as error:
        exit_code = 4 if error.code == "OFFLINE_RESOURCE_MISSING" else 3
        raise CliError(error.code, error.message, exit_code=exit_code) from error
    _write_report(arguments.json_report, parity_json_bytes(report))
    _write_report(arguments.markdown_report, parity_markdown(report).encode("utf-8"))
    if report["status"] != "pass":
        raise CliError("SCORER_PARITY_FAILED", "official scorer parity exceeded tolerance")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the public CLI and return a process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    output_format: str = arguments.output_format
    try:
        handlers: dict[str, Any] = {"doctor": _run_doctor, "scorer": _run_scorer}
        report = handlers[arguments.command](arguments)
    except CliError as error:
        sys.stderr.write(_render_error(error, output_format))
        return error.exit_code
    sys.stdout.write(_render_report(report, output_format))
    return 0
