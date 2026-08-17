"""Public command-line entrypoint for the LegalQA system."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from legal_rag.config import load_config
from legal_rag.doctor import DoctorReport, run_doctor
from legal_rag.errors import CliError


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _operational_timestamp() -> str:
    """Return an ISO timestamp at the project's fixed UTC+07 operational offset."""

    return datetime.now(timezone(timedelta(hours=7))).isoformat()


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
    pipeline = commands.add_parser(
        "pipeline", help="run a typed offline pipeline", allow_abbrev=False
    )
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    pipeline_run = pipeline_commands.add_parser(
        "run", help="run the MIL-002 synthetic fixture", allow_abbrev=False
    )
    pipeline_run.add_argument("--config", required=True)
    pipeline_run.add_argument("--questions", required=True)
    pipeline_run.add_argument("--contexts", required=True)
    pipeline_run.add_argument("--aliases", required=True)
    pipeline_run.add_argument("--resource-manifest", required=True)
    pipeline_run.add_argument("--output-directory", required=True, type=Path)
    pipeline_run.add_argument("--execution-mode", required=True, choices=("local-offline",))

    submit = commands.add_parser(
        "submit", help="build or validate an organizer submission", allow_abbrev=False
    )
    submit_commands = submit.add_subparsers(dest="submit_command", required=True)
    submit_build = submit_commands.add_parser(
        "build", help="build predictions from internal answers", allow_abbrev=False
    )
    submit_build.add_argument("--questions", required=True, type=Path)
    submit_build.add_argument("--answers", required=True, type=Path)
    submit_build.add_argument("--output", required=True, type=Path)
    submit_build.add_argument("--profile", required=True, choices=("competition",))
    submit_validate = submit_commands.add_parser(
        "validate", help="validate organizer predictions", allow_abbrev=False
    )
    submit_validate.add_argument("--questions", required=True, type=Path)
    submit_validate.add_argument("--predictions", required=True, type=Path)
    submit_validate.add_argument("--profile", required=True, choices=("competition",))

    split = commands.add_parser(
        "split", help="build an immutable organizer-question split", allow_abbrev=False
    )
    split_commands = split.add_subparsers(dest="split_command", required=True)
    split_build = split_commands.add_parser(
        "build", help="build split.v1 and its public overlap audit", allow_abbrev=False
    )
    split_build.add_argument("--questions", required=True, type=Path)
    split_build.add_argument("--public-questions", type=Path)
    split_build.add_argument("--output", required=True, type=Path)

    organizer = commands.add_parser(
        "organizer", help="convert immutable organizer artifacts", allow_abbrev=False
    )
    organizer_commands = organizer.add_subparsers(dest="organizer_command", required=True)
    import_questions = organizer_commands.add_parser(
        "import-questions", help="convert organizer questions to typed JSONL", allow_abbrev=False
    )
    import_questions.add_argument("--kind", required=True, choices=("train", "public"))
    import_questions.add_argument("--input", required=True, type=Path)
    import_questions.add_argument("--output", required=True, type=Path)
    import_questions.add_argument("--errors", required=True, type=Path)
    import_questions.add_argument("--strict", action="store_true", required=True)
    import_contexts = organizer_commands.add_parser(
        "import-contexts", help="convert organizer contexts to typed JSONL", allow_abbrev=False
    )
    import_contexts.add_argument("--input-dir", required=True, type=Path)
    import_contexts.add_argument("--pattern", required=True)
    import_contexts.add_argument("--output", required=True, type=Path)
    import_contexts.add_argument("--manifest", required=True, type=Path)
    import_contexts.add_argument("--errors", required=True, type=Path)
    import_contexts.add_argument("--strict", action="store_true", required=True)

    corpus = commands.add_parser(
        "corpus", help="build deterministic local corpus artifacts", allow_abbrev=False
    )
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_report = corpus_commands.add_parser(
        "report", help="chunk and audit an imported context corpus", allow_abbrev=False
    )
    corpus_report.add_argument("--contexts", required=True, type=Path)
    corpus_report.add_argument("--context-manifest", required=True, type=Path)
    corpus_report.add_argument("--chunks", required=True, type=Path)
    corpus_report.add_argument("--chunk-manifest", required=True, type=Path)
    corpus_report.add_argument("--json-report", required=True, type=Path)
    corpus_report.add_argument("--markdown-report", required=True, type=Path)

    aliases = commands.add_parser(
        "aliases", help="manage offline legal-reference alias review", allow_abbrev=False
    )
    alias_commands = aliases.add_subparsers(dest="aliases_command", required=True)
    alias_propose = alias_commands.add_parser(
        "propose", help="create a draft offset-backed alias review queue", allow_abbrev=False
    )
    alias_propose.add_argument("--contexts", required=True, type=Path)
    alias_propose.add_argument("--context-manifest", required=True, type=Path)
    alias_propose.add_argument("--output", required=True, type=Path)
    alias_propose.add_argument("--report", required=True, type=Path)

    grounding = commands.add_parser(
        "grounding", help="manage private grounding contracts", allow_abbrev=False
    )
    grounding_commands = grounding.add_subparsers(dest="grounding_command", required=True)
    grounding_sample = grounding_commands.add_parser(
        "sample", help="freeze the pre-index grounding sample", allow_abbrev=False
    )
    grounding_sample.add_argument("--questions", required=True, type=Path)
    grounding_sample.add_argument("--split-manifest", required=True, type=Path)
    grounding_sample.add_argument("--output", required=True, type=Path)

    baseline = commands.add_parser(
        "baseline", help="build the corpus-free MIL-003 baseline", allow_abbrev=False
    )
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_build = baseline_commands.add_parser(
        "build", help="build public and development fixed-refusal artifacts", allow_abbrev=False
    )
    baseline_build.add_argument("--train", required=True, type=Path)
    baseline_build.add_argument("--public", required=True, type=Path)
    baseline_build.add_argument("--split-manifest", required=True, type=Path)
    baseline_build.add_argument("--output-directory", required=True, type=Path)

    evaluate = commands.add_parser(
        "evaluate", help="run deterministic offline evaluations", allow_abbrev=False
    )
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_competition = evaluate_commands.add_parser(
        "competition", help="run the reviewed official-exact scorer", allow_abbrev=False
    )
    evaluate_competition.add_argument("--predictions", required=True, type=Path)
    evaluate_competition.add_argument("--references", required=True, type=Path)
    evaluate_competition.add_argument("--mode", required=True, choices=("official_exact",))
    evaluate_competition.add_argument("--per-query", required=True, type=Path)
    evaluate_competition.add_argument("--report", required=True, type=Path)
    evaluate_competition.add_argument(
        "--scorer-root", type=Path, default=Path("Scoring-Program-Task-LegalQA")
    )
    evaluate_competition.add_argument("--nltk-data", type=Path, default=Path("resources/nltk_data"))

    benchmark = commands.add_parser(
        "benchmark", help="record local operational measurements", allow_abbrev=False
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_baseline = benchmark_commands.add_parser(
        "baseline", help="measure the fixed-refusal public workload", allow_abbrev=False
    )
    benchmark_baseline.add_argument("--public", required=True, type=Path)
    benchmark_baseline.add_argument("--run-manifest", required=True, type=Path)
    benchmark_baseline.add_argument("--output", required=True, type=Path)
    benchmark_baseline.add_argument("--warm-samples", type=int, default=7)
    return parser


def _render_report(report: DoctorReport | dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        value = report.as_dict() if isinstance(report, DoctorReport) else report
        return _json_line(value)
    if isinstance(report, dict):
        if "summary" in report:
            return str(report["summary"]) + "\n"
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


def _run_pipeline(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.pipeline.fixture import (
        FixturePipelineError,
        FixturePipelineRequest,
        run_fixture_pipeline,
    )

    if arguments.pipeline_command != "run" or arguments.execution_mode != "local-offline":
        raise CliError("PIPELINE_COMMAND_INVALID", "pipeline command is invalid")
    request = FixturePipelineRequest(
        project_root=Path.cwd(),
        config_path=arguments.config,
        questions_path=arguments.questions,
        contexts_directory=arguments.contexts,
        aliases_path=arguments.aliases,
        resource_manifest_path=arguments.resource_manifest,
        output_directory=arguments.output_directory,
    )
    try:
        result = run_fixture_pipeline(request)
    except FixturePipelineError as error:
        raise CliError(error.code, error.message) from error
    return {"summary": result.summary}


def _read_bytes(path: Path, *, code: str, message: str, exit_code: int) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CliError(code, message, exit_code=exit_code) from error


def _run_submit(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.submission.writer import (
        SubmissionError,
        load_answers_jsonl,
        validate_submission,
        write_submission,
    )

    source = _read_bytes(
        arguments.questions,
        code="SUB_SOURCE_INVALID",
        message="question source is unavailable",
        exit_code=2,
    )
    try:
        if arguments.submit_command == "build":
            answer_bytes = _read_bytes(
                arguments.answers,
                code="SUB_ANSWER_ARTIFACT_INVALID",
                message="answer artifact is unavailable",
                exit_code=2,
            )
            answers = load_answers_jsonl(answer_bytes)
            validation = write_submission(arguments.output, source, answers)
        elif arguments.submit_command == "validate":
            predictions = _read_bytes(
                arguments.predictions,
                code="SUB_PREDICTIONS_INVALID",
                message="prediction artifact is unavailable",
                exit_code=2,
            )
            validation = validate_submission(source, predictions)
        else:
            raise CliError("SUB_COMMAND_INVALID", "submission command is invalid", exit_code=2)
    except SubmissionError as error:
        raise CliError(error.code, error.message, exit_code=2) from error
    return {"summary": validation.summary}


def _run_split(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.checksums import checksum_bytes
    from legal_rag.evaluation.split import (
        SplitError,
        build_split_manifest,
        load_split_questions_jsonl,
        write_split_manifest,
    )

    if arguments.split_command != "build":
        raise CliError("SPLIT_COMMAND_INVALID", "split command is invalid", exit_code=2)
    train_bytes = _read_bytes(
        arguments.questions,
        code="SPLIT_TRAIN_SOURCE_INVALID",
        message="train question source is unavailable",
        exit_code=2,
    )
    public_bytes = (
        _read_bytes(
            arguments.public_questions,
            code="SPLIT_PUBLIC_SOURCE_INVALID",
            message="public question source is unavailable",
            exit_code=2,
        )
        if arguments.public_questions is not None
        else b""
    )
    try:
        train = load_split_questions_jsonl(train_bytes, expected_answer_state="gold")
        public = (
            load_split_questions_jsonl(public_bytes, expected_answer_state="unlabeled")
            if public_bytes
            else ()
        )
        manifest = build_split_manifest(
            train,
            public,
            source_checksum=checksum_bytes(train_bytes),
            public_source_checksum=checksum_bytes(public_bytes),
        )
        checksum = write_split_manifest(arguments.output, manifest)
    except SplitError as error:
        raise CliError(error.code, error.message) from error
    return {
        "summary": (
            f"SPLIT COMPLETE questions={len(manifest.rows)} "
            f"groups={manifest.group_count} {checksum}"
        )
    }


def _run_organizer(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
    from legal_rag.ingestion.organizer import (
        OrganizerContextReader,
        OrganizerDataError,
        OrganizerQuestionReader,
        discover_context_files,
    )

    if not arguments.strict:
        raise CliError("ORGANIZER_COMMAND_INVALID", "organizer command is invalid", exit_code=2)
    try:
        if arguments.organizer_command == "import-questions":
            source = _read_bytes(
                arguments.input,
                code="ORGANIZER_SOURCE_INVALID",
                message="organizer source is unavailable",
                exit_code=2,
            )
            imported = OrganizerQuestionReader().read_bytes(
                source, kind=arguments.kind, artifact_path=f"{arguments.kind}-source"
            )
            rendered = imported.jsonl_bytes()
            _write_report(arguments.output, rendered)
            _write_report(arguments.errors, b"")
            return {
                "summary": (
                    f"IMPORT COMPLETE kind={arguments.kind} count={len(imported.records)} "
                    f"{checksum_bytes(rendered)}"
                )
            }
        if arguments.organizer_command == "import-contexts":
            files = discover_context_files(arguments.input_dir, pattern=arguments.pattern)
            imported_contexts = OrganizerContextReader().read_files(files)
            rendered = imported_contexts.jsonl_bytes()
            _write_report(arguments.output, rendered)
            _write_report(arguments.manifest, imported_contexts.manifest_bytes())
            _write_report(arguments.errors, b"")
            indexable = sum(record.indexable for record in imported_contexts.records)
            return {
                "summary": (
                    f"IMPORT COMPLETE kind=contexts count={len(imported_contexts.records)} "
                    f"indexable={indexable} "
                    f"quarantined={len(imported_contexts.records) - indexable} "
                    f"{checksum_bytes(rendered)}"
                )
            }
        raise CliError("ORGANIZER_COMMAND_INVALID", "organizer command is invalid", exit_code=2)
    except OrganizerDataError as error:
        _write_report(
            arguments.errors,
            canonical_json_bytes(
                {
                    "schema_version": "organizer.import.error.v1",
                    "code": error.code,
                    "json_path": error.json_path,
                    "raw_id": error.raw_id,
                    "message": error.message,
                }
            ),
        )
        raise CliError(error.code, error.message, exit_code=2) from error


def _run_corpus(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.artifacts import ImmutableArtifactError
    from legal_rag.domain.checksums import checksum_bytes
    from legal_rag.ingestion.corpus import (
        CorpusBuildError,
        corpus_checksum_from_import_manifest,
        write_corpus_artifacts,
    )

    if arguments.corpus_command != "report":
        raise CliError("CORPUS_COMMAND_INVALID", "corpus command is invalid", exit_code=2)
    manifest_bytes = _read_bytes(
        arguments.context_manifest,
        code="CORPUS_IMPORT_MANIFEST_INVALID",
        message="context import manifest is unavailable",
        exit_code=2,
    )
    try:
        summary = write_corpus_artifacts(
            contexts_path=arguments.contexts,
            chunks_path=arguments.chunks,
            manifest_path=arguments.chunk_manifest,
            report_path=arguments.json_report,
            markdown_path=arguments.markdown_report,
            corpus_checksum=corpus_checksum_from_import_manifest(manifest_bytes),
            context_import_manifest_checksum=checksum_bytes(manifest_bytes),
        )
    except (CorpusBuildError, ImmutableArtifactError) as error:
        raise CliError(error.code, error.message, exit_code=2) from error
    return {
        "summary": (
            f"CORPUS COMPLETE contexts={summary.context_count} "
            f"indexable={summary.indexable_context_count} "
            f"quarantined={summary.quarantined_context_count} "
            f"chunks={summary.chunk_count} {summary.chunks_checksum}"
        )
    }


def _run_aliases(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.artifacts import ImmutableArtifactError
    from legal_rag.ingestion.aliases import AliasProposalError, write_alias_proposals
    from legal_rag.ingestion.corpus import CorpusBuildError, corpus_checksum_from_import_manifest

    if arguments.aliases_command != "propose":
        raise CliError("ALIAS_COMMAND_INVALID", "alias command is invalid", exit_code=2)
    manifest_bytes = _read_bytes(
        arguments.context_manifest,
        code="CORPUS_IMPORT_MANIFEST_INVALID",
        message="context import manifest is unavailable",
        exit_code=2,
    )
    try:
        summary = write_alias_proposals(
            contexts_path=arguments.contexts,
            proposals_path=arguments.output,
            report_path=arguments.report,
            corpus_checksum=corpus_checksum_from_import_manifest(manifest_bytes),
        )
    except (AliasProposalError, CorpusBuildError, ImmutableArtifactError) as error:
        raise CliError(error.code, error.message, exit_code=2) from error
    return {
        "summary": (
            f"ALIAS PROPOSAL COMPLETE contexts={summary.context_count} "
            f"proposals={summary.proposal_count} {summary.proposals_checksum}"
        )
    }


def _run_grounding(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.checksums import checksum_bytes
    from legal_rag.evaluation.grounding import (
        GroundingError,
        GroundingQuestion,
        build_grounding_sample,
        write_grounding_sample,
    )
    from legal_rag.evaluation.split import (
        SplitError,
        load_split_manifest_rows,
        load_split_questions_jsonl,
    )

    if arguments.grounding_command != "sample":
        raise CliError("GROUNDING_COMMAND_INVALID", "grounding command is invalid", exit_code=2)
    question_bytes = _read_bytes(
        arguments.questions,
        code="GROUNDING_QUESTION_SOURCE_INVALID",
        message="grounding question source is unavailable",
        exit_code=2,
    )
    split_bytes = _read_bytes(
        arguments.split_manifest,
        code="GROUNDING_SPLIT_SOURCE_INVALID",
        message="grounding split manifest is unavailable",
        exit_code=2,
    )
    try:
        questions = load_split_questions_jsonl(question_bytes, expected_answer_state="gold")
        split_rows = load_split_manifest_rows(
            split_bytes,
            expected_source_checksum=checksum_bytes(question_bytes),
            expected_question_ids=tuple(question.question_id for question in questions),
        )
        development_ids = {row.question_id for row in split_rows if row.split == "development"}
        development = tuple(
            GroundingQuestion(question.question_id, question.question)
            for question in questions
            if question.question_id in development_ids
        )
        manifest = build_grounding_sample(development, split_checksum=checksum_bytes(split_bytes))
        checksum = write_grounding_sample(arguments.output, manifest)
    except (SplitError, GroundingError) as error:
        raise CliError(error.code, error.message) from error
    return {
        "summary": (
            f"GROUNDING SAMPLE COMPLETE selected={len(manifest.selected_question_ids)} "
            f"eligible={len(manifest.rows)} {checksum}"
        )
    }


def _run_baseline(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.domain.checksums import (
        canonical_json_bytes,
        checksum_bytes,
        checksum_file_set,
    )
    from legal_rag.evaluation.baseline import (
        BASELINE_SOURCE_PATHS,
        BaselineError,
        build_development_inputs,
        build_fixed_refusal_run,
        question_jsonl_bytes,
        write_baseline_artifacts,
    )
    from legal_rag.evaluation.split import SplitError, load_split_manifest_rows
    from legal_rag.ingestion.organizer import OrganizerDataError, OrganizerQuestionReader
    from legal_rag.submission.writer import (
        SubmissionError,
        build_submission,
        validate_submission,
    )

    if arguments.baseline_command != "build":
        raise CliError("BASELINE_COMMAND_INVALID", "baseline command is invalid", exit_code=2)
    train_source = _read_bytes(
        arguments.train,
        code="BASELINE_TRAIN_SOURCE_INVALID",
        message="baseline train source is unavailable",
        exit_code=2,
    )
    public_source = _read_bytes(
        arguments.public,
        code="BASELINE_PUBLIC_SOURCE_INVALID",
        message="baseline public source is unavailable",
        exit_code=2,
    )
    split_bytes = _read_bytes(
        arguments.split_manifest,
        code="BASELINE_SPLIT_SOURCE_INVALID",
        message="baseline split manifest is unavailable",
        exit_code=2,
    )
    try:
        reader = OrganizerQuestionReader()
        train = reader.read_bytes(train_source, kind="train", artifact_path="train-source")
        public = reader.read_bytes(public_source, kind="public", artifact_path="public-source")
        train_internal = train.jsonl_bytes()
        rows = load_split_manifest_rows(
            split_bytes,
            expected_source_checksum=checksum_bytes(train_internal),
            expected_question_ids=tuple(question.question_id for question in train.records),
        )
        train_by_id = {question.question_id: question for question in train.records}
        development = tuple(
            train_by_id[row.question_id] for row in rows if row.split == "development"
        )
        split_checksum = checksum_bytes(split_bytes)
        source_tree = checksum_file_set(Path.cwd(), BASELINE_SOURCE_PATHS)
        public_internal = public.jsonl_bytes()
        development_internal = question_jsonl_bytes(development)
        public_run = build_fixed_refusal_run(
            public.records,
            question_bytes=public_internal,
            split_checksum=split_checksum,
            source_tree=source_tree,
        )
        development_run = build_fixed_refusal_run(
            development,
            question_bytes=development_internal,
            split_checksum=split_checksum,
            source_tree=source_tree,
        )
        public_predictions = build_submission(public_source, public_run.answers)
        validate_submission(public_source, public_predictions)
        development_predictions, development_references = build_development_inputs(
            development, development_run.answers
        )
        artifacts = {
            **{f"public/{name}": data for name, data in public_run.artifacts.items()},
            "public/questions.jsonl": public_internal,
            "public/predictions.json": public_predictions,
            **{f"development/{name}": data for name, data in development_run.artifacts.items()},
            "development/questions.jsonl": development_internal,
            "development/predictions.json": development_predictions,
            "development/references.json": development_references,
        }
        artifact_rows = [
            {"path": path, "checksum": checksum_bytes(data)}
            for path, data in sorted(artifacts.items(), key=lambda item: item[0].encode())
        ]
        artifacts["baseline.manifest.json"] = canonical_json_bytes(
            {
                "schema_version": "mil003.baseline.manifest.v1",
                "baseline_kind": "plumbing_baseline",
                "generator_id": "fixture-extractive-v1",
                "public_run_id": public_run.run_id,
                "development_run_id": development_run.run_id,
                "split_checksum": split_checksum,
                "limitation": "no_real_context_index_until_mil_004",
                "artifacts": artifact_rows,
            }
        )
        write_baseline_artifacts(arguments.output_directory, artifacts)
    except (OrganizerDataError, SplitError, BaselineError, SubmissionError) as error:
        raise CliError(error.code, error.message) from error
    return {
        "summary": (
            f"BASELINE COMPLETE public={len(public.records)} "
            f"development={len(development)} public_run={public_run.run_id} "
            f"development_run={development_run.run_id}"
        )
    }


def _run_evaluate(arguments: argparse.Namespace) -> dict[str, str]:
    from legal_rag.evaluation.competition import (
        CompetitionEvaluationError,
        evaluate_competition_bytes,
        write_competition_evaluation,
    )
    from legal_rag.evaluation.official_exact import ScorerError

    if arguments.evaluate_command != "competition" or arguments.mode != "official_exact":
        raise CliError("EVAL_COMMAND_INVALID", "evaluation command is invalid", exit_code=2)
    predictions = _read_bytes(
        arguments.predictions,
        code="EVAL_PREDICTIONS_SOURCE_INVALID",
        message="evaluation predictions are unavailable",
        exit_code=2,
    )
    references = _read_bytes(
        arguments.references,
        code="EVAL_REFERENCES_SOURCE_INVALID",
        message="evaluation references are unavailable",
        exit_code=2,
    )
    try:
        evaluation = evaluate_competition_bytes(
            predictions,
            references,
            scorer_root=arguments.scorer_root,
            nltk_data_root=arguments.nltk_data,
        )
        write_competition_evaluation(
            evaluation,
            per_query_path=arguments.per_query,
            report_path=arguments.report,
        )
    except CompetitionEvaluationError as error:
        raise CliError(error.code, error.message) from error
    except ScorerError as error:
        exit_code = 4 if error.code == "OFFLINE_RESOURCE_MISSING" else 3
        raise CliError(error.code, error.message, exit_code=exit_code) from error
    return {
        "summary": (
            f"EVALUATION COMPLETE questions={evaluation.question_count} "
            f"meteor={evaluation.macro_meteor} rouge_l={evaluation.macro_rouge_l}"
        )
    }


def _run_benchmark(arguments: argparse.Namespace) -> dict[str, str]:
    import json as json_module
    import platform

    from pydantic import ValidationError

    from legal_rag.domain.checksums import (
        DeterminismError,
        checksum_bytes,
        validate_run_manifest_identity,
    )
    from legal_rag.domain.models import OperationalTelemetry, RunManifest
    from legal_rag.domain.telemetry import validate_telemetry_link
    from legal_rag.evaluation.performance import measure_fixed_refusal
    from legal_rag.ingestion.organizer import OrganizerDataError, OrganizerQuestionReader

    if arguments.benchmark_command != "baseline":
        raise CliError("BENCHMARK_COMMAND_INVALID", "benchmark command is invalid", exit_code=2)
    public_source = _read_bytes(
        arguments.public,
        code="BENCHMARK_PUBLIC_SOURCE_INVALID",
        message="benchmark public source is unavailable",
        exit_code=2,
    )
    manifest_bytes = _read_bytes(
        arguments.run_manifest,
        code="BENCHMARK_RUN_MANIFEST_INVALID",
        message="benchmark run manifest is unavailable",
        exit_code=2,
    )
    try:
        manifest = RunManifest.model_validate_json(manifest_bytes)
        validate_run_manifest_identity(manifest)
        imported = OrganizerQuestionReader().read_bytes(
            public_source, kind="public", artifact_path="public-source"
        )
        if manifest.question_checksum != checksum_bytes(imported.jsonl_bytes()):
            raise CliError(
                "BENCHMARK_QUESTION_MISMATCH",
                "benchmark questions do not match the run manifest",
            )
        measurement = measure_fixed_refusal(
            public_source,
            run_id=manifest.run_id,
            warm_samples=arguments.warm_samples,
        )
        run_manifest_checksum = checksum_bytes(manifest_bytes)
        telemetry = OperationalTelemetry.model_validate(
            {
                "schema_version": "operational.telemetry.v1",
                "run_id": manifest.run_id,
                "run_instance_id": uuid4(),
                "run_manifest_checksum": run_manifest_checksum,
            }
        )
        validate_telemetry_link(telemetry, manifest)
    except OrganizerDataError as error:
        raise CliError(error.code, error.message, exit_code=2) from error
    except (ValidationError, DeterminismError, ValueError) as error:
        raise CliError(
            "BENCHMARK_CONTRACT_INVALID", "benchmark inputs or settings are invalid"
        ) from error

    report = {
        "schema_version": "performance.cost.report.v1",
        "run_id": manifest.run_id,
        "run_instance_id": str(telemetry.run_instance_id),
        "run_manifest_checksum": run_manifest_checksum,
        "recorded_at": _operational_timestamp(),
        "timezone": "Asia/Ho_Chi_Minh",
        "hardware": {
            "execution_mode": "local-offline",
            "device": "cpu",
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "paid_service_used": False,
        },
        "measurement": measurement,
        "resource_caps_state": "awaiting_owner_oq_005",
    }
    rendered = (
        json_module.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    _write_report(arguments.output, rendered)
    return {
        "summary": (
            f"BENCHMARK COMPLETE questions={measurement['question_count']} "
            f"warm_samples={measurement['warm_sample_count']} run_id={manifest.run_id}"
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the public CLI and return a process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    output_format: str = getattr(arguments, "output_format", "text")
    try:
        handlers: dict[str, Any] = {
            "aliases": _run_aliases,
            "baseline": _run_baseline,
            "benchmark": _run_benchmark,
            "corpus": _run_corpus,
            "doctor": _run_doctor,
            "evaluate": _run_evaluate,
            "grounding": _run_grounding,
            "organizer": _run_organizer,
            "pipeline": _run_pipeline,
            "scorer": _run_scorer,
            "split": _run_split,
            "submit": _run_submit,
        }
        report = handlers[arguments.command](arguments)
    except CliError as error:
        sys.stderr.write(_render_error(error, output_format))
        return error.exit_code
    sys.stdout.write(_render_report(report, output_format))
    return 0
