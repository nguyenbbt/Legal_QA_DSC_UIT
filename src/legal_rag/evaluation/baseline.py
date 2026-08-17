"""Corpus-free deterministic fixed-refusal baseline for the MIL-003 gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from legal_rag.domain.artifacts import ImmutableArtifactError, write_immutable_bytes
from legal_rag.domain.checksums import (
    FileSetChecksum,
    canonical_json_bytes,
    checksum_bytes,
    compute_run_id,
    validate_run_manifest_identity,
)
from legal_rag.domain.models import AnswerRecord, QuestionRecord, RunManifest
from legal_rag.generation.fixture import FixtureExtractiveGenerator
from legal_rag.submission.writer import answers_jsonl_bytes

BASELINE_PIPELINE_VERSION = "fixed-refusal-v1"
BASELINE_SOURCE_PATHS = tuple(
    sorted(
        (
            "src/legal_rag/domain/checksums.py",
            "src/legal_rag/domain/models.py",
            "src/legal_rag/evaluation/baseline.py",
            "src/legal_rag/evaluation/split.py",
            "src/legal_rag/generation/fixture.py",
            "src/legal_rag/ingestion/organizer.py",
            "src/legal_rag/submission/writer.py",
        ),
        key=str.encode,
    )
)
_ZERO_RUN_ID = "run_" + "0" * 24
_ZERO_CHECKSUM = "sha256:" + "0" * 64


class BaselineError(Exception):
    """Stable safe failure at the corpus-free baseline boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class BaselineRun:
    run_id: str
    answers: tuple[AnswerRecord, ...]
    artifacts: dict[str, bytes]


def _fail(code: str, message: str) -> NoReturn:
    raise BaselineError(code, message)


def _jsonl_bytes(records: tuple[dict[str, object], ...]) -> bytes:
    lines = (
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for record in records
    )
    return ("\n".join(lines) + "\n").encode()


def _ordered_json_bytes(value: object) -> bytes:
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (rendered + "\n").encode()


def question_jsonl_bytes(questions: tuple[QuestionRecord, ...]) -> bytes:
    """Serialize an ordered typed question subset without organizer reconstruction."""

    return _jsonl_bytes(tuple(question.model_dump(mode="json") for question in questions))


def build_fixed_refusal_run(
    questions: tuple[QuestionRecord, ...],
    *,
    question_bytes: bytes,
    split_checksum: str,
    source_tree: FileSetChecksum,
) -> BaselineRun:
    """Build answer/run artifacts without corpus, retrieval, model, verifier, or network."""

    if not questions:
        _fail("BASELINE_QUESTIONS_EMPTY", "fixed-refusal baseline requires questions")
    question_ids = tuple(question.question_id for question in questions)
    if len(question_ids) != len(set(question_ids)):
        _fail("BASELINE_QUESTION_ID_DUPLICATE", "baseline question IDs must be unique")

    generator = FixtureExtractiveGenerator()
    generated = tuple(generator.generate(question, ()) for question in questions)
    diagnostics_bytes = canonical_json_bytes(
        {
            "schema_version": "evidence.diagnostics.batch.v1",
            "questions": [
                {"question_id": question.question_id, "items": []} for question in questions
            ],
        }
    )
    config_bytes = canonical_json_bytes(
        {
            "schema_version": "fixed.refusal.config.v1",
            "pipeline_version": BASELINE_PIPELINE_VERSION,
            "execution_mode": "local-offline",
            "competition_policy": "baseline.v1",
            "corpus_access": False,
        }
    )
    resource_bytes = canonical_json_bytes(
        {"schema_version": "fixed.refusal.resources.v1", "resources": []}
    )
    placeholder = RunManifest.model_validate(
        {
            "schema_version": "run.manifest.v1",
            "run_id": _ZERO_RUN_ID,
            "pipeline_version": BASELINE_PIPELINE_VERSION,
            "code_revision": f"tree:{source_tree.checksum}",
            "source_tree_checksum": source_tree.checksum,
            "scoped_source_paths": source_tree.paths,
            "config_checksum": checksum_bytes(config_bytes),
            "question_checksum": checksum_bytes(question_bytes),
            "corpus_checksum": checksum_bytes(b""),
            "index_checksum": None,
            "split_checksum": split_checksum,
            "model_id": None,
            "model_revision": None,
            "tokenizer_id": None,
            "tokenizer_revision": None,
            "prompt_revision": None,
            "seed": "dsc2026-fixed-refusal-v1",
            "execution_mode": "local-offline",
            "competition_policy": "baseline.v1",
            "comparison_type": "baseline",
            "resolved_as_of_date": None,
            "as_of_timezone": None,
            "resource_manifest_checksum": checksum_bytes(resource_bytes),
            "evidence_diagnostics_checksum": _ZERO_CHECKSUM,
            "answer_artifact_checksum": _ZERO_CHECKSUM,
        }
    )
    run_id = compute_run_id(placeholder)
    answers = tuple(
        AnswerRecord.model_validate(
            {
                "schema_version": "internal.answer.v1",
                "question_id": generated_answer.question_id,
                "answer": generated_answer.answer_text,
                "generator_id": generated_answer.generator_id,
                "evidence_ids": generated_answer.used_evidence_ids,
                "run_id": run_id,
            }
        )
        for generated_answer in generated
    )
    answer_bytes = answers_jsonl_bytes(answers)
    completed = placeholder.model_copy(
        update={
            "run_id": run_id,
            "evidence_diagnostics_checksum": checksum_bytes(diagnostics_bytes),
            "answer_artifact_checksum": checksum_bytes(answer_bytes),
        }
    )
    validate_run_manifest_identity(completed)
    return BaselineRun(
        run_id=run_id,
        answers=answers,
        artifacts={
            "answers.jsonl": answer_bytes,
            "evidence.diagnostics.json": diagnostics_bytes,
            "generated-answers.jsonl": _jsonl_bytes(
                tuple(answer.model_dump(mode="json") for answer in generated)
            ),
            "run.manifest.json": canonical_json_bytes(completed.model_dump(mode="json")),
        },
    )


def build_development_inputs(
    questions: tuple[QuestionRecord, ...], answers: tuple[AnswerRecord, ...]
) -> tuple[bytes, bytes]:
    """Render ordered official-exact predictions and gold references."""

    if tuple(question.question_id for question in questions) != tuple(
        answer.question_id for answer in answers
    ):
        _fail("BASELINE_ID_MISMATCH", "development answer IDs must match question order")
    if any(question.answer_state != "gold" or question.answer is None for question in questions):
        _fail("BASELINE_REFERENCE_INVALID", "development questions require gold answers")
    predictions = {answer.question_id: {"answer": answer.answer} for answer in answers}
    references = {question.question_id: cast(str, question.answer) for question in questions}
    return _ordered_json_bytes(predictions), _ordered_json_bytes(references)


def write_baseline_artifacts(output_directory: Path, artifacts: dict[str, bytes]) -> dict[str, str]:
    """Preflight then immutably create one complete baseline artifact bundle."""

    destinations: dict[str, Path] = {}
    for relative_path, data in artifacts.items():
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or "\\" in relative_path
        ):
            _fail("BASELINE_ARTIFACT_PATH_INVALID", "baseline artifact path is invalid")
        destination = output_directory.joinpath(*pure_path.parts)
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != data):
            _fail("BASELINE_ARTIFACT_IMMUTABLE", "baseline artifact cannot be replaced")
        destinations[relative_path] = destination

    checksums: dict[str, str] = {}
    try:
        for relative_path in sorted(artifacts, key=str.encode):
            checksums[relative_path] = write_immutable_bytes(
                destinations[relative_path], artifacts[relative_path]
            )
    except ImmutableArtifactError as error:
        raise BaselineError("BASELINE_ARTIFACT_WRITE_FAILED", error.message) from error
    return checksums
