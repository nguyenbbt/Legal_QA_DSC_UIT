"""MIL-002 local-offline synthetic-fixture vertical slice."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn, Self

from pydantic import model_validator

from legal_rag.domain.checksums import (
    canonical_json_bytes,
    checksum_bytes,
    checksum_file_set,
    compute_run_id,
    validate_run_manifest_identity,
    validate_run_output_checksums,
)
from legal_rag.domain.models import (
    AnswerRecord,
    FrozenStrictModel,
    NonNegativeInt,
    RunManifest,
    SafeRelativePath,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.generation.fixture import FixtureExtractiveGenerator
from legal_rag.ingestion.chunking import ChunkRecord, chunk_context
from legal_rag.ingestion.organizer import (
    OrganizerContextReader,
    OrganizerFile,
    OrganizerQuestionReader,
)
from legal_rag.retrieval.bm25 import APPROVED_BM25_RUNTIME_ID, build_bm25_index
from legal_rag.retrieval.exact import (
    load_alias_artifact,
    parse_legal_reference,
    resolve_exact_reference,
)
from legal_rag.retrieval.fusion import union_rank_candidates
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic
from legal_rag.retrieval.tokenizer import (
    RETRIEVAL_TOKENIZER_ID,
    RETRIEVAL_TOKENIZER_REVISION,
    retrieval_tokens,
)
from legal_rag.submission.writer import answers_jsonl_bytes, build_submission
from legal_rag.verification.evidence import (
    EvidenceSelectionConfig,
    EvidenceTokenizer,
    validate_and_admit_evidence,
)

PIPELINE_VERSION = "pipeline.v1"
_ZERO_CHECKSUM = "sha256:" + ("0" * 64)
_ZERO_RUN_ID = "run_" + ("0" * 24)
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}\Z")
_MAX_FIXTURE_RESOURCE_BYTES = 1024 * 1024
_ARTIFACT_NAMES = (
    "questions.import.jsonl",
    "contexts.import.jsonl",
    "context.import.manifest.json",
    "chunks.manifest.json",
    "aliases.manifest.json",
    "bm25.index.manifest.json",
    "retrieval.jsonl",
    "evidence.jsonl",
    "evidence.diagnostics.json",
    "generated-answers.jsonl",
    "answers.jsonl",
    "predictions.json",
    "run.manifest.json",
)


class FixturePipelineError(Exception):
    """Stable stage-aware failure that never exposes input contents or local paths."""

    def __init__(self, code: str, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


@dataclass(frozen=True, slots=True)
class FixturePipelineRequest:
    project_root: Path
    config_path: str
    questions_path: str
    contexts_directory: str
    aliases_path: str
    resource_manifest_path: str
    output_directory: Path


@dataclass(frozen=True, slots=True)
class FixturePipelineResult:
    run_id: str
    answer_count: int
    predictions_checksum: str
    artifact_names: tuple[str, ...]

    @property
    def summary(self) -> str:
        return (
            f"PIPELINE COMPLETE run_id={self.run_id} answers={self.answer_count} "
            f"{self.predictions_checksum}"
        )


class FixtureResourceFile(FrozenStrictModel, frozen=True):
    role: Literal["aliases", "context", "questions"]
    path: SafeRelativePath
    checksum: Sha256
    size_bytes: NonNegativeInt


class FixtureResourceManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["mil-002.fixture.resources.v1"]
    approval_state: Literal["approved"]
    license: Literal["CC0-1.0"]
    runtime_download_allowed: Literal[False]
    files: tuple[FixtureResourceFile, ...]

    @model_validator(mode="after")
    def _validate_inventory(self) -> Self:
        paths = tuple(item.path for item in self.files)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("fixture resource paths must be non-empty and unique")
        if paths != tuple(sorted(paths, key=lambda value: value.encode("utf-8"))):
            raise ValueError("fixture resource paths must use raw UTF-8 order")
        roles = tuple(item.role for item in self.files)
        if roles.count("questions") != 1 or roles.count("aliases") != 1:
            raise ValueError("fixture manifest requires one questions and one aliases file")
        if "context" not in roles:
            raise ValueError("fixture manifest requires at least one context file")
        return self


@dataclass(frozen=True, slots=True)
class _FixtureTokenizer(EvidenceTokenizer):
    tokenizer_id: str = RETRIEVAL_TOKENIZER_ID
    tokenizer_revision: str = RETRIEVAL_TOKENIZER_REVISION

    def count_tokens(self, text: str) -> int:
        return len(retrieval_tokens(text))


@dataclass(frozen=True, slots=True)
class _BuiltPipeline:
    result: FixturePipelineResult
    artifacts: dict[str, bytes]


def _fail(code: str, message: str, *, stage: str) -> NoReturn:
    raise FixturePipelineError(code, message, stage=stage)


def _safe_relative(value: str, *, stage: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"^[A-Za-z]:", value) is not None
    ):
        _fail(
            "PIPELINE_PATH_UNSAFE", "pipeline input path must be repository-relative", stage=stage
        )
    return path


def _resolve_input(root: Path, value: str, *, stage: str, directory: bool = False) -> Path:
    relative = _safe_relative(value, stage=stage)
    try:
        candidate = root.joinpath(*relative.parts)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail("PIPELINE_INPUT_MISSING", "pipeline input is unavailable", stage=stage)
    if candidate.is_symlink() or (resolved.is_dir() if directory else resolved.is_file()) is False:
        _fail("PIPELINE_INPUT_UNSUPPORTED", "pipeline input has an unsupported type", stage=stage)
    return resolved


def _read_input(root: Path, value: str, *, stage: str) -> bytes:
    try:
        path = _resolve_input(root, value, stage=stage)
        if path.stat().st_size > _MAX_FIXTURE_RESOURCE_BYTES:
            _fail(
                "FIXTURE_RESOURCE_TOO_LARGE",
                "synthetic fixture resource exceeds the 1048576-byte limit",
                stage=stage,
            )
        return path.read_bytes()
    except OSError:
        _fail("PIPELINE_INPUT_READ_FAILED", "pipeline input cannot be read", stage=stage)


def _load_resource_manifest(
    request: FixturePipelineRequest,
    root: Path,
) -> tuple[FixtureResourceManifest, bytes]:
    data = _read_input(root, request.resource_manifest_path, stage="inputs")
    try:
        manifest = parse_record_json(
            data,
            FixtureResourceManifest,
            artifact_path="fixture-resource-manifest.json",
        )
    except RecordValidationError as exc:
        raise FixturePipelineError(
            "FIXTURE_RESOURCE_MANIFEST_INVALID",
            "fixture resource manifest is invalid",
            stage="inputs",
        ) from exc
    for resource in manifest.files:
        resource_bytes = _read_input(root, resource.path, stage="inputs")
        if (
            len(resource_bytes) != resource.size_bytes
            or checksum_bytes(resource_bytes) != resource.checksum
        ):
            _fail(
                "FIXTURE_RESOURCE_CHECKSUM_MISMATCH",
                "fixture resource differs from its approved manifest",
                stage="inputs",
            )

    expected_questions = tuple(item.path for item in manifest.files if item.role == "questions")
    expected_aliases = tuple(item.path for item in manifest.files if item.role == "aliases")
    expected_contexts = tuple(item.path for item in manifest.files if item.role == "context")
    context_directory = _safe_relative(request.contexts_directory, stage="inputs").as_posix() + "/"
    requested_contexts = tuple(
        item.path for item in manifest.files if item.path.startswith(context_directory)
    )
    if (
        expected_questions != (request.questions_path,)
        or expected_aliases != (request.aliases_path,)
        or requested_contexts != expected_contexts
    ):
        _fail(
            "FIXTURE_RESOURCE_SCOPE_MISMATCH",
            "pipeline inputs do not equal the approved fixture inventory",
            stage="inputs",
        )
    return manifest, data


def _context_files(
    request: FixturePipelineRequest,
    root: Path,
    manifest: FixtureResourceManifest,
) -> tuple[OrganizerFile, ...]:
    _resolve_input(root, request.contexts_directory, stage="ingestion", directory=True)
    return tuple(
        OrganizerFile(relative_path=item.path, data=_read_input(root, item.path, stage="ingestion"))
        for item in manifest.files
        if item.role == "context"
    )


def _source_paths(root: Path, config_path: str) -> tuple[str, ...]:
    _resolve_input(root, config_path, stage="inputs")
    source_root = _resolve_input(root, "src/legal_rag", stage="inputs", directory=True)
    paths = [config_path]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file() and not path.is_symlink()
    )
    return tuple(sorted(paths, key=lambda value: value.encode("utf-8")))


def _code_revision(root: Path, source_tree_checksum: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return f"tree:{source_tree_checksum}"
    commit = completed.stdout.strip().casefold()
    return (
        f"git:{commit}"
        if completed.returncode == 0 and _GIT_COMMIT.fullmatch(commit)
        else f"tree:{source_tree_checksum}"
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)


def _retrieval_record(
    question_id: str,
    candidates: tuple[RetrievalCandidate, ...],
    diagnostics: tuple[RetrievalDiagnostic, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "retrieval.result.v1",
        "question_id": question_id,
        "candidates": [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "exact_reference_match": candidate.exact_reference_match,
                "sparse_score": candidate.sparse_score,
            }
            for candidate in candidates
        ],
        "diagnostics": [
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "candidate_count": diagnostic.candidate_count,
                "alias_manifest_checksum": diagnostic.alias_manifest_checksum,
            }
            for diagnostic in diagnostics
        ],
    }


def _call_hook(hook: Callable[[str], None] | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


def _build_pipeline(
    request: FixturePipelineRequest,
    hook: Callable[[str], None] | None,
    stage_state: list[str],
) -> _BuiltPipeline:
    root = request.project_root.resolve(strict=True)
    manifest, resource_manifest_bytes = _load_resource_manifest(request, root)
    source_paths = _source_paths(root, request.config_path)
    source_tree = checksum_file_set(root, source_paths)
    question_bytes = _read_input(root, request.questions_path, stage="inputs")
    alias_bytes = _read_input(root, request.aliases_path, stage="inputs")
    context_paths = tuple(item.path for item in manifest.files if item.role == "context")
    corpus_checksum = checksum_file_set(root, context_paths).checksum
    _call_hook(hook, "inputs")

    stage_state[0] = "ingestion"
    questions = OrganizerQuestionReader().read_bytes(
        question_bytes,
        kind="public",
        artifact_path=request.questions_path,
    )
    contexts = OrganizerContextReader().read_files(_context_files(request, root, manifest))
    _call_hook(hook, "ingestion")

    stage_state[0] = "chunking"
    chunk_results = tuple(chunk_context(context) for context in contexts.records)
    chunks: tuple[ChunkRecord, ...] = tuple(
        chunk for result in chunk_results for chunk in result.chunks
    )
    _call_hook(hook, "chunking")

    stage_state[0] = "indexing"
    aliases = load_alias_artifact(
        alias_bytes,
        contexts=contexts.records,
        corpus_checksum=corpus_checksum,
        artifact_path=request.aliases_path,
    )
    bm25 = build_bm25_index(
        chunks,
        corpus_checksum=corpus_checksum,
        alias_manifest_checksum=aliases.manifest_checksum,
        runtime_compatibility_id=APPROVED_BM25_RUNTIME_ID,
    )
    _call_hook(hook, "indexing")

    stage_state[0] = "retrieval"
    evidence_config = EvidenceSelectionConfig(
        max_evidence=12,
        evidence_token_budget=2048,
        reserve_tokens=0,
        template_id="fixture-evidence-template-v1",
        template_revision="1",
        template="[{rank}] {display_text}",
        separator="\n",
    )
    tokenizer = _FixtureTokenizer()
    admitted_by_question = []
    retrieval_records: list[dict[str, Any]] = []
    evidence_diagnostics: list[dict[str, Any]] = []
    for question in questions.records:
        parsed = parse_legal_reference(question.question)
        exact_candidates: tuple[RetrievalCandidate, ...] = ()
        exact_diagnostics = parsed.diagnostics
        if parsed.reference is not None:
            exact = resolve_exact_reference(parsed.reference, aliases=aliases, chunks=chunks)
            exact_candidates = exact.candidates
            exact_diagnostics = exact.diagnostics
        sparse = bm25.retrieve(question.question)
        candidates = union_rank_candidates(exact=exact_candidates, sparse=sparse.candidates)
        diagnostics = (*exact_diagnostics, *sparse.diagnostics)
        admitted = validate_and_admit_evidence(
            candidates,
            contexts=contexts.records,
            chunks=chunks,
            config=evidence_config,
            tokenizer=tokenizer,
        )
        admitted_by_question.append(admitted)
        retrieval_records.append(_retrieval_record(question.question_id, candidates, diagnostics))
        evidence_diagnostics.append(
            {
                "question_id": question.question_id,
                "items": [
                    {
                        "evidence_id": item.evidence_id,
                        "original_candidate_rank": item.original_candidate_rank,
                        "token_cost": item.token_cost,
                        "remaining_budget_before": item.remaining_budget_before,
                        "accepted_token_total_after": item.accepted_token_total_after,
                        "decision": item.decision,
                        "reason": item.reason,
                        "template_id": item.template_id,
                        "template_revision": item.template_revision,
                        "tokenizer_id": item.tokenizer_id,
                        "tokenizer_revision": item.tokenizer_revision,
                        "reserve_tokens": item.reserve_tokens,
                    }
                    for item in admitted.diagnostics
                ],
            }
        )
    _call_hook(hook, "retrieval")

    stage_state[0] = "generation"
    generator = FixtureExtractiveGenerator()
    generated = tuple(
        generator.generate(question, admitted.accepted)
        for question, admitted in zip(questions.records, admitted_by_question, strict=True)
    )
    _call_hook(hook, "generation")

    stage_state[0] = "serialization"
    diagnostics_bytes = canonical_json_bytes(
        {
            "schema_version": "evidence.diagnostics.batch.v1",
            "questions": evidence_diagnostics,
        }
    )
    placeholder = RunManifest.model_validate(
        {
            "schema_version": "run.manifest.v1",
            "run_id": _ZERO_RUN_ID,
            "pipeline_version": PIPELINE_VERSION,
            "code_revision": _code_revision(root, source_tree.checksum),
            "source_tree_checksum": source_tree.checksum,
            "scoped_source_paths": source_tree.paths,
            "config_checksum": checksum_bytes(
                _read_input(root, request.config_path, stage="inputs")
            ),
            "question_checksum": checksum_bytes(question_bytes),
            "corpus_checksum": corpus_checksum,
            "index_checksum": bm25.index_checksum,
            "split_checksum": None,
            "model_id": None,
            "model_revision": None,
            "tokenizer_id": RETRIEVAL_TOKENIZER_ID,
            "tokenizer_revision": RETRIEVAL_TOKENIZER_REVISION,
            "prompt_revision": None,
            "seed": "fixture-v1",
            "execution_mode": "local-offline",
            "competition_policy": "baseline.v1",
            "comparison_type": "baseline",
            "resolved_as_of_date": None,
            "as_of_timezone": None,
            "resource_manifest_checksum": checksum_bytes(resource_manifest_bytes),
            "evidence_diagnostics_checksum": _ZERO_CHECKSUM,
            "answer_artifact_checksum": _ZERO_CHECKSUM,
        }
    )
    run_id = compute_run_id(placeholder)
    answers = tuple(
        AnswerRecord.model_validate(
            {
                "schema_version": "internal.answer.v1",
                "question_id": answer.question_id,
                "answer": answer.answer_text,
                "generator_id": answer.generator_id,
                "evidence_ids": answer.used_evidence_ids,
                "run_id": run_id,
            }
        )
        for answer in generated
    )
    answer_bytes = answers_jsonl_bytes(answers)
    predictions_bytes = build_submission(question_bytes, answers)
    completed_manifest = placeholder.model_copy(
        update={
            "run_id": run_id,
            "evidence_diagnostics_checksum": checksum_bytes(diagnostics_bytes),
            "answer_artifact_checksum": checksum_bytes(answer_bytes),
        }
    )
    validate_run_manifest_identity(completed_manifest)

    chunk_manifest_bytes = canonical_json_bytes(
        {
            "schema_version": "chunk.corpus.manifest.v1",
            "contexts": [
                {
                    "context_id": context.context_id,
                    "manifest_checksum": checksum_bytes(result.manifest_bytes()),
                    "chunk_ids": [chunk.chunk_id for chunk in result.chunks],
                }
                for context, result in zip(contexts.records, chunk_results, strict=True)
            ],
        }
    )
    evidence_records = [
        evidence.model_dump(mode="json")
        for admitted in admitted_by_question
        for evidence in admitted.accepted
    ]
    generated_records = [answer.model_dump(mode="json") for answer in generated]
    artifacts = {
        "questions.import.jsonl": questions.jsonl_bytes(),
        "contexts.import.jsonl": contexts.jsonl_bytes(),
        "context.import.manifest.json": contexts.manifest_bytes(),
        "chunks.manifest.json": chunk_manifest_bytes,
        "aliases.manifest.json": aliases.manifest_bytes(),
        "bm25.index.manifest.json": bm25.manifest_bytes(),
        "retrieval.jsonl": _jsonl_bytes(retrieval_records),
        "evidence.jsonl": _jsonl_bytes(evidence_records),
        "evidence.diagnostics.json": diagnostics_bytes,
        "generated-answers.jsonl": _jsonl_bytes(generated_records),
        "answers.jsonl": answer_bytes,
        "predictions.json": predictions_bytes,
        "run.manifest.json": canonical_json_bytes(completed_manifest.model_dump(mode="json")),
    }
    _call_hook(hook, "serialization")
    return _BuiltPipeline(
        result=FixturePipelineResult(
            run_id=run_id,
            answer_count=len(answers),
            predictions_checksum=checksum_bytes(predictions_bytes),
            artifact_names=_ARTIFACT_NAMES,
        ),
        artifacts=artifacts,
    )


def _write_atomic(path: Path, data: bytes) -> None:
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
        temporary_path = None
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _write_failure(output_directory: Path, error: FixturePipelineError) -> None:
    data = canonical_json_bytes(
        {
            "schema_version": "pipeline.failure.v1",
            "code": error.code,
            "stage": error.stage,
        }
    )
    with contextlib.suppress(OSError):
        _write_atomic(output_directory / "failure.json", data)


def run_fixture_pipeline(
    request: FixturePipelineRequest,
    *,
    stage_hook: Callable[[str], None] | None = None,
) -> FixturePipelineResult:
    """Run every MIL-002 stage offline and publish only fully built artifacts."""

    stage_state = ["inputs"]
    try:
        built = _build_pipeline(request, stage_hook, stage_state)
        write_order = tuple(name for name in _ARTIFACT_NAMES if name != "predictions.json")
        for name in write_order:
            _write_atomic(request.output_directory / name, built.artifacts[name])
        validate_run_output_checksums(
            RunManifest.model_validate_json(built.artifacts["run.manifest.json"]),
            request.output_directory / "evidence.diagnostics.json",
            request.output_directory / "answers.jsonl",
        )
        _write_atomic(
            request.output_directory / "predictions.json",
            built.artifacts["predictions.json"],
        )
        return built.result
    except FixturePipelineError as caught_error:
        _write_failure(request.output_directory, caught_error)
        raise
    except Exception as exc:
        wrapped_error = FixturePipelineError(
            getattr(exc, "code", "PIPELINE_STAGE_FAILED"),
            "fixture pipeline stage failed",
            stage=stage_state[0],
        )
        _write_failure(request.output_directory, wrapped_error)
        raise wrapped_error from exc
