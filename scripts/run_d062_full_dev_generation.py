"""Run resumable fixed-G1A512 generation for one D-062 retrieval arm."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, checksum_file, content_json_bytes
from legal_rag.evaluation.competition import evaluate_competition_bytes
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.huggingface_local import Qwen3GeneratorBackend

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QUESTION_COUNT = 716
MAXIMUM_INPUT_TOKENS = 2048
MAXIMUM_NEW_TOKENS = 512
_MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def checkpoint_checksum(directory: Path) -> str:
    """Bind a sharded checkpoint index, every referenced shard, and runtime metadata."""

    index_path = directory / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("D-062 model checkpoint index is invalid") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("D-062 model checkpoint weight map is invalid")
    shards = set(weight_map.values())
    if any(
        not isinstance(name, str) or not name.endswith(".safetensors") or Path(name).name != name
        for name in shards
    ):
        raise ValueError("D-062 model checkpoint shard identity is invalid")
    names = ["model.safetensors.index.json", *sorted(shards, key=str.encode)]
    names.extend(name for name in _MODEL_METADATA_FILES if (directory / name).is_file())
    files = [
        {"name": name, "checksum": checksum_file(directory / name)}
        for name in sorted(names, key=str.encode)
    ]
    return checksum_bytes(content_json_bytes(files))


class _GeneratorBackend(Protocol):
    model_id: str
    model_revision: str

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str: ...


@dataclass(frozen=True, slots=True)
class GenerationExpectation:
    position: int
    question_id: str
    question: str
    evidence_ids: tuple[str, ...]
    evidence: tuple[str, ...]


class CheckpointedGeneratorBackend:
    """Strict per-question checkpoint adapter for deterministic local inference."""

    def __init__(
        self,
        *,
        backend: _GeneratorBackend,
        checkpoint_directory: Path,
        run_id: str,
        run_fingerprint: str,
        expectations: Sequence[GenerationExpectation],
    ) -> None:
        self._backend = backend
        self._checkpoint_directory = checkpoint_directory
        self._run_id = run_id
        self._run_fingerprint = run_fingerprint
        self._expectations = tuple(expectations)
        self._position = 0
        self.generated_count = 0
        self.resumed_count = 0
        self.checkpoint_elapsed_seconds = 0.0

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def model_revision(self) -> str:
        return self._backend.model_revision

    def _checkpoint_path(self, expectation: GenerationExpectation) -> Path:
        digest = checksum_bytes(expectation.question_id.encode("utf-8")).removeprefix("sha256:")
        return self._checkpoint_directory / f"{expectation.position:04d}-{digest}.json"

    def _load(self, path: Path, expectation: GenerationExpectation) -> tuple[str, float]:
        data = path.read_bytes()
        try:
            value = json.loads(data)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("D-062 generation checkpoint is invalid") from error
        if not isinstance(value, dict) or content_json_bytes(value) != data:
            raise ValueError("D-062 generation checkpoint is not canonical")
        evidence_checksum = checksum_bytes(content_json_bytes(list(expectation.evidence)))
        expected = {
            "schema_version": "d062.generation.checkpoint.v1",
            "run_id": self._run_id,
            "run_fingerprint": self._run_fingerprint,
            "position": expectation.position,
            "question_id": expectation.question_id,
            "question_checksum": checksum_bytes(expectation.question.encode("utf-8")),
            "evidence_ids": list(expectation.evidence_ids),
            "evidence_checksum": evidence_checksum,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("D-062 generation checkpoint differs from frozen inputs")
        answer = value.get("answer")
        elapsed = value.get("elapsed_seconds")
        if set(value) != {*expected, "answer", "elapsed_seconds"}:
            raise ValueError("D-062 generation checkpoint shape is invalid")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("D-062 generation checkpoint answer is invalid")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
            raise ValueError("D-062 generation checkpoint telemetry is invalid")
        return answer, float(elapsed)

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str:
        if self._position >= len(self._expectations):
            raise ValueError("D-062 generation invoked more rows than frozen")
        expectation = self._expectations[self._position]
        self._position += 1
        evidence_tuple = tuple(evidence)
        if question != expectation.question or evidence_tuple != expectation.evidence:
            raise ValueError("D-062 generation call differs from frozen inputs")
        path = self._checkpoint_path(expectation)
        if path.exists():
            answer, elapsed = self._load(path, expectation)
            self.resumed_count += 1
            self.checkpoint_elapsed_seconds += elapsed
            return answer

        started = time.perf_counter()
        answer = self._backend.generate(
            system_prompt=system_prompt,
            question=question,
            evidence=evidence_tuple,
        ).strip()
        elapsed = time.perf_counter() - started
        if not answer:
            raise ValueError("D-062 generator returned an empty answer")
        checkpoint = {
            "schema_version": "d062.generation.checkpoint.v1",
            "run_id": self._run_id,
            "run_fingerprint": self._run_fingerprint,
            "position": expectation.position,
            "question_id": expectation.question_id,
            "question_checksum": checksum_bytes(expectation.question.encode("utf-8")),
            "evidence_ids": list(expectation.evidence_ids),
            "evidence_checksum": checksum_bytes(content_json_bytes(list(expectation.evidence))),
            "answer": answer,
            "elapsed_seconds": elapsed,
        }
        write_immutable_bytes(path, content_json_bytes(checkpoint))
        self.generated_count += 1
        self.checkpoint_elapsed_seconds += elapsed
        return answer


class _ReplayBackend:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str:
        raise AssertionError("D-062 replay must not invoke the generator")


def build_generation_expectations(
    annotation_queue_data: bytes, retrieval_output_data: bytes
) -> tuple[GenerationExpectation, ...]:
    queue = tuple(json.loads(line) for line in annotation_queue_data.splitlines())
    retrieval = tuple(json.loads(line) for line in retrieval_output_data.splitlines())
    if len(queue) != len(retrieval):
        raise ValueError("D-062 generation artifacts have different row counts")
    expectations: list[GenerationExpectation] = []
    for position, (item, ranked) in enumerate(zip(queue, retrieval, strict=True)):
        if item["question_id"] != ranked["question_id"]:
            raise ValueError("D-062 generation artifacts have different order")
        candidates = {candidate["evidence_id"]: candidate for candidate in item["candidates"]}
        evidence_ids = tuple(candidate["evidence_id"] for candidate in ranked["candidates"][:3])
        if any(evidence_id not in candidates for evidence_id in evidence_ids):
            raise ValueError("D-062 retrieval references unknown evidence")
        expectations.append(
            GenerationExpectation(
                position=position,
                question_id=item["question_id"],
                question=item["question"],
                evidence_ids=evidence_ids,
                evidence=tuple(
                    candidates[evidence_id]["display_text"] for evidence_id in evidence_ids
                ),
            )
        )
    return tuple(expectations)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", choices=("r0", "base-reranker"), required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = Path(__file__).resolve().parents[1]
    retrieval_directory = (
        root / "artifacts/evaluations/recovery/R-009/D062-full-development-v1" / arguments.role
    )
    output = retrieval_directory / "generation"
    checkpoint_directory = root / ".local/runs/d062-full-development-v1/generation" / arguments.role
    model_checkpoint = root / ".local/models/qwen3-1.7b" / MODEL_REVISION
    queue_data = (retrieval_directory / "annotation-queue.v1.jsonl").read_bytes()
    retrieval_data = (retrieval_directory / "retrieval.v1.jsonl").read_bytes()
    expectations = build_generation_expectations(queue_data, retrieval_data)
    if len(expectations) != QUESTION_COUNT:
        raise RuntimeError("D-062 generation requires exactly 716 rows")
    run_id = f"D062-{arguments.role}-G1A512-development-716-v1"
    run_fingerprint = checksum_bytes(
        content_json_bytes(
            {
                "schema_version": "d062.generation.fingerprint.v1",
                "run_id": run_id,
                "annotation_queue_checksum": checksum_bytes(queue_data),
                "retrieval_output_checksum": checksum_bytes(retrieval_data),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_checkpoint_checksum": checkpoint_checksum(model_checkpoint),
                "prompt_checksum": checksum_bytes(PROMPT_A.encode("utf-8")),
                "evidence_limit": 3,
                "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
                "maximum_new_tokens": MAXIMUM_NEW_TOKENS,
                "do_sample": False,
                "enable_thinking": False,
            }
        )
    )
    backend = Qwen3GeneratorBackend(
        model_checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device="cuda",
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
    )
    checkpointed = CheckpointedGeneratorBackend(
        backend=backend,
        checkpoint_directory=checkpoint_directory,
        run_id=run_id,
        run_fingerprint=run_fingerprint,
        expectations=expectations,
    )
    torch.cuda.reset_peak_memory_stats()
    predictions, references, manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        backend=checkpointed,
        system_prompt=PROMPT_A,
        run_id=run_id,
        evidence_limit=3,
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id=("D062-r0-G1A512-development-716-v1" if arguments.role != "r0" else None),
        changed_axes=(("retrieval",) if arguments.role != "r0" else ()),
        profile_state="diagnostic_non_promotable",
    )
    replay_backend = CheckpointedGeneratorBackend(
        backend=_ReplayBackend(),
        checkpoint_directory=checkpoint_directory,
        run_id=run_id,
        run_fingerprint=run_fingerprint,
        expectations=expectations,
    )
    replayed, replay_references, replay_manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        backend=replay_backend,
        system_prompt=PROMPT_A,
        run_id=run_id,
        evidence_limit=3,
        maximum_input_tokens=MAXIMUM_INPUT_TOKENS,
        maximum_new_tokens=MAXIMUM_NEW_TOKENS,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id=("D062-r0-G1A512-development-716-v1" if arguments.role != "r0" else None),
        changed_axes=(("retrieval",) if arguments.role != "r0" else ()),
        profile_state="diagnostic_non_promotable",
    )
    if replayed != predictions or replay_references != references:
        raise RuntimeError("D-062 generation replay differs")
    evaluation = evaluate_competition_bytes(
        predictions,
        references,
        scorer_root=root / "Scoring-Program-Task-LegalQA",
        nltk_data_root=root / "resources/nltk_data",
        baseline_kind="d062_full_development_fixed_generator",
        limitation="development_only_retrieval_single_axis",
    )
    manifest_value = json.loads(manifest)
    telemetry = content_json_bytes(
        {
            "schema_version": "d062.generation.telemetry.v1",
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "execution_mode": "local-offline",
            "paid_service_used": False,
            "question_count": evaluation.question_count,
            "generated_question_count": checkpointed.generated_count,
            "resumed_question_count": checkpointed.resumed_count,
            "checkpoint_elapsed_seconds": checkpointed.checkpoint_elapsed_seconds,
            "wall_seconds": manifest_value["elapsed_seconds"],
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "macro_meteor": evaluation.macro_meteor,
            "macro_rouge_l": evaluation.macro_rouge_l,
            "byte_identical_replay": True,
        }
    )
    outputs = {
        "predictions.json": predictions,
        "references.json": references,
        "manifest.json": manifest,
        "replay-manifest.json": replay_manifest,
        "evaluation-per-query.jsonl": evaluation.per_query_bytes,
        "evaluation-report.json": evaluation.report_bytes,
        "telemetry.json": telemetry,
    }
    checksums = {name: write_immutable_bytes(output / name, data) for name, data in outputs.items()}
    print(
        json.dumps(
            {
                "role": arguments.role,
                "macro_meteor": evaluation.macro_meteor,
                "macro_rouge_l": evaluation.macro_rouge_l,
                "checksums": checksums,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
