"""Normalize candidate queues and compare stored fixed-generator answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.generator_comparison import compare_retrieval_generation_experiments
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_A

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--baseline-queue", required=True, type=Path)
    parser.add_argument("--candidate-queue", required=True, type=Path)
    parser.add_argument("--baseline-retrieval", required=True, type=Path)
    parser.add_argument("--candidate-retrieval", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--baseline-per-query", required=True, type=Path)
    parser.add_argument("--candidate-per-query", required=True, type=Path)
    parser.add_argument("--baseline-runtime", required=True, type=float)
    parser.add_argument("--candidate-runtime", required=True, type=float)
    parser.add_argument(
        "--candidate-runtime-kind",
        choices=("exact_full_run", "conservative_upper_bound"),
        required=True,
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def _rows(data: bytes) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in data.splitlines())


def _normalize_queues(first_data: bytes, second_data: bytes) -> bytes:
    first = _rows(first_data)
    second_by_id = {row["question_id"]: row for row in _rows(second_data)}
    if len(first) != len(second_by_id):
        raise ValueError("comparison queues have different question counts")
    output: list[dict[str, object]] = []
    for row in first:
        question_id = row["question_id"]
        other = second_by_id.get(question_id)
        if other is None or any(
            row[field] != other[field] for field in ("question", "question_checksum", "gold_answer")
        ):
            raise ValueError("comparison queue question/reference mismatch")
        candidates: dict[str, dict[str, object]] = {}
        for source in (row["candidates"], other["candidates"]):
            for candidate in source:
                evidence_id = candidate["evidence_id"]
                minimal = {
                    "evidence_id": evidence_id,
                    "display_text": candidate["display_text"],
                }
                if evidence_id in candidates and candidates[evidence_id] != minimal:
                    raise ValueError("comparison queue changed evidence text")
                candidates[evidence_id] = minimal
        output.append(
            {
                "schema_version": "r008.generator-comparison-queue.v1",
                "question_id": question_id,
                "question_checksum": row["question_checksum"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "candidates": [
                    candidates[evidence_id] for evidence_id in sorted(candidates, key=str.encode)
                ],
            }
        )
    return b"".join(content_json_bytes(row) for row in output)


def _replay_manifest(
    *,
    queue_data: bytes,
    retrieval_data: bytes,
    predictions_data: bytes,
    run_id: str,
    baseline_run_id: str | None,
) -> tuple[bytes, bytes]:
    queue = _rows(queue_data)
    retrieval_by_id = {row["question_id"]: row for row in _rows(retrieval_data)}
    stored = json.loads(predictions_data)
    answers: dict[tuple[str, tuple[str, ...]], str] = {}
    for row in queue:
        candidates = {item["evidence_id"]: item for item in row["candidates"]}
        evidence_ids = tuple(
            item["evidence_id"] for item in retrieval_by_id[row["question_id"]]["candidates"][:3]
        )
        evidence = tuple(candidates[evidence_id]["display_text"] for evidence_id in evidence_ids)
        key = (row["question"], evidence)
        answer = stored[row["question_id"]]["answer"]
        if key in answers and answers[key] != answer:
            raise ValueError("stored replay answers conflict")
        answers[key] = answer

    class ReplayBackend:
        model_id = MODEL_ID
        model_revision = MODEL_REVISION

        def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
            if system_prompt != PROMPT_A:
                raise ValueError("replay prompt changed")
            return answers[(question, evidence)]

    replayed, _, manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue_data,
        retrieval_output_data=retrieval_data,
        backend=ReplayBackend(),
        system_prompt=PROMPT_A,
        run_id=run_id,
        evidence_limit=3,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id=baseline_run_id,
        changed_axes=("retrieval",) if baseline_run_id is not None else (),
        profile_state="diagnostic_non_promotable",
    )
    if replayed != predictions_data:
        raise ValueError("normalized queue did not replay exact prediction bytes")
    return replayed, manifest


def main() -> int:
    arguments = _arguments()
    baseline_queue_data = arguments.baseline_queue.read_bytes()
    candidate_queue_data = arguments.candidate_queue.read_bytes()
    queue_data = _normalize_queues(baseline_queue_data, candidate_queue_data)
    baseline_predictions = arguments.baseline_predictions.read_bytes()
    candidate_predictions = arguments.candidate_predictions.read_bytes()
    _, baseline_manifest = _replay_manifest(
        queue_data=queue_data,
        retrieval_data=arguments.baseline_retrieval.read_bytes(),
        predictions_data=baseline_predictions,
        run_id=arguments.baseline_run_id,
        baseline_run_id=None,
    )
    _, candidate_manifest = _replay_manifest(
        queue_data=queue_data,
        retrieval_data=arguments.candidate_retrieval.read_bytes(),
        predictions_data=candidate_predictions,
        run_id=arguments.candidate_run_id,
        baseline_run_id=arguments.baseline_run_id,
    )
    core_data = compare_retrieval_generation_experiments(
        baseline_per_query_data=arguments.baseline_per_query.read_bytes(),
        candidate_per_query_data=arguments.candidate_per_query.read_bytes(),
        baseline_manifest_data=baseline_manifest,
        candidate_manifest_data=candidate_manifest,
        baseline_runtime_seconds=arguments.baseline_runtime,
        candidate_runtime_seconds=arguments.candidate_runtime,
    )
    core = json.loads(core_data)
    blockers = list(core["promotion_blockers"])
    if arguments.candidate_runtime_kind != "exact_full_run":
        blockers.append("FULL_CANDIDATE_RUNTIME_UNAVAILABLE")
    comparison_data = content_json_bytes(
        {
            **core,
            "normalized_queue_checksum": checksum_bytes(queue_data),
            "baseline_source_queue_checksum": checksum_bytes(baseline_queue_data),
            "candidate_source_queue_checksum": checksum_bytes(candidate_queue_data),
            "baseline_prediction_bytes_replayed": True,
            "candidate_prediction_bytes_replayed": True,
            "candidate_runtime_kind": arguments.candidate_runtime_kind,
            "promotion_state": "rejected_preserved" if blockers else core["promotion_state"],
            "promotion_blockers": blockers,
        }
    )
    outputs = {
        "normalized-comparison-queue.v1.jsonl": queue_data,
        "baseline.replay-manifest.v1.json": baseline_manifest,
        "candidate.replay-manifest.v1.json": candidate_manifest,
        "comparison.v1.json": comparison_data,
    }
    for name, data in outputs.items():
        write_immutable_bytes(arguments.output_directory / name, data)
    print(comparison_data.decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
