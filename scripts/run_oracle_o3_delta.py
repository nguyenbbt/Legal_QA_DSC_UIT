"""Generate only O3 rows whose selected evidence differs from completed O2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.competition import (
    evaluate_competition_bytes,
    write_competition_evaluation,
)
from legal_rag.evaluation.model_generation import run_grounded_generation_experiment
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.huggingface_local import Qwen3GeneratorBackend

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _jsonl(data: bytes) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in data.splitlines())


def _ranking(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(candidate["evidence_id"] for candidate in row["candidates"])


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--o2-retrieval", required=True, type=Path)
    parser.add_argument("--o2-predictions", required=True, type=Path)
    parser.add_argument("--o3-retrieval", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scorer-root", required=True, type=Path)
    parser.add_argument("--nltk-data", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()

    queue_data = arguments.queue.read_bytes()
    o2_data = arguments.o2_retrieval.read_bytes()
    o3_data = arguments.o3_retrieval.read_bytes()
    queue = _jsonl(queue_data)
    o2_by_id = {row["question_id"]: row for row in _jsonl(o2_data)}
    o3_by_id = {row["question_id"]: row for row in _jsonl(o3_data)}
    changed_ids = tuple(
        row["question_id"]
        for row in queue
        if _ranking(o2_by_id[row["question_id"]]) != _ranking(o3_by_id[row["question_id"]])
    )
    changed = frozenset(changed_ids)
    delta_queue = b"".join(
        content_json_bytes(row) for row in queue if row["question_id"] in changed
    )
    delta_retrieval = b"".join(
        content_json_bytes(o3_by_id[row["question_id"]])
        for row in queue
        if row["question_id"] in changed
    )
    if not changed_ids:
        raise ValueError("O3 has no changed evidence selections relative to O2")

    backend = Qwen3GeneratorBackend(
        arguments.checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device="cuda",
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
    )
    torch.cuda.reset_peak_memory_stats()
    delta_predictions, _, delta_manifest = run_grounded_generation_experiment(
        annotation_queue_data=delta_queue,
        retrieval_output_data=delta_retrieval,
        backend=backend,
        system_prompt=PROMPT_A,
        run_id="R-002A-O3-delta-G1A512-v1",
        evidence_limit=3,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        profile_state="diagnostic_non_promotable",
    )
    baseline_predictions = json.loads(arguments.o2_predictions.read_bytes())
    baseline_predictions.update(json.loads(delta_predictions))
    predictions = content_json_bytes(baseline_predictions)
    references = content_json_bytes({row["question_id"]: row["gold_answer"] for row in queue})
    output = arguments.output_directory
    write_immutable_bytes(output / "delta-predictions.json", delta_predictions)
    write_immutable_bytes(output / "delta-manifest.json", delta_manifest)
    write_immutable_bytes(output / "predictions.json", predictions)
    write_immutable_bytes(output / "references.json", references)
    evaluation = evaluate_competition_bytes(
        predictions,
        references,
        scorer_root=arguments.scorer_root,
        nltk_data_root=arguments.nltk_data,
    )
    write_competition_evaluation(
        evaluation,
        per_query_path=output / "evaluation-per-query.jsonl",
        report_path=output / "evaluation-report.json",
    )
    delta_value = json.loads(delta_manifest)
    manifest = {
        "schema_version": "retrieval.oracle-reuse-generation.manifest.v1",
        "run_id": "R-002A-O3-G1A512-v1",
        "profile_state": "diagnostic_non_promotable",
        "label_scope": "bounded_labeled_candidate_oracle",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_checksum": checksum_bytes(PROMPT_A.encode("utf-8")),
        "o2_retrieval_checksum": checksum_bytes(o2_data),
        "o3_retrieval_checksum": checksum_bytes(o3_data),
        "o2_predictions_checksum": checksum_bytes(arguments.o2_predictions.read_bytes()),
        "delta_queue_checksum": checksum_bytes(delta_queue),
        "delta_retrieval_checksum": checksum_bytes(delta_retrieval),
        "delta_predictions_checksum": checksum_bytes(delta_predictions),
        "predictions_checksum": checksum_bytes(predictions),
        "references_checksum": checksum_bytes(references),
        "question_count": len(queue),
        "generated_question_count": len(changed_ids),
        "reused_question_count": len(queue) - len(changed_ids),
        "ordered_changed_question_ids": changed_ids,
        "decoding": delta_value["decoding"],
        "wall_seconds": delta_value["elapsed_seconds"],
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "macro_meteor": evaluation.macro_meteor,
        "macro_rouge_l": evaluation.macro_rouge_l,
        "paid_service_used": False,
    }
    write_immutable_bytes(output / "manifest.json", content_json_bytes(manifest))
    print(content_json_bytes(manifest).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
