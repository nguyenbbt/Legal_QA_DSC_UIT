"""Local grounded-generation experiment over a frozen retrieval output."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Literal, Protocol

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes


class GeneratorBackendProtocol(Protocol):
    model_id: str
    model_revision: str

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str: ...


def run_grounded_generation_experiment(
    *,
    annotation_queue_data: bytes,
    retrieval_output_data: bytes,
    backend: GeneratorBackendProtocol,
    system_prompt: str,
    run_id: str,
    evidence_limit: int,
    maximum_input_tokens: int,
    maximum_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
    baseline_run_id: str | None = None,
    changed_axes: tuple[str, ...] = (),
    profile_state: Literal[
        "exploratory_non_promotable",
        "btc_approved_local_ablation",
        "g3_qlora_local_evaluation",
        "clean_reproducibility",
        "diagnostic_non_promotable",
    ] = "exploratory_non_promotable",
) -> tuple[bytes, bytes, bytes]:
    """Generate answers from exactly the frozen ranked evidence IDs."""

    if evidence_limit < 1 or not system_prompt.strip():
        raise ValueError("generation evidence limit and prompt must be valid")
    if maximum_input_tokens < 1 or maximum_new_tokens < 1:
        raise ValueError("generation token limits must be positive")
    if do_sample or enable_thinking:
        raise ValueError("competition generation must be deterministic with thinking disabled")
    if baseline_run_id is None and changed_axes:
        raise ValueError("changed axes require a baseline run")
    if baseline_run_id is not None and len(changed_axes) != 1:
        raise ValueError("an ablation must declare exactly one changed axis")
    queue = tuple(json.loads(line) for line in annotation_queue_data.splitlines())
    retrieval = tuple(json.loads(line) for line in retrieval_output_data.splitlines())
    ranking_by_id = {
        row["question_id"]: tuple(candidate["evidence_id"] for candidate in row["candidates"])
        for row in retrieval
    }
    predictions: dict[str, dict[str, str]] = {}
    references: dict[str, str] = {}
    answer_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for item in queue:
        question_id = item["question_id"]
        candidates = {candidate["evidence_id"]: candidate for candidate in item["candidates"]}
        ranked_ids = ranking_by_id[question_id][:evidence_limit]
        if any(evidence_id not in candidates for evidence_id in ranked_ids):
            raise ValueError("frozen retrieval references unknown evidence")
        answer = backend.generate(
            system_prompt=system_prompt,
            question=item["question"],
            evidence=tuple(candidates[evidence_id]["display_text"] for evidence_id in ranked_ids),
        ).strip()
        if not answer:
            answer = "Không đủ căn cứ trong dữ liệu được cung cấp để trả lời câu hỏi này."
        predictions[question_id] = {"answer": answer}
        references[question_id] = item["gold_answer"]
        answer_rows.append(
            {
                "schema_version": "model.generated_answer.v1",
                "run_id": run_id,
                "question_id": question_id,
                "answer": answer,
                "evidence_ids": ranked_ids,
            }
        )
    predictions_data = content_json_bytes(predictions)
    references_data = content_json_bytes(references)
    manifest_value: dict[str, object] = {
        "schema_version": "model.generation.experiment.manifest.v1",
        "run_id": run_id,
        "profile_state": profile_state,
        "model_id": backend.model_id,
        "model_revision": backend.model_revision,
        "prompt_checksum": checksum_bytes(system_prompt.encode("utf-8")),
        "retrieval_output_checksum": checksum_bytes(retrieval_output_data),
        "annotation_queue_checksum": checksum_bytes(annotation_queue_data),
        "predictions_checksum": checksum_bytes(predictions_data),
        "references_checksum": checksum_bytes(references_data),
        "question_count": len(answer_rows),
        "evidence_limit": evidence_limit,
        "decoding": {
            "maximum_input_tokens": maximum_input_tokens,
            "maximum_new_tokens": maximum_new_tokens,
            "do_sample": do_sample,
            "enable_thinking": enable_thinking,
        },
        "comparison": {
            "baseline_run_id": baseline_run_id,
            "changed_axes": changed_axes,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    adapter_id = getattr(backend, "adapter_id", None)
    if adapter_id is not None:
        adapter_checksum = getattr(backend, "adapter_checksum", None)
        adapter_config_checksum = getattr(backend, "adapter_config_checksum", None)
        if not all(
            isinstance(value, str)
            for value in (adapter_id, adapter_checksum, adapter_config_checksum)
        ):
            raise ValueError("adapter provenance must contain string identities")
        manifest_value["adapter"] = {
            "adapter_id": adapter_id,
            "adapter_checksum": adapter_checksum,
            "adapter_config_checksum": adapter_config_checksum,
        }
    manifest = content_json_bytes(manifest_value)
    return predictions_data, references_data, manifest


__all__ = ["GeneratorBackendProtocol", "run_grounded_generation_experiment"]
