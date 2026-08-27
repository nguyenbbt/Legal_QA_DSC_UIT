"""Build the fresh D-061 public evidence freeze entirely on the local host."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, checksum_file, content_json_bytes
from legal_rag.evaluation.legal_reranker_contract import (
    LEGAL_EVIDENCE_INSTRUCTION,
    LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
)
from legal_rag.evaluation.public_dry_run import build_public_evidence_queue
from legal_rag.ingestion.organizer import OrganizerQuestionReader
from legal_rag.models.huggingface_local import Qwen3RerankerBackend
from legal_rag.providers.public_campaign import D061_PUBLIC_CAMPAIGN
from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index
from legal_rag.retrieval.exact import load_frozen_alias_artifact
from legal_rag.retrieval.qwen3_reranker_prompt import QWEN3_RERANKER_SYSTEM

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RETRIEVAL_RUN_ID = "D061-base-reranker-public-1000-top50-v1"
MAXIMUM_LENGTH = 1536
BATCH_SIZE = 2
CANDIDATE_LIMIT = 50
EVIDENCE_LIMIT = 3
COMPETITION_PARAMETER_LIMIT = 4_000_000_000


class _ReplayOnlyReranker:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        raise AssertionError("D-061 evidence replay must not call the reranker")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path(root: Path, relative: object) -> Path:
    return root.joinpath(*str(relative).split("/"))


def _active_parameter_count(parameter_manifest: dict[str, object]) -> int:
    models = parameter_manifest.get("models")
    if not isinstance(models, list):
        raise RuntimeError("D-061 parameter manifest has no model entries")
    roles = {"generator", "reranker"}
    active = [entry for entry in models if isinstance(entry, dict) and entry.get("role") in roles]
    if {entry.get("role") for entry in active} != roles:
        raise RuntimeError("D-061 parameter manifest lacks the generator or reranker")
    count = sum(int(entry["exact_parameter_count"]) for entry in active)
    if count >= COMPETITION_PARAMETER_LIMIT:
        raise RuntimeError("D-061 active system violates the exclusive 4B gate")
    return count


def main() -> int:
    root = _project_root()
    public_path = root / "data" / "public-official.json"
    chunks_path = root / "artifacts" / "corpus" / "chunks.v1.jsonl"
    index_path = root / "artifacts" / "indices" / "bm25.v1.active.sqlite3"
    index_manifest_path = root / "artifacts" / "manifests" / "bm25.index.active.v1.json"
    aliases_path = root / "artifacts" / "governance" / "aliases.active.v1.jsonl"
    alias_manifest_path = root / "artifacts" / "manifests" / "aliases.active.v1.json"
    checkpoint = root / ".local" / "models" / "qwen3-reranker-0.6b" / MODEL_REVISION
    parameter_path = root / "artifacts" / "models" / "qwen3-btc-approved-parameter-manifest.v1.json"
    selection_path = _path(root, D061_PUBLIC_CAMPAIGN.selection_evidence_relative_path)
    evidence_path = _path(root, D061_PUBLIC_CAMPAIGN.evidence_relative_path)
    output_directory = evidence_path.parent
    checkpoint_directory = (
        root / ".local" / "runs" / "d061-base-reranker-public-v1" / "evidence-checkpoints"
    )

    public_data = public_path.read_bytes()
    index_manifest_data = index_manifest_path.read_bytes()
    alias_data = aliases_path.read_bytes()
    alias_manifest_data = alias_manifest_path.read_bytes()
    parameter_data = parameter_path.read_bytes()
    parameter_manifest = json.loads(parameter_data)
    active_parameter_count = _active_parameter_count(parameter_manifest)
    declared_system_count = int(parameter_manifest["system_parameter_count"])
    if declared_system_count >= COMPETITION_PARAMETER_LIMIT:
        raise RuntimeError("D-061 declared system violates the exclusive 4B gate")

    questions = (
        OrganizerQuestionReader()
        .read_bytes(public_data, kind="public", artifact_path="data/public-official.json")
        .records
    )
    if len(questions) != 1000:
        raise RuntimeError("D-061 requires the exact 1,000-question public source")

    torch.cuda.reset_peak_memory_stats()
    backend = Qwen3RerankerBackend(
        checkpoint,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        instruction=LEGAL_EVIDENCE_INSTRUCTION,
        device="cuda",
        batch_size=BATCH_SIZE,
        maximum_length=MAXIMUM_LENGTH,
    )
    frozen_inputs = {
        "alias_artifact": checksum_bytes(alias_data),
        "alias_manifest": checksum_bytes(alias_manifest_data),
        "bm25_database": checksum_file(index_path),
        "bm25_manifest": checksum_bytes(index_manifest_data),
        "chunk_artifact": str(json.loads(index_manifest_data)["chunks_artifact_checksum"]),
        "parameter_manifest": checksum_bytes(parameter_data),
        "public_source": checksum_bytes(public_data),
        "reranker_checkpoint": checksum_file(checkpoint / "model.safetensors"),
        "reranker_instruction": LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
        "reranker_system_prompt": checksum_bytes(QWEN3_RERANKER_SYSTEM.encode("utf-8")),
        "selection_evidence": checksum_bytes(selection_path.read_bytes()),
    }

    started = time.perf_counter()
    with open_disk_bm25_index(
        database_path=index_path,
        chunks_path=chunks_path,
        manifest_data=index_manifest_data,
    ) as index:
        aliases = load_frozen_alias_artifact(
            alias_data,
            manifest_data=alias_manifest_data,
            corpus_checksum=index.manifest.corpus_checksum,
            artifact_path=aliases_path.name,
        )
        artifacts = build_public_evidence_queue(
            questions,
            index=index,
            aliases=aliases,
            reranker=backend,
            retrieval_run_id=RETRIEVAL_RUN_ID,
            evidence_limit=EVIDENCE_LIMIT,
            reranker_candidate_limit=CANDIDATE_LIMIT,
            checkpoint_directory=checkpoint_directory,
            frozen_inputs=frozen_inputs,
        )
        replay = build_public_evidence_queue(
            questions,
            index=index,
            aliases=aliases,
            reranker=_ReplayOnlyReranker(),
            retrieval_run_id=RETRIEVAL_RUN_ID,
            evidence_limit=EVIDENCE_LIMIT,
            reranker_candidate_limit=CANDIDATE_LIMIT,
            checkpoint_directory=checkpoint_directory,
            frozen_inputs=frozen_inputs,
        )
    elapsed_seconds = time.perf_counter() - started
    if artifacts.queue_data != replay.queue_data:
        raise RuntimeError("D-061 evidence replay is not byte-identical")

    report = json.loads(artifacts.report_data)
    replay_report = json.loads(replay.report_data)
    if report["question_count"] != 1000 or replay_report["resumed_question_count"] != 1000:
        raise RuntimeError("D-061 evidence freeze is incomplete")
    if report["questions_without_evidence"]:
        raise RuntimeError("D-061 evidence freeze contains an empty evidence row")

    manifest_data = content_json_bytes(
        {
            "schema_version": "d061.public-evidence.manifest.v1",
            "campaign_id": D061_PUBLIC_CAMPAIGN.campaign_id,
            "run_id": RETRIEVAL_RUN_ID,
            "profile_state": "diagnostic_reporting_only",
            "public_results_usage": D061_PUBLIC_CAMPAIGN.public_results_usage,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "batch_size": BATCH_SIZE,
            "maximum_length": MAXIMUM_LENGTH,
            "candidate_limit": CANDIDATE_LIMIT,
            "evidence_limit": EVIDENCE_LIMIT,
            "active_generator_plus_reranker_parameter_count": active_parameter_count,
            "declared_manifest_system_parameter_count": declared_system_count,
            "competition_parameter_limit_exclusive": COMPETITION_PARAMETER_LIMIT,
            "passes_parameter_gate": True,
            "question_count": report["question_count"],
            "questions_without_evidence": report["questions_without_evidence"],
            "retrieval_fingerprint": report["retrieval_fingerprint"],
            "queue_checksum": checksum_bytes(artifacts.queue_data),
            "report_checksum": checksum_bytes(artifacts.report_data),
            "replay_queue_checksum": checksum_bytes(replay.queue_data),
            "byte_identical_replay": True,
            "generated_question_count": report["generated_question_count"],
            "replayed_question_count": replay_report["resumed_question_count"],
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "execution_mode": "local-offline",
            "paid_service_used": False,
            "frozen_inputs": frozen_inputs,
        }
    )
    outputs = {
        "public.evidence.v1.jsonl": artifacts.queue_data,
        "public.evidence.report.v1.json": artifacts.report_data,
        "public.evidence.replay-report.v1.json": replay.report_data,
        "d061.public-evidence.manifest.v1.json": manifest_data,
    }
    checksums = {
        name: write_immutable_bytes(output_directory / name, data) for name, data in outputs.items()
    }
    print(
        json.dumps(
            {
                "campaign_id": D061_PUBLIC_CAMPAIGN.campaign_id,
                "question_count": report["question_count"],
                "queue_checksum": checksum_bytes(artifacts.queue_data),
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
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
