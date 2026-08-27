"""Run R-003 legal sparse discovery on the frozen 60-question benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.real_retrieval import (
    build_real_retrieval_artifacts,
    retrieve_question,
)
from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index
from legal_rag.retrieval.exact import load_frozen_alias_artifact
from legal_rag.retrieval.legal_sparse import LegalSparseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--annotation-queue", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--index-database", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--aliases", required=True, type=Path)
    parser.add_argument("--alias-manifest", required=True, type=Path)
    parser.add_argument("--discovery-limit", type=int, default=100)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()

    queue = tuple(json.loads(line) for line in arguments.annotation_queue.read_bytes().splitlines())
    index_manifest_data = arguments.index_manifest.read_bytes()
    alias_manifest_data = arguments.alias_manifest.read_bytes()
    started = time.perf_counter()
    with open_disk_bm25_index(
        database_path=arguments.index_database,
        chunks_path=arguments.chunks,
        manifest_data=index_manifest_data,
    ) as index:
        aliases = load_frozen_alias_artifact(
            arguments.aliases.read_bytes(),
            manifest_data=alias_manifest_data,
            corpus_checksum=index.manifest.corpus_checksum,
            artifact_path=arguments.aliases.name,
        )
        legal_index = LegalSparseRetriever(index, discovery_limit=arguments.discovery_limit)
        questions = tuple(
            QuestionRecord.model_validate(
                {
                    "schema_version": "internal.question.v1",
                    "question_id": item["question_id"],
                    "original_id": item["question_id"],
                    "original_id_kind": "object_key_string",
                    "source_position": position,
                    "source_artifact": "artifacts/private/recovery-grounding-queue.jsonl",
                    "source_checksum": item["question_checksum"],
                    "question": item["question"],
                    "answer": item["gold_answer"],
                    "answer_state": "gold",
                }
            )
            for position, item in enumerate(queue)
        )
        results = tuple(
            retrieve_question(
                question,
                index=legal_index,
                aliases=aliases,
                candidate_limit=arguments.discovery_limit,
            )
            for question in questions
        )
        artifacts = build_real_retrieval_artifacts(
            results,
            selected_question_ids=tuple(item["question_id"] for item in queue),
            split_checksum=queue[0]["split_checksum"],
            index_checksum=legal_index.index_checksum,
            chunks_checksum=index.manifest.chunks_artifact_checksum,
            alias_manifest_checksum=aliases.manifest_checksum,
        )
    wall_seconds = time.perf_counter() - started
    write_immutable_bytes(
        arguments.output_directory / "retrieval.v1.jsonl", artifacts.retrieval_output
    )
    write_immutable_bytes(
        arguments.output_directory / "annotation-queue.v1.jsonl", artifacts.annotation_queue
    )
    write_immutable_bytes(arguments.output_directory / "report.v1.json", artifacts.report)
    write_immutable_bytes(
        arguments.output_directory / "telemetry.v1.json",
        content_json_bytes(
            {
                "schema_version": "legal-sparse.telemetry.v1",
                "execution_mode": "local-offline",
                "question_count": len(results),
                "candidate_limit": arguments.discovery_limit,
                "wall_seconds": wall_seconds,
                "peak_memory_bytes": None,
                "peak_memory_unavailable_reason": "WINDOWS_RSS_SAMPLER_NOT_MANIFESTED",
                "source_index_bytes": arguments.index_database.stat().st_size,
                "retrieval_output_bytes": len(artifacts.retrieval_output),
                "annotation_queue_bytes": len(artifacts.annotation_queue),
                "paid_service_used": False,
                "cost_usd": 0.0,
            },
        ),
    )
    print(
        json.dumps(
            {
                "question_count": len(results),
                "candidate_limit": arguments.discovery_limit,
                "index_checksum": legal_index.index_checksum,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
