from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index
from legal_rag.training.local_reranker_dataset import (
    RerankerSeedBuildConfig,
    build_reranker_dataset_from_selections,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build closed official-train R-008 groups")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--selections", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--index-database", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--maximum-negatives", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    question_data = arguments.questions.read_bytes()
    split_data = arguments.split_manifest.read_bytes()
    selection_data = arguments.selections.read_bytes()
    index_manifest_data = arguments.index_manifest.read_bytes()
    with open_disk_bm25_index(
        database_path=arguments.index_database,
        chunks_path=arguments.chunks,
        manifest_data=index_manifest_data,
    ) as index:
        artifacts = build_reranker_dataset_from_selections(
            question_data=question_data,
            split_manifest_data=split_data,
            selection_data=selection_data,
            chunks=index,
            chunks_checksum=index.manifest.chunks_artifact_checksum,
            index_checksum=index.index_checksum,
            config=RerankerSeedBuildConfig(maximum_negatives=arguments.maximum_negatives),
        )

    paths = {
        "groups": arguments.output_directory / "training-groups.v1.jsonl",
        "provenance": arguments.output_directory / "training-examples.v1.jsonl",
        "manifest": arguments.output_directory / "training-manifest.v1.json",
    }
    checksums = {
        "groups": write_immutable_bytes(paths["groups"], artifacts.groups_data),
        "provenance": write_immutable_bytes(paths["provenance"], artifacts.provenance_data),
        "manifest": write_immutable_bytes(paths["manifest"], artifacts.manifest_data),
    }
    report_data = content_json_bytes(
        {
            "schema_version": "r008.dataset-build.report.v1",
            "group_count": artifacts.group_count,
            "pair_count": artifacts.pair_count,
            "execution_mode": "local-offline",
            "contains_generated_text": False,
            "checksums": checksums,
        }
    )
    report_checksum = write_immutable_bytes(
        arguments.output_directory / "dataset-build.report.v1.json", report_data
    )
    print(
        f"R008 DATASET groups={artifacts.group_count} pairs={artifacts.pair_count} "
        f"manifest={checksums['manifest']} report={report_checksum}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
