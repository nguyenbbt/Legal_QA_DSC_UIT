"""Freeze train-calibrated R-006B packing and apply it once to development R0."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.token_counting import Qwen3InputTokenCounter
from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index
from legal_rag.retrieval.packing_calibration import (
    PackingCalibrationCandidate,
    PackingCalibrationGroup,
    calibrate_relative_sparse_score,
)
from legal_rag.retrieval.selection_artifacts import build_evidence_selection_artifacts
from legal_rag.training.rag_sft import load_gold_questions

_SELECTION_FIELDS = {
    "schema_version",
    "question_id",
    "question_checksum",
    "evidence_ids",
    "evidence_checksums",
    "support_score",
    "support_policy_version",
}
_EXPECTED_QUESTION_CHECKSUM = (
    "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f"
)
_EXPECTED_SPLIT_CHECKSUM = "sha256:e1ff7fbca4ed9c9434eee5625b64a2743b904977d25c0c1a75378fcbc84210f6"
_EXPECTED_SELECTION_CHECKSUM = (
    "sha256:526906e3efceacd96535be14bb51d1e26bfc23bfef490a6c0224258e73e54011"
)
_EXPECTED_INDEX_CHECKSUM = "sha256:a21d01063f28cd445daa377eec7b4088fc959e3a38ec5b5cdad860f6de73275f"
_EXPECTED_DEVELOPMENT_QUEUE_CHECKSUM = (
    "sha256:bf1402e9679d3c0460679db6456000cf8995281d3e62a0df2cbc06b5b6922989"
)
_EXPECTED_R0_RETRIEVAL_CHECKSUM = (
    "sha256:d42ff25966e745a8e9eb47a87f1f0477cb8850fe80fcf873cb691cc51f04a54b"
)
_EXPECTED_SELECTION_COUNT = 187


def _require_checksum(data: bytes, expected: str, name: str) -> None:
    if checksum_bytes(data) != expected:
        raise ValueError(f"D-058 {name} checksum changed")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--train-questions", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--train-selections", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--index-database", required=True, type=Path)
    parser.add_argument("--index-manifest", required=True, type=Path)
    parser.add_argument("--development-queue", required=True, type=Path)
    parser.add_argument("--r0-retrieval", required=True, type=Path)
    parser.add_argument("--tokenizer-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def _selection_rows(data: bytes) -> tuple[dict[str, Any], ...]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("train selection JSONL framing is invalid")
    rows: list[dict[str, Any]] = []
    for line in data.splitlines(keepends=True):
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or set(value) != _SELECTION_FIELDS
            or value.get("schema_version") != "training.evidence.selection.v1"
            or content_json_bytes(value) != line
        ):
            raise ValueError("train selection row contract is invalid")
        question_id = value.get("question_id")
        evidence_ids = value.get("evidence_ids")
        if (
            not isinstance(question_id, str)
            or not question_id
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(not isinstance(item, str) or not item for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise ValueError("train selection identity or relevance is invalid")
        rows.append(value)
    if tuple(row["question_id"] for row in rows) != tuple(
        sorted((row["question_id"] for row in rows), key=str.encode)
    ):
        raise ValueError("train selections must be bytewise ordered")
    return tuple(rows)


def main() -> int:
    arguments = _arguments()
    question_data = arguments.train_questions.read_bytes()
    split_data = arguments.split_manifest.read_bytes()
    selection_data = arguments.train_selections.read_bytes()
    index_manifest_data = arguments.index_manifest.read_bytes()
    development_queue_data = arguments.development_queue.read_bytes()
    r0_retrieval_data = arguments.r0_retrieval.read_bytes()
    _require_checksum(question_data, _EXPECTED_QUESTION_CHECKSUM, "train questions")
    _require_checksum(split_data, _EXPECTED_SPLIT_CHECKSUM, "split manifest")
    _require_checksum(selection_data, _EXPECTED_SELECTION_CHECKSUM, "train selections")
    _require_checksum(
        development_queue_data,
        _EXPECTED_DEVELOPMENT_QUEUE_CHECKSUM,
        "development queue",
    )
    _require_checksum(r0_retrieval_data, _EXPECTED_R0_RETRIEVAL_CHECKSUM, "R0 retrieval")
    questions = load_gold_questions(question_data)
    questions_by_id = {question.question_id: question for question in questions}
    source_checksums = {question.source_checksum for question in questions}
    if len(source_checksums) != 1:
        raise ValueError("official train question source checksum is inconsistent")
    split_rows = load_split_manifest_rows(
        split_data,
        expected_source_checksum=next(iter(source_checksums)),
        expected_question_ids=tuple(question.question_id for question in questions),
    )
    split_by_id = {row.question_id: row.split for row in split_rows}
    selections = _selection_rows(selection_data)
    if len(selections) != _EXPECTED_SELECTION_COUNT:
        raise ValueError("D-058 requires exactly 187 approved train selections")

    group_rows: list[dict[str, Any]] = []
    groups: list[PackingCalibrationGroup] = []
    with open_disk_bm25_index(
        database_path=arguments.index_database,
        chunks_path=arguments.chunks,
        manifest_data=index_manifest_data,
    ) as index:
        if index.index_checksum != _EXPECTED_INDEX_CHECKSUM:
            raise ValueError("D-058 active R0 index checksum changed")
        for selection in selections:
            question_id = selection["question_id"]
            question = questions_by_id.get(question_id)
            if question is None or split_by_id.get(question_id) != "train":
                raise ValueError("packing calibration accepts official train selections only")
            if selection["question_checksum"] != checksum_bytes(question.question.encode()):
                raise ValueError("packing calibration question checksum changed")
            retrieval = index.retrieve(question.question, candidate_limit=12)
            candidates = tuple(
                PackingCalibrationCandidate(
                    evidence_id=candidate.chunk.chunk_id,
                    sparse_score=float(candidate.sparse_score),
                )
                for candidate in retrieval.candidates
                if candidate.sparse_score is not None
            )
            if not candidates:
                raise ValueError("packing calibration R0 candidate set is empty")
            relevant_ids = tuple(selection["evidence_ids"])
            groups.append(
                PackingCalibrationGroup(
                    question_id=question_id,
                    split="train",
                    relevant_evidence_ids=relevant_ids,
                    candidates=candidates,
                )
            )
            group_rows.append(
                {
                    "schema_version": "evidence-packing-calibration.group.v1",
                    "question_id": question_id,
                    "question_checksum": selection["question_checksum"],
                    "split": "train",
                    "relevant_evidence_ids": relevant_ids,
                    "candidates": [asdict(candidate) for candidate in candidates],
                }
            )

        candidate_universe_data = b"".join(content_json_bytes(row) for row in group_rows)
        calibration = calibrate_relative_sparse_score(tuple(groups))
        calibration_data = content_json_bytes(
            {
                **asdict(calibration),
                "construction_version": "r006b-train-relative-bm25-f1.v1",
                "split": "train",
                "candidate_limit": 12,
                "maximum_evidence_count": 3,
                "contains_generated_text": False,
                "question_source_checksum": checksum_bytes(question_data),
                "split_manifest_checksum": checksum_bytes(split_data),
                "selection_checksum": checksum_bytes(selection_data),
                "candidate_universe_checksum": checksum_bytes(candidate_universe_data),
                "chunks_checksum": index.manifest.chunks_artifact_checksum,
                "index_checksum": index.index_checksum,
            }
        )

    calibration_checksum = checksum_bytes(calibration_data)
    counter = Qwen3InputTokenCounter.from_checkpoint(
        arguments.tokenizer_checkpoint,
        system_prompt=PROMPT_A,
    )
    development = build_evidence_selection_artifacts(
        annotation_queue_data=development_queue_data,
        retrieval_output_data=r0_retrieval_data,
        source_run_id="R0",
        selected_run_id="R006B-P3A-R0-train-calibrated-v1",
        maximum_input_tokens=2048,
        token_counter=counter,
        minimum_relative_sparse_score=calibration.minimum_relative_sparse_score,
        calibration_checksum=calibration_checksum,
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train-candidate-universe.v1.jsonl": candidate_universe_data,
        "packing-calibration.v1.json": calibration_data,
        "development-retrieval.v1.jsonl": development.retrieval_output,
        "development-selection-report.v1.json": development.selection_report,
    }
    checksums = {
        name: write_immutable_bytes(arguments.output_directory / name, data)
        for name, data in outputs.items()
    }
    print(
        content_json_bytes(
            {
                "schema_version": "r006b.packing-recovery.result.v1",
                "minimum_relative_sparse_score": format(
                    calibration.minimum_relative_sparse_score, ".17g"
                ),
                "micro_evidence_set_f1": format(calibration.micro_f1, ".17g"),
                "group_count": calibration.group_count,
                "checksums": checksums,
            }
        ).decode(),
        end="",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
