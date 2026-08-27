"""Replay frozen R-006B calibration and selection without rerunning the BM25 scan."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.models.token_counting import Qwen3InputTokenCounter
from legal_rag.retrieval.packing_calibration import (
    PackingCalibrationCandidate,
    PackingCalibrationGroup,
    calibrate_relative_sparse_score,
)
from legal_rag.retrieval.selection_artifacts import build_evidence_selection_artifacts


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate-universe", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--development-queue", required=True, type=Path)
    parser.add_argument("--r0-retrieval", required=True, type=Path)
    parser.add_argument("--tokenizer-checkpoint", required=True, type=Path)
    parser.add_argument("--development-retrieval", required=True, type=Path)
    parser.add_argument("--development-selection-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _groups(data: bytes) -> tuple[PackingCalibrationGroup, ...]:
    groups: list[PackingCalibrationGroup] = []
    for line in data.splitlines(keepends=True):
        value = json.loads(line)
        if content_json_bytes(value) != line:
            raise ValueError("candidate universe row is not canonical")
        groups.append(
            PackingCalibrationGroup(
                question_id=value["question_id"],
                split=value["split"],
                relevant_evidence_ids=tuple(value["relevant_evidence_ids"]),
                candidates=tuple(
                    PackingCalibrationCandidate(**candidate) for candidate in value["candidates"]
                ),
            )
        )
    return tuple(groups)


def main() -> int:
    arguments = _arguments()
    universe_data = arguments.candidate_universe.read_bytes()
    calibration_data = arguments.calibration.read_bytes()
    stored_calibration = json.loads(calibration_data)
    replayed_calibration = calibrate_relative_sparse_score(_groups(universe_data))
    for key, value in asdict(replayed_calibration).items():
        if stored_calibration.get(key) != value:
            raise ValueError(f"calibration replay differs at {key}")
    if stored_calibration.get("candidate_universe_checksum") != checksum_bytes(universe_data):
        raise ValueError("candidate universe checksum does not match calibration")

    calibration_checksum = checksum_bytes(calibration_data)
    counter = Qwen3InputTokenCounter.from_checkpoint(
        arguments.tokenizer_checkpoint,
        system_prompt=PROMPT_A,
    )
    replayed_selection = build_evidence_selection_artifacts(
        annotation_queue_data=arguments.development_queue.read_bytes(),
        retrieval_output_data=arguments.r0_retrieval.read_bytes(),
        source_run_id="R0",
        selected_run_id="R006B-P3A-R0-train-calibrated-v1",
        maximum_input_tokens=2048,
        token_counter=counter,
        minimum_relative_sparse_score=replayed_calibration.minimum_relative_sparse_score,
        calibration_checksum=calibration_checksum,
    )
    stored_retrieval = arguments.development_retrieval.read_bytes()
    stored_report = arguments.development_selection_report.read_bytes()
    if (
        replayed_selection.retrieval_output != stored_retrieval
        or replayed_selection.selection_report != stored_report
    ):
        raise ValueError("development selection replay is not byte-identical")

    verification = content_json_bytes(
        {
            "schema_version": "r006b.packing-replay-verification.v1",
            "candidate_universe_checksum": checksum_bytes(universe_data),
            "calibration_checksum": calibration_checksum,
            "development_retrieval_checksum": checksum_bytes(stored_retrieval),
            "development_selection_report_checksum": checksum_bytes(stored_report),
            "calibration_result_identical": True,
            "development_retrieval_byte_identical": True,
            "development_selection_report_byte_identical": True,
        }
    )
    verification_checksum = write_immutable_bytes(arguments.output, verification)
    print(f"R006B REPLAY VERIFIED {verification_checksum}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
