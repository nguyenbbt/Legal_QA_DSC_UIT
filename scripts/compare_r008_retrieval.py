from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.retrieval_comparison import compare_retrieval_experiments


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-universe R-008 retrieval")
    parser.add_argument("--grounding-benchmark", required=True, type=Path)
    parser.add_argument("--baseline-output", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    benchmark_data = arguments.grounding_benchmark.read_bytes()
    baseline_data = arguments.baseline_output.read_bytes()
    candidate_data = arguments.candidate_output.read_bytes()
    baseline_report = json.loads(arguments.baseline_report.read_bytes())
    candidate_report = json.loads(arguments.candidate_report.read_bytes())
    core_data = compare_retrieval_experiments(
        grounding_benchmark_data=benchmark_data,
        baseline_output_data=baseline_data,
        candidate_output_data=candidate_data,
        baseline_run_id=baseline_report["run_id"],
        candidate_run_id=candidate_report["run_id"],
    )
    core = json.loads(core_data)
    runtime_ratio = candidate_report["elapsed_seconds"] / baseline_report["elapsed_seconds"]
    blockers = list(core["promotion_blockers"])
    if runtime_ratio > 1.25:
        blockers.append("RUNTIME_REGRESSION_EXCEEDS_25_PERCENT")
    comparison_data = content_json_bytes(
        {
            **core,
            "baseline_output_checksum": checksum_bytes(baseline_data),
            "candidate_output_checksum": checksum_bytes(candidate_data),
            "grounding_benchmark_checksum": checksum_bytes(benchmark_data),
            "runtime": {
                "baseline_seconds": baseline_report["elapsed_seconds"],
                "candidate_seconds": candidate_report["elapsed_seconds"],
                "candidate_to_baseline_ratio": runtime_ratio,
            },
            "promotion_state": "rejected_preserved" if blockers else "retrieval_gate_passed",
            "promotion_blockers": blockers,
        }
    )
    checksum = write_immutable_bytes(arguments.output, comparison_data)
    print(f"R008 RETRIEVAL COMPARISON {checksum}", flush=True)
    print(comparison_data.decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
