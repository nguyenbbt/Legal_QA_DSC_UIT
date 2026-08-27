"""Finalize and compare one approved answer-grounding assessment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.grounding.answer_assessment import (
    compare_answer_grounding,
    export_answer_assessments,
    import_labeled_answer_assessment_queue,
    load_approved_answer_assessments,
)

_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\Z")


def finalize_answer_grounding_artifacts(
    *,
    artifact_stem: str,
    queue_data: bytes,
    labeled_data: bytes,
    benchmark_manifest_data: bytes,
    benchmark_data: bytes,
    baseline_manifest_data: bytes,
    baseline_assessment_data: bytes,
) -> dict[str, bytes]:
    """Bind labels, export the candidate, and apply the grounding guards."""

    if _SAFE_STEM.fullmatch(artifact_stem) is None:
        raise ValueError("artifact stem must contain only letters, digits, and hyphens")
    imported = import_labeled_answer_assessment_queue(queue_data, labeled_data)
    if imported.evaluated_run_id != artifact_stem:
        raise ValueError("artifact stem must equal the assessed run identity")
    assessment_name = f"{artifact_stem}.grounding.v1.jsonl"
    manifest_name = f"{artifact_stem}.grounding.manifest.v1.json"
    report_name = f"{artifact_stem}.grounding.report.v1.json"
    comparison_name = f"{artifact_stem}.vs-baseline.grounding-comparison.v1.json"
    exported = export_answer_assessments(
        imported,
        queue_data=queue_data,
        benchmark_manifest_data=benchmark_manifest_data,
        benchmark_data=benchmark_data,
        assessment_path=f"assessments/{assessment_name}",
    )
    baseline = load_approved_answer_assessments(
        manifest_data=baseline_manifest_data,
        assessment_data=baseline_assessment_data,
        benchmark_manifest_data=benchmark_manifest_data,
        benchmark_data=benchmark_data,
    )
    candidate = load_approved_answer_assessments(
        manifest_data=exported.manifest_data,
        assessment_data=exported.assessment_data,
        benchmark_manifest_data=benchmark_manifest_data,
        benchmark_data=benchmark_data,
    )
    return {
        assessment_name: exported.assessment_data,
        manifest_name: exported.manifest_data,
        report_name: exported.report_data,
        comparison_name: compare_answer_grounding(baseline, candidate),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--labeled", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--baseline-assessment", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    artifacts = finalize_answer_grounding_artifacts(
        artifact_stem=arguments.artifact_stem,
        queue_data=arguments.queue.read_bytes(),
        labeled_data=arguments.labeled.read_bytes(),
        benchmark_manifest_data=arguments.benchmark_manifest.read_bytes(),
        benchmark_data=arguments.benchmark.read_bytes(),
        baseline_manifest_data=arguments.baseline_manifest.read_bytes(),
        baseline_assessment_data=arguments.baseline_assessment.read_bytes(),
    )
    checksums = {
        name: write_immutable_bytes(arguments.output_directory / name, data)
        for name, data in artifacts.items()
    }
    report = json.loads(artifacts[f"{arguments.artifact_stem}.grounding.report.v1.json"])
    comparison = json.loads(
        artifacts[f"{arguments.artifact_stem}.vs-baseline.grounding-comparison.v1.json"]
    )
    print(
        json.dumps(
            {
                "run_id": arguments.artifact_stem,
                "rates": report["rates"],
                "grounding_gate": comparison["grounding_gate"],
                "promotion_blockers": comparison["promotion_blockers"],
                "checksums": checksums,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
