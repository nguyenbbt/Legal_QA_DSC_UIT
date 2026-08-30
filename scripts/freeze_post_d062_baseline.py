"""Freeze D-063 from completed D-062 files without running inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.post_d062 import reconcile_d063

_D062_ROOT = Path("artifacts/evaluations/recovery/R-009/D062-full-development-v1")


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _markdown(report: dict[str, Any], json_checksum: str) -> bytes:
    metrics = report["metrics"]
    grounding = report["grounding"]
    meteor = report["meteor_outcomes"]
    rouge = report["rouge_l_outcomes"]
    baseline_scores = f"{metrics['baseline_meteor']:.10f} / {metrics['baseline_rouge_l']:.10f}"
    candidate_scores = f"{metrics['candidate_meteor']:.10f} / {metrics['candidate_rouge_l']:.10f}"
    meteor_outcomes = f"{meteor['candidate_wins']}/{meteor['ties']}/{meteor['candidate_losses']}"
    rouge_outcomes = f"{rouge['candidate_wins']}/{rouge['ties']}/{rouge['candidate_losses']}"
    lines = [
        "# D-063 Post-D-062 Baseline Freeze",
        "",
        f"- `D063_STATUS = {report['d063_status']}`",
        "- `POST_D062_BASELINE = current base-Qwen3-reranker + Qwen3-1.7B G1A512`",
        f"- Question count: {report['question_count']}",
        f"- New inference runs: {report['new_inference_runs']}",
        f"- Whole-system parameters: {report['system_parameter_count']:,} (< 4,000,000,000)",
        f"- JSON checksum: `{json_checksum}`",
        "",
        "## Frozen D-062 measurements",
        "",
        f"- Baseline METEOR / ROUGE-L: {baseline_scores}",
        f"- Candidate METEOR / ROUGE-L: {candidate_scores}",
        f"- METEOR delta / CI95: {metrics['meteor_mean_delta']:+.10f} / {metrics['meteor_ci95']}",
        f"- ROUGE-L delta: {metrics['rouge_l_mean_delta']:+.10f}",
        f"- METEOR wins/ties/losses: {meteor_outcomes}",
        f"- ROUGE-L wins/ties/losses: {rouge_outcomes}",
        f"- Fully supported: {grounding['candidate_fully_supported_rate']:.6f}",
        f"- Unsupported: {grounding['candidate_unsupported_answer_rate']:.6f}",
        "- Numeric, grounding, resource, and byte-replay gates: PASS",
        "",
        "No D-062 file was modified and no retrieval or generation inference was run.",
        "Exact final model registration and OQ-001 packaging remain separate blockers.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/post-d062"))
    args = parser.parse_args()

    report = reconcile_d063(
        comparison_data=_read(_D062_ROOT / "comparison/comparison.v1.json"),
        grounding_data=_read(
            Path(
                "artifacts/evaluations/grounding/assessments/"
                "D062-base-reranker-G1A512-development-716-v1.vs-baseline."
                "grounding-comparison.v1.json"
            )
        ),
        baseline_metrics_data=_read(_D062_ROOT / "r0/generation/evaluation-per-query.jsonl"),
        candidate_metrics_data=_read(
            _D062_ROOT / "base-reranker/generation/evaluation-per-query.jsonl"
        ),
        baseline_predictions_data=_read(_D062_ROOT / "r0/generation/predictions.json"),
        candidate_predictions_data=_read(_D062_ROOT / "base-reranker/generation/predictions.json"),
        parameter_manifest_data=_read(
            Path("artifacts/models/qwen3-btc-approved-parameter-manifest.v1.json")
        ),
        expected_question_count=716,
    )
    json_data = content_json_bytes(report)
    json_checksum = checksum_bytes(json_data)
    markdown_data = _markdown(report, json_checksum)
    json_path = args.output_dir / "D063-post-d062-baseline-freeze.v1.json"
    markdown_path = args.output_dir / "D063-post-d062-baseline-freeze.v1.md"
    written_json = write_immutable_bytes(json_path, json_data)
    written_markdown = write_immutable_bytes(markdown_path, markdown_data)
    evidence = content_json_bytes(
        {
            "schema_version": "post_d062.baseline.freeze.checksums.v1",
            "json_checksum": written_json,
            "markdown_checksum": written_markdown,
        }
    )
    evidence_checksum = write_immutable_bytes(
        args.output_dir / "D063-post-d062-baseline-freeze.checksums.v1.json", evidence
    )
    print(f"D063_STATUS={report['d063_status']}")
    print(f"json={written_json}")
    print(f"markdown={written_markdown}")
    print(f"checksums={evidence_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
