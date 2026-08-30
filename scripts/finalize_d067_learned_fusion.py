"""Freeze the measured D-067 learned-fusion outcome and stopping report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_file, content_json_bytes
from legal_rag.evaluation.learned_fusion import FEATURE_NAMES

_ROOT = Path("artifacts/training/learned-fusion/d067")
_REPORT = Path("artifacts/reports/post-d062/D067-completion-report.v1.md")
_RUN_MANIFEST = _ROOT / "D067.run-manifest.v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"D-067 expected a JSON object: {path.name}")
    return cast(dict[str, Any], value)


def _verify_checksum(path: Path, expected: object) -> str:
    actual = checksum_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise SystemExit(f"D-067 evidence checksum drift: {path.name}")
    return actual


def _metric(value: dict[str, Any], name: str, cutoff: int) -> float:
    raw = value.get(name)
    if not isinstance(raw, dict):
        raise SystemExit(f"D-067 metric is invalid: {name}")
    result = raw.get(str(cutoff))
    if not isinstance(result, int | float):
        raise SystemExit(f"D-067 cutoff metric is invalid: {name}@{cutoff}")
    return float(result)


def _delta(comparison: dict[str, Any], name: str) -> float:
    raw = comparison.get("metric_deltas")
    if not isinstance(raw, dict) or not isinstance(raw.get(name), int | float):
        raise SystemExit(f"D-067 metric delta is invalid: {name}")
    return float(raw[name])


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _top_features(model_manifest: dict[str, Any]) -> tuple[tuple[str, float, int], ...]:
    audit = model_manifest.get("model_audit")
    if not isinstance(audit, dict):
        raise SystemExit("D-067 model audit is absent")
    names = audit.get("feature_names")
    gains = audit.get("feature_gain_importance")
    splits = audit.get("feature_split_counts")
    if names != list(FEATURE_NAMES) or not isinstance(gains, list) or not isinstance(splits, list):
        raise SystemExit("D-067 feature audit is invalid")
    if len(gains) != len(FEATURE_NAMES) or len(splits) != len(FEATURE_NAMES):
        raise SystemExit("D-067 feature-importance shape drift")
    rows = tuple(
        (name, float(gain), int(split))
        for name, gain, split in zip(names, gains, splits, strict=True)
    )
    return tuple(sorted(rows, key=lambda item: (-item[1], item[0].encode())))


def main() -> int:
    run_manifest = _load_json(_RUN_MANIFEST)
    if (
        run_manifest.get("schema_version") != "evaluation.d067-run-manifest.v1"
        or run_manifest.get("status") != "COMPLETE_STOP_BEFORE_D068"
        or run_manifest.get("fit_data") != "official-train-only"
        or run_manifest.get("development_or_public_data_used") is not False
        or run_manifest.get("gpu_used") is not False
        or run_manifest.get("modal_used") is not False
        or run_manifest.get("d068_status") != "CLOSED"
        or run_manifest.get("post_d062_baseline_changed") is not False
    ):
        raise SystemExit("D-067 run manifest is invalid")

    evidence = {
        "features": _verify_checksum(
            _ROOT / "D067.features.v1.jsonl", run_manifest["feature_checksum"]
        ),
        "feature_manifest": _verify_checksum(
            _ROOT / "D067.features.manifest.v1.json", run_manifest["feature_manifest_checksum"]
        ),
        "split": _verify_checksum(
            _ROOT / "D067.group-split.v1.json", run_manifest["split_checksum"]
        ),
        "model": _verify_checksum(
            _ROOT / "D067.lambda-mart.v1.txt", run_manifest["model_checksum"]
        ),
        "model_manifest": _verify_checksum(
            _ROOT / "D067.lambda-mart.model-manifest.v1.json",
            run_manifest["model_manifest_checksum"],
        ),
        "rankings": _verify_checksum(
            _ROOT / "D067.lambda-mart.validation.rankings.v1.jsonl",
            run_manifest["ranking_checksum"],
        ),
        "rrf_evaluation": _verify_checksum(
            _ROOT / "D067.rrf60.validation.evaluation.v1.json",
            run_manifest["rrf_evaluation_checksum"],
        ),
        "learned_evaluation": _verify_checksum(
            _ROOT / "D067.lambda-mart.validation.evaluation.v1.json",
            run_manifest["learned_evaluation_checksum"],
        ),
        "comparison": _verify_checksum(
            _ROOT / "D067.validation.comparison.v1.json", run_manifest["comparison_checksum"]
        ),
        "telemetry": _verify_checksum(
            _ROOT / "D067.telemetry.v1.json", run_manifest["telemetry_checksum"]
        ),
        "run_manifest": checksum_file(_RUN_MANIFEST),
    }
    feature_manifest = _load_json(_ROOT / "D067.features.manifest.v1.json")
    model_manifest = _load_json(_ROOT / "D067.lambda-mart.model-manifest.v1.json")
    telemetry = _load_json(_ROOT / "D067.telemetry.v1.json")
    rrf = _load_json(_ROOT / "D067.rrf60.validation.evaluation.v1.json")
    learned = _load_json(_ROOT / "D067.lambda-mart.validation.evaluation.v1.json")
    comparison = _load_json(_ROOT / "D067.validation.comparison.v1.json")

    fit_groups = int(feature_manifest["fit_group_count"])
    validation_groups = int(feature_manifest["validation_group_count"])
    if (
        fit_groups + validation_groups != 2_391
        or int(feature_manifest["question_group_count"]) != 2_391
        or rrf.get("question_count") != validation_groups
        or learned.get("question_count") != validation_groups
        or model_manifest.get("whole_system_strictly_below_4b") is not True
        or telemetry.get("prediction_replay_byte_identical") is not True
        or telemetry.get("ranking_replay_byte_identical") is not True
    ):
        raise SystemExit("D-067 completeness, resource, or replay gate drift")
    gate_passed = comparison.get("passes_retrieval_gate") is True
    expected_winner = "D067-LAMBDAMART-PROVISIONAL" if gate_passed else "R-DISC-4B-FIXED-RRF-60"
    if run_manifest.get("provisional_fusion_winner") != expected_winner:
        raise SystemExit("D-067 recorded winner is inconsistent with the frozen gate")

    stage_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-stage-state.v1",
            "status": "COMPLETE_STOP_BEFORE_D068",
            "official_train_fit_count": 5_582,
            "positive_group_count": 2_391,
            "fit_group_count": fit_groups,
            "validation_group_count": validation_groups,
            "candidate_row_count": feature_manifest["candidate_row_count"],
            "positive_candidate_row_count": feature_manifest["positive_candidate_row_count"],
            "all_negative_candidate_group_count": feature_manifest[
                "all_negative_candidate_group_count"
            ],
            "fixed_baseline": "R-DISC-4B-FIXED-RRF-60",
            "learned_candidate": "D067-LAMBDAMART",
            "retrieval_gate_passed": gate_passed,
            "provisional_fusion_winner": expected_winner,
            "downstream_gate_status": "PENDING_LATER_FIXED_PIPELINE_EVALUATION",
            "post_d062_baseline_changed": False,
            "d068_status": "CLOSED",
            "fit_data": "official-train-only",
            "development_or_public_data_used": False,
            "gpu_used": False,
            "modal_used": False,
            "cost_usd": 0,
            "evidence_checksums": evidence,
        }
    )
    stage_checksum = write_immutable_bytes(_ROOT / "D067.stage-state.v1.json", stage_data)

    table_rows = []
    for label, value in (("Fixed RRF-60", rrf), ("LambdaMART", learned)):
        cells = [label]
        cells.extend(_format_metric(_metric(value, "recall_at", k)) for k in (5, 10, 20, 50))
        cells.extend(
            _format_metric(_metric(value, "evidence_set_recall_at", k)) for k in (5, 10, 20, 50)
        )
        cells.append(_format_metric(float(value["mrr_at_50"])))
        table_rows.append("| " + " | ".join(cells) + " |")
    top_features = _top_features(model_manifest)[:8]
    feature_lines = [
        f"- `{name}`: gain `{gain:.6f}`, splits `{split_count}`."
        for name, gain, split_count in top_features
    ]
    novel = comparison.get("novel_positive_recovery_at_50")
    lost = comparison.get("lost_positive_recovery_at_50")
    if not isinstance(novel, list) or not isinstance(lost, list):
        raise SystemExit("D-067 recovery diagnostic is invalid")

    report_lines = [
        "# D-067 Train-only Learned Fusion Completion Report",
        "",
        "D-067 is complete and stops before D-068. Exactly one fixed CPU LightGBM",
        "LambdaMART candidate was fitted using only official-train retrieval-supervision.v2",
        "groups and answer-independent features. Development/public data, GPU, Modal, paid",
        "services, leaderboard feedback, early stopping, and parameter sweeps were not used.",
        "",
        "## Frozen data and split",
        "",
        "- Authoritative official train-fit rows: 5,582.",
        "- Positive D-065 retrieval groups: 2,391.",
        f"- Deterministic group-disjoint fit/validation groups: {fit_groups}/{validation_groups}.",
        f"- Candidate feature rows: {int(feature_manifest['candidate_row_count']):,}.",
        f"- Positive candidate rows: {int(feature_manifest['positive_candidate_row_count']):,}.",
        (
            "- Candidate pools with no frozen positive: "
            f"{int(feature_manifest['all_negative_candidate_group_count']):,}; retained without "
            "fabricating positives."
        ),
        "- Candidate universe: deterministic set union of frozen sparse Top-50 and dense Top-50.",
        "- Labels: D-065 canonical positive-chunk membership only. Answer-derived legal",
        "  coordinates, mapping confidence, and mapping class were not features.",
        "",
        "## Held-out retrieval result",
        "",
        (
            "| Candidate | R@5 | R@10 | R@20 | R@50 | Set R@5 | Set R@10 | "
            "Set R@20 | Set R@50 | MRR@50 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        f"- Recall@10 delta: `{_delta(comparison, 'recall_at_10'):+.6f}`.",
        f"- Recall@50 delta: `{_delta(comparison, 'recall_at_50'):+.6f}`.",
        (
            "- Evidence-set Recall@50 delta: "
            f"`{_delta(comparison, 'evidence_set_recall_at_50'):+.6f}`."
        ),
        f"- MRR@50 delta: `{_delta(comparison, 'mrr_at_50'):+.6f}`.",
        f"- Novel/lost positive groups at Top-50: `{len(novel)}/{len(lost)}`.",
        f"- Frozen retrieval gate: `{'PASS' if gate_passed else 'FAIL'}`.",
        f"- Provisional fusion winner: `{expected_winner}`.",
        "",
        "The result is retrieval-only. It does not promote or replace the frozen",
        "`POST_D062_BASELINE`; METEOR, ROUGE-L, and grounding were intentionally not run.",
        "The downstream gate remains pending a separately governed fixed-pipeline stage.",
        "",
        "## Model and feature audit",
        "",
        "- LightGBM 4.7.0 (MIT), `lambdarank`, exactly 200 trees; scikit-learn 1.9.0",
        "  (BSD-3-Clause).",
        (
            "- Tree/split/leaf/learned-value counts: "
            f"{model_manifest['model_audit']['tree_count']}/"
            f"{model_manifest['model_audit']['split_count']}/"
            f"{model_manifest['model_audit']['leaf_count']}/"
            f"{model_manifest['model_audit']['learned_value_count']}."
        ),
        "- Highest gain features (diagnostic only; no post-hoc tuning):",
        *feature_lines,
        "- Neural parameters added: 0.",
        (
            "- Conservative whole-system learned-value count: "
            f"{int(model_manifest['conservative_whole_system_learned_parameter_count']):,}, "
            "strictly below 4B."
        ),
        "",
        "## Replay, resources, and checksums",
        "",
        (
            "- Feature construction: "
            f"{float(feature_manifest['construction_wall_seconds']):.3f} seconds, "
            "4 CPU workers."
        ),
        f"- LambdaMART fit: {float(telemetry['fit_wall_seconds']):.3f} seconds, 4 CPU threads.",
        f"- Recorded process RSS after fit: {int(telemetry['rss_after_fit_bytes']):,} bytes.",
        "- Prediction and ranking replay: byte-identical PASS.",
        "- GPU/Modal/paid service: no/no/no; cost USD 0.",
        f"- Feature checksum: `{evidence['features']}`.",
        f"- Group-split checksum: `{evidence['split']}`.",
        f"- Model checksum: `{evidence['model']}`.",
        f"- Validation ranking checksum: `{evidence['rankings']}`.",
        f"- Run-manifest checksum: `{evidence['run_manifest']}`.",
        f"- Stage-state checksum: `{stage_checksum}`.",
        "",
        "## Boundary and exact reproduction",
        "",
        "D-068 remains CLOSED. No reranker tournament, model zoo, fine-tuning,",
        "development/public generation, or public submission is authorized by D-067.",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe scripts\build_d067_features.py",
        r".\.venv\Scripts\python.exe scripts\run_d067_learned_fusion.py",
        r".\.venv\Scripts\python.exe scripts\finalize_d067_learned_fusion.py",
        r".\.venv\Scripts\ruff.exe format --check .",
        r".\.venv\Scripts\ruff.exe check .",
        (
            r".\.venv\Scripts\mypy.exe src scripts/build_d067_features.py "
            "scripts/run_d067_learned_fusion.py scripts/finalize_d067_learned_fusion.py"
        ),
        (
            r".\.venv\Scripts\python.exe -m pytest -q "
            r'-m "not integration and not gpu" --basetemp .local\pytest-d067-final'
        ),
        "```",
        "",
    ]
    report_checksum = write_immutable_bytes(_REPORT, "\n".join(report_lines).encode("utf-8"))
    checksums_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-checksums.v1",
            "evidence_checksums": evidence,
            "stage_state_checksum": stage_checksum,
            "completion_report_checksum": report_checksum,
        }
    )
    checksums_checksum = write_immutable_bytes(_ROOT / "D067.checksums.v1.json", checksums_data)
    print(f"stage={stage_checksum}")
    print(f"report={report_checksum}")
    print(f"checksums={checksums_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
