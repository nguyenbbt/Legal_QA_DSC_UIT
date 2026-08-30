"""Freeze the completed D-066 dense-discovery evidence and stopping report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_file, content_json_bytes

_ROOT = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1")
_REPORT = Path("artifacts/reports/post-d062/D066-completion-report.v1.md")
_INDEX = Path("artifacts/indices/dense/d066-r-disc-1-116f6cf195224b12")

_EXPECTED_CHECKSUMS = {
    "R-DISC-0-BM25.evaluation.v1.json": (
        "sha256:2795b02443813a9fc0d4ffce92d921bac029d5292a90d63262289f5e106a9358"
    ),
    "R-DISC-0-BM25.rankings.v1.jsonl": (
        "sha256:a3c10cd01274efc1cf93b81efd23a9c8756b67acef4119dd165966aa7a866463"
    ),
    "R-DISC-1.index-build.v1.json": (
        "sha256:1da5e85d49b37036209579cf07bfd48f56cdf7f31c3dff85cf81051836244167"
    ),
    "R-DISC-1.evaluation.v1.json": (
        "sha256:2ee122bdb11bb4d36893b5c6650fe40c3f453e9aaf845c6552228fcca644c452"
    ),
    "R-DISC-1.rankings.v1.jsonl": (
        "sha256:001ce3f281bd1db774ff6ab3db551fa3d4a058f2c5c18a5f3b1b8da9acc69d20"
    ),
    "R-DISC-1.sparse-dense-diagnostics.v1.json": (
        "sha256:94e60e3e2bf8e2180790ff8730e7bae88a954fd26f7651ebe613d140c4781234"
    ),
    "R-DISC-4A.union-evaluation.v1.json": (
        "sha256:05947ed98cbeaaf07faa15daf45aa46bac2c75afdd93a9f38cd7c8b702c0bc49"
    ),
    "R-DISC-4B-RRF60.evaluation.v1.json": (
        "sha256:703a22ffc1fe11c0ccc4fd7b8520975cc79c79ea3c6054b47ba1bde4bf54fd06"
    ),
    "R-DISC-4B-RRF60.rankings.v1.jsonl": (
        "sha256:ce5de92658c94b00a262f5ed6a1dc637ed296cb930ec6464e28307e0907a182a"
    ),
    "R-DISC-1-R-DISC-4.telemetry.v1.json": (
        "sha256:2ebf2abbfeb4e78bb00515ca222e332e464710e2e39a165fafcc9c85cc0925ba"
    ),
    "R-DISC-1-R-DISC-4.manifest.v1.json": (
        "sha256:e216e9bd258e364c47fa076b8294723692ae48c2063bd81f1fec359f5f906a5b"
    ),
}
_INDEX_CHECKSUMS = {
    "manifest.json": "sha256:84cbd768eabb021a2eb472dcede730e25c045af37e145ea8a75b5149410b05e0",
    "vectors.f16.npy": "sha256:f15b803dbae9dc0e2aa9cdf157861017e714c31a6dea65762df1393728dada04",
    "chunk-ids.jsonl": "sha256:218973fe0b4c0589eddfe9e3a9bf2d379a2843e4d327d158ed58e03dc31a8c39",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def _verify_evidence() -> None:
    for name, expected in _EXPECTED_CHECKSUMS.items():
        if checksum_file(_ROOT / name) != expected:
            raise SystemExit(f"D-066 evidence checksum drift: {name}")
    for name, expected in _INDEX_CHECKSUMS.items():
        if checksum_file(_INDEX / name) != expected:
            raise SystemExit(f"D-066 dense-index checksum drift: {name}")


def _metric_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall_at": value["recall_at"],
        "evidence_set_recall_at": value["evidence_set_recall_at"],
        "mrr_at_50": value["mrr_at_50"],
        "answer_bearing_coverage_at": value["answer_bearing_coverage_at"],
    }


def main() -> int:
    _verify_evidence()
    sparse = _load_json(_ROOT / "R-DISC-0-BM25.evaluation.v1.json")
    dense = _load_json(_ROOT / "R-DISC-1.evaluation.v1.json")
    union = _load_json(_ROOT / "R-DISC-4A.union-evaluation.v1.json")
    rrf = _load_json(_ROOT / "R-DISC-4B-RRF60.evaluation.v1.json")
    diagnostics = _load_json(_ROOT / "R-DISC-1.sparse-dense-diagnostics.v1.json")
    index = _load_json(_ROOT / "R-DISC-1.index-build.v1.json")
    telemetry = _load_json(_ROOT / "R-DISC-1-R-DISC-4.telemetry.v1.json")
    manifest = _load_json(_ROOT / "R-DISC-1-R-DISC-4.manifest.v1.json")

    expected_counts = {"BOTH": 1356, "DENSE_ONLY": 409, "NEITHER": 517, "SPARSE_ONLY": 109}
    if diagnostics["classification_counts"] != expected_counts:
        raise SystemExit("D-066 paired classification drift")
    if index["completed_count"] != 641_118 or index["store_audit"]["missing_chunk_count"]:
        raise SystemExit("D-066 dense-index completeness drift")
    if manifest["recommended_d066_standing_winner"] != "R-DISC-4A-SPARSE-DENSE-UNION":
        raise SystemExit("D-066 standing-winner drift")

    stage_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-stage-state.v2",
            "status": "COMPLETE_STOP_BEFORE_D067",
            "positive_group_count": 2_391,
            "canonical_chunk_count": 641_118,
            "r_disc_0_standing_winner": "EXACT_PLUS_BM25",
            "r_disc_0": {
                "status": "COMPLETE_FROZEN",
                "metrics": _metric_summary(sparse),
                "manifest_checksum": (
                    "sha256:3a029a974a578ba55ce7c3c983f45c892dd262f3fadac42c28b353d63c396170"
                ),
            },
            "r_disc_1": {
                "status": "COMPLETE",
                "model_id": "Qwen/Qwen3-Embedding-0.6B",
                "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                "metrics": _metric_summary(dense),
                "dense_only_groups": 409,
                "sparse_only_groups": 109,
                "both_groups": 1_356,
                "neither_groups": 517,
                "index_build_checksum": _EXPECTED_CHECKSUMS["R-DISC-1.index-build.v1.json"],
                "index_manifest_checksum": _INDEX_CHECKSUMS["manifest.json"],
                "index_vector_checksum": _INDEX_CHECKSUMS["vectors.f16.npy"],
                "index_ids_checksum": _INDEX_CHECKSUMS["chunk-ids.jsonl"],
            },
            "r_disc_4a": {
                "status": "COMPLETE",
                "candidate": "SPARSE_DENSE_TOP50_SET_UNION",
                "metrics": union,
            },
            "r_disc_4b": {
                "status": "COMPLETE_RETAINED_NOT_STANDING_WINNER",
                "candidate": "UNWEIGHTED_RRF_60",
                "metrics": _metric_summary(rrf),
                "novel_positive_recovery_at_50": 337,
                "lost_positive_recovery_at_50": 30,
            },
            "recommended_d066_standing_winner": "R-DISC-4A-SPARSE-DENSE-UNION",
            "post_d062_baseline_changed": False,
            "d067": {
                "status": "CLOSED",
                "learned_fusion_should_open": True,
                "reason": "UNION_COVERAGE_GAIN_NOT_PRESERVED_BY_FIXED_RRF60",
            },
            "dense_replay": {
                "top_k_ids_and_order_identical": True,
                "maximum_score_delta": telemetry["replay_maximum_score_delta"],
                "score_tolerance": telemetry["replay_score_tolerance"],
            },
            "whole_system_parameter_count": 3_223_292_928,
            "whole_system_strictly_below_4b": True,
            "fit_performed": False,
            "development_or_public_data_used": False,
            "modal_used": False,
            "cost_usd": 0,
        }
    )
    stage_checksum = write_immutable_bytes(_ROOT / "D066.stage-state.v2.json", stage_data)

    report_lines = [
        "# D-066 Qwen3 Dense Discovery and Fixed Fusion Completion Report",
        "",
        "D-066 is complete and stops before D-067. The frozen sparse winner remains",
        "Exact+BM25, while the standing discovery candidate is the deterministic",
        "Top-50 sparse+dense set union. No model was fitted and no development, public,",
        "Modal, paid-service, or leaderboard feedback was used.",
        "",
        "## Frozen inputs and index audit",
        "",
        "- Retrieval supervision: 2,391 positive official-train groups.",
        "- Canonical corpus: exactly 641,118 chunks.",
        "- Dense model: `Qwen/Qwen3-Embedding-0.6B` at revision",
        "  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, dimension 1,024, float16.",
        "- Full store audit: 641,118 rows; zero missing/duplicate IDs; zero NaN/Inf,",
        "  zero, or non-unit vectors; deterministic chunk-ID-to-row mapping.",
        "- Historical 12,147-row partial: rejected and not read or resumed.",
        f"- Vector checksum: `{_INDEX_CHECKSUMS['vectors.f16.npy']}`.",
        f"- Chunk-ID checksum: `{_INDEX_CHECKSUMS['chunk-ids.jsonl']}`.",
        f"- Index manifest checksum: `{_INDEX_CHECKSUMS['manifest.json']}`.",
        "",
        "## Discovery metrics",
        "",
        (
            "| Candidate | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Set R@5 | "
            "Set R@10 | Set R@20 | Set R@50 | MRR@50 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| Exact+BM25 | 0.348808 | 0.429527 | 0.507319 | 0.612714 | 0.131744 | "
            "0.157256 | 0.193225 | 0.247177 | 0.253798 |"
        ),
        (
            "| Qwen3 dense | 0.444584 | 0.537850 | 0.630280 | 0.738185 | 0.153910 | "
            "0.199916 | 0.246759 | 0.317022 | 0.327388 |"
        ),
        (
            "| Top-50 set union | 0.520703 | 0.611460 | 0.694688 | 0.783772 | "
            "0.191552 | 0.235885 | 0.285236 | 0.350481 | n/a |"
        ),
        (
            "| Fixed RRF-60 | 0.446257 | 0.543706 | 0.635299 | 0.741113 | 0.163948 | "
            "0.208699 | 0.252614 | 0.314095 | 0.334301 |"
        ),
        "",
        "Sparse/dense Top-50 paired classes are `BOTH=1,356`, `DENSE_ONLY=409`,",
        "`SPARSE_ONLY=109`, and `NEITHER=517`. Dense therefore recovers 409 novel",
        "positive groups but does not subsume sparse. At Top-50, the union recovers",
        "3,615 positive chunk assignments and 838 complete multi-positive evidence",
        "sets. Its mean raw pool size is 86.234 candidates.",
        "",
        "Fixed RRF-60 improves over either standalone ranking but loses 30 sparse",
        "positive groups and does not preserve the union ceiling. It is retained as",
        "negative/diagnostic evidence, not chosen as the standing D-066 candidate.",
        "",
        "## Replay and resources",
        "",
        (
            f"- Full index: {index['wall_seconds']:.3f} seconds at "
            f"{index['documents_per_second']:.3f} chunks/s."
        ),
        (
            f"- Query encoding: {telemetry['query_encode_wall_seconds']:.3f} seconds; "
            f"peak VRAM {telemetry['query_encode_peak_vram_bytes']:,} bytes."
        ),
        (
            f"- Dense search: {telemetry['dense_search_wall_seconds']:.3f} seconds; "
            f"peak VRAM {telemetry['dense_search_peak_vram_bytes']:,} bytes."
        ),
        f"- Recorded process RSS: {telemetry['process_rss_bytes']:,} bytes.",
        "- Index payload: 1,350,725,525 bytes (vectors + IDs + manifest).",
        "- Exact preflight peak VRAM: 2,579,660,800 bytes. The full-index runner did",
        "  not separately publish a peak-VRAM sample, so no larger value is claimed.",
        "- Replay: identical Top-K IDs/order; maximum score delta",
        (
            f"  `{telemetry['replay_maximum_score_delta']}` within tolerance "
            f"`{telemetry['replay_score_tolerance']}`."
        ),
        "- Cost USD 0; local GPU only; no Modal or paid service.",
        "- Exact whole-system count: 3,223,292,928 parameters, strictly below 4B.",
        "",
        "## Decision and remaining boundary",
        "",
        "`R-DISC-4A-SPARSE-DENSE-UNION` is the D-066 standing discovery candidate.",
        "This is candidate coverage evidence, not an end-to-end promotion: METEOR,",
        "ROUGE-L, and grounding were intentionally not rerun. The frozen post-D-062",
        "baseline therefore remains unchanged.",
        "",
        "D-067 learned fusion should be opened under a separate bounded authorization",
        "because dense adds 409 novel groups, sparse retains 109 unique groups, and",
        "fixed RRF-60 does not preserve the full union. D-067 remains CLOSED here.",
        "GTE/BGE, fitting, generator changes, and development/public inference remain",
        "outside D-066.",
        "",
        "## Verification",
        "",
        "- Ruff format/lint: PASS.",
        "- mypy: PASS over 116 source/script files.",
        "- CPU suite: 681 PASS, 2 Windows symlink SKIP, 5 DESELECTED.",
        (
            "- Dense/fusion manifest checksum: "
            f"`{_EXPECTED_CHECKSUMS['R-DISC-1-R-DISC-4.manifest.v1.json']}`."
        ),
        f"- Stage-state checksum: `{stage_checksum}`.",
        "",
        "## Exact reproduction commands",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe scripts\run_d066_dense_index.py",
        r".\.venv\Scripts\python.exe scripts\run_d066_dense_discovery.py",
        r".\.venv\Scripts\python.exe scripts\finalize_d066_dense.py",
        r".\.venv\Scripts\ruff.exe format --check .",
        r".\.venv\Scripts\ruff.exe check .",
        (
            r".\.venv\Scripts\mypy.exe src scripts/run_d066_dense_index.py "
            "scripts/run_d066_dense_discovery.py"
        ),
        (
            r'.\.venv\Scripts\python.exe -m pytest -q -m "not integration and not gpu" '
            "--basetemp .local\\pytest-d066-rdisc1-final"
        ),
        "```",
        "",
    ]
    report_checksum = write_immutable_bytes(_REPORT, "\n".join(report_lines).encode("utf-8"))
    checksums_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-checksums.v3",
            "stage_state_checksum": stage_checksum,
            "completion_report_checksum": report_checksum,
            "dense_fusion_manifest_checksum": _EXPECTED_CHECKSUMS[
                "R-DISC-1-R-DISC-4.manifest.v1.json"
            ],
            "index_build_checksum": _EXPECTED_CHECKSUMS["R-DISC-1.index-build.v1.json"],
            "index_manifest_checksum": _INDEX_CHECKSUMS["manifest.json"],
            "index_vector_checksum": _INDEX_CHECKSUMS["vectors.f16.npy"],
            "index_ids_checksum": _INDEX_CHECKSUMS["chunk-ids.jsonl"],
        }
    )
    checksums_checksum = write_immutable_bytes(_ROOT / "D066.checksums.v3.json", checksums_data)
    print(f"stage={stage_checksum}")
    print(f"report={report_checksum}")
    print(f"checksums={checksums_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
