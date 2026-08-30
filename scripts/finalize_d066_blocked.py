"""Freeze the measured D-066 resource-blocked stopping report."""

from __future__ import annotations

from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_file, content_json_bytes

_ROOT = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1")
_SPARSE = _ROOT / "preflight/R-DISC-0.sparse-preflight-timeout.v1.json"
_DENSE = _ROOT / "preflight/R-DISC-1.dense-preflight.v1.json"
_EXPECTED_SPARSE = "sha256:4223ca4927dab310c7728b8be7c1dc03a0238cf3a142dbee87913a685ad22fce"
_EXPECTED_DENSE = "sha256:70fd25ab9f486057a1b33cb083aa65033dc39f56a1577b28adc46be972f2838b"


def main() -> int:
    sparse_checksum = checksum_file(_SPARSE)
    dense_checksum = checksum_file(_DENSE)
    if sparse_checksum != _EXPECTED_SPARSE or dense_checksum != _EXPECTED_DENSE:
        raise SystemExit("D-066 preflight evidence checksum drift")

    stage_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-stage-state.v1",
            "status": "BLOCKED_RESOURCE_GATE",
            "authoritative_positive_group_count": 2_391,
            "excluded_ambiguous_count": 108,
            "excluded_document_only_count": 161,
            "excluded_unresolved_count": 2_922,
            "r_disc_0": {
                "status": "BLOCKED",
                "blocker_code": "OPS002_PREFLIGHT_TIMEOUT",
                "sample_question_count": 100,
                "sample_timeout_seconds": 904,
                "sample_complete": False,
                "rankings_published": False,
                "quality_metrics_computed": False,
                "legal_sparse_started": False,
                "evidence_checksum": sparse_checksum,
            },
            "r_disc_1": {
                "preflight_status": "PASS",
                "full_index_status": "NOT_OPENED_SEQUENTIAL_PARENT_BLOCKED",
                "documents_per_second_milli": 37_144,
                "projected_runtime_seconds": 17_261,
                "peak_vram_bytes": 2_579_660_800,
                "projected_vector_bytes": 1_313_009_664,
                "exact_model_parameter_count": 595_776_512,
                "whole_system_parameter_count": 3_223_292_928,
                "historical_partial_disposition": "REJECT_STALE_PARTIAL",
                "evidence_checksum": dense_checksum,
            },
            "r_disc_2": {
                "status": "BLOCKED",
                "blocker_code": "MODEL_PARAMETER_AUDIT_MISSING",
            },
            "r_disc_3": {
                "status": "BLOCKED",
                "blocker_code": "MODEL_PARAMETER_AUDIT_MISSING",
            },
            "r_disc_4": {
                "status": "BLOCKED",
                "blocker_code": "DENSE_RANKINGS_MISSING",
            },
            "candidate_winner": None,
            "fallback_changed": False,
            "d067_opened": False,
            "fit_performed": False,
            "development_or_public_data_used": False,
            "modal_used": False,
            "cost_usd": 0,
        }
    )
    stage_checksum = write_immutable_bytes(_ROOT / "D066.stage-state.v1.json", stage_data)

    report_lines = [
        "# D-066 Candidate Discovery Stopping Report",
        "",
        "This v2 report supersedes v1 only to correct PowerShell reproduction quoting;",
        "all stage evidence and decisions are unchanged.",
        "",
        "D-066 remains incomplete and is blocked at the frozen R-DISC-0 OPS-002",
        "resource gate. D-067 was not opened. No retrieval winner was selected, no",
        "fallback changed, and no fitting, Modal, development/public inference, or",
        "leaderboard feedback was used.",
        "",
        "## Repository state entering D-066",
        "",
        "- D-063 through D-065 were complete and frozen.",
        "- Retrieval supervision v2 contained 2,391 positive train groups over",
        "  641,118 immutable chunks.",
        "- The post-D-062 parent remained base Qwen3 reranker + Qwen3-1.7B G1A512.",
        "- The old Qwen dense partial contained 12,147 rows and was rejected evidence.",
        "",
        "## Implemented infrastructure",
        "",
        "- Train-positive provenance loader with non-train/checksum fail-closed guards.",
        "- Recall and evidence-set Recall at 5/10/20/50, MRR@50, exact normalized",
        "  answer-sentence-bearing coverage, novelty/loss, stable ordering and bytes.",
        "- Bounded local execution and immutable context/coordinate lookup caching.",
        "- Sparse and dense OPS-002 preflight contracts.",
        "- Length-stratified dense sampling and checksum-bound resumable dense-store",
        "  construction with tail truncation and completed-file tamper rejection.",
        "",
        "## Measured arm state",
        "",
        f"- R-DISC-0: BLOCKED (`OPS002_PREFLIGHT_TIMEOUT`), checksum `{sparse_checksum}`.",
        "  The fixed first 100 positive groups did not complete by 904 seconds with",
        "  four workers. No ranking or quality metric was published; legal sparse",
        "  was not started.",
        f"- R-DISC-1 preflight: PASS, checksum `{dense_checksum}`.",
        "  Throughput 37.144 documents/s; projected runtime 17,261 seconds; peak VRAM",
        "  2,579,660,800 bytes; projected vector bytes 1,313,009,664. The exact model",
        "  count is 595,776,512 and whole-system count is 3,223,292,928 (<4B). Full",
        "  indexing was not opened because the sequential sparse parent is absent.",
        "- R-DISC-2/R-DISC-3: BLOCKED; no exact GTE/BGE revision/license/checksum/",
        "  parameter manifests exist.",
        "- R-DISC-4: BLOCKED; fixed RRF-60 requires completed sparse and dense rankings.",
        "",
        "## Metrics and winner",
        "",
        "Recall@5/10/20/50, evidence-set Recall, MRR@50, answer-bearing coverage,",
        "novel recovery, METEOR, ROUGE-L, and grounding are unavailable because no",
        "complete paired arm was published. No proxy winner was promoted. The frozen",
        "post-D-062 fallback remains unchanged.",
        "",
        "## Verification",
        "",
        "- Focused D-066 tests: 22 PASS.",
        "- Ruff format/lint: PASS.",
        "- mypy over 110 source files: PASS.",
        "- CPU suite: 665 PASS, 2 SKIP, 5 DESELECTED.",
        f"- Stage-state checksum: `{stage_checksum}`.",
        "- Cost: USD 0; Modal/network: unused.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        (
            r".\.venv\Scripts\python.exe -m pytest "
            "tests/unit/evaluation/test_discovery_tournament.py "
            "tests/unit/evaluation/test_bounded_parallel.py "
            "tests/unit/retrieval/test_lookup_cache.py "
            "tests/unit/retrieval/test_sparse_preflight.py "
            "tests/unit/retrieval/test_dense_preflight.py "
            "tests/unit/retrieval/test_dense_sampling.py "
            "tests/unit/retrieval/test_resumable_dense_store.py -q "
            "--basetemp=.local/pytest-d066-focused-final"
        ),
        r".\.venv\Scripts\ruff.exe format --check .",
        r".\.venv\Scripts\ruff.exe check .",
        r".\.venv\Scripts\mypy.exe src",
        (
            r'.\.venv\Scripts\python.exe -m pytest -q -m "not integration and not gpu" '
            "--basetemp=.local/pytest-d066-full-final"
        ),
        "```",
        "",
        "The next safe recovery is a parity-proven execution optimization for the",
        "frozen BM25 semantics followed by a fresh OPS-002 preflight. Do not run the",
        "full dense-index command or open D-067 while R-DISC-0 remains blocked.",
        "",
    ]
    report_data = "\n".join(report_lines).encode("utf-8")
    report_checksum = write_immutable_bytes(
        Path("artifacts/reports/post-d062/D066-blocker-report.v2.md"), report_data
    )
    checksums_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-checksums.v2",
            "stage_state_checksum": stage_checksum,
            "report_checksum": report_checksum,
            "sparse_preflight_checksum": sparse_checksum,
            "dense_preflight_checksum": dense_checksum,
        }
    )
    checksums_checksum = write_immutable_bytes(_ROOT / "D066.checksums.v2.json", checksums_data)
    print(f"stage={stage_checksum}")
    print(f"report={report_checksum}")
    print(f"checksums={checksums_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
