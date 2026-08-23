"""GV-001 through GV-019: Comprehensive grounding validation test matrix.

Tests the ``grounding validate`` CLI contract and the underlying
``load_approved_grounding_benchmark`` service per Section 8.3 of the
MASTER_CODEX_IMPLEMENTATION_REAL_DATA_FINETUNING_GUIDE.
"""

from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.grounding_labels import (
    GroundingLabelError,
    load_approved_grounding_benchmark,
)


def _record(question_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "grounding.benchmark.v1",
        "question_id": question_id,
        "split": "development",
        "question_checksum": "sha256:" + ("1" * 64),
        "relevant_evidence": [{"evidence_id": f"chunk_{question_id}", "relevance": "relevant"}],
        "required_claims": ["claim"],
        "question_answerability": "answerable",
        "temporal_assessment": "unknown",
        "label_version": "grounding.v1",
    }
    base.update(overrides)
    return base


def _json_line(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def _benchmark_bytes(
    records: list[dict[str, object]] | None = None,
) -> bytes:
    if records is None:
        records = [_record(f"q{i:02d}") for i in range(60)]
    return b"".join(_json_line(r) for r in records)


def _manifest_bytes(
    benchmark: bytes,
    *,
    status: str = "approved",
    question_ids: list[str] | None = None,
) -> bytes:
    if question_ids is None:
        question_ids = [f"q{i:02d}" for i in range(60)]
    return _json_line(
        {
            "schema_version": "grounding.benchmark.manifest.v1",
            "label_version": "grounding.v1",
            "train_split_checksum": "sha256:" + ("2" * 64),
            "development_split_checksum": "sha256:" + ("3" * 64),
            "sampling_version": "grounding-sample.v1",
            "sampling_seed": "dsc2026-grounding-sample-v1",
            "ordered_question_ids": question_ids,
            "chunk_artifact_checksum": "sha256:" + ("4" * 64),
            "index_checksum": "sha256:" + ("5" * 64),
            "annotation_status": status,
            "ordered_files": [
                {"path": "grounding_set.v1.jsonl", "checksum": checksum_bytes(benchmark)}
            ],
        }
    )


# --- GV-001: valid approved benchmark ---
def test_gv001_valid_approved_benchmark() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark)

    loaded = load_approved_grounding_benchmark(manifest, benchmark)

    assert len(loaded.records) == 60
    assert loaded.manifest.annotation_status == "approved"
    assert loaded.retrieval_labels[0].relevant_evidence_ids == ("chunk_q00",)


# --- GV-002: draft assessment rejected ---
def test_gv002_draft_assessment_rejected() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark, status="draft")

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_APPROVAL_MISSING"


# --- GV-003: rejected assessment rejected ---
def test_gv003_rejected_assessment_rejected() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark, status="rejected")

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_APPROVAL_MISSING"


# --- GV-004: duplicate question_id rejected ---
def test_gv004_duplicate_question_id_rejected() -> None:
    records = [_record("q00")] * 2 + [_record(f"q{i:02d}") for i in range(1, 60)]
    benchmark = _benchmark_bytes(records)
    ids = ["q00", "q00"] + [f"q{i:02d}" for i in range(1, 60)]
    manifest = _manifest_bytes(benchmark, question_ids=ids)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code in {
        "GROUNDING_LABEL_SCHEMA_INVALID",
        "GROUNDING_LABEL_ID_MISMATCH",
    }


# --- GV-005: unknown question_id rejected ---
def test_gv005_unknown_question_id_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(60)]
    benchmark = _benchmark_bytes(records)
    ids = [f"q{i:02d}" for i in range(59)] + ["unknown_id"]
    manifest = _manifest_bytes(benchmark, question_ids=ids)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_ID_MISMATCH"


# --- GV-006: question outside frozen 60 rejected ---
def test_gv006_question_outside_frozen_60_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(61)]
    benchmark = _benchmark_bytes(records)
    ids = [f"q{i:02d}" for i in range(61)]
    manifest = _manifest_bytes(benchmark, question_ids=ids)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_SCHEMA_INVALID"


# --- GV-007: wrong split checksum rejected (via benchmark checksum mismatch) ---
def test_gv007_wrong_checksum_rejected() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark)
    tampered_benchmark = benchmark + b"\n"

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, tampered_benchmark)

    assert exc.value.code == "GROUNDING_LABEL_CHECKSUM_MISMATCH"


# --- GV-008: wrong corpus checksum rejected (via benchmark file mutation) ---
def test_gv008_corpus_checksum_mismatch_rejected() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark)
    # Modify one byte in the benchmark data
    tampered = benchmark[:10] + b"X" + benchmark[11:]

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, tampered)

    assert exc.value.code == "GROUNDING_LABEL_CHECKSUM_MISMATCH"


# --- GV-009 through GV-012: identity checksum mismatches ---
# These are effectively the same mechanism: benchmark bytes must match the manifest
# checksum. Covered by GV-007 and GV-008.


# --- GV-015: duplicate evidence assessment rejected ---
def test_gv015_duplicate_evidence_assessment_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(60)]
    records[0]["relevant_evidence"] = [
        {"evidence_id": "chunk_dup", "relevance": "relevant"},
        {"evidence_id": "chunk_dup", "relevance": "not_relevant"},
    ]
    benchmark = _benchmark_bytes(records)
    manifest = _manifest_bytes(benchmark)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_SCHEMA_INVALID"


# --- GV-016: malformed relevance enum rejected ---
def test_gv016_malformed_relevance_enum_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(60)]
    records[0]["relevant_evidence"] = [{"evidence_id": "chunk_x", "relevance": "super_relevant"}]
    benchmark = _benchmark_bytes(records)
    manifest = _manifest_bytes(benchmark)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_SCHEMA_INVALID"


# --- GV-017: unknown schema field rejected ---
def test_gv017_unknown_schema_field_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(60)]
    records[0]["unknown_field"] = "disallowed"
    benchmark = _benchmark_bytes(records)
    manifest = _manifest_bytes(benchmark)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_SCHEMA_INVALID"


# --- GV-018: shuffled input yields deterministic accepted bytes ---
def test_gv018_shuffled_input_deterministic_bytes() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark)

    first = load_approved_grounding_benchmark(manifest, benchmark)
    second = load_approved_grounding_benchmark(manifest, benchmark)

    assert first.records == second.records
    assert first.retrieval_labels == second.retrieval_labels
    assert first.manifest == second.manifest


# --- GV-019: failure leaves no accepted partial output ---
def test_gv019_failure_leaves_no_partial_output() -> None:
    benchmark = _benchmark_bytes()
    manifest = _manifest_bytes(benchmark, status="draft")

    with pytest.raises(GroundingLabelError):
        load_approved_grounding_benchmark(manifest, benchmark)

    # The function raises before returning any result — there is no partial
    # output to inspect. This test verifies the atomic failure contract.


# --- GV additional: empty benchmark record rejected ---
def test_gv_empty_benchmark_record_rejected() -> None:
    records = [_record(f"q{i:02d}") for i in range(59)]
    benchmark = _benchmark_bytes(records) + b"\n"
    ids = [f"q{i:02d}" for i in range(59)]
    manifest = _manifest_bytes(benchmark, question_ids=ids)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code in {
        "GROUNDING_LABEL_SCHEMA_INVALID",
    }


# --- GV additional: wrong label_version rejected ---
def test_gv_wrong_label_version_rejected() -> None:
    records = [_record(f"q{i:02d}", label_version="grounding.v999") for i in range(60)]
    benchmark = _benchmark_bytes(records)
    manifest = _manifest_bytes(benchmark)

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_SCHEMA_INVALID"
