"""REVAL-001 through REVAL-015: Comprehensive retrieval evaluation test matrix.

Tests the label-backed ``evaluate retrieval`` service per Section 9.4 of the
MASTER_CODEX_IMPLEMENTATION_REAL_DATA_FINETUNING_GUIDE.
"""

from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.retrieval_metrics import (
    ContainmentInputRow,
    RetrievalEvaluationError,
    RetrievalLabelRow,
    RetrievalOutputRow,
    evaluate_answer_containment,
    evaluate_retrieval,
)


# --- REVAL-001: hand-computed Recall@1 ---
def test_reval001_hand_computed_recall_at_1() -> None:
    """Two evaluable questions: q1 has relevant 'a' at rank 1, q2 has relevant 'c' at rank 2."""
    labels = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q2", ("c",)),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a", "b")),
        RetrievalOutputRow("q2", ("b", "c")),
    )

    report = evaluate_retrieval(labels, outputs)

    # q1: a at rank 1 → recall@1 = 1/1 = 1.0
    # q2: c at rank 2 → recall@1 = 0/1 = 0.0
    # macro = (1.0 + 0.0) / 2 = 0.5
    assert report.recall_at_1 == 0.5


# --- REVAL-002: hand-computed Recall@5 ---
def test_reval002_hand_computed_recall_at_5() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a", "b")),
        RetrievalLabelRow("q2", ("c",)),
    )
    outputs = (
        RetrievalOutputRow("q1", ("x", "a", "y", "b", "z")),
        RetrievalOutputRow("q2", ("x", "y", "z", "w", "c")),
    )

    report = evaluate_retrieval(labels, outputs)

    # q1: {a,b} in top-5 → both found → recall@5 = 2/2 = 1.0
    # q2: {c} in top-5 → found at rank 5 → recall@5 = 1/1 = 1.0
    # macro = (1.0 + 1.0) / 2 = 1.0
    assert report.recall_at_5 == 1.0


# --- REVAL-003: hand-computed Recall@10 ---
def test_reval003_hand_computed_recall_at_10() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q2", ("c",)),
    )
    outputs = (
        RetrievalOutputRow("q1", tuple(f"x{i}" for i in range(9)) + ("a",)),
        RetrievalOutputRow("q2", tuple(f"x{i}" for i in range(10)) + ("c",)),
    )

    report = evaluate_retrieval(labels, outputs)

    # q1: a at rank 10 → in top-10 → recall@10 = 1.0
    # q2: c at rank 11 → NOT in top-10 → recall@10 = 0.0
    # macro = (1.0 + 0.0) / 2 = 0.5
    assert report.recall_at_10 == 0.5


# --- REVAL-004: hand-computed MRR@10 ---
def test_reval004_hand_computed_mrr_at_10() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q2", ("c",)),
        RetrievalLabelRow("q3", ("e",)),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a", "b")),  # relevant at rank 1 → 1/1
        RetrievalOutputRow("q2", ("b", "c")),  # relevant at rank 2 → 1/2
        RetrievalOutputRow("q3", ("x", "y", "e")),  # relevant at rank 3 → 1/3
    )

    report = evaluate_retrieval(labels, outputs)

    # MRR@10 = (1/1 + 1/2 + 1/3) / 3 = (6/6 + 3/6 + 2/6) / 3 = 11/18
    assert report.mrr_at_10 == pytest.approx(11.0 / 18.0)


# --- REVAL-005: multi-evidence evidence-set Recall@10 ---
def test_reval005_multi_evidence_set_recall_at_10() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a", "b")),
        RetrievalLabelRow("q2", ("c", "d")),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a", "b", "x")),  # both in top-10 → set hit
        RetrievalOutputRow("q2", ("c", "x", "y")),  # only c, missing d → no set hit
    )

    report = evaluate_retrieval(labels, outputs)

    # q1: {a,b} ⊆ top-10 → 1.0
    # q2: {c,d} ⊄ top-10 → 0.0
    # macro = (1.0 + 0.0) / 2 = 0.5
    assert report.evidence_set_recall_at_10 == 0.5


# --- REVAL-006: empty relevance exclusion ---
def test_reval006_empty_relevance_exclusion() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q2", ()),  # empty → excluded from metrics
        RetrievalLabelRow("q3", ("c",)),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a",)),
        RetrievalOutputRow("q2", ("b",)),
        RetrievalOutputRow("q3", ("c",)),
    )

    report = evaluate_retrieval(labels, outputs)

    assert report.retrieval_evaluable_count == 2
    assert report.retrieval_unevaluable_count == 1
    assert report.unevaluable_question_ids == ("q2",)
    assert report.recall_at_1 == 1.0  # both evaluable hit at rank 1


# --- REVAL-007: mixed denominator ---
def test_reval007_mixed_denominator() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a", "b", "c")),  # 3 relevant
        RetrievalLabelRow("q2", ("d",)),  # 1 relevant
    )
    outputs = (
        RetrievalOutputRow("q1", ("a",)),  # recall@1 = 1/3
        RetrievalOutputRow("q2", ("d",)),  # recall@1 = 1/1
    )

    report = evaluate_retrieval(labels, outputs)

    # macro recall@1 = (1/3 + 1) / 2 = 2/3
    assert report.recall_at_1 == pytest.approx(2.0 / 3.0)


# --- REVAL-008: empty evaluation rejected ---
def test_reval008_empty_evaluation_rejected() -> None:
    labels = (
        RetrievalLabelRow("q1", ()),
        RetrievalLabelRow("q2", ()),
    )
    outputs = (
        RetrievalOutputRow("q1", ("a",)),
        RetrievalOutputRow("q2", ("b",)),
    )

    with pytest.raises(RetrievalEvaluationError) as exc:
        evaluate_retrieval(labels, outputs)

    assert exc.value.code == "RETRIEVAL_EVAL_EMPTY"


# --- REVAL-009: benchmark ID mismatch rejected ---
def test_reval009_benchmark_id_mismatch_rejected() -> None:
    labels = (RetrievalLabelRow("q1", ("a",)),)
    outputs = (RetrievalOutputRow("q2", ("a",)),)

    with pytest.raises(RetrievalEvaluationError) as exc:
        evaluate_retrieval(labels, outputs)

    assert exc.value.code == "RETRIEVAL_EVAL_ID_MISMATCH"


# --- REVAL-010: duplicate question results rejected ---
def test_reval010_duplicate_question_results_rejected() -> None:
    labels = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q1", ("b",)),
    )
    outputs = (RetrievalOutputRow("q1", ("a",)),)

    with pytest.raises(RetrievalEvaluationError) as exc:
        evaluate_retrieval(labels, outputs)

    assert exc.value.code == "RETRIEVAL_LABEL_ID_DUPLICATE"


# --- REVAL-011: containment namespace isolated ---
def test_reval011_containment_namespace_isolated() -> None:
    rows = (
        ContainmentInputRow("q1", "answer text", ("evidence with answer text",)),
        ContainmentInputRow("q2", "other", ("no match",)),
    )

    report = evaluate_answer_containment(rows)

    assert report.metric_namespace == "diagnostic_answer_containment"
    assert report.containment_at_1 == 0.5
    assert report.total_question_count == 2
    assert report.eligible_question_count == 2


# --- REVAL-012: approval state required (test via grounding labels) ---
def test_reval012_approval_state_required() -> None:
    from legal_rag.evaluation.grounding_labels import (
        GroundingLabelError,
        load_approved_grounding_benchmark,
    )

    records = [
        {
            "schema_version": "grounding.benchmark.v1",
            "question_id": f"q{i:02d}",
            "split": "development",
            "question_checksum": "sha256:" + ("1" * 64),
            "relevant_evidence": [{"evidence_id": f"chunk_{i}", "relevance": "relevant"}],
            "required_claims": ["claim"],
            "question_answerability": "answerable",
            "temporal_assessment": "unknown",
            "label_version": "grounding.v1",
        }
        for i in range(60)
    ]
    benchmark = b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in records)
    manifest = (
        json.dumps(
            {
                "schema_version": "grounding.benchmark.manifest.v1",
                "label_version": "grounding.v1",
                "train_split_checksum": "sha256:" + ("2" * 64),
                "development_split_checksum": "sha256:" + ("3" * 64),
                "sampling_version": "grounding-sample.v1",
                "sampling_seed": "dsc2026-grounding-sample-v1",
                "ordered_question_ids": [f"q{i:02d}" for i in range(60)],
                "chunk_artifact_checksum": "sha256:" + ("4" * 64),
                "index_checksum": "sha256:" + ("5" * 64),
                "annotation_status": "draft",
                "ordered_files": [
                    {
                        "path": "grounding_set.v1.jsonl",
                        "checksum": checksum_bytes(benchmark),
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()

    with pytest.raises(GroundingLabelError) as exc:
        load_approved_grounding_benchmark(manifest, benchmark)

    assert exc.value.code == "GROUNDING_LABEL_APPROVAL_MISSING"


# --- REVAL-013: checksum mismatch rejected ---
def test_reval013_checksum_mismatch_rejected() -> None:
    # Build minimal valid inputs then tamper the annotation queue
    from tests.unit.evaluation.test_retrieval_evaluation import _evaluation_inputs

    from legal_rag.evaluation.retrieval_evaluation import (
        LabeledRetrievalError,
        evaluate_labeled_retrieval,
    )

    retrieval, queue, manifest, benchmark = _evaluation_inputs()
    tampered_queue = queue.replace(
        checksum_bytes(b"chunks").encode(),
        checksum_bytes(b"wrong-chunks").encode(),
        1,
    )

    with pytest.raises(LabeledRetrievalError) as exc:
        evaluate_labeled_retrieval(
            retrieval_output_data=retrieval,
            annotation_queue_data=tampered_queue,
            benchmark_manifest_data=manifest,
            benchmark_data=benchmark,
        )

    assert exc.value.code == "RETRIEVAL_EVAL_CHUNK_MISMATCH"


# --- REVAL-014: deterministic input permutation ---
def test_reval014_deterministic_input_permutation() -> None:
    labels = (
        RetrievalLabelRow("q2", ("b",)),
        RetrievalLabelRow("q1", ("a",)),
    )
    outputs = (
        RetrievalOutputRow("q2", ("b",)),
        RetrievalOutputRow("q1", ("a",)),
    )
    labels_reversed = (
        RetrievalLabelRow("q1", ("a",)),
        RetrievalLabelRow("q2", ("b",)),
    )
    outputs_reversed = (
        RetrievalOutputRow("q1", ("a",)),
        RetrievalOutputRow("q2", ("b",)),
    )

    report_a = evaluate_retrieval(labels, outputs)
    report_b = evaluate_retrieval(labels_reversed, outputs_reversed)

    assert report_a.recall_at_1 == report_b.recall_at_1
    assert report_a.recall_at_5 == report_b.recall_at_5
    assert report_a.recall_at_10 == report_b.recall_at_10
    assert report_a.mrr_at_10 == report_b.mrr_at_10
    assert report_a.evidence_set_recall_at_10 == report_b.evidence_set_recall_at_10


# --- REVAL-015: final report byte-identical across clean runs ---
def test_reval015_report_byte_identical_across_runs() -> None:
    from tests.unit.evaluation.test_retrieval_evaluation import _evaluation_inputs

    from legal_rag.evaluation.retrieval_evaluation import evaluate_labeled_retrieval

    retrieval, queue, manifest, benchmark = _evaluation_inputs()

    first = evaluate_labeled_retrieval(
        retrieval_output_data=retrieval,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=benchmark,
    )
    second = evaluate_labeled_retrieval(
        retrieval_output_data=retrieval,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=benchmark,
    )

    assert first == second
    assert json.loads(first) == json.loads(second)
