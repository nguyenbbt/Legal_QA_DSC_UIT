from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.grounding.answer_assessment import (
    AnswerAssessmentError,
    compare_answer_grounding,
    export_answer_assessments,
    import_labeled_answer_assessment_queue,
    load_approved_answer_assessments,
)


def _queue_data(*, labeled: bool, run_id: str = "G1A512-fixture") -> bytes:
    rows: list[bytes] = []
    for index in range(60):
        answer = f"Generated answer {index}"
        row = {
            "schema_version": "grounding.answer-assessment.work-item.v1",
            "question_id": f"q{index:02d}",
            "evaluated_run_id": run_id,
            "answer_checksum": checksum_bytes(answer.encode()),
            "question": f"Question {index}",
            "gold_answer": f"Gold answer {index}",
            "generated_answer": answer,
            "evidence": [
                {
                    "evidence_id": f"chunk_{index:02d}",
                    "display_text": f"Evidence {index}",
                }
            ],
            "answer_support": "fully_supported" if labeled else None,
            "unsupported_claim_count": 0 if labeled else None,
            "annotator_id": "reviewer" if labeled else None,
            "adjudicator_id": "owner" if labeled else None,
            "adjudication_state": "approved" if labeled else "pending",
            "label_version": "grounding.v1",
        }
        rows.append(content_json_bytes(row))
    return b"".join(rows)


def _benchmark() -> tuple[bytes, bytes]:
    rows = b"".join(
        content_json_bytes(
            {
                "schema_version": "grounding.benchmark.v1",
                "question_id": f"q{index:02d}",
                "split": "development",
                "question_checksum": checksum_bytes(f"Question {index}".encode()),
                "relevant_evidence": [
                    {"evidence_id": f"chunk_{index:02d}", "relevance": "relevant"}
                ],
                "required_claims": [],
                "question_answerability": "answerable",
                "temporal_assessment": "unknown",
                "label_version": "grounding.v1",
            }
        )
        for index in range(60)
    )
    manifest = content_json_bytes(
        {
            "schema_version": "grounding.benchmark.manifest.v1",
            "label_version": "grounding.v1",
            "train_split_checksum": checksum_bytes(b"train"),
            "development_split_checksum": checksum_bytes(b"development"),
            "sampling_version": "grounding-sample.v1",
            "sampling_seed": "dsc2026-grounding-sample-v1",
            "ordered_question_ids": [f"q{index:02d}" for index in range(60)],
            "chunk_artifact_checksum": checksum_bytes(b"chunks"),
            "index_checksum": checksum_bytes(b"index"),
            "annotation_status": "approved",
            "ordered_files": [{"path": "grounding_set.v1.jsonl", "checksum": checksum_bytes(rows)}],
        }
    )
    return rows, manifest


def _approved(*, run_id: str, partial_count: int = 0):
    queue = _queue_data(labeled=False, run_id=run_id)
    labeled_rows = [
        json.loads(line) for line in _queue_data(labeled=True, run_id=run_id).splitlines()
    ]
    for row in labeled_rows[:partial_count]:
        row["answer_support"] = "partially_supported"
        row["unsupported_claim_count"] = 1
    labeled = b"".join(content_json_bytes(row) for row in labeled_rows)
    benchmark, benchmark_manifest = _benchmark()
    exported = export_answer_assessments(
        import_labeled_answer_assessment_queue(queue, labeled),
        queue_data=queue,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        assessment_path=f"assessments/{run_id}.grounding.v1.jsonl",
    )
    return load_approved_answer_assessments(
        manifest_data=exported.manifest_data,
        assessment_data=exported.assessment_data,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
    )


def test_import_labeled_answer_assessments_binds_every_identity_field() -> None:
    labeled = _queue_data(labeled=True)

    imported = import_labeled_answer_assessment_queue(_queue_data(labeled=False), labeled)

    assert imported.evaluated_run_id == "G1A512-fixture"
    assert imported.source_labeled_checksum == checksum_bytes(labeled)
    assert len(imported.records) == 60
    assert imported.records[0].answer_support == "fully_supported"


def test_import_labeled_answer_assessments_rejects_identity_changes() -> None:
    rows = [json.loads(line) for line in _queue_data(labeled=True).splitlines()]
    rows[0]["generated_answer"] = "tampered"
    rows[0]["answer_checksum"] = checksum_bytes(b"tampered")
    changed = b"".join(content_json_bytes(row) for row in rows)

    with pytest.raises(AnswerAssessmentError) as caught:
        import_labeled_answer_assessment_queue(_queue_data(labeled=False), changed)

    assert caught.value.code == "GROUNDING_ASSESSMENT_QUEUE_MISMATCH"


def test_import_labeled_answer_assessments_rejects_inconsistent_support_count() -> None:
    rows = [json.loads(line) for line in _queue_data(labeled=True).splitlines()]
    rows[0]["answer_support"] = "unsupported"
    rows[0]["unsupported_claim_count"] = 0
    invalid = b"".join(content_json_bytes(row) for row in rows)

    with pytest.raises(AnswerAssessmentError) as caught:
        import_labeled_answer_assessment_queue(_queue_data(labeled=False), invalid)

    assert caught.value.code == "GROUNDING_ASSESSMENT_SCHEMA_INVALID"


def test_import_requires_digest_sample_adjudication_for_fully_supported_rows() -> None:
    rows = [json.loads(line) for line in _queue_data(labeled=True).splitlines()]
    for row in rows:
        row["adjudicator_id"] = None
    incomplete = b"".join(content_json_bytes(row) for row in rows)

    with pytest.raises(AnswerAssessmentError) as caught:
        import_labeled_answer_assessment_queue(_queue_data(labeled=False), incomplete)

    assert caught.value.code == "GROUNDING_ASSESSMENT_ADJUDICATION_INCOMPLETE"


def test_export_answer_assessments_is_approved_checksum_bound_and_deterministic() -> None:
    queue = _queue_data(labeled=False)
    imported = import_labeled_answer_assessment_queue(queue, _queue_data(labeled=True))
    benchmark, benchmark_manifest = _benchmark()

    first = export_answer_assessments(
        imported,
        queue_data=queue,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        assessment_path="assessments/G1A512-fixture.grounding.v1.jsonl",
    )
    second = export_answer_assessments(
        imported,
        queue_data=queue,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        assessment_path="assessments/G1A512-fixture.grounding.v1.jsonl",
    )

    assert first == second
    approved = load_approved_answer_assessments(
        manifest_data=first.manifest_data,
        assessment_data=first.assessment_data,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
    )
    assert len(approved.records) == 60
    assert approved.rates.fully_supported_rate == 1.0
    assert approved.rates.unsupported_answer_rate == 0.0
    assert json.loads(first.report_data)["annotation_status"] == "approved"


def test_load_answer_assessments_rejects_changed_assessment_bytes() -> None:
    queue = _queue_data(labeled=False)
    benchmark, benchmark_manifest = _benchmark()
    exported = export_answer_assessments(
        import_labeled_answer_assessment_queue(queue, _queue_data(labeled=True)),
        queue_data=queue,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        assessment_path="assessments/G1A512-fixture.grounding.v1.jsonl",
    )

    with pytest.raises(AnswerAssessmentError) as caught:
        load_approved_answer_assessments(
            manifest_data=exported.manifest_data,
            assessment_data=exported.assessment_data + b"\n",
            benchmark_manifest_data=benchmark_manifest,
            benchmark_data=benchmark,
        )

    assert caught.value.code == "GROUNDING_ASSESSMENT_CHECKSUM_MISMATCH"


def test_grounding_comparison_applies_unsupported_and_fully_supported_guards() -> None:
    comparison = json.loads(
        compare_answer_grounding(
            _approved(run_id="G1-baseline"),
            _approved(run_id="G1A512-candidate", partial_count=2),
        )
    )

    assert comparison["grounding_gate"] == "failed"
    assert comparison["promotion_blockers"] == ["FULLY_SUPPORTED_RATE_REGRESSION_EXCEEDS_0_02"]
    assert comparison["rates"]["fully_supported_delta"] == pytest.approx(-2 / 60)
