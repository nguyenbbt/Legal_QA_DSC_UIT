from __future__ import annotations

import json

import pytest
from scripts.finalize_answer_grounding import finalize_answer_grounding_artifacts
from scripts.prepare_answer_grounding import prepare_answer_grounding_artifacts

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.grounding.answer_assessment import (
    export_answer_assessments,
    import_labeled_answer_assessment_queue,
)


def _fixture_sources() -> tuple[bytes, bytes, bytes, bytes, bytes]:
    annotations: list[bytes] = []
    retrieval: list[bytes] = []
    predictions: dict[str, dict[str, str]] = {}
    source_queue: list[bytes] = []
    source_labeled: list[bytes] = []
    for index in range(60):
        question_id = f"q{index:02d}"
        evidence_id = f"chunk_{index:02d}"
        answer = f"Generated answer {index}"
        annotations.append(
            content_json_bytes(
                {
                    "question_id": question_id,
                    "question": f"Question {index}",
                    "gold_answer": f"Gold answer {index}",
                    "candidates": [
                        {"evidence_id": evidence_id, "display_text": f"Evidence {index}"}
                    ],
                }
            )
        )
        retrieval.append(
            content_json_bytes(
                {"question_id": question_id, "candidates": [{"evidence_id": evidence_id}]}
            )
        )
        predictions[question_id] = {"answer": answer}
        base = {
            "schema_version": "grounding.answer-assessment.work-item.v1",
            "question_id": question_id,
            "evaluated_run_id": "source-run",
            "answer_checksum": checksum_bytes(answer.encode()),
            "question": f"Question {index}",
            "gold_answer": f"Gold answer {index}",
            "generated_answer": answer,
            "evidence": [{"evidence_id": evidence_id, "display_text": f"Evidence {index}"}],
            "answer_support": None,
            "unsupported_claim_count": None,
            "annotator_id": None,
            "adjudicator_id": None,
            "adjudication_state": "pending",
            "label_version": "grounding.v1",
        }
        source_queue.append(content_json_bytes(base))
        source_labeled.append(
            content_json_bytes(
                {
                    **base,
                    "answer_support": "fully_supported",
                    "unsupported_claim_count": 0,
                    "annotator_id": "reviewer",
                    "adjudicator_id": "adjudicator",
                    "adjudication_state": "approved",
                }
            )
        )
    return (
        b"".join(annotations),
        b"".join(retrieval),
        content_json_bytes(predictions),
        b"".join(source_queue),
        b"".join(source_labeled),
    )


def test_prepare_answer_grounding_artifacts_builds_and_prefills_frozen_queue() -> None:
    annotation, retrieval, predictions, source_queue, source_labeled = _fixture_sources()

    artifacts = prepare_answer_grounding_artifacts(
        artifact_prefix="D061",
        evaluated_run_id="R008-GC0-base-G1A512-v1",
        annotation_queue_data=annotation,
        retrieval_output_data=retrieval,
        predictions_data=predictions,
        source_queue_data=source_queue,
        source_labeled_data=source_labeled,
        evidence_limit=3,
    )

    assert set(artifacts) == {
        "D061-answer-grounding-work-queue.v1.jsonl",
        "D061-answer-grounding-work-queue.v1.prefilled.jsonl",
        "D061-answer-grounding-prefill-report.v1.json",
    }
    queue_rows = [
        json.loads(line)
        for line in artifacts["D061-answer-grounding-work-queue.v1.jsonl"].splitlines()
    ]
    prefilled_rows = [
        json.loads(line)
        for line in artifacts["D061-answer-grounding-work-queue.v1.prefilled.jsonl"].splitlines()
    ]
    report = json.loads(artifacts["D061-answer-grounding-prefill-report.v1.json"])
    assert len(queue_rows) == 60
    assert queue_rows[0]["evaluated_run_id"] == "R008-GC0-base-G1A512-v1"
    assert len(queue_rows[0]["evidence"]) == 1
    assert all(row["adjudication_state"] == "approved" for row in prefilled_rows)
    assert report["prefilled_count"] == 60
    assert report["pending_count"] == 0


def test_prepare_answer_grounding_artifacts_rejects_unsafe_output_prefix() -> None:
    with pytest.raises(ValueError, match="artifact prefix"):
        prepare_answer_grounding_artifacts(
            artifact_prefix="../outside",
            evaluated_run_id="candidate-run",
            annotation_queue_data=b"",
            retrieval_output_data=b"",
            predictions_data=b"",
            source_queue_data=b"",
            source_labeled_data=b"",
            evidence_limit=3,
        )


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


def test_finalize_answer_grounding_exports_and_compares_approved_candidate() -> None:
    annotation, retrieval, predictions, source_queue, source_labeled = _fixture_sources()
    queue = prepare_answer_grounding_artifacts(
        artifact_prefix="D061",
        evaluated_run_id="candidate-run",
        annotation_queue_data=annotation,
        retrieval_output_data=retrieval,
        predictions_data=predictions,
        source_queue_data=source_queue,
        source_labeled_data=source_labeled,
        evidence_limit=3,
    )["D061-answer-grounding-work-queue.v1.jsonl"]
    labeled_rows = []
    for line in queue.splitlines():
        row = json.loads(line)
        row.update(
            {
                "answer_support": "fully_supported",
                "unsupported_claim_count": 0,
                "annotator_id": "reviewer",
                "adjudicator_id": "adjudicator",
                "adjudication_state": "approved",
            }
        )
        labeled_rows.append(content_json_bytes(row))
    labeled = b"".join(labeled_rows)
    benchmark, benchmark_manifest = _benchmark()
    baseline = export_answer_assessments(
        import_labeled_answer_assessment_queue(source_queue, source_labeled),
        queue_data=source_queue,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        assessment_path="assessments/source-run.grounding.v1.jsonl",
    )

    artifacts = finalize_answer_grounding_artifacts(
        artifact_stem="candidate-run",
        queue_data=queue,
        labeled_data=labeled,
        benchmark_manifest_data=benchmark_manifest,
        benchmark_data=benchmark,
        baseline_manifest_data=baseline.manifest_data,
        baseline_assessment_data=baseline.assessment_data,
    )

    assert set(artifacts) == {
        "candidate-run.grounding.v1.jsonl",
        "candidate-run.grounding.manifest.v1.json",
        "candidate-run.grounding.report.v1.json",
        "candidate-run.vs-baseline.grounding-comparison.v1.json",
    }
    assert (
        json.loads(artifacts["candidate-run.vs-baseline.grounding-comparison.v1.json"])[
            "grounding_gate"
        ]
        == "passed"
    )
