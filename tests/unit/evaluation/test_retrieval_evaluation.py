from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.retrieval_evaluation import (
    LabeledRetrievalError,
    evaluate_labeled_retrieval,
)

SPLIT_CHECKSUM = checksum_bytes(b"split")
CHUNKS_CHECKSUM = checksum_bytes(b"chunks")
INDEX_CHECKSUM = checksum_bytes(b"index")


def _json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _candidate(question_index: int, evidence_id: str, rank: int) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "context_id": str(question_index + 1),
        "hierarchy_path": ["article:1"],
        "canonical_start": rank * 10,
        "canonical_end": rank * 10 + 8,
        "chunk_checksum": checksum_bytes(evidence_id.encode()),
        "exact_reference_match": False,
        "sparse_score": 2.0 - rank,
    }


def _evaluation_inputs() -> tuple[bytes, bytes, bytes, bytes]:
    retrieval_rows: list[bytes] = []
    queue_rows: list[bytes] = []
    benchmark_rows: list[bytes] = []
    question_ids: list[str] = []
    for index in range(60):
        question_id = f"q{index:02d}"
        question_ids.append(question_id)
        question_checksum = checksum_bytes(f"question {question_id}".encode())
        first_id = f"chunk_{index:024x}"
        second_id = f"chunk_{index + 1000:024x}"
        candidates = [_candidate(index, first_id, 0), _candidate(index, second_id, 1)]
        diagnostics = [
            {
                "code": "EXACT_REFERENCE_MALFORMED" if index == 0 else "EXACT_COORDINATE_ABSENT",
                "candidate_count": 0,
                "parser_version": "legal-reference-parser.v1",
                "document_key_version": "legal-document-number-key.v1",
                "alias_manifest_checksum": None,
            }
        ]
        retrieval_rows.append(
            _json_line(
                {
                    "schema_version": "retrieval.output.v1",
                    "question_id": question_id,
                    "question_checksum": question_checksum,
                    "candidates": candidates,
                    "diagnostics": diagnostics,
                }
            )
        )
        queue_candidates = [
            {**candidate, "display_text": f"Evidence with gold {question_id}"}
            for candidate in candidates
        ]
        queue_rows.append(
            _json_line(
                {
                    "schema_version": "grounding.annotation.work-item.v1",
                    "question_id": question_id,
                    "split": "development",
                    "question_checksum": question_checksum,
                    "question": f"question {question_id}",
                    "gold_answer": f"gold {question_id}",
                    "split_checksum": SPLIT_CHECKSUM,
                    "chunk_artifact_checksum": CHUNKS_CHECKSUM,
                    "index_checksum": INDEX_CHECKSUM,
                    "candidates": queue_candidates,
                    "diagnostics": diagnostics,
                    "relevant_evidence": None,
                    "required_claims": None,
                    "question_answerability": None,
                    "temporal_assessment": None,
                    "annotation_state": "pending_primary_annotation",
                }
            )
        )
        remainder = index % 4
        relevant = (
            []
            if remainder == 3
            else [
                {
                    "evidence_id": (
                        first_id
                        if remainder == 0
                        else second_id
                        if remainder == 1
                        else f"chunk_{index + 2000:024x}"
                    ),
                    "relevance": "partially_relevant" if remainder == 1 else "relevant",
                }
            ]
        )
        benchmark_rows.append(
            _json_line(
                {
                    "schema_version": "grounding.benchmark.v1",
                    "question_id": question_id,
                    "split": "development",
                    "question_checksum": question_checksum,
                    "relevant_evidence": relevant,
                    "required_claims": ["claim"],
                    "question_answerability": "answerable",
                    "temporal_assessment": "unknown",
                    "label_version": "grounding.v1",
                }
            )
        )
    benchmark = b"".join(benchmark_rows)
    manifest = _json_line(
        {
            "schema_version": "grounding.benchmark.manifest.v1",
            "label_version": "grounding.v1",
            "train_split_checksum": checksum_bytes(b"train-split"),
            "development_split_checksum": SPLIT_CHECKSUM,
            "sampling_version": "grounding-sample.v1",
            "sampling_seed": "dsc2026-grounding-sample-v1",
            "ordered_question_ids": question_ids,
            "chunk_artifact_checksum": CHUNKS_CHECKSUM,
            "index_checksum": INDEX_CHECKSUM,
            "annotation_status": "approved",
            "ordered_files": [
                {
                    "path": "grounding_set.v1.jsonl",
                    "checksum": checksum_bytes(benchmark),
                }
            ],
        }
    )
    return b"".join(retrieval_rows), b"".join(queue_rows), manifest, benchmark


def test_approved_labels_produce_deterministic_metrics_and_failure_taxonomy() -> None:
    retrieval, queue, manifest, benchmark = _evaluation_inputs()

    rendered = evaluate_labeled_retrieval(
        retrieval_output_data=retrieval,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=benchmark,
    )
    report = json.loads(rendered)

    assert report["metrics_status"] == "complete_owner_approved_labels"
    assert report["metrics"] == {
        "benchmark_question_count": 60,
        "evidence_set_recall_at_10": pytest.approx(2 / 3),
        "mrr_at_10": 0.5,
        "recall_at_1": pytest.approx(1 / 3),
        "recall_at_10": pytest.approx(2 / 3),
        "recall_at_5": pytest.approx(2 / 3),
        "retrieval_evaluable_count": 45,
        "retrieval_unevaluable_count": 15,
        "unevaluable": [
            {"question_id": f"q{index:02d}", "reason": "NO_RELEVANT_EVIDENCE"}
            for index in range(3, 60, 4)
        ],
        "unevaluable_question_ids": [f"q{index:02d}" for index in range(3, 60, 4)],
    }
    assert report["containment"]["metric_namespace"] == "diagnostic_answer_containment"
    assert report["containment"]["containment_at_10"] == 1.0
    taxonomy = {row["category"]: row for row in report["failure_taxonomy"]}
    assert taxonomy["parser_or_alias_identity_error"]["question_count"] == 1
    assert taxonomy["missing_top_10_evidence"]["question_count"] == 15
    assert taxonomy["poor_top_rank_ordering"]["question_count"] == 15
    assert rendered == evaluate_labeled_retrieval(
        retrieval_output_data=retrieval,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=benchmark,
    )


def test_evaluation_rejects_annotation_identity_tampering() -> None:
    retrieval, queue, manifest, benchmark = _evaluation_inputs()
    tampered = queue.replace(INDEX_CHECKSUM.encode(), checksum_bytes(b"other-index").encode(), 1)

    with pytest.raises(LabeledRetrievalError, match="index checksum") as captured:
        evaluate_labeled_retrieval(
            retrieval_output_data=retrieval,
            annotation_queue_data=tampered,
            benchmark_manifest_data=manifest,
            benchmark_data=benchmark,
        )

    assert captured.value.code == "RETRIEVAL_EVAL_INDEX_MISMATCH"


def test_grounding_validate_cli_accepts_only_the_approved_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, manifest, benchmark = _evaluation_inputs()
    manifest_path = tmp_path / "grounding.benchmark.manifest.json"
    benchmark_path = tmp_path / "grounding_set.v1.jsonl"
    report_path = tmp_path / "grounding.validation.report.json"
    manifest_path.write_bytes(manifest)
    benchmark_path.write_bytes(benchmark)

    exit_code = main(
        [
            "grounding",
            "validate",
            "--manifest",
            str(manifest_path),
            "--benchmark",
            str(benchmark_path),
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("GROUNDING VALIDATION COMPLETE questions=60 sha256:")
    assert captured.err == ""
    report = json.loads(report_path.read_bytes())
    assert report == {
        "annotation_status": "approved",
        "benchmark_checksum": checksum_bytes(benchmark),
        "benchmark_question_count": 60,
        "label_version": "grounding.v1",
        "manifest_checksum": checksum_bytes(manifest),
        "schema_version": "grounding.validation.report.v1",
        "validation_errors": [],
        "validation_result": "valid",
    }


def test_grounding_validate_cli_retries_the_same_immutable_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, manifest, benchmark = _evaluation_inputs()
    manifest_path = tmp_path / "grounding.benchmark.manifest.json"
    benchmark_path = tmp_path / "grounding_set.v1.jsonl"
    report_path = tmp_path / "grounding.validation.report.json"
    manifest_path.write_bytes(manifest)
    benchmark_path.write_bytes(benchmark)
    arguments = [
        "grounding",
        "validate",
        "--manifest",
        str(manifest_path),
        "--benchmark",
        str(benchmark_path),
        "--output",
        str(report_path),
    ]

    first_exit = main(arguments)
    first_output = capsys.readouterr()
    first_report = report_path.read_bytes()
    second_exit = main(arguments)
    second_output = capsys.readouterr()

    assert first_exit == second_exit == 0
    assert first_output == second_output
    assert report_path.read_bytes() == first_report


def test_evaluate_retrieval_cli_writes_an_immutable_labeled_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    retrieval, queue, manifest, benchmark = _evaluation_inputs()
    paths = {
        "retrieval": tmp_path / "retrieval.jsonl",
        "queue": tmp_path / "annotation-queue.jsonl",
        "manifest": tmp_path / "grounding.manifest.json",
        "benchmark": tmp_path / "grounding.jsonl",
        "report": tmp_path / "retrieval-report.json",
    }
    for name, data in (
        ("retrieval", retrieval),
        ("queue", queue),
        ("manifest", manifest),
        ("benchmark", benchmark),
    ):
        paths[name].write_bytes(data)

    arguments = [
        "evaluate",
        "retrieval",
        "--retrieval-output",
        str(paths["retrieval"]),
        "--annotation-queue",
        str(paths["queue"]),
        "--grounding-manifest",
        str(paths["manifest"]),
        "--grounding-benchmark",
        str(paths["benchmark"]),
        "--report",
        str(paths["report"]),
    ]
    first_exit = main(arguments)
    first_output = capsys.readouterr()
    second_exit = main(arguments)
    second_output = capsys.readouterr()

    assert first_exit == second_exit == 0
    assert first_output.out.startswith(
        "RETRIEVAL EVALUATION COMPLETE benchmark=60 evaluable=45 sha256:"
    )
    assert first_output == second_output
    assert json.loads(paths["report"].read_bytes())["metrics"]["recall_at_10"] == pytest.approx(
        2 / 3
    )
