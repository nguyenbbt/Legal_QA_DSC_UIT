from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.grounding_labels import load_approved_grounding_benchmark
from legal_rag.grounding.annotation import (
    APPROVAL_CONFIRMATION,
    GroundingAnnotationError,
    QuestionAnnotation,
    annotation_progress_bytes,
    export_grounding_benchmark,
    import_labeled_annotation_queue,
    load_annotation_progress,
    load_annotation_queue,
    new_annotation_progress,
    record_question_annotation,
)


def _queue_bytes(*, suffix: str = "") -> bytes:
    rows: list[bytes] = []
    for index in range(60):
        question_id = f"q{index:02d}"
        rows.append(
            content_json_bytes(
                {
                    "schema_version": "grounding.annotation.work-item.v1",
                    "question_id": question_id,
                    "split": "development",
                    "question_checksum": checksum_bytes(f"Question {index}{suffix}".encode()),
                    "question": f"Question {index}{suffix}",
                    "gold_answer": f"Answer {index}",
                    "split_checksum": checksum_bytes(b"split"),
                    "chunk_artifact_checksum": checksum_bytes(b"chunks"),
                    "index_checksum": checksum_bytes(b"index"),
                    "candidates": [
                        {
                            "evidence_id": f"chunk_{index:02d}",
                            "context_id": str(index),
                            "hierarchy_path": ["root"],
                            "canonical_start": 0,
                            "canonical_end": 10,
                            "chunk_checksum": checksum_bytes(f"chunk {index}".encode()),
                            "exact_reference_match": False,
                            "sparse_score": 1.0,
                            "display_text": f"Evidence {index}",
                        }
                    ],
                    "diagnostics": [],
                    "relevant_evidence": None,
                    "required_claims": None,
                    "question_answerability": None,
                    "temporal_assessment": None,
                    "annotation_state": "pending_primary_annotation",
                }
            )
        )
    return b"".join(rows)


def _complete_progress(queue_data: bytes):
    queue = load_annotation_queue(queue_data)
    progress = new_annotation_progress(queue, annotator_id="owner")
    for index, item in enumerate(queue.items):
        progress = record_question_annotation(
            queue,
            progress,
            QuestionAnnotation(
                question_id=item.question_id,
                evidence_labels=((item.candidates[0].evidence_id, "relevant"),),
                required_claims=(f"Claim {index}",),
                question_answerability="answerable",
                temporal_assessment="unknown",
            ),
        )
    return queue, progress


def _labeled_queue_bytes(queue_data: bytes) -> bytes:
    rows: list[bytes] = []
    for line in queue_data.splitlines():
        row = json.loads(line)
        for index, candidate in enumerate(row["candidates"]):
            candidate["relevance_label"] = ("r", "p", "n")[index % 3]
        rows.append(content_json_bytes(row))
    return b"".join(rows)


def test_import_labeled_queue_maps_short_labels_and_records_provenance() -> None:
    queue_data = _queue_bytes()
    labeled_data = _labeled_queue_bytes(queue_data)
    queue = load_annotation_queue(queue_data)

    imported = import_labeled_annotation_queue(
        queue,
        labeled_data,
        annotator_id="llm-proposal",
    )

    assert imported.defaulted_metadata_question_count == 60
    assert imported.progress.annotation_origin == "imported_labeled_queue"
    assert imported.progress.source_labeled_checksum == checksum_bytes(labeled_data)
    assert imported.progress.metadata_completion == "retrieval_only_defaults"
    assert len(imported.progress.annotations) == 60
    assert imported.progress.annotations[0].evidence_labels == (("chunk_00", "relevant"),)
    assert imported.progress.annotations[0].required_claims == ()
    assert imported.progress.annotations[0].question_answerability == "unknown"
    assert imported.progress.annotations[0].temporal_assessment == "unknown"


def test_import_labeled_queue_rejects_changed_queue_content_and_missing_label() -> None:
    queue_data = _queue_bytes()
    queue = load_annotation_queue(queue_data)
    changed_rows = [json.loads(line) for line in _labeled_queue_bytes(queue_data).splitlines()]
    changed_rows[0]["candidates"][0]["display_text"] = "tampered"
    changed_data = b"".join(content_json_bytes(row) for row in changed_rows)

    with pytest.raises(GroundingAnnotationError) as changed:
        import_labeled_annotation_queue(queue, changed_data, annotator_id="llm-proposal")
    assert changed.value.code == "GROUNDING_LABELED_QUEUE_MISMATCH"

    missing_rows = [json.loads(line) for line in _labeled_queue_bytes(queue_data).splitlines()]
    del missing_rows[0]["candidates"][0]["relevance_label"]
    missing_data = b"".join(content_json_bytes(row) for row in missing_rows)

    with pytest.raises(GroundingAnnotationError) as missing:
        import_labeled_annotation_queue(queue, missing_data, annotator_id="llm-proposal")
    assert missing.value.code == "GROUNDING_ANNOTATION_SCHEMA_INVALID"


def test_import_labeled_cli_writes_private_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "artifacts" / "evaluations" / "grounding"
    private_root.mkdir(parents=True)
    queue_path = private_root / "annotation-work-queue.v1.jsonl"
    labeled_path = private_root / "annotation-work-queue.v1.labeled.jsonl"
    progress_path = private_root / "annotation-progress.v1.json"
    queue_data = _queue_bytes()
    queue_path.write_bytes(queue_data)
    labeled_path.write_bytes(_labeled_queue_bytes(queue_data))
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "grounding",
            "import-labeled",
            "--queue",
            str(queue_path),
            "--labeled-queue",
            str(labeled_path),
            "--progress",
            str(progress_path),
            "--annotator-id",
            "llm-proposal",
        ]
    )

    captured = capsys.readouterr()
    progress = load_annotation_progress(
        progress_path.read_bytes(), load_annotation_queue(queue_data)
    )
    assert exit_code == 0
    assert "GROUNDING LABEL IMPORT COMPLETE questions=60 metadata_defaults=60" in captured.out
    assert captured.err == ""
    assert progress.annotation_origin == "imported_labeled_queue"


def test_import_labeled_cli_does_not_overwrite_different_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "artifacts" / "evaluations" / "grounding"
    private_root.mkdir(parents=True)
    queue_path = private_root / "annotation-work-queue.v1.jsonl"
    labeled_path = private_root / "annotation-work-queue.v1.labeled.jsonl"
    progress_path = private_root / "annotation-progress.v1.json"
    queue_data = _queue_bytes()
    queue_path.write_bytes(queue_data)
    labeled_path.write_bytes(_labeled_queue_bytes(queue_data))
    progress_path.write_bytes(b"existing manual progress")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "grounding",
            "import-labeled",
            "--queue",
            str(queue_path),
            "--labeled-queue",
            str(labeled_path),
            "--progress",
            str(progress_path),
            "--annotator-id",
            "llm-proposal",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "GROUNDING_PROGRESS_EXISTS" in captured.err
    assert progress_path.read_bytes() == b"existing manual progress"


def test_complete_progress_exports_deterministic_approved_benchmark() -> None:
    queue_data = _queue_bytes()
    queue, progress = _complete_progress(queue_data)

    first = export_grounding_benchmark(
        queue,
        progress,
        benchmark_path="grounding_set.v1.jsonl",
        approval_state="approved",
        owner_confirmation=APPROVAL_CONFIRMATION,
    )
    second = export_grounding_benchmark(
        queue,
        progress,
        benchmark_path="grounding_set.v1.jsonl",
        approval_state="approved",
        owner_confirmation=APPROVAL_CONFIRMATION,
    )

    assert first == second
    benchmark_data, manifest_data = first
    approved = load_approved_grounding_benchmark(manifest_data, benchmark_data)
    assert len(approved.records) == 60
    assert approved.manifest.annotation_status == "approved"
    assert approved.manifest.chunk_artifact_checksum == checksum_bytes(b"chunks")
    assert approved.retrieval_labels[0].relevant_evidence_ids == ("chunk_00",)


def test_export_rejects_incomplete_progress_and_unconfirmed_approval() -> None:
    queue = load_annotation_queue(_queue_bytes())
    incomplete = new_annotation_progress(queue, annotator_id="owner")

    with pytest.raises(GroundingAnnotationError) as incomplete_error:
        export_grounding_benchmark(
            queue,
            incomplete,
            benchmark_path="grounding_set.v1.jsonl",
            approval_state="draft",
            owner_confirmation=None,
        )
    assert incomplete_error.value.code == "GROUNDING_ANNOTATION_INCOMPLETE"

    _, complete = _complete_progress(_queue_bytes())
    with pytest.raises(GroundingAnnotationError) as approval_error:
        export_grounding_benchmark(
            queue,
            complete,
            benchmark_path="grounding_set.v1.jsonl",
            approval_state="approved",
            owner_confirmation=None,
        )
    assert approval_error.value.code == "GROUNDING_APPROVAL_CONFIRMATION_REQUIRED"


def test_progress_is_bound_to_the_exact_queue() -> None:
    queue = load_annotation_queue(_queue_bytes())
    progress = new_annotation_progress(queue, annotator_id="owner")
    changed_queue = load_annotation_queue(_queue_bytes(suffix=" changed"))

    with pytest.raises(GroundingAnnotationError) as mismatch:
        load_annotation_progress(annotation_progress_bytes(progress), changed_queue)

    assert mismatch.value.code == "GROUNDING_PROGRESS_QUEUE_MISMATCH"


def test_progress_rejects_inconsistent_import_provenance() -> None:
    queue = load_annotation_queue(_queue_bytes())
    progress = new_annotation_progress(queue, annotator_id="owner")
    payload = json.loads(annotation_progress_bytes(progress))
    payload["annotation_origin"] = "imported_labeled_queue"

    with pytest.raises(GroundingAnnotationError) as invalid:
        load_annotation_progress(content_json_bytes(payload), queue)

    assert invalid.value.code == "GROUNDING_ANNOTATION_SCHEMA_INVALID"


def test_annotate_cli_labels_one_selected_question_and_saves_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "artifacts" / "evaluations" / "grounding"
    private_root.mkdir(parents=True)
    queue_path = private_root / "annotation-work-queue.v1.jsonl"
    progress_path = private_root / "annotation-progress.v1.json"
    queue_data = _queue_bytes()
    queue_path.write_bytes(queue_data)
    responses = iter(("r", "claim one", "a", "u"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "grounding",
            "annotate",
            "--queue",
            str(queue_path),
            "--progress",
            str(progress_path),
            "--annotator-id",
            "owner",
            "--question-id",
            "q00",
        ]
    )

    captured = capsys.readouterr()
    progress = load_annotation_progress(
        progress_path.read_bytes(), load_annotation_queue(queue_data)
    )
    assert exit_code == 0
    assert "completed=1/60" in captured.out
    assert captured.err == ""
    assert progress.annotations[0].question_id == "q00"


def test_annotate_cli_rejects_progress_outside_private_grounding_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_root = tmp_path / "artifacts" / "evaluations" / "grounding"
    private_root.mkdir(parents=True)
    queue_path = private_root / "annotation-work-queue.v1.jsonl"
    queue_path.write_bytes(_queue_bytes())
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "grounding",
            "annotate",
            "--queue",
            str(queue_path),
            "--progress",
            str(tmp_path / "progress.json"),
            "--annotator-id",
            "owner",
            "--question-id",
            "q00",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "GROUNDING_PRIVATE_PATH_REQUIRED" in captured.err
    assert not (tmp_path / "progress.json").exists()


def test_export_cli_writes_owner_approved_private_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "artifacts" / "evaluations" / "grounding"
    private_root.mkdir(parents=True)
    queue_path = private_root / "annotation-work-queue.v1.jsonl"
    progress_path = private_root / "annotation-progress.v1.json"
    benchmark_path = private_root / "grounding_set.v1.jsonl"
    manifest_path = private_root / "grounding.benchmark.manifest.v1.json"
    queue_data = _queue_bytes()
    queue, progress = _complete_progress(queue_data)
    queue_path.write_bytes(queue_data)
    progress_path.write_bytes(annotation_progress_bytes(progress))
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "grounding",
            "export",
            "--queue",
            str(queue_path),
            "--progress",
            str(progress_path),
            "--benchmark",
            str(benchmark_path),
            "--manifest",
            str(manifest_path),
            "--approval-state",
            "approved",
            "--owner-confirmation",
            APPROVAL_CONFIRMATION,
        ]
    )

    captured = capsys.readouterr()
    approved = load_approved_grounding_benchmark(
        manifest_path.read_bytes(), benchmark_path.read_bytes()
    )
    assert exit_code == 0
    assert "GROUNDING EXPORT COMPLETE questions=60 status=approved" in captured.out
    assert captured.err == ""
    assert approved.manifest.ordered_files[0].path == benchmark_path.name
