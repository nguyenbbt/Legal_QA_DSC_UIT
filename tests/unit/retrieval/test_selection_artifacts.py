from __future__ import annotations

import json

from legal_rag.domain.checksums import content_json_bytes
from legal_rag.retrieval.selection_artifacts import build_evidence_selection_artifacts


def test_selection_artifact_joins_ranked_ids_and_is_byte_stable() -> None:
    queue = content_json_bytes(
        {
            "question_id": "q",
            "question_checksum": "sha256:" + "a" * 64,
            "question": "question",
            "candidates": [
                {
                    "evidence_id": "a",
                    "context_id": "1",
                    "hierarchy_path": ["Điều 1"],
                    "canonical_start": 0,
                    "canonical_end": 100,
                    "display_text": "A",
                },
                {
                    "evidence_id": "b",
                    "context_id": "1",
                    "hierarchy_path": ["Điều 1", "Khoản 1"],
                    "canonical_start": 10,
                    "canonical_end": 90,
                    "display_text": "B",
                },
                {
                    "evidence_id": "c",
                    "context_id": "1",
                    "hierarchy_path": ["Điều 2"],
                    "canonical_start": 101,
                    "canonical_end": 150,
                    "display_text": "C",
                },
            ],
        }
    )
    ranking = content_json_bytes(
        {
            "question_id": "q",
            "question_checksum": "sha256:" + "a" * 64,
            "candidates": [
                {"evidence_id": "a"},
                {"evidence_id": "b"},
                {"evidence_id": "c"},
            ],
        }
    )
    kwargs = {
        "annotation_queue_data": queue,
        "retrieval_output_data": ranking,
        "source_run_id": "R0",
        "selected_run_id": "P1",
        "maximum_input_tokens": 100,
        "token_counter": lambda question, evidence: 10 + 20 * len(evidence),
    }

    first = build_evidence_selection_artifacts(**kwargs)
    second = build_evidence_selection_artifacts(**kwargs)

    assert first == second
    output = json.loads(first.retrieval_output)
    assert [row["evidence_id"] for row in output["candidates"]] == ["a", "c"]
    report = json.loads(first.selection_report)
    assert report["selector_version"] == "evidence-set-selector.v2"
    assert report["selected_count_distribution"] == {"2": 1}
    assert report["reason_counts"]["SKIP_PARENT_CHILD_REDUNDANCY"] == 1


def test_selection_artifact_freezes_relative_score_policy() -> None:
    queue = content_json_bytes(
        {
            "question_id": "q",
            "question_checksum": "sha256:" + "a" * 64,
            "question": "question",
            "candidates": [
                {
                    "evidence_id": evidence_id,
                    "context_id": evidence_id,
                    "hierarchy_path": [evidence_id],
                    "canonical_start": index * 10,
                    "canonical_end": index * 10 + 9,
                    "display_text": evidence_id,
                    "sparse_score": score,
                }
                for index, (evidence_id, score) in enumerate((("a", 10.0), ("b", 6.0), ("c", 5.0)))
            ],
        }
    )
    ranking = content_json_bytes(
        {
            "question_id": "q",
            "question_checksum": "sha256:" + "a" * 64,
            "candidates": [{"evidence_id": value} for value in ("a", "b", "c")],
        }
    )

    artifacts = build_evidence_selection_artifacts(
        annotation_queue_data=queue,
        retrieval_output_data=ranking,
        source_run_id="R0",
        selected_run_id="P3A",
        maximum_input_tokens=100,
        token_counter=lambda question, evidence: 10 * len(evidence),
        minimum_relative_sparse_score=0.6,
        calibration_checksum="sha256:" + "b" * 64,
    )

    output = json.loads(artifacts.retrieval_output)
    assert [row["evidence_id"] for row in output["candidates"]] == ["a", "b"]
    report = json.loads(artifacts.selection_report)
    assert report["minimum_relative_sparse_score"] == 0.6
    assert report["calibration_checksum"] == "sha256:" + "b" * 64
    assert report["selected_count_distribution"] == {"2": 1}
