from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.retrieval_set_evaluation import (
    RetrievalSetArtifactError,
    evaluate_stored_retrieval_set,
)


def _inputs(*, reranked: bool = False) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    queue_rows = []
    label_rows = []
    ranking_rows = []
    answer_rows = []
    question_ids = [f"q{index:02d}" for index in range(60)]
    for index, question_id in enumerate(question_ids):
        question_checksum = "sha256:" + f"{index + 1:064x}"
        candidates = [
            {
                "evidence_id": f"a{index}",
                "context_id": "1",
                "hierarchy_path": ["Điều 1"],
                "canonical_start": 0,
                "canonical_end": 10,
                "chunk_checksum": "sha256:" + "a" * 64,
                "exact_reference_match": False,
                "sparse_score": 2.0,
                "display_text": "A",
            },
            {
                "evidence_id": f"b{index}",
                "context_id": "1",
                "hierarchy_path": ["Điều 2"],
                "canonical_start": 11,
                "canonical_end": 20,
                "chunk_checksum": "sha256:" + "b" * 64,
                "exact_reference_match": False,
                "sparse_score": 1.0,
                "display_text": "B",
            },
        ]
        queue_rows.append(
            {
                "schema_version": "grounding.annotation.work-item.v1",
                "question_id": question_id,
                "question_checksum": question_checksum,
                "candidates": candidates,
            }
        )
        label_rows.append(
            {
                "schema_version": "grounding.benchmark.v1",
                "question_id": question_id,
                "split": "development",
                "question_checksum": question_checksum,
                "relevant_evidence": [
                    {"evidence_id": f"b{index}", "relevance": "relevant"},
                    {"evidence_id": f"a{index}", "relevance": "partially_relevant"},
                ],
                "required_claims": [],
                "question_answerability": "answerable",
                "temporal_assessment": "valid",
                "label_version": "grounding.v1",
            }
        )
        ordered = reversed(candidates) if reranked else candidates
        ranking_rows.append(
            {
                "schema_version": (
                    "model.retrieval.output.v1" if reranked else "retrieval.output.v1"
                ),
                "question_id": question_id,
                "question_checksum": question_checksum,
                "candidates": [
                    (
                        {"evidence_id": row["evidence_id"], "reranker_score": 0.9 - rank}
                        if reranked
                        else {key: value for key, value in row.items() if key != "display_text"}
                    )
                    for rank, row in enumerate(ordered)
                ],
            }
        )
        answer_rows.append(
            {
                "schema_version": "competition.per_query.v1",
                "question_id": question_id,
                "meteor": index / 100.0,
                "rouge_l": index / 120.0,
            }
        )
    queue = b"".join(content_json_bytes(row) for row in queue_rows)
    labels = b"".join(content_json_bytes(row) for row in label_rows)
    manifest = content_json_bytes(
        {
            "schema_version": "grounding.benchmark.manifest.v1",
            "label_version": "grounding.v1",
            "train_split_checksum": "sha256:" + "d" * 64,
            "development_split_checksum": "sha256:" + "e" * 64,
            "sampling_version": "grounding-sample.v1",
            "sampling_seed": "dsc2026-grounding-sample-v1",
            "chunk_artifact_checksum": "sha256:" + "f" * 64,
            "index_checksum": "sha256:" + "1" * 64,
            "annotation_status": "approved",
            "ordered_question_ids": question_ids,
            "ordered_files": [{"path": "grounding.jsonl", "checksum": checksum_bytes(labels)}],
        }
    )
    ranking = b"".join(content_json_bytes(row) for row in ranking_rows)
    answer = b"".join(content_json_bytes(row) for row in answer_rows)
    return ranking, queue, manifest, labels, answer


@pytest.mark.parametrize("reranked", [False, True])
def test_stored_set_evaluation_joins_canonical_metadata_and_is_deterministic(
    reranked: bool,
) -> None:
    ranking, queue, manifest, labels, answer = _inputs(reranked=reranked)

    first = evaluate_stored_retrieval_set(
        retrieval_output_data=ranking,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=labels,
        answer_per_query_data=answer,
        run_id="R2R" if reranked else "R0",
    )
    second = evaluate_stored_retrieval_set(
        retrieval_output_data=ranking,
        annotation_queue_data=queue,
        benchmark_manifest_data=manifest,
        benchmark_data=labels,
        answer_per_query_data=answer,
        run_id="R2R" if reranked else "R0",
    )

    assert first == second
    report = json.loads(first)
    assert report["schema_version"] == "retrieval.set-evaluation.artifact.v1"
    assert report["metrics"]["questions"][0]["question_id"] == "q00"
    assert report["metrics"]["questions"][0]["token_cost_at_3"] is None
    assert report["metrics"]["questions"][0]["metadata_unavailable_reason"] is None
    assert report["metrics"]["questions"][0]["unique_legal_coordinate_coverage_at_3"] == 1.0
    assert report["metrics"]["questions"][0]["document_context_hit_at_3"] == 1.0
    assert report["metrics"]["evidence_count_distribution_at_3"] == [0, 0, 60, 0]
    expected = 1.0 if reranked else 0.0
    assert report["metrics"]["questions"][0]["required_evidence_coverage_at_1"] == expected


def test_stored_set_evaluation_rejects_ranking_id_outside_queue() -> None:
    ranking, queue, manifest, labels, answer = _inputs(reranked=True)
    tampered = ranking.replace(b'"evidence_id":"b0"', b'"evidence_id":"x0"', 1)

    with pytest.raises(RetrievalSetArtifactError, match="candidate universe") as error:
        evaluate_stored_retrieval_set(
            retrieval_output_data=tampered,
            annotation_queue_data=queue,
            benchmark_manifest_data=manifest,
            benchmark_data=labels,
            answer_per_query_data=answer,
            run_id="R2R",
        )
    assert error.value.code == "RETRIEVAL_SET_CANDIDATE_MISMATCH"
