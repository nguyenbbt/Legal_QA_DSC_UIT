from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.legal_reranker_contract import (
    LEGAL_EVIDENCE_INSTRUCTION,
    LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM,
)
from legal_rag.evaluation.model_retrieval import (
    ModelRetrievalExperimentError,
    run_labeled_reranker_experiment,
)


class _Backend:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        del query
        return tuple(1.0 if value == "relevant" else 0.0 for value in documents)


def _line(value: object) -> bytes:
    return (json.dumps(value) + "\n").encode()


def test_labeled_reranker_emits_no_private_text_and_exact_metrics() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "private question",
            "question_checksum": "sha256:" + "1" * 64,
            "candidates": [
                {"evidence_id": "wrong", "display_text": "irrelevant"},
                {"evidence_id": "right", "display_text": "relevant"},
            ],
        }
    )
    labels = _line(
        {
            "question_id": "q1",
            "relevant_evidence": [{"evidence_id": "right", "relevance": "relevant"}],
        }
    )

    output, report = run_labeled_reranker_experiment(
        annotation_queue_data=queue,
        grounding_benchmark_data=labels,
        backend=_Backend(),
        run_id="R2-fixture",
    )

    assert b"private question" not in output
    assert json.loads(output)["candidates"][0]["evidence_id"] == "right"
    assert json.loads(report)["metrics"]["recall_at_1"] == 1.0


class _NonFiniteBackend(_Backend):
    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        del query
        return tuple(float("nan") for _ in documents)


def test_labeled_reranker_rejects_duplicate_candidate_identity() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "question",
            "question_checksum": "sha256:" + "1" * 64,
            "candidates": [
                {"evidence_id": "same", "display_text": "one"},
                {"evidence_id": "same", "display_text": "two"},
            ],
        }
    )
    labels = _line({"question_id": "q1", "relevant_evidence": []})

    with pytest.raises(ModelRetrievalExperimentError) as caught:
        run_labeled_reranker_experiment(
            annotation_queue_data=queue,
            grounding_benchmark_data=labels,
            backend=_Backend(),
            run_id="R2-fixture",
        )
    assert caught.value.code == "RERANK_CANDIDATE_DUPLICATE"


def test_labeled_reranker_rejects_nonfinite_scores_and_oversized_limit() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "question",
            "question_checksum": "sha256:" + "1" * 64,
            "candidates": [{"evidence_id": "e1", "display_text": "text"}],
        }
    )
    labels = _line({"question_id": "q1", "relevant_evidence": []})
    with pytest.raises(ModelRetrievalExperimentError) as caught:
        run_labeled_reranker_experiment(
            annotation_queue_data=queue,
            grounding_benchmark_data=labels,
            backend=_NonFiniteBackend(),
            run_id="R2-fixture",
        )
    assert caught.value.code == "RERANK_SCORE_NONFINITE"

    with pytest.raises(ModelRetrievalExperimentError) as caught:
        run_labeled_reranker_experiment(
            annotation_queue_data=queue,
            grounding_benchmark_data=labels,
            backend=_Backend(),
            run_id="R2-fixture",
            candidate_limit=101,
        )
    assert caught.value.code == "RERANK_LIMIT_INVALID"


def test_legal_reranker_instruction_checksum_is_frozen() -> None:
    assert (
        checksum_bytes(LEGAL_EVIDENCE_INSTRUCTION.encode("utf-8"))
        == LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM
    )
    assert LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM == (
        "sha256:2649c60c3bcbf0b01f74569c7edfecc9e488a1b4fa54476f794df8f52cd062b2"
    )
