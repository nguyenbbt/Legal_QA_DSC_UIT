from __future__ import annotations

import json

from legal_rag.evaluation.model_retrieval import run_labeled_reranker_experiment


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
