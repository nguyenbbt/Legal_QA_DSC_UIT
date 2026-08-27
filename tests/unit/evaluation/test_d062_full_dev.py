from __future__ import annotations

import json

from scripts.run_d062_full_dev_retrieval import (
    build_development_generation_inputs,
    load_gold_development_questions,
    rank_frozen_candidates,
)

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.ingestion.chunking import ChunkRecord


class _Reranker:
    model_id = "reranker"
    model_revision = "revision"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        assert query == "Question"
        assert documents == ("first retrieval", "second retrieval")
        return (0.0, 1.0)


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        context_id="1",
        source_url="https://example.com/law",
        hierarchy_path=("Điều 1",),
        hierarchy_rule_id="HIER_ARTICLE",
        hierarchy_kind="article",
        hierarchy_ordinal="1",
        canonical_start=0,
        canonical_end=len(text),
        display_text=text,
        retrieval_text=text + " retrieval",
        window_index=0,
        chunk_checksum="sha256:" + ("2" if chunk_id == "c1" else "3") * 64,
        context_checksum="sha256:" + "4" * 64,
    )


def test_rank_frozen_candidates_uses_only_persisted_candidate_order() -> None:
    row = {
        "candidates": [
            {"evidence_id": "c1", "exact_reference_match": False, "sparse_score": 2.0},
            {"evidence_id": "c2", "exact_reference_match": True, "sparse_score": None},
        ]
    }
    chunks = {"c1": _chunk("c1", "first"), "c2": _chunk("c2", "second")}

    baseline = rank_frozen_candidates("Question", row, chunks, reranker=None, limit=2)
    reranked = rank_frozen_candidates("Question", row, chunks, reranker=_Reranker(), limit=2)

    assert tuple(candidate.chunk.chunk_id for candidate in baseline) == ("c1", "c2")
    assert tuple(candidate.chunk.chunk_id for candidate in reranked) == ("c2", "c1")


def test_load_gold_development_questions_preserves_answers_in_split_order() -> None:
    first = _question("q2", 1, "Second", "Gold 2")
    second = _question("q1", 0, "First", "Gold 1")
    data = b"".join(
        content_json_bytes(question.model_dump(mode="json")) for question in (first, second)
    )

    loaded = load_gold_development_questions(data)

    assert tuple(question.question_id for question in loaded) == ("q1", "q2")
    assert tuple(question.answer for question in loaded) == ("Gold 1", "Gold 2")


def _question(question_id: str, position: int, question: str, answer: str) -> QuestionRecord:
    return QuestionRecord(
        schema_version="internal.question.v1",
        question_id=question_id,
        original_id=question_id,
        original_id_kind="object_key_string",
        source_position=position,
        source_artifact="fixtures/questions.json",
        source_checksum="sha256:" + "1" * 64,
        question=question,
        answer=answer,
        answer_state="gold",
    )


def test_build_development_generation_inputs_binds_gold_to_frozen_evidence() -> None:
    question = QuestionRecord(
        schema_version="internal.question.v1",
        question_id="q1",
        original_id="q1",
        original_id_kind="object_key_string",
        source_position=0,
        source_artifact="fixtures/questions.json",
        source_checksum="sha256:" + "1" * 64,
        question="Question",
        answer="Gold",
        answer_state="gold",
    )
    evidence = content_json_bytes(
        {
            "schema_version": "public.evidence.v1",
            "retrieval_run_id": "d062-r0",
            "retrieval_fingerprint": "sha256:" + "2" * 64,
            "question_id": "q1",
            "question_checksum": checksum_bytes(b"Question"),
            "question": "Question",
            "evidence": [
                {
                    "evidence_id": "chunk-1",
                    "context_id": "c1",
                    "hierarchy_path": ["h1"],
                    "canonical_start": 0,
                    "canonical_end": 8,
                    "display_text": "Evidence",
                    "chunk_checksum": "sha256:" + "3" * 64,
                    "exact_reference_match": False,
                    "sparse_score": 1.0,
                    "reranker_score": None,
                    "rank": 1,
                }
            ],
        }
    )

    queue, retrieval = build_development_generation_inputs((question,), evidence)

    assert json.loads(queue)["gold_answer"] == "Gold"
    assert json.loads(queue)["candidates"][0] == {
        "evidence_id": "chunk-1",
        "display_text": "Evidence",
    }
    assert json.loads(retrieval)["candidates"] == [{"evidence_id": "chunk-1"}]
