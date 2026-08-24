from __future__ import annotations

import json

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.training.evidence_mining import (
    EvidenceMiningConfig,
    mine_evidence_selections,
)


class _Backend:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        assert query == "Mức phạt là 10 triệu đồng."
        return tuple(0.99 if "10 triệu" in document else 0.20 for document in documents)


class _CoverageAwareBackend:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        scores = {"A": 0.99, "B": 0.98, "C": 0.97, "FULL": 0.96}
        return tuple(scores[document] for document in documents)


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id,
        chunk_id,
        "https://example.invalid",
        ("Điều 1",),
        "ARTICLE",
        "article",
        "1",
        0,
        len(text),
        text,
        text,
        0,
        checksum_bytes(text.encode()),
        checksum_bytes(chunk_id.encode()),
    )


def _question() -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q1",
            "original_id": "q1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/train.questions.jsonl",
            "source_checksum": checksum_bytes(b"source"),
            "question": "Mức phạt là bao nhiêu?",
            "answer": "Mức phạt là 10 triệu đồng.",
            "answer_state": "gold",
        }
    )


def test_miner_retains_only_high_confidence_official_evidence() -> None:
    supported = _chunk("a", "Theo Điều 1, mức phạt là 10 triệu đồng.")
    distractor = _chunk("b", "Điều 2 quy định thời hạn 30 ngày.")

    result = mine_evidence_selections(
        questions=(_question(),),
        split_by_question={"q1": "train"},
        retrieve=lambda _: (
            RetrievalCandidate(distractor, False, 2.0),
            RetrievalCandidate(supported, False, 1.0),
        ),
        backend=_Backend(),
        config=EvidenceMiningConfig(
            minimum_support_score=0.95,
            minimum_answer_token_coverage=0.8,
            maximum_candidates=8,
            maximum_evidence=3,
        ),
    )

    row = json.loads(result.selection_data)
    assert row["question_id"] == "q1"
    assert row["evidence_ids"] == ["a"]
    assert result.report.accepted_rows == 1
    assert result.report.rejected_rows == 0


def test_miner_reports_unsupported_rows_without_emitting_training_text() -> None:
    distractor = _chunk("b", "Điều 2 quy định thời hạn 30 ngày.")

    result = mine_evidence_selections(
        questions=(_question(),),
        split_by_question={"q1": "train"},
        retrieve=lambda _: (RetrievalCandidate(distractor, False, 2.0),),
        backend=_Backend(),
        config=EvidenceMiningConfig(
            minimum_support_score=0.95,
            minimum_answer_token_coverage=0.8,
            maximum_candidates=8,
            maximum_evidence=3,
        ),
    )

    assert result.selection_data == b""
    assert result.report.accepted_rows == 0
    assert result.report.rejected_by_reason == (("support_below_threshold", 1),)


def test_miner_chooses_supported_evidence_by_marginal_answer_coverage() -> None:
    candidates = tuple(
        RetrievalCandidate(_chunk(chunk_id, text), False, 1.0)
        for chunk_id, text in (
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
            ("full", "FULL"),
        )
    )
    full = candidates[-1].chunk
    candidates = (
        *candidates[:-1],
        RetrievalCandidate(
            ChunkRecord(
                full.chunk_id,
                full.context_id,
                full.source_url,
                full.hierarchy_path,
                full.hierarchy_rule_id,
                full.hierarchy_kind,
                full.hierarchy_ordinal,
                full.canonical_start,
                len("Mức phạt là 10 triệu đồng."),
                "Mức phạt là 10 triệu đồng.",
                "FULL",
                full.window_index,
                checksum_bytes("Mức phạt là 10 triệu đồng.".encode()),
                full.context_checksum,
            ),
            False,
            1.0,
        ),
    )

    result = mine_evidence_selections(
        questions=(_question(),),
        split_by_question={"q1": "train"},
        retrieve=lambda _: candidates,
        backend=_CoverageAwareBackend(),
        config=EvidenceMiningConfig(
            minimum_support_score=0.95,
            minimum_answer_token_coverage=0.8,
            maximum_candidates=3,
            maximum_evidence=3,
        ),
    )

    assert json.loads(result.selection_data)["evidence_ids"] == ["full"]
