from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.generation.qwen3 import Qwen3LegalGenerator
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.pipeline.model import run_model_questions
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.exact import AliasIndex
from legal_rag.retrieval.models import RetrievalCandidate


class _Index:
    def __init__(self, candidate: RetrievalCandidate) -> None:
        self._candidate = candidate

    def retrieve(self, query: str) -> SparseRetrievalResult:
        return SparseRetrievalResult(
            query, ("query",), (self._candidate,), (), "sha256:" + "3" * 64
        )

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return ()

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return ()


class _Reranker:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return (1.0,) * len(documents)


class _GeneratorBackend:
    model_id = "fixture/generator"
    model_revision = "revision-1"

    def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
        return evidence[0]


def test_model_pipeline_preserves_question_and_evidence_identity() -> None:
    text = "Căn cứ thử nghiệm."
    chunk = ChunkRecord(
        "chunk_a",
        "1",
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
        checksum_bytes(b"context"),
    )
    candidate = RetrievalCandidate(chunk, False, 1.0)
    question = QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q1",
            "original_id": "q1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/questions.json",
            "source_checksum": checksum_bytes(b"questions"),
            "question": "Nội dung thử nghiệm là gì?",
            "answer": None,
            "answer_state": "unlabeled",
        }
    )
    aliases = AliasIndex(
        (),
        checksum_bytes(b"corpus"),
        "fixtures/aliases.jsonl",
        checksum_bytes(b"aliases"),
    )

    answers = run_model_questions(
        (question,),
        index=_Index(candidate),
        aliases=aliases,
        reranker=_Reranker(),
        generator=Qwen3LegalGenerator(_GeneratorBackend()),
        run_id="run_" + "1" * 24,
    )

    assert answers[0].question_id == "q1"
    assert answers[0].answer == text
    assert answers[0].evidence_ids == ("chunk_a",)
