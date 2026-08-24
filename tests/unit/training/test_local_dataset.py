from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.exact import AliasIndex, LegalReferenceAlias, document_number_key
from legal_rag.training.local_dataset import _answer_exact_candidates


class _Index:
    def __init__(self, chunks: tuple[ChunkRecord, ...]) -> None:
        self._chunks = chunks

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return self._chunks if context_id == "1" else ()


def _chunk(chunk_id: str, text: str, window: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id,
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
        window,
        checksum_bytes(text.encode()),
        checksum_bytes(b"context"),
    )


def test_answer_exact_retrieval_preserves_same_coordinate_fragments() -> None:
    document_number = "08/2022/NQ-HĐND"
    aliases = AliasIndex(
        (
            LegalReferenceAlias.model_validate(
                {
                    "schema_version": "legal.reference.alias.v1",
                    "document_number": document_number,
                    "document_number_key": document_number_key(document_number),
                    "context_id": "1",
                    "source_kind": "owner_override",
                    "canonical_start": None,
                    "canonical_end": None,
                    "review_state": "approved",
                }
            ),
        ),
        checksum_bytes(b"corpus"),
        "aliases.jsonl",
        checksum_bytes(b"aliases"),
    )
    chunks = (_chunk("a", "Nội dung phần một.", 0), _chunk("b", "Nội dung phần hai.", 1))
    question = QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q1",
            "original_id": "q1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/train.questions.jsonl",
            "source_checksum": checksum_bytes(b"questions"),
            "question": "Quy định là gì?",
            "answer": f"Căn cứ Điều 1 của {document_number}, nội dung được quy định.",
            "answer_state": "gold",
        }
    )

    candidates = _answer_exact_candidates(question, index=_Index(chunks), aliases=aliases)  # type: ignore[arg-type]

    assert tuple(candidate.chunk.chunk_id for candidate in candidates) == ("a", "b")
