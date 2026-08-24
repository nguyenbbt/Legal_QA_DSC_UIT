from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import Evidence, QuestionRecord
from legal_rag.generation.qwen3 import Qwen3LegalGenerator


class _Backend:
    model_id = "fixture/generator"
    model_revision = "revision-1"

    def __init__(self) -> None:
        self.evidence: tuple[str, ...] = ()

    def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
        assert system_prompt
        assert question
        self.evidence = evidence
        return "Câu trả lời thử nghiệm."


def _question() -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "1",
            "original_id": "1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/questions.json",
            "source_checksum": checksum_bytes(b"questions"),
            "question": "Câu hỏi thử nghiệm?",
            "answer": None,
            "answer_state": "unlabeled",
        }
    )


def _evidence(evidence_id: str, rank: int, text: str) -> Evidence:
    return Evidence.model_validate(
        {
            "schema_version": "internal.evidence.v1",
            "evidence_id": evidence_id,
            "context_id": "1",
            "source_url": "https://example.invalid/legal",
            "hierarchy_path": ("Điều 1",),
            "canonical_start": 0,
            "canonical_end": len(text),
            "display_text": text,
            "retrieval_text": text,
            "rank": rank,
            "component_scores": {
                "exact_reference_match": False,
                "sparse_score": None,
                "dense_score": 0.5,
                "reranker_score": None,
            },
            "chunk_checksum": "sha256:" + "1" * 64,
            "context_checksum": "sha256:" + "2" * 64,
            "integrity_status": "valid",
            "claim_support": "unknown",
            "version_validity": "unknown",
        }
    )


def test_generator_passes_only_bounded_ranked_evidence() -> None:
    backend = _Backend()
    generator = Qwen3LegalGenerator(backend, maximum_evidence_count=2)

    answer = generator.generate(
        _question(),
        (_evidence("e3", 3, "ba"), _evidence("e1", 1, "một"), _evidence("e2", 2, "hai")),
    )

    assert backend.evidence == ("một", "hai")
    assert answer.used_evidence_ids == ("e1", "e2")
    assert answer.answer_text == "Câu trả lời thử nghiệm."


def test_generator_fails_closed_without_evidence_and_does_not_call_backend() -> None:
    backend = _Backend()

    answer = Qwen3LegalGenerator(backend).generate(_question(), ())

    assert answer.used_evidence_ids == ()
    assert answer.answer_text.startswith("Không đủ căn cứ")
    assert backend.evidence == ()
