"""Evidence-bounded Qwen3 generator implementing the shared domain protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from legal_rag.domain.models import Evidence, GeneratedAnswer, QuestionRecord

PROMPT_A = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.
Chỉ trả lời dựa trên các căn cứ được cung cấp. Không suy diễn quy định không có trong căn cứ.
Trả lời trực tiếp, chính xác và súc tích bằng tiếng Việt. Nêu điều, khoản hoặc điểm khi căn cứ
có thông tin đó. Nếu căn cứ không đủ để trả lời, hãy nói rõ rằng chưa đủ căn cứ."""


class LocalGeneratorBackend(Protocol):
    model_id: str
    model_revision: str

    def generate(self, *, system_prompt: str, question: str, evidence: Sequence[str]) -> str: ...


class Qwen3LegalGenerator:
    """Use only integrity-valid evidence in deterministic rank order."""

    generator_id = "qwen3-1.7b-prompt-a-v1"

    def __init__(self, backend: LocalGeneratorBackend, *, maximum_evidence_count: int = 6) -> None:
        if maximum_evidence_count < 1:
            raise ValueError("maximum evidence count must be positive")
        self._backend = backend
        self._maximum_evidence_count = maximum_evidence_count

    def generate(
        self,
        question: QuestionRecord,
        evidence: Sequence[Evidence],
    ) -> GeneratedAnswer:
        valid = tuple(
            sorted(
                (item for item in evidence if item.integrity_status == "valid"),
                key=lambda item: (item.rank, item.evidence_id.encode("utf-8")),
            )[: self._maximum_evidence_count]
        )
        if not valid:
            answer = "Không đủ căn cứ trong dữ liệu được cung cấp để trả lời câu hỏi này."
        else:
            answer = self._backend.generate(
                system_prompt=PROMPT_A,
                question=question.question,
                evidence=tuple(item.display_text for item in valid),
            ).strip()
        if not answer:
            raise ValueError("generator returned an empty answer")
        return GeneratedAnswer.model_validate(
            {
                "schema_version": "internal.generated_answer.v1",
                "question_id": question.question_id,
                "answer_text": answer,
                "generator_id": self.generator_id,
                "competition_policy": "baseline.v1",
                "used_evidence_ids": tuple(item.evidence_id for item in valid),
                "material_claims": (),
            }
        )


__all__ = ["LocalGeneratorBackend", "PROMPT_A", "Qwen3LegalGenerator"]
