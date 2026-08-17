"""Offline deterministic extractive generator used by the CPU fixture slice."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from legal_rag.domain.models import Evidence, GeneratedAnswer, QuestionRecord
from legal_rag.ingestion.hierarchy import parse_hierarchy

FIXED_REFUSAL = "Không đủ căn cứ trong dữ liệu được cung cấp để trả lời câu hỏi này."

_TERMINATORS = frozenset(".?!…")
_WHITESPACE = re.compile(r"\s+")


def _without_heading_only_first_line(text: str) -> str:
    lines = text.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None or not any(line.strip() for line in lines[first_index + 1 :]):
        return text

    first_line = lines[first_index]
    parsed = parse_hierarchy(first_line)
    if (
        len(parsed.nodes) == 1
        and parsed.nodes[0].canonical_start == 0
        and not parsed.nodes[0].rule_id.startswith("IMPLICIT_")
        and parsed.nodes[0].heading_only
    ):
        del lines[first_index]
        return "\n".join(lines)
    return text


def _extract_answer(display_text: str) -> str:
    canonical = unicodedata.normalize("NFC", display_text)
    collapsed = _WHITESPACE.sub(" ", _without_heading_only_first_line(canonical)).strip()

    for index, character in enumerate(collapsed[:800]):
        if character in _TERMINATORS and (
            index + 1 == len(collapsed) or collapsed[index + 1].isspace()
        ):
            return collapsed[: index + 1]

    prefix = collapsed[:800]
    if any(character.isspace() for character in prefix):
        prefix = prefix.rsplit(maxsplit=1)[0]
    return prefix.strip().rstrip("…") + "…"


class FixtureExtractiveGenerator:
    """The exact network-free ``fixture-extractive-v1`` baseline."""

    generator_id = "fixture-extractive-v1"

    def generate(
        self,
        question: QuestionRecord,
        evidence: Sequence[Evidence],
    ) -> GeneratedAnswer:
        valid = tuple(item for item in evidence if item.integrity_status == "valid")
        if not valid:
            answer_text = FIXED_REFUSAL
            used_evidence_ids: tuple[str, ...] = ()
        else:
            selected = min(valid, key=lambda item: (item.rank, item.evidence_id))
            answer_text = _extract_answer(selected.display_text)
            used_evidence_ids = (selected.evidence_id,)

        return GeneratedAnswer.model_validate(
            {
                "schema_version": "internal.generated_answer.v1",
                "question_id": question.question_id,
                "answer_text": answer_text,
                "generator_id": self.generator_id,
                "competition_policy": "baseline.v1",
                "used_evidence_ids": used_evidence_ids,
                "material_claims": (),
            }
        )
