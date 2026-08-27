"""Frozen legal-evidence instruction for the R-005A reranker experiment."""

from legal_rag.domain.checksums import checksum_bytes

LEGAL_EVIDENCE_INSTRUCTION = (
    "Given a Vietnamese legal question, identify passages that directly provide the controlling "
    "legal rule, conditions, exceptions, procedures, deadlines, amounts, or enumerated items "
    "needed to answer it. Prefer exact article, clause, point, and document matches; reject "
    "keyword-only passages."
)
LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM = checksum_bytes(LEGAL_EVIDENCE_INSTRUCTION.encode("utf-8"))

__all__ = ["LEGAL_EVIDENCE_INSTRUCTION", "LEGAL_EVIDENCE_INSTRUCTION_CHECKSUM"]
