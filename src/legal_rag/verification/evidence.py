"""Two-phase evidence integrity validation and ordered token admission."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes, content_json_bytes
from legal_rag.domain.models import ComponentScores, ContextRecord, Evidence
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.models import RetrievalCandidate


class EvidenceManifestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EvidenceTokenizer(Protocol):
    tokenizer_id: str
    tokenizer_revision: str

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EvidenceSelectionConfig:
    max_evidence: int
    evidence_token_budget: int
    reserve_tokens: int
    template_id: str
    template_revision: str
    template: str
    separator: str

    def __post_init__(self) -> None:
        if not 1 <= self.max_evidence <= 12:
            raise ValueError("max_evidence must be between 1 and 12")
        if self.evidence_token_budget < 0 or self.reserve_tokens < 0:
            raise ValueError("evidence and reserve token budgets must be non-negative")
        if not all(
            value
            for value in (
                self.template_id,
                self.template_revision,
                self.template,
                self.separator,
            )
        ):
            raise ValueError("evidence template metadata and bytes must be non-empty")


@dataclass(frozen=True, slots=True)
class EvidenceDiagnostic:
    evidence_id: str
    original_candidate_rank: int
    token_cost: int
    remaining_budget_before: int
    accepted_token_total_after: int
    decision: Literal["accepted", "rejected"]
    reason: str | None
    template_id: str
    template_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    reserve_tokens: int


@dataclass(frozen=True, slots=True)
class EvidenceValidationResult:
    accepted: tuple[Evidence, ...]
    diagnostics: tuple[EvidenceDiagnostic, ...]

    def diagnostics_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "evidence.diagnostics.v1",
                "items": [
                    {
                        "evidence_id": item.evidence_id,
                        "original_candidate_rank": item.original_candidate_rank,
                        "token_cost": item.token_cost,
                        "remaining_budget_before": item.remaining_budget_before,
                        "accepted_token_total_after": item.accepted_token_total_after,
                        "decision": item.decision,
                        "reason": item.reason,
                        "template_id": item.template_id,
                        "template_revision": item.template_revision,
                        "tokenizer_id": item.tokenizer_id,
                        "tokenizer_revision": item.tokenizer_revision,
                        "reserve_tokens": item.reserve_tokens,
                    }
                    for item in self.diagnostics
                ],
            }
        )


def _manifest_failure(message: str) -> EvidenceManifestError:
    return EvidenceManifestError("EVIDENCE_MANIFEST_INTEGRITY", message)


def _unique_contexts(contexts: tuple[ContextRecord, ...]) -> dict[str, ContextRecord]:
    by_id: dict[str, ContextRecord] = {}
    for context in contexts:
        if context.context_id in by_id:
            raise _manifest_failure("active context manifest contains a duplicate ID")
        by_id[context.context_id] = context
    return by_id


def _unique_chunks(chunks: tuple[ChunkRecord, ...]) -> dict[str, ChunkRecord]:
    by_id: dict[str, ChunkRecord] = {}
    for chunk in chunks:
        if chunk.chunk_id in by_id:
            raise _manifest_failure("active chunk manifest contains a duplicate ID")
        by_id[chunk.chunk_id] = chunk
    return by_id


def _candidate_order_key(candidate: RetrievalCandidate) -> tuple[int, bool, float, str]:
    if candidate.sparse_score is not None and not math.isfinite(candidate.sparse_score):
        raise _manifest_failure("candidate sparse score is non-finite")
    return (
        -int(candidate.exact_reference_match),
        candidate.sparse_score is None,
        -(candidate.sparse_score or 0.0),
        candidate.chunk.chunk_id,
    )


def _validate_candidate_sequence(candidates: tuple[RetrievalCandidate, ...]) -> None:
    if len(candidates) > 12:
        raise _manifest_failure("incoming evidence candidate sequence exceeds 12")
    ids = tuple(candidate.chunk.chunk_id for candidate in candidates)
    if len(ids) != len(set(ids)):
        raise _manifest_failure("incoming evidence candidate sequence contains duplicates")
    if candidates != tuple(sorted(candidates, key=_candidate_order_key)):
        raise _manifest_failure("incoming evidence candidate order violates RET-004")


def _context_checksum(context: ContextRecord) -> str:
    return checksum_bytes(content_json_bytes(context.model_dump(mode="json")))


def _intrinsic_rejection(
    candidate: RetrievalCandidate,
    contexts: dict[str, ContextRecord],
    chunks: dict[str, ChunkRecord],
) -> tuple[str | None, ChunkRecord | None, ContextRecord | None]:
    active_chunk = chunks.get(candidate.chunk.chunk_id)
    if active_chunk is None:
        return "EVIDENCE_ID_MISSING", None, None
    context = contexts.get(active_chunk.context_id)
    if context is None:
        raise _manifest_failure("active chunk references a missing context manifest entry")
    if not context.indexable or context.quarantine_reason is not None:
        return "EVIDENCE_QUARANTINED", active_chunk, context
    if candidate.chunk != active_chunk:
        raise _manifest_failure("candidate chunk differs from the active chunk manifest")
    if active_chunk.context_checksum != _context_checksum(context):
        raise _manifest_failure("active context checksum does not match the chunk manifest")
    if not (0 <= active_chunk.canonical_start < active_chunk.canonical_end <= len(context.passage)):
        return "EVIDENCE_OFFSET_INVALID", active_chunk, context
    if (
        context.passage[active_chunk.canonical_start : active_chunk.canonical_end]
        != active_chunk.display_text
    ):
        return "EVIDENCE_OFFSET_INVALID", active_chunk, context
    if not active_chunk.display_text.strip():
        return "EVIDENCE_EMPTY", active_chunk, context
    if active_chunk.context_id != context.context_id:
        raise _manifest_failure("chunk and context identity differ")
    return None, active_chunk, context


def _diagnostic(
    *,
    candidate: RetrievalCandidate,
    candidate_rank: int,
    token_cost: int,
    remaining_budget: int,
    accepted_total: int,
    decision: Literal["accepted", "rejected"],
    reason: str | None,
    config: EvidenceSelectionConfig,
    tokenizer: EvidenceTokenizer,
) -> EvidenceDiagnostic:
    return EvidenceDiagnostic(
        evidence_id=candidate.chunk.chunk_id,
        original_candidate_rank=candidate_rank,
        token_cost=token_cost,
        remaining_budget_before=remaining_budget,
        accepted_token_total_after=accepted_total,
        decision=decision,
        reason=reason,
        template_id=config.template_id,
        template_revision=config.template_revision,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        reserve_tokens=config.reserve_tokens,
    )


def validate_and_admit_evidence(
    candidates: tuple[RetrievalCandidate, ...],
    *,
    contexts: tuple[ContextRecord, ...],
    chunks: tuple[ChunkRecord, ...],
    config: EvidenceSelectionConfig,
    tokenizer: EvidenceTokenizer,
) -> EvidenceValidationResult:
    """Validate intrinsic integrity, then greedily admit without reordering."""

    _validate_candidate_sequence(candidates)
    contexts_by_id = _unique_contexts(contexts)
    chunks_by_id = _unique_chunks(chunks)
    accepted: list[Evidence] = []
    diagnostics: list[EvidenceDiagnostic] = []
    accepted_total = 0
    for candidate_rank, candidate in enumerate(candidates, start=1):
        remaining_budget = config.evidence_token_budget - accepted_total
        rejection, active_chunk, context = _intrinsic_rejection(
            candidate,
            contexts_by_id,
            chunks_by_id,
        )
        if rejection is not None:
            diagnostics.append(
                _diagnostic(
                    candidate=candidate,
                    candidate_rank=candidate_rank,
                    token_cost=0,
                    remaining_budget=remaining_budget,
                    accepted_total=accepted_total,
                    decision="rejected",
                    reason=rejection,
                    config=config,
                    tokenizer=tokenizer,
                )
            )
            continue
        assert active_chunk is not None
        assert context is not None
        rendered = (
            config.template.format(
                evidence_id=active_chunk.chunk_id,
                context_id=active_chunk.context_id,
                display_text=active_chunk.display_text,
                rank=candidate_rank,
            )
            + config.separator
        )
        token_cost = tokenizer.count_tokens(rendered)
        if token_cost < 0:
            raise _manifest_failure("evidence tokenizer returned a negative token count")
        if len(accepted) >= config.max_evidence:
            reason = "EVIDENCE_COUNT_LIMIT"
        elif token_cost > remaining_budget:
            reason = "EVIDENCE_OUTSIDE_BUDGET"
        else:
            reason = None
        if reason is not None:
            diagnostics.append(
                _diagnostic(
                    candidate=candidate,
                    candidate_rank=candidate_rank,
                    token_cost=token_cost,
                    remaining_budget=remaining_budget,
                    accepted_total=accepted_total,
                    decision="rejected",
                    reason=reason,
                    config=config,
                    tokenizer=tokenizer,
                )
            )
            continue
        accepted_total += token_cost
        accepted.append(
            Evidence.model_validate(
                {
                    "schema_version": "internal.evidence.v1",
                    "evidence_id": active_chunk.chunk_id,
                    "context_id": active_chunk.context_id,
                    "source_url": active_chunk.source_url,
                    "hierarchy_path": active_chunk.hierarchy_path,
                    "canonical_start": active_chunk.canonical_start,
                    "canonical_end": active_chunk.canonical_end,
                    "display_text": active_chunk.display_text,
                    "retrieval_text": active_chunk.retrieval_text,
                    "rank": len(accepted) + 1,
                    "component_scores": ComponentScores(
                        exact_reference_match=candidate.exact_reference_match,
                        sparse_score=candidate.sparse_score,
                        dense_score=None,
                        reranker_score=None,
                    ),
                    "chunk_checksum": active_chunk.chunk_checksum,
                    "context_checksum": active_chunk.context_checksum,
                    "integrity_status": "valid",
                    "claim_support": "unknown",
                    "version_validity": "unknown",
                }
            )
        )
        diagnostics.append(
            _diagnostic(
                candidate=candidate,
                candidate_rank=candidate_rank,
                token_cost=token_cost,
                remaining_budget=remaining_budget,
                accepted_total=accepted_total,
                decision="accepted",
                reason=None,
                config=config,
                tokenizer=tokenizer,
            )
        )
    return EvidenceValidationResult(tuple(accepted), tuple(diagnostics))
