"""Offline alias proposals that cannot enter the active runtime without review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes, write_immutable_chunks
from legal_rag.domain.checksums import canonical_json_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.retrieval.exact import document_number_key, find_document_numbers

ALIAS_PROPOSAL_RULE_ID = "passage-header-unique-document-number.v1"
_HEADER_CODE_POINT_LIMIT = 256


class AliasProposalError(Exception):
    """Stable failure while creating a local alias review queue."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AliasProposalSummary:
    context_count: int
    indexable_context_count: int
    quarantined_context_count: int
    proposal_count: int
    multiple_candidate_context_count: int
    proposals_checksum: str
    report_checksum: str


@dataclass(frozen=True, slots=True)
class _ProposalBuild:
    proposals: list[dict[str, object]]
    context_count: int
    indexable_context_count: int
    quarantined_context_count: int
    multiple_candidate_context_count: int


def _proposal_id(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"aliasprop_{digest[:24]}"


def _read_contexts(path: Path) -> _ProposalBuild:
    proposals: list[dict[str, object]] = []
    context_count = 0
    indexable_count = 0
    quarantined_count = 0
    multiple_candidate_count = 0
    seen_context_ids: set[str] = set()
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    context = parse_record_json(
                        line,
                        ContextRecord,
                        artifact_path="contexts.jsonl",
                        record_identity=str(line_number),
                    )
                except RecordValidationError as error:
                    issue = error.issues[0]
                    raise AliasProposalError(
                        "ALIAS_PROPOSAL_CONTEXT_INVALID", issue.message
                    ) from error
                if context.source_position != line_number - 1:
                    raise AliasProposalError(
                        "ALIAS_PROPOSAL_SOURCE_ORDER_INVALID",
                        "context source positions must be consecutive JSONL order",
                    )
                if context.context_id in seen_context_ids:
                    raise AliasProposalError(
                        "ALIAS_PROPOSAL_CONTEXT_DUPLICATE",
                        "context JSONL contains a duplicate context ID",
                    )
                seen_context_ids.add(context.context_id)
                context_count += 1
                if not context.indexable:
                    quarantined_count += 1
                    continue
                indexable_count += 1
                matches = find_document_numbers(context.passage[:_HEADER_CODE_POINT_LIMIT])
                distinct_keys = {document_number_key(match.document_number) for match in matches}
                if len(distinct_keys) > 1:
                    multiple_candidate_count += 1
                    continue
                if not matches:
                    continue
                match = matches[0]
                identity = {
                    "proposal_rule_id": ALIAS_PROPOSAL_RULE_ID,
                    "document_number": match.document_number,
                    "document_number_key": document_number_key(match.document_number),
                    "context_id": context.context_id,
                    "source_kind": "passage_header",
                    "canonical_start": match.canonical_start,
                    "canonical_end": match.canonical_end,
                }
                proposals.append(
                    {
                        "schema_version": "legal.reference.alias.proposal.v1",
                        "proposal_id": _proposal_id(identity),
                        **identity,
                        "review_state": "draft",
                    }
                )
    except OSError as error:
        raise AliasProposalError(
            "ALIAS_PROPOSAL_CONTEXT_SOURCE_INVALID", "context JSONL cannot be read"
        ) from error
    proposals.sort(
        key=lambda row: (
            str(row["document_number_key"]).encode("utf-8"),
            int(str(row["context_id"])),
            str(row["document_number"]).encode("utf-8"),
        )
    )
    return _ProposalBuild(
        proposals=proposals,
        context_count=context_count,
        indexable_context_count=indexable_count,
        quarantined_context_count=quarantined_count,
        multiple_candidate_context_count=multiple_candidate_count,
    )


def write_alias_proposals(
    *,
    contexts_path: Path,
    proposals_path: Path,
    report_path: Path,
    corpus_checksum: str,
) -> AliasProposalSummary:
    """Create immutable draft proposals and a metadata-only review report."""

    build = _read_contexts(contexts_path)
    proposals_checksum = write_immutable_chunks(
        proposals_path,
        (
            (
                json.dumps(
                    proposal,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for proposal in build.proposals
        ),
    )
    report_checksum = write_immutable_bytes(
        report_path,
        canonical_json_bytes(
            {
                "schema_version": "legal.reference.alias.proposal.report.v1",
                "proposal_rule_id": ALIAS_PROPOSAL_RULE_ID,
                "header_code_point_limit": _HEADER_CODE_POINT_LIMIT,
                "corpus_checksum": corpus_checksum,
                "context_count": build.context_count,
                "indexable_context_count": build.indexable_context_count,
                "quarantined_context_count": build.quarantined_context_count,
                "proposal_count": len(build.proposals),
                "multiple_candidate_context_count": build.multiple_candidate_context_count,
                "proposals_checksum": proposals_checksum,
                "activation_state": "owner_review_required",
            }
        ),
    )
    return AliasProposalSummary(
        context_count=build.context_count,
        indexable_context_count=build.indexable_context_count,
        quarantined_context_count=build.quarantined_context_count,
        proposal_count=len(build.proposals),
        multiple_candidate_context_count=build.multiple_candidate_context_count,
        proposals_checksum=proposals_checksum,
        report_checksum=report_checksum,
    )
