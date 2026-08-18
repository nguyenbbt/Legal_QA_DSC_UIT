"""Offline alias proposals that cannot enter the active runtime without review."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from legal_rag.domain.artifacts import write_immutable_bytes, write_immutable_chunks
from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import (
    CanonicalIntegerString,
    ContextRecord,
    FrozenStrictModel,
    NfcString,
    NonNegativeInt,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.retrieval.exact import (
    DOCUMENT_KEY_VERSION,
    LegalReferenceAlias,
    document_number_key,
    find_document_numbers,
)

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
class AliasFreezeSummary:
    alias_count: int
    proposals_checksum: str
    aliases_checksum: str
    manifest_checksum: str
    report_checksum: str


class AliasProposalRecord(FrozenStrictModel, frozen=True):
    schema_version: Literal["legal.reference.alias.proposal.v1"]
    proposal_id: NfcString
    proposal_rule_id: Literal["passage-header-unique-document-number.v1"]
    document_number: NfcString
    document_number_key: NfcString
    context_id: CanonicalIntegerString
    source_kind: Literal["passage_header"]
    canonical_start: NonNegativeInt
    canonical_end: NonNegativeInt
    review_state: Literal["draft"]

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        identity = {
            "proposal_rule_id": self.proposal_rule_id,
            "document_number": self.document_number,
            "document_number_key": self.document_number_key,
            "context_id": self.context_id,
            "source_kind": self.source_kind,
            "canonical_start": self.canonical_start,
            "canonical_end": self.canonical_end,
        }
        if self.document_number_key != document_number_key(self.document_number):
            raise ValueError("document_number_key does not match its canonical identity")
        if self.canonical_start >= self.canonical_end:
            raise ValueError("proposal offsets must form a non-empty canonical span")
        if self.proposal_id != _proposal_id(identity):
            raise ValueError("proposal_id does not match the proposal identity")
        return self


class AliasProposalReport(FrozenStrictModel, frozen=True):
    schema_version: Literal["legal.reference.alias.proposal.report.v1"]
    proposal_rule_id: Literal["passage-header-unique-document-number.v1"]
    header_code_point_limit: Literal[256]
    corpus_checksum: Sha256
    context_count: NonNegativeInt
    indexable_context_count: NonNegativeInt
    quarantined_context_count: NonNegativeInt
    proposal_count: NonNegativeInt
    multiple_candidate_context_count: NonNegativeInt
    proposals_checksum: Sha256
    activation_state: Literal["owner_review_required"]


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


def _read_proposal_artifacts(
    proposals_path: Path,
    report_path: Path,
) -> tuple[bytes, AliasProposalReport, tuple[AliasProposalRecord, ...]]:
    try:
        proposals_data = proposals_path.read_bytes()
        report_data = report_path.read_bytes()
    except OSError as error:
        raise AliasProposalError(
            "ALIAS_PROPOSAL_SOURCE_INVALID", "proposal artifacts cannot be read"
        ) from error
    try:
        report = parse_record_json(
            report_data,
            AliasProposalReport,
            artifact_path="aliases.proposed.report.json",
            record_identity="report",
        )
        records = tuple(
            parse_record_json(
                line,
                AliasProposalRecord,
                artifact_path="aliases.proposed.jsonl",
                record_identity=str(line_number),
            )
            for line_number, line in enumerate(proposals_data.splitlines(keepends=True), start=1)
        )
    except RecordValidationError as error:
        issue = error.issues[0]
        raise AliasProposalError("ALIAS_PROPOSAL_SCHEMA_INVALID", issue.message) from error
    return proposals_data, report, records


def _active_aliases(records: tuple[AliasProposalRecord, ...]) -> tuple[LegalReferenceAlias, ...]:
    aliases = tuple(
        LegalReferenceAlias.model_validate(
            {
                "schema_version": "legal.reference.alias.v1",
                "document_number": record.document_number,
                "document_number_key": record.document_number_key,
                "context_id": record.context_id,
                "source_kind": record.source_kind,
                "canonical_start": record.canonical_start,
                "canonical_end": record.canonical_end,
                "review_state": "approved",
            }
        )
        for record in records
    )
    order = lambda alias: (  # noqa: E731 - exact contract order
        alias.document_number_key.encode("utf-8"),
        int(alias.context_id),
        alias.document_number.encode("utf-8"),
    )
    if aliases != tuple(sorted(aliases, key=order)):
        raise AliasProposalError(
            "ALIAS_PROPOSAL_ORDER_INVALID", "proposal records are not in canonical order"
        )
    if len(aliases) != len(set(aliases)):
        raise AliasProposalError(
            "ALIAS_PROPOSAL_DUPLICATE", "proposal artifact contains a duplicate alias"
        )
    return aliases


def _validate_alias_provenance(
    contexts_path: Path,
    aliases: tuple[LegalReferenceAlias, ...],
) -> None:
    aliases_by_context: dict[str, list[LegalReferenceAlias]] = {}
    for alias in aliases:
        aliases_by_context.setdefault(alias.context_id, []).append(alias)
    validated_contexts: set[str] = set()
    try:
        with contexts_path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                context = parse_record_json(
                    line,
                    ContextRecord,
                    artifact_path="contexts.jsonl",
                    record_identity=str(line_number),
                )
                context_aliases = aliases_by_context.get(context.context_id, ())
                if not context_aliases:
                    continue
                if not context.indexable:
                    raise AliasProposalError(
                        "ALIAS_PROPOSAL_CONTEXT_INVALID",
                        "approved alias points to a quarantined context",
                    )
                for alias in context_aliases:
                    assert alias.canonical_start is not None
                    assert alias.canonical_end is not None
                    if (
                        alias.canonical_end > len(context.passage)
                        or context.passage[alias.canonical_start : alias.canonical_end]
                        != alias.document_number
                    ):
                        raise AliasProposalError(
                            "ALIAS_PROPOSAL_PROVENANCE_INVALID",
                            "approved alias offsets do not reconstruct the document number",
                        )
                validated_contexts.add(context.context_id)
    except RecordValidationError as error:
        issue = error.issues[0]
        raise AliasProposalError("ALIAS_PROPOSAL_CONTEXT_INVALID", issue.message) from error
    except OSError as error:
        raise AliasProposalError(
            "ALIAS_PROPOSAL_CONTEXT_SOURCE_INVALID", "context JSONL cannot be read"
        ) from error
    if validated_contexts != set(aliases_by_context):
        raise AliasProposalError(
            "ALIAS_PROPOSAL_CONTEXT_INVALID",
            "approved alias does not resolve to an active corpus context",
        )


def _alias_jsonl_bytes(aliases: tuple[LegalReferenceAlias, ...]) -> bytes:
    return b"".join(canonical_json_bytes(alias.model_dump(mode="json")) for alias in aliases)


def freeze_alias_proposals(
    *,
    contexts_path: Path,
    proposals_path: Path,
    proposal_report_path: Path,
    aliases_path: Path,
    manifest_path: Path,
    report_path: Path,
    expected_proposals_checksum: str,
    expected_proposal_count: int,
) -> AliasFreezeSummary:
    """Activate exactly one owner-approved proposal artifact after provenance validation."""

    proposals_data, proposal_report, proposal_records = _read_proposal_artifacts(
        proposals_path, proposal_report_path
    )
    proposals_checksum = checksum_bytes(proposals_data)
    if (
        proposals_checksum != expected_proposals_checksum
        or proposal_report.proposals_checksum != expected_proposals_checksum
    ):
        raise AliasProposalError(
            "ALIAS_PROPOSAL_CHECKSUM_MISMATCH",
            "proposal artifact does not match the owner-approved checksum",
        )
    if (
        len(proposal_records) != expected_proposal_count
        or proposal_report.proposal_count != expected_proposal_count
    ):
        raise AliasProposalError(
            "ALIAS_PROPOSAL_COUNT_MISMATCH",
            "proposal artifact does not match the owner-approved record count",
        )
    aliases = _active_aliases(proposal_records)
    _validate_alias_provenance(contexts_path, aliases)
    alias_data = _alias_jsonl_bytes(aliases)
    aliases_checksum = checksum_bytes(alias_data)
    artifact_path = aliases_path.name
    manifest_data = canonical_json_bytes(
        {
            "schema_version": "legal.reference.alias.manifest.v1",
            "document_key_version": DOCUMENT_KEY_VERSION,
            "unicode_version": unicodedata.unidata_version,
            "corpus_checksum": proposal_report.corpus_checksum,
            "ordered_files": [{"path": artifact_path, "checksum": aliases_checksum}],
            "record_count": len(aliases),
            "aggregate_checksum": aliases_checksum,
        }
    )
    manifest_checksum = checksum_bytes(manifest_data)
    no_candidate_count = (
        proposal_report.indexable_context_count
        - proposal_report.proposal_count
        - proposal_report.multiple_candidate_context_count
    )
    report_data = canonical_json_bytes(
        {
            "schema_version": "legal.reference.alias.activation.report.v1",
            "proposal_rule_id": proposal_report.proposal_rule_id,
            "corpus_checksum": proposal_report.corpus_checksum,
            "approved_proposals_checksum": proposals_checksum,
            "approved_proposal_count": len(aliases),
            "ambiguous_context_count": proposal_report.multiple_candidate_context_count,
            "no_candidate_context_count": no_candidate_count,
            "quarantined_context_count": proposal_report.quarantined_context_count,
            "aliases_checksum": aliases_checksum,
            "alias_manifest_checksum": manifest_checksum,
            "activation_state": "active",
        }
    )
    write_immutable_bytes(aliases_path, alias_data)
    write_immutable_bytes(manifest_path, manifest_data)
    report_checksum = write_immutable_bytes(report_path, report_data)
    return AliasFreezeSummary(
        alias_count=len(aliases),
        proposals_checksum=proposals_checksum,
        aliases_checksum=aliases_checksum,
        manifest_checksum=manifest_checksum,
        report_checksum=report_checksum,
    )
