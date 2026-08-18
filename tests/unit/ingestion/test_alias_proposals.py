from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.aliases import (
    AliasProposalError,
    freeze_alias_proposals,
    write_alias_proposals,
)
from legal_rag.ingestion.organizer import ContextImport, ContextImportEntry
from legal_rag.retrieval.exact import load_alias_artifact


def _context(position: int, passage: str, *, indexable: bool = True) -> ContextRecord:
    return ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": str(position + 1),
            "original_id": str(position + 1),
            "original_id_kind": "json_integer",
            "source_position": position,
            "source_artifact": f"context_{position + 1}.json",
            "source_checksum": checksum_bytes(passage.encode()),
            "name": None,
            "source_url": f"https://example.invalid/{position + 1}",
            "passage": passage if indexable else "",
            "indexable": indexable,
            "quarantine_reason": None if indexable else "EMPTY_PASSAGE",
        }
    )


def _jsonl(contexts: tuple[ContextRecord, ...]) -> bytes:
    return b"".join(
        (
            json.dumps(
                context.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for context in contexts
    )


def test_alias_proposals_are_draft_offset_backed_and_immutable(tmp_path: Path) -> None:
    passage = "NGHỊ QUYẾT\nSố 08/2022/NQ-HĐND về nội dung thử nghiệm."
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_bytes(_jsonl((_context(0, passage), _context(1, "", indexable=False))))
    proposals = tmp_path / "aliases.proposed.jsonl"
    report = tmp_path / "aliases.proposed.report.json"

    first = write_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        report_path=report,
        corpus_checksum="sha256:" + ("1" * 64),
    )
    second = write_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        report_path=report,
        corpus_checksum="sha256:" + ("1" * 64),
    )

    assert first == second
    row = json.loads(proposals.read_bytes())
    assert row["review_state"] == "draft"
    assert row["source_kind"] == "passage_header"
    assert passage[row["canonical_start"] : row["canonical_end"]] == row["document_number"]
    assert first.proposal_count == 1
    assert first.quarantined_context_count == 1


def test_alias_propose_cli_uses_the_import_manifest_corpus_identity(tmp_path: Path, capsys) -> None:
    context = _context(0, "Số 12/2024/NQ-HĐND về nội dung thử nghiệm.")
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_bytes(_jsonl((context,)))
    import_manifest = tmp_path / "contexts.import.json"
    import_manifest.write_bytes(
        ContextImport(
            records=(context,),
            entries=(
                ContextImportEntry(
                    source_artifact="context_1.json",
                    context_id="1",
                    source_checksum=context.source_checksum,
                    indexable=True,
                    quarantine_reason=None,
                    source_position=0,
                ),
            ),
            warnings=(),
        ).manifest_bytes()
    )
    proposals = tmp_path / "aliases.proposed.jsonl"
    report = tmp_path / "aliases.proposed.report.json"

    exit_code = main(
        [
            "aliases",
            "propose",
            "--contexts",
            str(contexts),
            "--context-manifest",
            str(import_manifest),
            "--output",
            str(proposals),
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("ALIAS PROPOSAL COMPLETE contexts=1 proposals=1 sha256:")
    assert captured.err == ""


def test_alias_freeze_activates_only_the_checksum_approved_proposals(tmp_path: Path) -> None:
    context = _context(0, "Số 12/2024/NQ-HĐND về nội dung thử nghiệm.")
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_bytes(_jsonl((context,)))
    proposals = tmp_path / "aliases.proposed.jsonl"
    proposal_report = tmp_path / "aliases.proposed.report.json"
    proposal_summary = write_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        report_path=proposal_report,
        corpus_checksum="sha256:" + ("1" * 64),
    )
    active = tmp_path / "aliases.active.jsonl"
    manifest = tmp_path / "aliases.active.manifest.json"
    freeze_report = tmp_path / "aliases.active.report.json"

    summary = freeze_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        proposal_report_path=proposal_report,
        aliases_path=active,
        manifest_path=manifest,
        report_path=freeze_report,
        expected_proposals_checksum=proposal_summary.proposals_checksum,
        expected_proposal_count=1,
    )

    row = json.loads(active.read_bytes())
    assert row == {
        "canonical_end": 18,
        "canonical_start": 3,
        "context_id": "1",
        "document_number": "12/2024/NQ-HĐND",
        "document_number_key": "12/2024/nq-hdnd",
        "review_state": "approved",
        "schema_version": "legal.reference.alias.v1",
        "source_kind": "passage_header",
    }
    assert summary.alias_count == 1
    assert summary.proposals_checksum == proposal_summary.proposals_checksum
    loaded = load_alias_artifact(
        active.read_bytes(),
        contexts=(context,),
        corpus_checksum="sha256:" + ("1" * 64),
        artifact_path="aliases.active.jsonl",
    )
    assert manifest.read_bytes() == loaded.manifest_bytes()


def test_alias_freeze_rejects_an_unapproved_proposal_checksum(tmp_path: Path) -> None:
    context = _context(0, "Số 12/2024/NQ-HĐND về nội dung thử nghiệm.")
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_bytes(_jsonl((context,)))
    proposals = tmp_path / "aliases.proposed.jsonl"
    proposal_report = tmp_path / "aliases.proposed.report.json"
    write_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        report_path=proposal_report,
        corpus_checksum="sha256:" + ("1" * 64),
    )

    with pytest.raises(AliasProposalError, match="approved checksum") as mismatch:
        freeze_alias_proposals(
            contexts_path=contexts,
            proposals_path=proposals,
            proposal_report_path=proposal_report,
            aliases_path=tmp_path / "aliases.active.jsonl",
            manifest_path=tmp_path / "aliases.active.manifest.json",
            report_path=tmp_path / "aliases.active.report.json",
            expected_proposals_checksum="sha256:" + ("0" * 64),
            expected_proposal_count=1,
        )

    assert mismatch.value.code == "ALIAS_PROPOSAL_CHECKSUM_MISMATCH"


def test_alias_freeze_cli_requires_the_approved_checksum_and_count(tmp_path: Path, capsys) -> None:
    context = _context(0, "Số 12/2024/NQ-HĐND về nội dung thử nghiệm.")
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_bytes(_jsonl((context,)))
    proposals = tmp_path / "aliases.proposed.jsonl"
    proposal_report = tmp_path / "aliases.proposed.report.json"
    proposed = write_alias_proposals(
        contexts_path=contexts,
        proposals_path=proposals,
        report_path=proposal_report,
        corpus_checksum="sha256:" + ("1" * 64),
    )

    exit_code = main(
        [
            "aliases",
            "freeze",
            "--contexts",
            str(contexts),
            "--proposals",
            str(proposals),
            "--proposal-report",
            str(proposal_report),
            "--expected-proposals-checksum",
            proposed.proposals_checksum,
            "--expected-proposal-count",
            "1",
            "--output",
            str(tmp_path / "aliases.active.jsonl"),
            "--manifest",
            str(tmp_path / "aliases.active.manifest.json"),
            "--report",
            str(tmp_path / "aliases.active.report.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("ALIAS FREEZE COMPLETE aliases=1 sha256:")
    assert captured.err == ""
