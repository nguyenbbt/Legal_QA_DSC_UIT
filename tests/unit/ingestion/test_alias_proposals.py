from __future__ import annotations

import json
from pathlib import Path

from legal_rag.cli import main
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.aliases import write_alias_proposals
from legal_rag.ingestion.organizer import ContextImport, ContextImportEntry


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
