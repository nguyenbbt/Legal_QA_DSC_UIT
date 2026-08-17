from __future__ import annotations

from dataclasses import replace

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, ChunkRecord, chunk_context
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.verification.evidence import (
    EvidenceManifestError,
    EvidenceSelectionConfig,
    EvidenceTokenizer,
    validate_and_admit_evidence,
)


class CharacterTokenizer(EvidenceTokenizer):
    tokenizer_id = "character-fixture-v1"
    tokenizer_revision = "1"

    def count_tokens(self, text: str) -> int:
        return len(text)


class DoubleCharacterTokenizer(EvidenceTokenizer):
    tokenizer_id = "double-character-fixture-v1"
    tokenizer_revision = "2"

    def count_tokens(self, text: str) -> int:
        return len(text) * 2


def context_and_chunk(context_id: int, passage: str) -> tuple[ContextRecord, ChunkRecord]:
    context = ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": str(context_id),
            "original_id": str(context_id),
            "original_id_kind": "json_integer",
            "source_position": context_id,
            "source_artifact": f"fixtures/context_{context_id}.json",
            "source_checksum": checksum_bytes(passage.encode()),
            "name": None,
            "source_url": f"https://example.invalid/{context_id}",
            "passage": passage,
            "indexable": True,
            "quarantine_reason": None,
        }
    )
    chunk = chunk_context(
        context,
        config=ChunkingConfig(minimum_fragment_tokens=1),
    ).chunks[0]
    return context, chunk


def candidate(chunk: ChunkRecord, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(chunk, False, score)


def selection_config(**changes: object) -> EvidenceSelectionConfig:
    values: dict[str, object] = {
        "max_evidence": 12,
        "evidence_token_budget": 10_000,
        "reserve_tokens": 512,
        "template_id": "evidence-template.v1",
        "template_revision": "1",
        "template": "{evidence_id}|{display_text}",
        "separator": "\n---\n",
    }
    values.update(changes)
    return EvidenceSelectionConfig(**values)  # type: ignore[arg-type]


def test_integrity_valid_candidates_are_admitted_in_original_order() -> None:
    first_context, first_chunk = context_and_chunk(1, "Nội dung thứ nhất.")
    second_context, second_chunk = context_and_chunk(2, "Nội dung thứ hai.")

    result = validate_and_admit_evidence(
        (candidate(first_chunk, 2.0), candidate(second_chunk, 1.0)),
        contexts=(first_context, second_context),
        chunks=(first_chunk, second_chunk),
        config=selection_config(),
        tokenizer=CharacterTokenizer(),
    )

    assert [evidence.evidence_id for evidence in result.accepted] == [
        first_chunk.chunk_id,
        second_chunk.chunk_id,
    ]
    assert [evidence.rank for evidence in result.accepted] == [1, 2]
    assert all(evidence.claim_support == "unknown" for evidence in result.accepted)
    assert all(evidence.version_validity == "unknown" for evidence in result.accepted)
    assert [diagnostic.decision for diagnostic in result.diagnostics] == ["accepted", "accepted"]


def test_missing_quarantined_offset_and_empty_candidates_are_item_local_rejections() -> None:
    context, chunk = context_and_chunk(1, " Nội dung hợp lệ.")
    missing = replace(chunk, chunk_id="chunk_000000000000000000000000")
    quarantined_context = ContextRecord.model_validate(
        {
            **context.model_dump(mode="json"),
            "passage": "",
            "indexable": False,
            "quarantine_reason": "EMPTY_PASSAGE",
        }
    )
    quarantined_chunk = replace(chunk, context_id="2", chunk_id="chunk_111111111111111111111111")
    quarantined_context = quarantined_context.model_copy(
        update={"context_id": "2", "original_id": "2"}
    )
    invalid_offset = replace(
        chunk,
        chunk_id="chunk_222222222222222222222222",
        canonical_end=len(context.passage) + 1,
    )
    empty = replace(
        chunk,
        chunk_id="chunk_333333333333333333333333",
        canonical_start=0,
        canonical_end=1,
        display_text=" ",
    )

    result = validate_and_admit_evidence(
        (
            candidate(missing, 4.0),
            candidate(quarantined_chunk, 3.0),
            candidate(invalid_offset, 2.0),
            candidate(empty, 1.0),
        ),
        contexts=(context, quarantined_context),
        chunks=(quarantined_chunk, invalid_offset, empty),
        config=selection_config(),
        tokenizer=CharacterTokenizer(),
    )

    assert result.accepted == ()
    assert [diagnostic.reason for diagnostic in result.diagnostics] == [
        "EVIDENCE_ID_MISSING",
        "EVIDENCE_QUARANTINED",
        "EVIDENCE_OFFSET_INVALID",
        "EVIDENCE_EMPTY",
    ]


def test_manifest_checksum_mismatch_and_rank_corruption_fail_systemically() -> None:
    first_context, first_chunk = context_and_chunk(1, "Một.")
    second_context, second_chunk = context_and_chunk(2, "Hai.")
    tampered = replace(first_chunk, chunk_checksum=checksum_bytes(b"tampered"))

    with pytest.raises(EvidenceManifestError) as checksum_error:
        validate_and_admit_evidence(
            (candidate(tampered, 2.0),),
            contexts=(first_context,),
            chunks=(first_chunk,),
            config=selection_config(),
            tokenizer=CharacterTokenizer(),
        )
    assert checksum_error.value.code == "EVIDENCE_MANIFEST_INTEGRITY"

    with pytest.raises(EvidenceManifestError) as rank_error:
        validate_and_admit_evidence(
            (candidate(second_chunk, 1.0), candidate(first_chunk, 2.0)),
            contexts=(first_context, second_context),
            chunks=(first_chunk, second_chunk),
            config=selection_config(),
            tokenizer=CharacterTokenizer(),
        )
    assert rank_error.value.code == "EVIDENCE_MANIFEST_INTEGRITY"


def test_candidate_bound_over_twelve_fails_systemically() -> None:
    pairs = tuple(context_and_chunk(index, f"Nội dung {index}.") for index in range(1, 14))
    candidates = tuple(
        candidate(chunk, float(14 - index)) for index, (_, chunk) in enumerate(pairs, 1)
    )

    with pytest.raises(EvidenceManifestError) as captured:
        validate_and_admit_evidence(
            candidates,
            contexts=tuple(context for context, _ in pairs),
            chunks=tuple(chunk for _, chunk in pairs),
            config=selection_config(),
            tokenizer=CharacterTokenizer(),
        )

    assert captured.value.code == "EVIDENCE_MANIFEST_INTEGRITY"


def test_oversized_first_candidate_is_rejected_and_later_candidate_can_fit() -> None:
    first_context, first_chunk = context_and_chunk(1, "Nội dung rất dài cho ngân sách.")
    second_context, second_chunk = context_and_chunk(2, "Ngắn.")
    tokenizer = CharacterTokenizer()
    second_cost = tokenizer.count_tokens(
        f"{second_chunk.chunk_id}|{second_chunk.display_text}\n---\n"
    )

    result = validate_and_admit_evidence(
        (candidate(first_chunk, 2.0), candidate(second_chunk, 1.0)),
        contexts=(first_context, second_context),
        chunks=(first_chunk, second_chunk),
        config=selection_config(evidence_token_budget=second_cost),
        tokenizer=tokenizer,
    )

    assert [diagnostic.reason for diagnostic in result.diagnostics] == [
        "EVIDENCE_OUTSIDE_BUDGET",
        None,
    ]
    assert [evidence.evidence_id for evidence in result.accepted] == [second_chunk.chunk_id]


def test_count_limit_rejects_later_candidates_without_reordering() -> None:
    first_context, first_chunk = context_and_chunk(1, "Một.")
    second_context, second_chunk = context_and_chunk(2, "Hai.")

    result = validate_and_admit_evidence(
        (candidate(first_chunk, 2.0), candidate(second_chunk, 1.0)),
        contexts=(first_context, second_context),
        chunks=(first_chunk, second_chunk),
        config=selection_config(max_evidence=1),
        tokenizer=CharacterTokenizer(),
    )

    assert [diagnostic.reason for diagnostic in result.diagnostics] == [
        None,
        "EVIDENCE_COUNT_LIMIT",
    ]
    assert result.diagnostics_bytes() == result.diagnostics_bytes()


def test_no_fit_and_changed_tokenizer_are_recorded_deterministically() -> None:
    context, chunk = context_and_chunk(1, "Không vừa ngân sách.")
    tokenizer = DoubleCharacterTokenizer()

    result = validate_and_admit_evidence(
        (candidate(chunk, 1.0),),
        contexts=(context,),
        chunks=(chunk,),
        config=selection_config(evidence_token_budget=1),
        tokenizer=tokenizer,
    )

    assert result.accepted == ()
    assert result.diagnostics[0].reason == "EVIDENCE_OUTSIDE_BUDGET"
    assert result.diagnostics[0].tokenizer_id == tokenizer.tokenizer_id
    assert result.diagnostics[0].tokenizer_revision == tokenizer.tokenizer_revision
