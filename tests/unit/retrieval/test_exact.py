from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.ingestion.chunking import ChunkingConfig, chunk_context
from legal_rag.retrieval.exact import (
    AliasArtifactError,
    document_number_key,
    load_alias_artifact,
    load_frozen_alias_artifact,
    parse_legal_reference,
    resolve_exact_reference,
)


def context_record(passage: str) -> ContextRecord:
    return ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": "740",
            "original_id": "740",
            "original_id_kind": "json_integer",
            "source_position": 0,
            "source_artifact": "fixtures/context_740.json",
            "source_checksum": checksum_bytes(passage.encode()),
            "name": "Synthetic fixture",
            "source_url": "https://example.invalid/740",
            "passage": passage,
            "indexable": True,
            "quarantine_reason": None,
        }
    )


def alias_bytes(document_number: str = "08/2022/NQ-HĐND") -> bytes:
    record = {
        "schema_version": "legal.reference.alias.v1",
        "document_number": document_number,
        "document_number_key": document_number_key(document_number),
        "context_id": "740",
        "source_kind": "organizer_name",
        "canonical_start": None,
        "canonical_end": None,
        "review_state": "approved",
    }
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def test_document_number_key_preserves_leading_zero_and_folds_accents() -> None:
    assert document_number_key("5868/QĐ-BYT") == "5868/qd-byt"
    assert document_number_key("08/2022/NQ-HĐND") == "08/2022/nq-hdnd"


def test_parser_accepts_coordinate_and_both_document_number_forms() -> None:
    first = parse_legal_reference("Điểm a khoản 1 Điều 2 theo số 5868/QĐ-BYT")
    second = parse_legal_reference("Khoản 03 Điều 04 của 08/2022/NQ-HĐND")

    assert first.reference is not None
    assert first.reference.point == "a"
    assert first.reference.clause == "1"
    assert first.reference.article == "2"
    assert first.reference.document_number == "5868/qđ-byt"
    assert second.reference is not None
    assert second.reference.document_number == "08/2022/nq-hđnd"


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("Khoản 1 điểm a Điều 2", "EXACT_REFERENCE_MALFORMED"),
        ("Điểm a Điều 2", "EXACT_REFERENCE_MALFORMED"),
        ("Điều 1 và Điều 2", "EXACT_COORDINATE_AMBIGUOUS"),
        ("Điều 1 của 01/QĐ-A và 02/QĐ-B", "EXACT_DOCUMENT_AMBIGUOUS"),
        ("Câu hỏi không viện dẫn", "EXACT_COORDINATE_ABSENT"),
    ],
)
def test_parser_fails_closed_for_malformed_or_ambiguous_references(
    question: str,
    code: str,
) -> None:
    result = parse_legal_reference(question)

    assert result.reference is None
    assert result.diagnostics[-1].code == code


def test_alias_artifact_is_closed_ordered_and_bound_to_context() -> None:
    context = context_record("Điều 1. Nội dung.")
    corpus_checksum = checksum_bytes(b"synthetic-corpus")

    index = load_alias_artifact(
        alias_bytes(),
        contexts=(context,),
        corpus_checksum=corpus_checksum,
        artifact_path="fixtures/aliases.jsonl",
    )

    assert index.context_ids_for("08/2022/nq-hdnd") == ("740",)
    assert index.corpus_checksum == corpus_checksum
    assert index.manifest_bytes() == index.manifest_bytes()


def test_frozen_alias_loader_requires_the_checksum_linked_active_manifest() -> None:
    data = alias_bytes()
    corpus_checksum = checksum_bytes(b"synthetic-corpus")
    validated = load_alias_artifact(
        data,
        contexts=(context_record("Điều 1. Nội dung."),),
        corpus_checksum=corpus_checksum,
        artifact_path="aliases.active.v1.jsonl",
    )

    frozen = load_frozen_alias_artifact(
        data,
        manifest_data=validated.manifest_bytes(),
        corpus_checksum=corpus_checksum,
        artifact_path="aliases.active.v1.jsonl",
    )

    assert frozen == validated

    with pytest.raises(AliasArtifactError, match="checksum") as changed:
        load_frozen_alias_artifact(
            data[:-1] + b" \n",
            manifest_data=validated.manifest_bytes(),
            corpus_checksum=corpus_checksum,
            artifact_path="aliases.active.v1.jsonl",
        )
    assert changed.value.code == "ALIAS_MANIFEST_MISMATCH"


def test_alias_artifact_rejects_bad_key_and_unknown_field() -> None:
    record = json.loads(alias_bytes())
    record["document_number_key"] = "wrong"
    record["extra"] = True
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode()

    with pytest.raises(AliasArtifactError) as captured:
        load_alias_artifact(
            data,
            contexts=(context_record("Điều 1."),),
            corpus_checksum=checksum_bytes(b"corpus"),
            artifact_path="fixtures/aliases.jsonl",
        )

    assert captured.value.code in {"ALIAS_SCHEMA_INVALID", "ALIAS_KEY_INVALID"}


def test_exact_resolution_returns_one_matching_point_chunk() -> None:
    context = context_record("Điều 1\n1. Khoản mẫu\na) Nội dung điểm.\n")
    chunks = chunk_context(
        context,
        config=ChunkingConfig(minimum_fragment_tokens=1),
    ).chunks
    aliases = load_alias_artifact(
        alias_bytes(),
        contexts=(context,),
        corpus_checksum=checksum_bytes(b"corpus"),
        artifact_path="fixtures/aliases.jsonl",
    )
    parsed = parse_legal_reference("Điểm a khoản 1 Điều 1 của 08/2022/NQ-HĐND")
    assert parsed.reference is not None

    result = resolve_exact_reference(parsed.reference, aliases=aliases, chunks=chunks)

    assert len(result.candidates) == 1
    assert result.candidates[0].exact_reference_match is True
    assert result.candidates[0].chunk.hierarchy_kind == "point"


def test_exact_resolution_does_not_infer_unknown_document_alias() -> None:
    context = context_record("Điều 1. Nội dung.")
    chunks = chunk_context(context).chunks
    aliases = load_alias_artifact(
        alias_bytes(),
        contexts=(context,),
        corpus_checksum=checksum_bytes(b"corpus"),
        artifact_path="fixtures/aliases.jsonl",
    )
    parsed = parse_legal_reference("Điều 1 của 99/QĐ-KHÔNG-CÓ")
    assert parsed.reference is not None

    result = resolve_exact_reference(parsed.reference, aliases=aliases, chunks=chunks)

    assert result.candidates == ()
    assert result.diagnostics[-1].code == "EXACT_DOCUMENT_UNRESOLVED"


def test_exact_resolution_abstains_for_zero_or_multiple_coordinate_chunks() -> None:
    context = context_record("Điều 1. " + " ".join(f"n{index}" for index in range(30)))
    chunks = chunk_context(
        context,
        config=ChunkingConfig(
            hierarchy_max_tokens=10,
            window_tokens=8,
            overlap_tokens=2,
            minimum_fragment_tokens=1,
        ),
    ).chunks
    aliases = load_alias_artifact(
        alias_bytes(),
        contexts=(context,),
        corpus_checksum=checksum_bytes(b"corpus"),
        artifact_path="fixtures/aliases.jsonl",
    )
    article_one = parse_legal_reference("Điều 1 của 08/2022/NQ-HĐND").reference
    article_two = parse_legal_reference("Điều 2 của 08/2022/NQ-HĐND").reference
    assert article_one is not None and article_two is not None

    multiple = resolve_exact_reference(article_one, aliases=aliases, chunks=chunks)
    missing = resolve_exact_reference(article_two, aliases=aliases, chunks=chunks)

    assert multiple.diagnostics[-1].code == "EXACT_COORDINATE_MULTI_CHUNK"
    assert missing.diagnostics[-1].code == "EXACT_COORDINATE_UNRESOLVED"
