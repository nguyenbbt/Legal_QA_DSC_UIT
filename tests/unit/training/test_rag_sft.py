from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import SplitQuestion, build_split_manifest
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.training.rag_sft import (
    RagSftBuildConfig,
    RagSftBuildError,
    build_rag_sft_dataset,
)


class _Chunks:
    def __init__(self, chunk: ChunkRecord) -> None:
        self._chunk = chunk

    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]:
        return tuple(self._chunk for chunk_id in chunk_ids if chunk_id == self._chunk.chunk_id)


def _question_id_for_split(split: str) -> str:
    for index in range(100):
        question_id = f"q{index:02d}"
        manifest = build_split_manifest(
            (SplitQuestion(question_id, "Mức phạt là bao nhiêu?"),),
            (),
            source_checksum=checksum_bytes(b"questions"),
            public_source_checksum=checksum_bytes(b"public"),
        )
        if manifest.rows[0].split == split:
            return question_id
    raise AssertionError(f"unable to construct a {split} fixture")


def _inputs(
    *, split: str = "train", support_score: float = 0.99
) -> tuple[bytes, bytes, bytes, _Chunks]:
    question_id = _question_id_for_split(split)
    question = "Mức phạt là bao nhiêu?"
    answer = "Mức phạt là 10 triệu đồng."
    question_data = content_json_bytes(
        {
            "schema_version": "internal.question.v1",
            "question_id": question_id,
            "original_id": question_id,
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/train.questions.jsonl",
            "source_checksum": checksum_bytes(b"source"),
            "question": question,
            "answer": answer,
            "answer_state": "gold",
        }
    )
    split_manifest = build_split_manifest(
        (SplitQuestion(question_id, question),),
        (),
        source_checksum=checksum_bytes(question_data),
        public_source_checksum=checksum_bytes(b"public"),
    ).json_bytes()
    text = "Theo Điều 1, mức phạt là 10 triệu đồng."
    chunk = ChunkRecord(
        "chunk_a",
        "1",
        "https://example.invalid",
        ("Điều 1",),
        "ARTICLE",
        "article",
        "1",
        0,
        len(text),
        text,
        text,
        0,
        checksum_bytes(text.encode()),
        checksum_bytes(b"context"),
    )
    selection_data = content_json_bytes(
        {
            "schema_version": "training.evidence.selection.v1",
            "question_id": question_id,
            "question_checksum": checksum_bytes(question.encode()),
            "evidence_ids": [chunk.chunk_id],
            "evidence_checksums": [chunk.chunk_checksum],
            "support_score": support_score,
            "support_policy_version": "official-reranker-plus-lexical.v1",
        }
    )
    return question_data, split_manifest, selection_data, _Chunks(chunk)


def _build(*, split: str = "train", support_score: float = 0.99):
    question_data, split_manifest, selection_data, chunks = _inputs(
        split=split, support_score=support_score
    )
    return build_rag_sft_dataset(
        question_data=question_data,
        split_manifest_data=split_manifest,
        selection_data=selection_data,
        chunks=chunks,
        config=RagSftBuildConfig(
            construction_version="rag-sft.v1",
            support_policy_checksum=checksum_bytes(b"policy"),
            chunks_checksum=checksum_bytes(b"chunks"),
            index_checksum=checksum_bytes(b"index"),
            minimum_support_score=0.95,
            minimum_answer_token_coverage=0.8,
        ),
    )


def test_builder_emits_exact_gold_target_and_complete_provenance() -> None:
    result = _build()

    provenance = json.loads(result.provenance_data)
    material = json.loads(result.material_data)
    manifest = json.loads(result.manifest_data)

    assert provenance["schema_version"] == "training.example.v1"
    assert provenance["split"] == "train"
    assert provenance["target_source"] == "official_train_answer"
    assert provenance["contains_generated_text"] is False
    assert material["target"] == "Mức phạt là 10 triệu đồng."
    assert material["evidence"][0]["evidence_id"] == "chunk_a"
    assert manifest["accepted_rows"] == 1
    assert manifest["provenance_checksum"] == checksum_bytes(result.provenance_data)
    assert manifest["material_checksum"] == checksum_bytes(result.material_data)


def test_builder_rejects_non_train_rows() -> None:
    with pytest.raises(RagSftBuildError) as caught:
        _build(split="development")

    assert caught.value.code == "RAG_SFT_SPLIT_REJECTED"


def test_builder_rejects_evidence_below_support_threshold() -> None:
    with pytest.raises(RagSftBuildError) as caught:
        _build(support_score=0.90)

    assert caught.value.code == "RAG_SFT_SUPPORT_REJECTED"
