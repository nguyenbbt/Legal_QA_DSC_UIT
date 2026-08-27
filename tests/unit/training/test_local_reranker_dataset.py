from __future__ import annotations

import json

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import SplitQuestion, build_split_manifest
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.training.local_reranker_dataset import (
    RerankerSeedBuildConfig,
    build_reranker_dataset_from_selections,
)


class _Chunks:
    def __init__(self, chunks: tuple[ChunkRecord, ...]) -> None:
        self._chunks = chunks

    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]:
        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        return tuple(by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return tuple(chunk for chunk in self._chunks if chunk.context_id == context_id)


def _chunk(chunk_id: str, article: str, start: int) -> ChunkRecord:
    text = f"Điều {article}. Nội dung {chunk_id}."
    return ChunkRecord(
        chunk_id,
        "ctx-law",
        "https://example.invalid/law",
        (f"Điều {article}",),
        "HIER_ARTICLE",
        "article",
        article,
        start,
        start + len(text),
        text,
        text,
        0,
        checksum_bytes(text.encode()),
        checksum_bytes(b"ctx-law"),
    )


def _train_id(question: str) -> str:
    for index in range(100):
        question_id = f"q{index:02d}"
        manifest = build_split_manifest(
            (SplitQuestion(question_id, question),),
            (),
            source_checksum=checksum_bytes(b"source"),
            public_source_checksum=checksum_bytes(b"public"),
        )
        if manifest.rows[0].split == "train":
            return question_id
    raise AssertionError("no train ID fixture")


def test_builds_upload_material_without_copying_official_train_answer() -> None:
    question = "Quy định nằm tại điều nào?"
    question_id = _train_id(question)
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
            "answer": "Bí mật không được đưa vào group.",
            "answer_state": "gold",
        }
    )
    split_data = build_split_manifest(
        (SplitQuestion(question_id, question),),
        (),
        source_checksum=checksum_bytes(question_data),
        public_source_checksum=checksum_bytes(b"public"),
    ).json_bytes()
    positive = _chunk("positive", "2", 100)
    wrong = _chunk("wrong", "3", 300)
    selection_data = content_json_bytes(
        {
            "schema_version": "training.evidence.selection.v1",
            "question_id": question_id,
            "question_checksum": checksum_bytes(question.encode()),
            "evidence_ids": [positive.chunk_id],
            "evidence_checksums": [positive.chunk_checksum],
            "support_score": 0.99,
            "support_policy_version": "official-reranker-plus-lexical.v1",
        }
    )

    artifacts = build_reranker_dataset_from_selections(
        question_data=question_data,
        split_manifest_data=split_data,
        selection_data=selection_data,
        chunks=_Chunks((positive, wrong)),
        chunks_checksum=checksum_bytes(b"chunks"),
        index_checksum=checksum_bytes(b"index"),
        config=RerankerSeedBuildConfig(),
    )

    material = json.loads(artifacts.groups_data)
    assert material["question_id"] == question_id
    assert material["positives"][0]["evidence_id"] == "positive"
    assert material["negatives"][0]["evidence_id"] == "wrong"
    assert b"B\xc3\xad m\xe1\xba\xadt" not in artifacts.groups_data
    assert json.loads(artifacts.manifest_data)["group_count"] == 1
