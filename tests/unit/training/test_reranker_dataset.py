from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.training.provenance import parse_training_example
from legal_rag.training.reranker_dataset import (
    RerankerDatasetError,
    RerankerTrainingSeed,
    build_reranker_training_artifacts,
)


def _chunk(
    chunk_id: str,
    *,
    context_id: str = "ctx-law",
    article: str,
    start: int,
    text: str | None = None,
) -> ChunkRecord:
    display = text or f"Điều {article}. Nội dung pháp luật {chunk_id}."
    return ChunkRecord(
        chunk_id=chunk_id,
        context_id=context_id,
        source_url="https://example.invalid/law",
        hierarchy_path=(f"Điều {article}",),
        hierarchy_rule_id="HIER_ARTICLE",
        hierarchy_kind="article",
        hierarchy_ordinal=article,
        canonical_start=start,
        canonical_end=start + len(display),
        display_text=display,
        retrieval_text=display,
        window_index=0,
        chunk_checksum=checksum_bytes(display.encode()),
        context_checksum=checksum_bytes(context_id.encode()),
    )


def _seed(*, split: str = "train", reverse: bool = False) -> RerankerTrainingSeed:
    positive = _chunk("positive", article="2", start=100, text="Điều 2. Người lao động được nghỉ.")
    candidates = (
        _chunk("wrong-3", article="3", start=300),
        _chunk("other-law", context_id="ctx-other", article="2", start=100),
        _chunk("wrong-1", article="1", start=0),
        positive,
    )
    if reverse:
        candidates = tuple(reversed(candidates))
    return RerankerTrainingSeed(
        question_id="q-1",
        question="Người lao động được nghỉ theo điều nào?",
        split=split,
        positives=(positive,),
        candidate_pool=candidates,
    )


def _build(seed: RerankerTrainingSeed):
    return build_reranker_training_artifacts(
        seeds=(seed,),
        question_source_checksum=checksum_bytes(b"questions"),
        split_manifest_checksum=checksum_bytes(b"split"),
        selection_checksum=checksum_bytes(b"selection"),
        chunks_checksum=checksum_bytes(b"chunks"),
        index_checksum=checksum_bytes(b"index"),
        construction_version="r008-reranker-groups.v1",
        maximum_negatives=8,
    )


def test_builds_only_same_document_wrong_coordinate_pairwise_groups() -> None:
    artifacts = _build(_seed())
    group = json.loads(artifacts.groups_data)

    assert artifacts.group_count == 1
    assert artifacts.pair_count == 2
    assert group["question_id"] == "q-1"
    assert group["split"] == "train"
    assert [item["evidence_id"] for item in group["positives"]] == ["positive"]
    assert {item["evidence_id"] for item in group["negatives"]} == {"wrong-1", "wrong-3"}
    assert {item["negative_type"] for item in group["negatives"]} == {"SAME_DOCUMENT_WRONG_ARTICLE"}
    assert "answer" not in group
    assert group["contains_generated_text"] is False

    provenance = parse_training_example(json.loads(artifacts.provenance_data))
    assert provenance.task == "reranking"
    assert provenance.target_source == "deterministic_relevance"
    assert provenance.contains_generated_text is False


def test_group_bytes_are_independent_of_seed_and_candidate_order() -> None:
    first = _build(_seed())
    second = _build(_seed(reverse=True))

    assert first.groups_data == second.groups_data
    assert first.provenance_data == second.provenance_data
    assert first.manifest_data == second.manifest_data


def test_non_train_seed_fails_closed_before_material_is_emitted() -> None:
    with pytest.raises(RerankerDatasetError) as raised:
        _build(_seed(split="development"))

    assert raised.value.code == "RERANKER_TRAIN_SPLIT_REJECTED"


def test_dataset_rejects_a_seed_without_deterministic_negative() -> None:
    seed = _seed()
    isolated = RerankerTrainingSeed(
        question_id=seed.question_id,
        question=seed.question,
        split="train",
        positives=seed.positives,
        candidate_pool=(_chunk("other-law", context_id="ctx-other", article="9", start=10),),
    )

    with pytest.raises(RerankerDatasetError) as raised:
        _build(isolated)

    assert raised.value.code == "RERANKER_TRAIN_DATASET_EMPTY"
