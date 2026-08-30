from __future__ import annotations

from dataclasses import replace

import pytest

from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryGroup,
    DiscoveryRanking,
    evaluate_discovery_arm,
)
from legal_rag.evaluation.learned_fusion import (
    FEATURE_NAMES,
    FusionCandidateSignals,
    FusionFeatureRow,
    LearnedFusionError,
    build_fusion_feature_values,
    build_group_split,
    build_query_legal_signals,
    compare_fusion_validation,
    deserialize_feature_rows,
    law_identity_key_from_context_name,
    rank_learned_fusion,
    serialize_feature_rows,
)
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.exact import document_number_key


def _chunk(chunk_id: str = "candidate-a") -> ChunkRecord:
    text = "Điều 10 quy định nghĩa vụ pháp lý của tổ chức."
    return ChunkRecord(
        chunk_id=chunk_id,
        context_id="ctx-12",
        source_url="https://example.invalid/legal",
        hierarchy_path=("Điều 10", "Khoản 2", "Điểm a"),
        hierarchy_rule_id="HIER_POINT",
        hierarchy_kind="point",
        hierarchy_ordinal="a",
        canonical_start=0,
        canonical_end=len(text),
        display_text=text,
        retrieval_text=text,
        window_index=0,
        chunk_checksum="sha256:" + "1" * 64,
        context_checksum="sha256:" + "2" * 64,
    )


def test_group_split_is_complete_disjoint_and_replay_stable() -> None:
    question_ids = tuple(str(value) for value in range(1, 101))

    first = build_group_split(question_ids)
    replay = build_group_split(tuple(reversed(question_ids)))

    assert first == replay
    assert len(first.fit_question_ids) + len(first.validation_question_ids) == 100
    assert set(first.fit_question_ids).isdisjoint(first.validation_question_ids)
    assert set(first.fit_question_ids) | set(first.validation_question_ids) == set(question_ids)
    assert first.split_version == "d067-group-split.v1"


def test_feature_order_and_values_use_only_answer_independent_signals() -> None:
    question = "điểm a khoản 2 Điều 10 Nghị định số 12/2022/NĐ-CP quy định thế nào?"
    query = build_query_legal_signals(
        question,
        document_aliases={document_number_key("12/2022/NĐ-CP"): ("ctx-12",)},
    )
    candidate = FusionCandidateSignals(
        chunk=_chunk(),
        sparse_score=7.5,
        sparse_rank=1,
        dense_score=0.75,
        dense_rank=2,
        exact_reference_flag=True,
        candidate_law_key=None,
    )

    values = build_fusion_feature_values(query, candidate)
    by_name = dict(zip(FEATURE_NAMES, values, strict=True))

    assert FEATURE_NAMES == (
        "bm25_score",
        "bm25_rank",
        "dense_score",
        "dense_rank",
        "exact_reference_flag",
        "document_id_match",
        "law_title_match",
        "article_number_match",
        "clause_match",
        "point_match",
        "query_length",
        "candidate_length",
        "lexical_overlap",
        "legal_term_overlap",
        "hierarchy_distance",
        "source_retriever_flags",
    )
    assert by_name["bm25_score"] == 7.5
    assert by_name["bm25_rank"] == 1.0
    assert by_name["dense_score"] == 0.75
    assert by_name["dense_rank"] == 2.0
    assert by_name["exact_reference_flag"] == 1.0
    assert by_name["document_id_match"] == 1.0
    assert by_name["article_number_match"] == 1.0
    assert by_name["clause_match"] == 1.0
    assert by_name["point_match"] == 1.0
    assert by_name["hierarchy_distance"] == 0.0
    assert by_name["source_retriever_flags"] == 7.0
    assert by_name["query_length"] > 0.0
    assert by_name["candidate_length"] > 0.0
    assert 0.0 <= by_name["lexical_overlap"] <= 1.0
    assert 0.0 <= by_name["legal_term_overlap"] <= 1.0


def test_feature_serialization_is_stable_and_labels_do_not_change_feature_values() -> None:
    features = tuple(float(index) for index in range(len(FEATURE_NAMES)))
    positive = FusionFeatureRow(
        "q2",
        "sha256:" + "2" * 64,
        "c2",
        "fit",
        1,
        features,
        "sha256:" + "2" * 64,
    )
    negative = replace(positive, label=0)
    earlier = FusionFeatureRow(
        "q1",
        "sha256:" + "1" * 64,
        "c1",
        "validation",
        0,
        features,
        "sha256:" + "1" * 64,
    )

    assert positive.feature_values == negative.feature_values
    serialized = serialize_feature_rows((positive, earlier))

    assert serialized == serialize_feature_rows((earlier, positive))
    assert deserialize_feature_rows(serialized) == (earlier, positive)


def test_feature_deserialization_rejects_coercible_malformed_fields() -> None:
    malformed = (
        b'{"schema_version":"evaluation.d067-feature-row.v1",'
        b'"question_id":1,"question_checksum":"sha256:'
        + b"1"
        * 64
        + b'","chunk_id":"c1","partition":"fit","label":false,'
        b'"feature_values":['
        + b",".join(b"0" for _ in FEATURE_NAMES)
        + b'],"chunk_checksum":"sha256:'
        + b"2" * 64
        + b'"}\n'
    )

    with pytest.raises(LearnedFusionError, match="feature field types are invalid") as error:
        deserialize_feature_rows(malformed)

    assert error.value.code == "D067_FEATURE_ARTIFACT_INVALID"


def test_context_law_identity_uses_the_same_normalized_legal_key() -> None:
    assert law_identity_key_from_context_name("Luật Đất đai 2024") == "luat dat dai 2024"
    assert law_identity_key_from_context_name(None) is None


def test_learned_ranking_uses_score_then_utf8_chunk_id_tie_break() -> None:
    features = (0.0,) * len(FEATURE_NAMES)
    rows = (
        FusionFeatureRow(
            "q2", "sha256:" + "3" * 64, "z", "validation", 0, features, "sha256:" + "3" * 64
        ),
        FusionFeatureRow(
            "q1", "sha256:" + "1" * 64, "b", "validation", 0, features, "sha256:" + "2" * 64
        ),
        FusionFeatureRow(
            "q1", "sha256:" + "1" * 64, "a", "validation", 1, features, "sha256:" + "1" * 64
        ),
    )

    rankings = rank_learned_fusion(rows, (0.5, 0.8, 0.8), limit=50)

    assert tuple(item.question_id for item in rankings) == ("q1", "q2")
    assert tuple(item.chunk_id for item in rankings[0].candidates) == ("a", "b")


def test_validation_gate_requires_preserved_top50_and_better_ranking() -> None:
    groups = (
        DiscoveryGroup("q1", "sha256:" + "1" * 64, "q", "sha256:" + "2" * 64, "a", ("p1",)),
        DiscoveryGroup("q2", "sha256:" + "3" * 64, "q", "sha256:" + "4" * 64, "a", ("p2",)),
    )
    baseline = (
        DiscoveryRanking("q1", (DiscoveryCandidate("n1", ""), DiscoveryCandidate("p1", ""))),
        DiscoveryRanking("q2", (DiscoveryCandidate("n2", ""), DiscoveryCandidate("p2", ""))),
    )
    learned = (
        DiscoveryRanking("q1", (DiscoveryCandidate("p1", ""), DiscoveryCandidate("n1", ""))),
        DiscoveryRanking("q2", (DiscoveryCandidate("p2", ""), DiscoveryCandidate("n2", ""))),
    )

    comparison = compare_fusion_validation(
        evaluate_discovery_arm("RRF", groups, baseline),
        evaluate_discovery_arm("LEARNED", groups, learned),
    )

    assert comparison.passes_retrieval_gate
    assert comparison.standing_winner == "LEARNED"
    assert comparison.lost_positive_recovery_at_50 == ()
