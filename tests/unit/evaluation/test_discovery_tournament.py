from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryRanking,
    DiscoveryTournamentError,
    compare_discovery_arms,
    evaluate_discovery_arm,
    load_discovery_groups,
    serialize_discovery_evaluation,
    serialize_discovery_rankings,
)


def _question(question_id: str, answer: str, position: int) -> dict[str, object]:
    return {
        "schema_version": "internal.question.v1",
        "question_id": question_id,
        "original_id": question_id,
        "original_id_kind": "object_key_string",
        "source_position": position,
        "source_artifact": "fixture.jsonl",
        "source_checksum": "sha256:" + "a" * 64,
        "question": f"Question {question_id}?",
        "answer": answer,
        "answer_state": "gold",
    }


def _supervision(
    question_id: str,
    question: str,
    answer: str,
    mapping_class: str,
    chunk_ids: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "training.retrieval-supervision.group.v2",
        "group_id": f"retrieval_supervision_v2:{question_id}",
        "question_id": question_id,
        "question_checksum": checksum_bytes(question.encode()),
        "source_answer_checksum": checksum_bytes(answer.encode()),
        "mapping_class": mapping_class,
        "ambiguity_state": "NONE" if chunk_ids else mapping_class,
        "canonical_chunk_ids": chunk_ids,
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(content_json_bytes(row) for row in rows)


def test_load_discovery_groups_uses_only_frozen_train_positive_rows() -> None:
    questions = [_question("q1", "Alpha answer.", 0), _question("dev", "Dev answer.", 1)]
    supervision = [
        _supervision(
            "q1",
            "Question q1?",
            "Alpha answer.",
            "EXACT_DOC_ARTICLE",
            ["c1"],
        ),
        _supervision("dev", "Question dev?", "Dev answer.", "UNRESOLVED", []),
    ]
    supervision_data = _jsonl(supervision)

    groups = load_discovery_groups(
        supervision_data=supervision_data,
        question_source_data=_jsonl(questions),
        train_question_ids=("q1",),
        expected_positive_count=1,
        expected_supervision_checksum=checksum_bytes(supervision_data),
    )

    assert len(groups) == 1
    assert groups[0].question_id == "q1"
    assert groups[0].positive_chunk_ids == ("c1",)
    assert groups[0].gold_answer == "Alpha answer."


def test_load_discovery_groups_fails_closed_on_leakage_or_checksum_drift() -> None:
    questions = [_question("q1", "Alpha.", 0), _question("dev", "Dev.", 1)]
    supervision_data = _jsonl(
        [
            _supervision("q1", "Question q1?", "Alpha.", "EXACT_DOC_ARTICLE", ["c1"]),
            _supervision("dev", "Question dev?", "Dev.", "EXACT_DOC_ARTICLE", ["c2"]),
        ]
    )

    with pytest.raises(DiscoveryTournamentError) as leakage:
        load_discovery_groups(
            supervision_data=supervision_data,
            question_source_data=_jsonl(questions),
            train_question_ids=("q1",),
            expected_positive_count=2,
            expected_supervision_checksum=checksum_bytes(supervision_data),
        )
    assert leakage.value.code == "D066_SPLIT_LEAKAGE"

    with pytest.raises(DiscoveryTournamentError) as stale:
        load_discovery_groups(
            supervision_data=supervision_data,
            question_source_data=_jsonl(questions),
            train_question_ids=("q1", "dev"),
            expected_positive_count=2,
            expected_supervision_checksum="sha256:" + "0" * 64,
        )
    assert stale.value.code == "D066_SUPERVISION_CHECKSUM_MISMATCH"


def test_discovery_metrics_are_set_aware_and_answer_bearing_is_diagnostic() -> None:
    question_data = _jsonl(
        [
            _question("q1", "The required rule applies. Second sentence.", 0),
            _question("q2", "No hit.", 1),
        ]
    )
    supervision_data = _jsonl(
        [
            _supervision(
                "q1",
                "Question q1?",
                "The required rule applies. Second sentence.",
                "SAME_COORDINATE_MULTICHUNK",
                ["a", "b"],
            ),
            _supervision("q2", "Question q2?", "No hit.", "EXACT_DOC_ARTICLE", ["c"]),
        ]
    )
    groups = load_discovery_groups(
        supervision_data=supervision_data,
        question_source_data=question_data,
        train_question_ids=("q1", "q2"),
        expected_positive_count=2,
        expected_supervision_checksum=checksum_bytes(supervision_data),
    )
    q1_candidates = tuple(
        DiscoveryCandidate(
            chunk_id=("a" if rank == 1 else "b" if rank == 6 else f"x{rank}"),
            display_text=(
                "Preamble. The required rule applies."
                if rank == 2
                else "Second sentence."
                if rank == 6
                else "irrelevant"
            ),
        )
        for rank in range(1, 51)
    )
    rankings = (
        DiscoveryRanking(
            "q2", tuple(DiscoveryCandidate(f"z{rank}", "irrelevant") for rank in range(50))
        ),
        DiscoveryRanking("q1", q1_candidates),
    )

    report = evaluate_discovery_arm("fixture-arm", groups, rankings)

    assert report.question_count == 2
    assert report.recall_at[5] == 0.5
    assert report.recall_at[10] == 0.5
    assert report.evidence_set_recall_at[5] == 0.0
    assert report.evidence_set_recall_at[10] == 0.5
    assert report.mrr_at_50 == 0.5
    assert report.answer_bearing_coverage_at[5] == 0.5
    assert report.answer_bearing_coverage_at[10] == 0.5
    assert report.rows[0].question_id == "q1"
    assert report.rows[0].first_positive_rank == 1


def test_arm_comparison_reports_novel_recovery_loss_and_replays_identically() -> None:
    question_data = _jsonl([_question("q1", "Answer one.", 0), _question("q2", "Answer two.", 1)])
    supervision_data = _jsonl(
        [
            _supervision("q1", "Question q1?", "Answer one.", "EXACT_DOC_ARTICLE", ["a"]),
            _supervision("q2", "Question q2?", "Answer two.", "EXACT_DOC_ARTICLE", ["b"]),
        ]
    )
    groups = load_discovery_groups(
        supervision_data=supervision_data,
        question_source_data=question_data,
        train_question_ids=("q1", "q2"),
        expected_positive_count=2,
        expected_supervision_checksum=checksum_bytes(supervision_data),
    )
    baseline = evaluate_discovery_arm(
        "baseline",
        groups,
        (
            DiscoveryRanking("q1", (DiscoveryCandidate("a", "Answer one."),)),
            DiscoveryRanking("q2", (DiscoveryCandidate("x", "none"),)),
        ),
    )
    candidate = evaluate_discovery_arm(
        "candidate",
        groups,
        (
            DiscoveryRanking("q1", (DiscoveryCandidate("x", "none"),)),
            DiscoveryRanking("q2", (DiscoveryCandidate("b", "Answer two."),)),
        ),
    )

    comparison = compare_discovery_arms(baseline, candidate)

    assert comparison.novel_positive_recovery_at_50 == ("q2",)
    assert comparison.lost_positive_recovery_at_50 == ("q1",)
    assert comparison.standing_winner == "baseline"
    assert serialize_discovery_evaluation(candidate) == serialize_discovery_evaluation(candidate)
    assert json.loads(serialize_discovery_evaluation(candidate))["schema_version"] == (
        "evaluation.discovery-arm.v1"
    )


def test_ranking_artifact_is_stably_ordered_and_omits_private_text() -> None:
    rankings = (
        DiscoveryRanking("q2", (DiscoveryCandidate("c2", "Private answer two"),)),
        DiscoveryRanking("q1", (DiscoveryCandidate("c1", "Private answer one"),)),
    )

    data = serialize_discovery_rankings("arm", rankings)
    rows = [json.loads(line) for line in data.splitlines()]

    assert [row["question_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["candidate_chunk_ids"] == ["c1"]
    assert b"Private answer" not in data
