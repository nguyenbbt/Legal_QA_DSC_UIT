from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.post_d062 import PostD062Error, reconcile_d063


def _json(value: object) -> bytes:
    return content_json_bytes(value)


def _metrics(rows: list[tuple[str, float, float]]) -> bytes:
    return b"".join(
        _json(
            {
                "schema_version": "competition.per_query.v1",
                "question_id": question_id,
                "rouge_l": rouge,
                "meteor": meteor,
            }
        )
        for question_id, meteor, rouge in rows
    )


def _predictions(rows: list[tuple[str, str]]) -> bytes:
    return _json({question_id: {"answer": answer} for question_id, answer in rows})


def _comparison(question_count: int = 2) -> bytes:
    return _json(
        {
            "schema_version": "generator.retrieval.comparison.v1",
            "question_count": question_count,
            "baseline_prediction_bytes_replayed": True,
            "candidate_prediction_bytes_replayed": True,
            "numeric_evaluation_gate": "passed",
            "resource_gate": "passed",
            "metrics": {
                "baseline_meteor": 0.2,
                "baseline_rouge_l": 0.3,
                "candidate_meteor": 0.25,
                "candidate_rouge_l": 0.35,
                "meteor_mean_delta": 0.05,
                "meteor_ci95": [0.01, 0.09],
                "rouge_l_mean_delta": 0.05,
            },
        }
    )


def _grounding() -> bytes:
    return _json(
        {
            "schema_version": "grounding.assessment.comparison.v1",
            "question_count": 60,
            "grounding_gate": "passed",
            "promotion_blockers": [],
            "rates": {
                "candidate_fully_supported_rate": 0.32,
                "candidate_unsupported_answer_rate": 0.38,
            },
        }
    )


def _parameters() -> bytes:
    return _json(
        {
            "schema_version": "model.parameter_manifest.v1",
            "system_parameter_count": 3_223_292_928,
            "competition_limit_exclusive": 4_000_000_000,
            "passes_parameter_gate": True,
            "models": [
                {"role": "generator", "model_id": "Qwen/Qwen3-1.7B"},
                {"role": "reranker", "model_id": "Qwen/Qwen3-Reranker-0.6B"},
            ],
        }
    )


def test_reconcile_d063_uses_frozen_d062_evidence_without_inference() -> None:
    report = reconcile_d063(
        comparison_data=_comparison(),
        grounding_data=_grounding(),
        baseline_metrics_data=_metrics([("q1", 0.2, 0.3), ("q2", 0.3, 0.4)]),
        candidate_metrics_data=_metrics([("q1", 0.3, 0.35), ("q2", 0.3, 0.45)]),
        baseline_predictions_data=_predictions([("q1", "một hai"), ("q2", "ba")]),
        candidate_predictions_data=_predictions([("q1", "một hai ba"), ("q2", "bốn")]),
        parameter_manifest_data=_parameters(),
        expected_question_count=2,
    )

    assert report["d063_status"] == "SATISFIED_BY_D062_FINALIZATION"
    assert report["post_d062_baseline"] == {
        "generator": "Qwen/Qwen3-1.7B G1A512",
        "retrieval": "Qwen/Qwen3-Reranker-0.6B base",
    }
    assert report["meteor_outcomes"] == {"candidate_wins": 1, "ties": 1, "candidate_losses": 0}
    assert report["answer_token_lengths"]["baseline"]["maximum"] == 2
    assert report["answer_token_lengths"]["candidate"]["maximum"] == 3
    assert report["new_inference_runs"] == 0


def test_reconcile_d063_fails_closed_on_non_identical_ids() -> None:
    with pytest.raises(PostD062Error) as captured:
        reconcile_d063(
            comparison_data=_comparison(),
            grounding_data=_grounding(),
            baseline_metrics_data=_metrics([("q1", 0.2, 0.3), ("q2", 0.3, 0.4)]),
            candidate_metrics_data=_metrics([("q1", 0.3, 0.35), ("qx", 0.3, 0.45)]),
            baseline_predictions_data=_predictions([("q1", "a"), ("q2", "b")]),
            candidate_predictions_data=_predictions([("q1", "c"), ("qx", "d")]),
            parameter_manifest_data=_parameters(),
            expected_question_count=2,
        )

    assert captured.value.code == "D063_ID_MISMATCH"


def test_reconcile_d063_requires_all_completed_gates() -> None:
    comparison = json.loads(_comparison())
    comparison["candidate_prediction_bytes_replayed"] = False

    with pytest.raises(PostD062Error) as captured:
        reconcile_d063(
            comparison_data=_json(comparison),
            grounding_data=_grounding(),
            baseline_metrics_data=_metrics([("q1", 0.2, 0.3), ("q2", 0.3, 0.4)]),
            candidate_metrics_data=_metrics([("q1", 0.3, 0.35), ("q2", 0.3, 0.45)]),
            baseline_predictions_data=_predictions([("q1", "a"), ("q2", "b")]),
            candidate_predictions_data=_predictions([("q1", "c"), ("q2", "d")]),
            parameter_manifest_data=_parameters(),
            expected_question_count=2,
        )

    assert captured.value.code == "D063_REQUIREMENT_UNSATISFIED"
