"""Fit and evaluate the one fixed train-only D-067 LambdaMART candidate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil  # type: ignore[import-untyped]

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryGroup,
    DiscoveryRanking,
    evaluate_discovery_arm,
    load_discovery_groups,
    serialize_discovery_evaluation,
    serialize_discovery_rankings,
)
from legal_rag.evaluation.learned_fusion import (
    FEATURE_NAMES,
    FusionFeatureRow,
    build_group_split,
    compare_fusion_validation,
    deserialize_feature_rows,
    rank_learned_fusion,
)
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.training.learned_fusion import (
    FIXED_D067_RECIPE,
    build_training_matrix,
    fit_lightgbm_ranker,
    lightgbm_runtime_audit,
    predict_lightgbm_model,
)
from legal_rag.training.rag_sft import load_gold_questions

_D066 = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1")
_ROOT = Path("artifacts/training/learned-fusion/d067")
_QUESTIONS = Path("artifacts/internal/train.questions.jsonl")
_SPLIT = Path("artifacts/splits/train-dev-test.v1.json")
_SUPERVISION = Path("artifacts/training/retrieval-supervision/v2/retrieval-supervision.v2.jsonl")
_FEATURES = _ROOT / "D067.features.v1.jsonl"
_FEATURE_MANIFEST = _ROOT / "D067.features.manifest.v1.json"
_GROUP_SPLIT = _ROOT / "D067.group-split.v1.json"
_RRF_RANKINGS = _D066 / "R-DISC-4B-RRF60.rankings.v1.jsonl"
_EXPECTED_STATIC = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "supervision": "sha256:affd0969261243f0718e9faaed5d9cc0617138714cc190171cb9e7bf7253c1d6",
    "rrf_rankings": "sha256:ce5de92658c94b00a262f5ed6a1dc637ed296cb930ec6464e28307e0907a182a",
}
_TRAIN_COUNT = 5_582
_GROUP_COUNT = 2_391
_WHOLE_SYSTEM_PARENT_PARAMETERS = 3_223_292_928


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"D-067 JSON object is invalid: {path.name}")
    return cast(dict[str, Any], value)


def _load_groups() -> tuple[DiscoveryGroup, ...]:
    actual = {
        "questions": _checksum(_QUESTIONS),
        "split": _checksum(_SPLIT),
        "supervision": _checksum(_SUPERVISION),
        "rrf_rankings": _checksum(_RRF_RANKINGS),
    }
    if actual != _EXPECTED_STATIC:
        raise SystemExit("D-067 training input checksum drift")
    question_data = _QUESTIONS.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        _SPLIT.read_bytes(),
        expected_source_checksum=_EXPECTED_STATIC["questions"],
        expected_question_ids=tuple(item.question_id for item in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    if len(train_ids) != _TRAIN_COUNT:
        raise SystemExit("D-067 train-fit count drift")
    return load_discovery_groups(
        supervision_data=_SUPERVISION.read_bytes(),
        question_source_data=question_data,
        train_question_ids=train_ids,
        expected_positive_count=_GROUP_COUNT,
        expected_supervision_checksum=_EXPECTED_STATIC["supervision"],
    )


def _load_rrf_rankings(validation_ids: frozenset[str]) -> tuple[DiscoveryRanking, ...]:
    rankings: list[DiscoveryRanking] = []
    for line in _RRF_RANKINGS.read_bytes().splitlines():
        row = json.loads(line)
        question_id = str(row.get("question_id"))
        if question_id not in validation_ids:
            continue
        raw_ids = row.get("candidate_chunk_ids")
        if (
            row.get("schema_version") != "retrieval.discovery-ranking.v1"
            or row.get("arm_id") != "R-DISC-4B-FIXED-RRF-60"
            or not isinstance(raw_ids, list)
            or len(raw_ids) != 50
            or len(raw_ids) != len(set(raw_ids))
            or not all(isinstance(item, str) and item for item in raw_ids)
        ):
            raise SystemExit("D-067 fixed-RRF ranking is invalid")
        rankings.append(
            DiscoveryRanking(
                question_id,
                tuple(DiscoveryCandidate(chunk_id, "") for chunk_id in cast(list[str], raw_ids)),
            )
        )
    rankings.sort(key=lambda item: item.question_id.encode())
    if {item.question_id for item in rankings} != validation_ids:
        raise SystemExit("D-067 fixed-RRF validation coverage is incomplete")
    return tuple(rankings)


def _positive_assignment_recall(
    groups: Sequence[DiscoveryGroup], rankings: Sequence[DiscoveryRanking]
) -> dict[int, float]:
    ranking_by_id = {item.question_id: item for item in rankings}
    denominator = sum(len(group.positive_chunk_ids) for group in groups)
    if denominator < 1 or set(ranking_by_id) != {group.question_id for group in groups}:
        raise SystemExit("D-067 positive-assignment evaluation is invalid")
    return {
        cutoff: sum(
            len(
                set(group.positive_chunk_ids)
                & {item.chunk_id for item in ranking_by_id[group.question_id].candidates[:cutoff]}
            )
            for group in groups
        )
        / denominator
        for cutoff in (5, 10, 20, 50)
    }


def _validate_features(
    groups: tuple[DiscoveryGroup, ...], rows: tuple[FusionFeatureRow, ...]
) -> tuple[tuple[FusionFeatureRow, ...], tuple[FusionFeatureRow, ...]]:
    split = build_group_split(tuple(group.question_id for group in groups))
    expected_partition = {
        **{question_id: "fit" for question_id in split.fit_question_ids},
        **{question_id: "validation" for question_id in split.validation_question_ids},
    }
    group_by_id = {group.question_id: group for group in groups}
    seen_groups = {row.question_id for row in rows}
    if seen_groups != set(group_by_id):
        raise SystemExit("D-067 feature group coverage is incomplete")
    for row in rows:
        group = group_by_id[row.question_id]
        if (
            row.partition != expected_partition[row.question_id]
            or row.question_checksum != group.question_checksum
            or row.label != int(row.chunk_id in frozenset(group.positive_chunk_ids))
        ):
            raise SystemExit("D-067 feature split, provenance, or label drift")
    fit = tuple(row for row in rows if row.partition == "fit")
    validation = tuple(row for row in rows if row.partition == "validation")
    if not fit or not validation:
        raise SystemExit("D-067 feature partitions are empty")
    return fit, validation


def main() -> int:
    groups = _load_groups()
    feature_manifest = _load_json(_FEATURE_MANIFEST)
    feature_checksum = _checksum(_FEATURES)
    split_checksum = _checksum(_GROUP_SPLIT)
    if (
        feature_manifest.get("schema_version") != "evaluation.d067-features.manifest.v1"
        or feature_manifest.get("status") != "COMPLETE"
        or feature_manifest.get("feature_names") != list(FEATURE_NAMES)
        or feature_manifest.get("feature_checksum") != feature_checksum
        or feature_manifest.get("split_checksum") != split_checksum
        or feature_manifest.get("answer_derived_fields_used_as_features") is not False
        or feature_manifest.get("development_or_public_data_used") is not False
    ):
        raise SystemExit("D-067 feature manifest is invalid")
    rows = deserialize_feature_rows(_FEATURES.read_bytes())
    fit_rows, validation_rows = _validate_features(groups, rows)
    fit_matrix = build_training_matrix(fit_rows)
    validation_matrix = build_training_matrix(validation_rows)

    process = psutil.Process()
    started = time.perf_counter()
    trained = fit_lightgbm_ranker(fit_matrix, recipe=FIXED_D067_RECIPE)
    fit_seconds = time.perf_counter() - started
    rss_after_fit = int(process.memory_info().rss)
    predictions = predict_lightgbm_model(trained.model_data, validation_matrix.features)
    replay_predictions = predict_lightgbm_model(trained.model_data, validation_matrix.features)
    if not np.array_equal(predictions, replay_predictions):
        raise SystemExit("D-067 prediction replay differs")
    learned_rankings = rank_learned_fusion(validation_matrix.rows, predictions.tolist(), limit=50)
    learned_ranking_data = serialize_discovery_rankings("D067-LAMBDAMART", learned_rankings)
    if learned_ranking_data != serialize_discovery_rankings(
        "D067-LAMBDAMART",
        rank_learned_fusion(validation_matrix.rows, replay_predictions.tolist(), limit=50),
    ):
        raise SystemExit("D-067 ranking replay differs")

    validation_ids = frozenset(validation_matrix.question_ids)
    validation_groups = tuple(group for group in groups if group.question_id in validation_ids)
    validation_groups = tuple(sorted(validation_groups, key=lambda item: item.question_id.encode()))
    rrf_rankings = _load_rrf_rankings(validation_ids)
    rrf_evaluation = evaluate_discovery_arm(
        "R-DISC-4B-FIXED-RRF-60-VALIDATION", validation_groups, rrf_rankings
    )
    learned_evaluation = evaluate_discovery_arm(
        "D067-LAMBDAMART-VALIDATION", validation_groups, learned_rankings
    )
    comparison = compare_fusion_validation(rrf_evaluation, learned_evaluation)
    rrf_assignment = _positive_assignment_recall(validation_groups, rrf_rankings)
    learned_assignment = _positive_assignment_recall(validation_groups, learned_rankings)

    model_checksum = write_immutable_bytes(_ROOT / "D067.lambda-mart.v1.txt", trained.model_data)
    ranking_checksum = write_immutable_bytes(
        _ROOT / "D067.lambda-mart.validation.rankings.v1.jsonl", learned_ranking_data
    )
    rrf_evaluation_checksum = write_immutable_bytes(
        _ROOT / "D067.rrf60.validation.evaluation.v1.json",
        serialize_discovery_evaluation(rrf_evaluation),
    )
    learned_evaluation_checksum = write_immutable_bytes(
        _ROOT / "D067.lambda-mart.validation.evaluation.v1.json",
        serialize_discovery_evaluation(learned_evaluation),
    )
    comparison_data = content_json_bytes(
        {
            **asdict(comparison),
            "baseline_positive_assignment_recall_at": rrf_assignment,
            "candidate_positive_assignment_recall_at": learned_assignment,
            "downstream_gate_status": "PENDING_LATER_FIXED_PIPELINE_EVALUATION",
            "post_d062_baseline_changed": False,
        }
    )
    comparison_checksum = write_immutable_bytes(
        _ROOT / "D067.validation.comparison.v1.json", comparison_data
    )
    runtime = lightgbm_runtime_audit()
    sklearn_metadata = importlib.metadata.metadata("scikit-learn")
    learned_component_count = trained.audit.learned_value_count
    conservative_system_count = _WHOLE_SYSTEM_PARENT_PARAMETERS + learned_component_count
    model_manifest_data = content_json_bytes(
        {
            "schema_version": "model.learned-fusion-manifest.v1",
            "role": "retrieval_fusion",
            "implementation": "LightGBM-LambdaMART",
            "library_version": runtime.version,
            "library_license": runtime.license_expression,
            "scikit_learn_version": importlib.metadata.version("scikit-learn"),
            "scikit_learn_license": sklearn_metadata.get("License-Expression", ""),
            "recipe": asdict(FIXED_D067_RECIPE),
            "feature_names": list(FEATURE_NAMES),
            "model_audit": asdict(trained.audit),
            "model_checksum": model_checksum,
            "neural_parameter_count": 0,
            "tree_learned_numeric_value_count": learned_component_count,
            "conservative_whole_system_learned_parameter_count": conservative_system_count,
            "whole_system_strictly_below_4b": conservative_system_count < 4_000_000_000,
            "uv_lock_checksum": _checksum(Path("uv.lock")),
        }
    )
    model_manifest_checksum = write_immutable_bytes(
        _ROOT / "D067.lambda-mart.model-manifest.v1.json", model_manifest_data
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-telemetry.v1",
            "execution_mode": "local-cpu-offline",
            "fit_wall_seconds": fit_seconds,
            "rss_after_fit_bytes": rss_after_fit,
            "fit_group_count": len(fit_matrix.question_ids),
            "fit_candidate_count": len(fit_matrix.rows),
            "validation_group_count": len(validation_matrix.question_ids),
            "validation_candidate_count": len(validation_matrix.rows),
            "prediction_replay_byte_identical": True,
            "ranking_replay_byte_identical": True,
            "gpu_used": False,
            "modal_used": False,
            "paid_service_used": False,
            "cost_usd": 0,
        }
    )
    telemetry_checksum = write_immutable_bytes(_ROOT / "D067.telemetry.v1.json", telemetry_data)
    standing_winner = (
        "D067-LAMBDAMART-PROVISIONAL"
        if comparison.passes_retrieval_gate
        else "R-DISC-4B-FIXED-RRF-60"
    )
    run_manifest_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-run-manifest.v1",
            "status": "COMPLETE_STOP_BEFORE_D068",
            "feature_checksum": feature_checksum,
            "feature_manifest_checksum": _checksum(_FEATURE_MANIFEST),
            "split_checksum": split_checksum,
            "model_checksum": model_checksum,
            "model_manifest_checksum": model_manifest_checksum,
            "ranking_checksum": ranking_checksum,
            "rrf_evaluation_checksum": rrf_evaluation_checksum,
            "learned_evaluation_checksum": learned_evaluation_checksum,
            "comparison_checksum": comparison_checksum,
            "telemetry_checksum": telemetry_checksum,
            "provisional_fusion_winner": standing_winner,
            "retrieval_gate_passed": comparison.passes_retrieval_gate,
            "downstream_gate_status": "PENDING_LATER_FIXED_PIPELINE_EVALUATION",
            "post_d062_baseline_changed": False,
            "d068_status": "CLOSED",
            "fit_data": "official-train-only",
            "development_or_public_data_used": False,
            "gpu_used": False,
            "modal_used": False,
        }
    )
    run_manifest_checksum = write_immutable_bytes(
        _ROOT / "D067.run-manifest.v1.json", run_manifest_data
    )
    print(
        json.dumps(
            {
                "comparison": asdict(comparison),
                "fit_wall_seconds": fit_seconds,
                "provisional_fusion_winner": standing_winner,
                "run_manifest_checksum": run_manifest_checksum,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
