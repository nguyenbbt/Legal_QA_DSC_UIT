from __future__ import annotations

from dataclasses import replace

import numpy as np

from legal_rag.evaluation.learned_fusion import FEATURE_NAMES, FusionFeatureRow
from legal_rag.training.learned_fusion import (
    FIXED_D067_RECIPE,
    build_training_matrix,
    fit_lightgbm_ranker,
    lightgbm_runtime_audit,
    predict_lightgbm_model,
)


def _rows() -> tuple[FusionFeatureRow, ...]:
    rows: list[FusionFeatureRow] = []
    for group in range(8):
        for candidate in range(4):
            label = int(candidate == group % 4)
            values = [0.0] * len(FEATURE_NAMES)
            values[0] = float(label * 10)
            values[1] = float(candidate + 1)
            rows.append(
                FusionFeatureRow(
                    question_id=f"q{group:02d}",
                    question_checksum="sha256:" + f"{group + 1:x}" * 64,
                    chunk_id=f"c{candidate}",
                    partition="fit",
                    label=label,
                    feature_values=tuple(values),
                    chunk_checksum="sha256:" + f"{candidate + 1}" * 64,
                )
            )
    return tuple(rows)


def test_training_matrix_preserves_complete_contiguous_groups() -> None:
    matrix = build_training_matrix(tuple(reversed(_rows())))

    assert matrix.features.shape == (32, len(FEATURE_NAMES))
    assert matrix.labels.shape == (32,)
    assert matrix.group_sizes == (4,) * 8
    assert matrix.question_ids == tuple(f"q{group:02d}" for group in range(8))
    assert sum(matrix.group_sizes) == len(matrix.rows)


def test_lightgbm_ranker_model_and_prediction_replay_are_deterministic() -> None:
    matrix = build_training_matrix(_rows())
    recipe = replace(
        FIXED_D067_RECIPE,
        n_estimators=8,
        min_child_samples=1,
        n_jobs=1,
    )

    trained = fit_lightgbm_ranker(matrix, recipe=recipe)
    first = predict_lightgbm_model(trained.model_data, matrix.features)
    replay = predict_lightgbm_model(trained.model_data, matrix.features)
    runtime = lightgbm_runtime_audit()

    assert np.array_equal(first, replay)
    assert trained.audit.tree_count == 8
    assert trained.audit.split_count > 0
    assert trained.audit.leaf_count > trained.audit.tree_count
    assert trained.audit.learned_value_count == trained.audit.split_count + trained.audit.leaf_count
    assert trained.audit.feature_names == FEATURE_NAMES
    assert len(trained.audit.feature_split_counts) == len(FEATURE_NAMES)
    assert len(trained.audit.feature_gain_importance) == len(FEATURE_NAMES)
    assert runtime.version == "4.7.0"
    assert runtime.license_expression == "MIT"
