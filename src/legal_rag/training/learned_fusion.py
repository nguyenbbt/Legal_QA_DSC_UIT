"""Fixed CPU LightGBM LambdaMART adapter for the bounded D-067 experiment."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from legal_rag.evaluation.learned_fusion import FEATURE_NAMES, FusionFeatureRow, LearnedFusionError


@dataclass(frozen=True, slots=True)
class LightGbmRecipe:
    objective: str = "lambdarank"
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    reg_lambda: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 20_260_830
    n_jobs: int = 4
    deterministic: bool = True
    force_col_wise: bool = True

    def __post_init__(self) -> None:
        if (
            self.objective != "lambdarank"
            or self.n_estimators < 1
            or not 0.0 < self.learning_rate <= 1.0
            or self.num_leaves < 2
            or self.min_child_samples < 1
            or self.reg_lambda < 0.0
            or self.subsample != 1.0
            or self.colsample_bytree != 1.0
            or self.n_jobs < 1
            or not self.deterministic
            or not self.force_col_wise
        ):
            raise LearnedFusionError("D067_RECIPE_INVALID", "LightGBM recipe is invalid")


FIXED_D067_RECIPE = LightGbmRecipe()


@dataclass(frozen=True, slots=True)
class FusionTrainingMatrix:
    rows: tuple[FusionFeatureRow, ...]
    features: NDArray[np.float64]
    labels: NDArray[np.int32]
    group_sizes: tuple[int, ...]
    question_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LightGbmRuntimeAudit:
    version: str
    license_expression: str


@dataclass(frozen=True, slots=True)
class LightGbmModelAudit:
    feature_names: tuple[str, ...]
    feature_split_counts: tuple[int, ...]
    feature_gain_importance: tuple[float, ...]
    tree_count: int
    split_count: int
    leaf_count: int
    learned_value_count: int


@dataclass(frozen=True, slots=True)
class TrainedLightGbmFusion:
    model_data: bytes
    audit: LightGbmModelAudit


def build_training_matrix(rows: tuple[FusionFeatureRow, ...]) -> FusionTrainingMatrix:
    """Sort and pack complete candidate groups for LambdaMART."""

    ordered = tuple(sorted(rows, key=lambda row: (row.question_id.encode(), row.chunk_id.encode())))
    if not ordered or len({row.partition for row in ordered}) != 1:
        raise LearnedFusionError("D067_MATRIX_INVALID", "one matrix must contain one partition")
    identities = tuple((row.question_id, row.chunk_id) for row in ordered)
    if len(identities) != len(set(identities)):
        raise LearnedFusionError("D067_MATRIX_INVALID", "matrix contains duplicate candidates")
    question_ids = tuple(dict.fromkeys(row.question_id for row in ordered))
    group_sizes = tuple(
        sum(row.question_id == question_id for row in ordered) for question_id in question_ids
    )
    features = np.asarray([row.feature_values for row in ordered], dtype=np.float64)
    labels = np.asarray([row.label for row in ordered], dtype=np.int32)
    if features.shape != (len(ordered), len(FEATURE_NAMES)) or sum(group_sizes) != len(ordered):
        raise LearnedFusionError("D067_MATRIX_INVALID", "matrix shape or groups are invalid")
    return FusionTrainingMatrix(ordered, features, labels, group_sizes, question_ids)


def _lightgbm() -> Any:
    return importlib.import_module("lightgbm")


def lightgbm_runtime_audit() -> LightGbmRuntimeAudit:
    metadata = importlib.metadata.metadata("lightgbm")
    return LightGbmRuntimeAudit(
        version=importlib.metadata.version("lightgbm"),
        license_expression=metadata.get("License-Expression", ""),
    )


def _tree_counts(node: dict[str, Any]) -> tuple[int, int]:
    if "leaf_value" in node:
        return 0, 1
    left_split, left_leaves = _tree_counts(node["left_child"])
    right_split, right_leaves = _tree_counts(node["right_child"])
    return 1 + left_split + right_split, left_leaves + right_leaves


def fit_lightgbm_ranker(
    matrix: FusionTrainingMatrix, *, recipe: LightGbmRecipe = FIXED_D067_RECIPE
) -> TrainedLightGbmFusion:
    """Fit one deterministic LambdaMART model without validation-driven adaptation."""

    lgb = _lightgbm()
    ranker = lgb.LGBMRanker(
        objective=recipe.objective,
        n_estimators=recipe.n_estimators,
        learning_rate=recipe.learning_rate,
        num_leaves=recipe.num_leaves,
        min_child_samples=recipe.min_child_samples,
        reg_lambda=recipe.reg_lambda,
        subsample=recipe.subsample,
        colsample_bytree=recipe.colsample_bytree,
        random_state=recipe.random_state,
        n_jobs=recipe.n_jobs,
        deterministic=recipe.deterministic,
        force_col_wise=recipe.force_col_wise,
        verbosity=-1,
    )
    ranker.fit(
        matrix.features,
        matrix.labels,
        group=list(matrix.group_sizes),
        feature_name=list(FEATURE_NAMES),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    booster = ranker.booster_
    model_data = booster.model_to_string(num_iteration=recipe.n_estimators).encode("utf-8")
    dumped = booster.dump_model()
    tree_info = dumped.get("tree_info", [])
    if not isinstance(tree_info, list) or len(tree_info) != recipe.n_estimators:
        raise LearnedFusionError("D067_MODEL_AUDIT_INVALID", "tree count differs from recipe")
    split_count = 0
    leaf_count = 0
    for tree in tree_info:
        split, leaves = _tree_counts(tree["tree_structure"])
        split_count += split
        leaf_count += leaves
    audit = LightGbmModelAudit(
        feature_names=tuple(booster.feature_name()),
        feature_split_counts=tuple(
            int(value) for value in booster.feature_importance(importance_type="split")
        ),
        feature_gain_importance=tuple(
            float(value) for value in booster.feature_importance(importance_type="gain")
        ),
        tree_count=len(tree_info),
        split_count=split_count,
        leaf_count=leaf_count,
        learned_value_count=split_count + leaf_count,
    )
    if audit.feature_names != FEATURE_NAMES or not model_data:
        raise LearnedFusionError("D067_MODEL_AUDIT_INVALID", "model feature identity drifted")
    return TrainedLightGbmFusion(model_data, audit)


def predict_lightgbm_model(model_data: bytes, features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Load immutable model bytes and return finite float64 raw ranking scores."""

    if not model_data or features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise LearnedFusionError("D067_PREDICTION_INVALID", "model or feature shape is invalid")
    booster = _lightgbm().Booster(model_str=model_data.decode("utf-8"))
    scores = np.asarray(booster.predict(features), dtype=np.float64)
    if scores.shape != (features.shape[0],) or not np.isfinite(scores).all():
        raise LearnedFusionError("D067_PREDICTION_INVALID", "model predictions are invalid")
    return scores


__all__ = [
    "FIXED_D067_RECIPE",
    "FusionTrainingMatrix",
    "LightGbmModelAudit",
    "LightGbmRecipe",
    "LightGbmRuntimeAudit",
    "TrainedLightGbmFusion",
    "build_training_matrix",
    "fit_lightgbm_ranker",
    "lightgbm_runtime_audit",
    "predict_lightgbm_model",
]
