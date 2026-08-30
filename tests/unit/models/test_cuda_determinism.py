from __future__ import annotations

import pytest

from legal_rag.models.cuda_determinism import (
    CudaDeterminismError,
    configure_cublas_workspace,
)


def test_configure_cublas_workspace_sets_stable_default_before_cuda() -> None:
    environment: dict[str, str] = {}

    selected = configure_cublas_workspace(environment)

    assert selected == ":4096:8"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_configure_cublas_workspace_preserves_valid_value_and_rejects_drift() -> None:
    assert configure_cublas_workspace({"CUBLAS_WORKSPACE_CONFIG": ":16:8"}) == ":16:8"

    with pytest.raises(CudaDeterminismError) as captured:
        configure_cublas_workspace({"CUBLAS_WORKSPACE_CONFIG": "invalid"})

    assert captured.value.code == "CUDA_DETERMINISM_CONFIG_INVALID"
