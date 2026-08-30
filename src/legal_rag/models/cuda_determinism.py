"""Fail-closed CUDA environment configuration for reproducible local matmul."""

from __future__ import annotations

from collections.abc import MutableMapping

_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
_DEFAULT = ":4096:8"
_VALID = frozenset({_DEFAULT, ":16:8"})


class CudaDeterminismError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def configure_cublas_workspace(environment: MutableMapping[str, str]) -> str:
    """Set one documented deterministic CuBLAS workspace before CUDA initializes."""

    configured = environment.get(_VARIABLE)
    if configured is None:
        environment[_VARIABLE] = _DEFAULT
        return _DEFAULT
    if configured not in _VALID:
        raise CudaDeterminismError(
            "CUDA_DETERMINISM_CONFIG_INVALID",
            "CUBLAS workspace configuration is not an approved deterministic value",
        )
    return configured


__all__ = ["CudaDeterminismError", "configure_cublas_workspace"]
