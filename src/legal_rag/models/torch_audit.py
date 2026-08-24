"""Exact PyTorch tensor audit adapter for the framework-neutral contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_rag.models.parameter_audit import (
    ParameterAuditReport,
    ParameterTensor,
    audit_parameters,
)


def audit_torch_module(module: Any) -> ParameterAuditReport:
    """Count unique named parameters; LoRA/adapter tensors remain separately visible."""

    tensors = tuple(
        ParameterTensor(
            name=name,
            shape=tuple(int(value) for value in parameter.shape),
            category="adapter" if "lora_" in name or ".adapter" in name else "base",
            trainable=bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters()
    )
    return audit_parameters(tensors)


def audit_safetensors_directory(path: Path) -> ParameterAuditReport:
    """Count tensor shapes directly from one immutable safetensors snapshot."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("locked safetensors dependency is unavailable") from error
    tensors: list[ParameterTensor] = []
    for file_path in sorted(path.glob("*.safetensors"), key=lambda item: item.name.encode("utf-8")):
        with safe_open(file_path, framework="pt", device="cpu") as stream:
            for name in stream.keys():  # noqa: SIM118 - safe_open is not iterable
                tensors.append(
                    ParameterTensor(
                        name=name,
                        shape=tuple(int(value) for value in stream.get_slice(name).get_shape()),
                        category="base",
                        trainable=False,
                    )
                )
    return audit_parameters(tuple(tensors))


__all__ = ["audit_safetensors_directory", "audit_torch_module"]
