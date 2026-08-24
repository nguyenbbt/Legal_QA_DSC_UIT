"""Framework-neutral exact tensor-shape parameter accounting."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Literal

from legal_rag.domain.checksums import canonical_json_bytes
from legal_rag.models.manifest import ModelManifestError


@dataclass(frozen=True, slots=True)
class ParameterTensor:
    """Minimal immutable metadata required to count one learned tensor."""

    name: str
    shape: tuple[int, ...]
    category: Literal["base", "adapter"]
    trainable: bool

    def __post_init__(self) -> None:
        if not self.name.strip() or unicodedata.normalize("NFC", self.name) != self.name:
            raise ModelManifestError(
                "MODEL_PARAMETER_AUDIT_INVALID_TENSOR", "tensor name must be non-empty NFC"
            )
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.shape
        ):
            raise ModelManifestError(
                "MODEL_PARAMETER_AUDIT_INVALID_TENSOR",
                "tensor shape dimensions must be positive integers",
            )

    @property
    def numel(self) -> int:
        """Exact product of dimensions; a scalar shape has one element."""
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class ParameterAuditReport:
    """Deterministic exact counts and checksum for one checkpoint artifact."""

    schema_version: Literal["model.parameter_audit.v1"]
    tensors: tuple[ParameterTensor, ...]
    exact_parameter_count: int
    adapter_parameter_count: int
    trainable_parameter_count: int
    parameter_audit_checksum: str


def _tensor_payload(tensor: ParameterTensor) -> dict[str, object]:
    return {
        "category": tensor.category,
        "name": tensor.name,
        "numel": tensor.numel,
        "shape": tensor.shape,
        "trainable": tensor.trainable,
    }


def audit_parameters(tensors: tuple[ParameterTensor, ...]) -> ParameterAuditReport:
    """Count exact `numel` values and emit a canonical, order-independent audit."""
    if not tensors:
        raise ModelManifestError(
            "MODEL_PARAMETER_AUDIT_MISSING", "parameter audit contains no tensors"
        )
    names = tuple(tensor.name for tensor in tensors)
    if len(names) != len(set(names)):
        raise ModelManifestError(
            "MODEL_PARAMETER_AUDIT_DUPLICATE_TENSOR",
            "parameter audit contains a duplicate tensor name",
        )

    ordered = tuple(sorted(tensors, key=lambda tensor: tensor.name.encode("utf-8")))
    base_count = sum(tensor.numel for tensor in ordered if tensor.category == "base")
    adapter_count = sum(tensor.numel for tensor in ordered if tensor.category == "adapter")
    trainable_count = sum(tensor.numel for tensor in ordered if tensor.trainable)
    payload = {
        "adapter_parameter_count": adapter_count,
        "exact_parameter_count": base_count,
        "schema_version": "model.parameter_audit.v1",
        "tensors": tuple(_tensor_payload(tensor) for tensor in ordered),
        "trainable_parameter_count": trainable_count,
    }
    checksum = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return ParameterAuditReport(
        schema_version="model.parameter_audit.v1",
        tensors=ordered,
        exact_parameter_count=base_count,
        adapter_parameter_count=adapter_count,
        trainable_parameter_count=trainable_count,
        parameter_audit_checksum=f"sha256:{checksum}",
    )


__all__ = ["ParameterAuditReport", "ParameterTensor", "audit_parameters"]
