from __future__ import annotations

import torch
from safetensors.torch import save_file

from legal_rag.models.torch_audit import (
    audit_adapter_safetensors_directory,
    audit_safetensors_directory,
    audit_torch_module,
)


class _FixtureModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3, 2)
        self.lora_A = torch.nn.Parameter(torch.zeros(2, 3))


def test_torch_audit_counts_base_and_adapter_exactly() -> None:
    report = audit_torch_module(_FixtureModule())

    assert report.exact_parameter_count == 8
    assert report.adapter_parameter_count == 6
    assert report.trainable_parameter_count == 14


def test_safetensors_audit_counts_shapes_without_loading_a_model(tmp_path) -> None:
    save_file({"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}, tmp_path / "model.safetensors")

    report = audit_safetensors_directory(tmp_path)

    assert report.exact_parameter_count == 8
    assert report.adapter_parameter_count == 0
    assert report.trainable_parameter_count == 0


def test_adapter_safetensors_audit_counts_every_tensor_as_adapter(tmp_path) -> None:
    save_file(
        {"base_model.layer.lora_A.weight": torch.zeros(2, 3)},
        tmp_path / "adapter_model.safetensors",
    )

    report = audit_adapter_safetensors_directory(tmp_path)

    assert report.exact_parameter_count == 0
    assert report.adapter_parameter_count == 6
    assert report.trainable_parameter_count == 0
