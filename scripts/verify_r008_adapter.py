from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.models.torch_audit import audit_adapter_safetensors_directory
from legal_rag.providers.modal_reranker_training import validate_r008_training_response
from legal_rag.training.reranker_lora import directory_checksum


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the downloaded R-008 PEFT adapter")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--adapter-directory", required=True, type=Path)
    parser.add_argument("--whole-system-base-parameters", type=int, default=3_223_292_928)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    request_data = arguments.request.read_bytes()
    response_data = arguments.response.read_bytes()
    response = validate_r008_training_response(
        request_data=request_data,
        response_data=response_data,
    )
    adapter_checksum = directory_checksum(arguments.adapter_directory)
    if adapter_checksum != response.adapter_checksum:
        raise RuntimeError("MODAL_R008_CHECKSUM_MISMATCH")
    audit = audit_adapter_safetensors_directory(arguments.adapter_directory)
    if audit.adapter_parameter_count != response.adapter_parameter_count:
        raise RuntimeError("MODEL_PARAMETER_AUDIT_MISMATCH")
    whole_system_parameters = arguments.whole_system_base_parameters + audit.adapter_parameter_count
    if whole_system_parameters != response.whole_system_parameter_count:
        raise RuntimeError("MODEL_PARAMETER_AUDIT_MISMATCH")
    if whole_system_parameters >= 4_000_000_000:
        raise RuntimeError("MODEL_PARAMETER_LIMIT")
    report_data = content_json_bytes(
        {
            "schema_version": "model.adapter-parameter-audit.v1",
            "run_id": response.run_id,
            "adapter_checksum": adapter_checksum,
            "adapter_parameter_count": audit.adapter_parameter_count,
            "parameter_audit_checksum": audit.parameter_audit_checksum,
            "tensor_count": len(audit.tensors),
            "whole_system_parameter_count": whole_system_parameters,
            "competition_limit_exclusive": 4_000_000_000,
            "passes_parameter_gate": True,
        }
    )
    report_checksum = write_immutable_bytes(
        arguments.adapter_directory.parent / "adapter.parameter-audit.v1.json",
        report_data,
    )
    print(
        f"R008 VERIFIED adapter={adapter_checksum} parameters={audit.adapter_parameter_count} "
        f"whole_system={whole_system_parameters} report={report_checksum}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
