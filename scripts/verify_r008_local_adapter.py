from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.models.torch_audit import audit_adapter_safetensors_directory
from legal_rag.training.reranker_lora import directory_checksum


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one local R-008 corrective adapter")
    parser.add_argument("--run-report", required=True, type=Path)
    parser.add_argument("--adapter-directory", required=True, type=Path)
    arguments = parser.parse_args()
    run_report = json.loads(arguments.run_report.read_bytes())
    adapter_checksum = directory_checksum(arguments.adapter_directory)
    audit = audit_adapter_safetensors_directory(arguments.adapter_directory)
    if adapter_checksum != run_report["adapter_checksum"]:
        raise RuntimeError("R008_LOCAL_ADAPTER_CHECKSUM_MISMATCH")
    if audit.adapter_parameter_count != run_report["adapter_parameter_count"]:
        raise RuntimeError("MODEL_PARAMETER_AUDIT_MISMATCH")
    report_data = content_json_bytes(
        {
            "schema_version": "model.adapter-parameter-audit.v1",
            "run_id": "R008-qwen3-reranker-corrective-epoch1-v1",
            "adapter_checksum": adapter_checksum,
            "adapter_parameter_count": audit.adapter_parameter_count,
            "parameter_audit_checksum": audit.parameter_audit_checksum,
            "tensor_count": len(audit.tensors),
            "whole_system_parameter_count": run_report["whole_system_parameter_count"],
            "competition_limit_exclusive": 4_000_000_000,
            "passes_parameter_gate": run_report["whole_system_parameter_count"] < 4_000_000_000,
        }
    )
    checksum = write_immutable_bytes(
        arguments.adapter_directory.parent / "adapter.parameter-audit.v1.json",
        report_data,
    )
    print(
        f"R008 CORRECTIVE VERIFIED adapter={adapter_checksum} "
        f"parameters={audit.adapter_parameter_count} report={checksum}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
