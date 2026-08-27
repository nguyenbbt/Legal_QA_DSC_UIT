from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.evaluation.grounding_labels import (
    GroundingLabelError,
    load_approved_grounding_benchmark,
)


def _benchmark_bytes() -> bytes:
    return b"".join(
        (
            json.dumps(
                {
                    "schema_version": "grounding.benchmark.v1",
                    "question_id": f"q{index:02d}",
                    "split": "development",
                    "question_checksum": "sha256:" + ("1" * 64),
                    "relevant_evidence": [
                        {
                            "evidence_id": f"chunk_{index:024x}",
                            "relevance": "relevant",
                        }
                    ],
                    "required_claims": ["claim"],
                    "question_answerability": "answerable",
                    "temporal_assessment": "unknown",
                    "label_version": "grounding.v1",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for index in range(60)
    )


def _manifest_bytes(benchmark: bytes, status: str = "approved") -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "grounding.benchmark.manifest.v1",
                "label_version": "grounding.v1",
                "train_split_checksum": "sha256:" + ("2" * 64),
                "development_split_checksum": "sha256:" + ("3" * 64),
                "sampling_version": "grounding-sample.v1",
                "sampling_seed": "dsc2026-grounding-sample-v1",
                "ordered_question_ids": [f"q{index:02d}" for index in range(60)],
                "chunk_artifact_checksum": "sha256:" + ("4" * 64),
                "index_checksum": "sha256:" + ("5" * 64),
                "annotation_status": status,
                "ordered_files": [
                    {
                        "path": "grounding_set.v1.jsonl",
                        "checksum": checksum_bytes(benchmark),
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def test_approved_grounding_benchmark_loads_exact_60_rows() -> None:
    benchmark = _benchmark_bytes()

    loaded = load_approved_grounding_benchmark(_manifest_bytes(benchmark), benchmark)

    assert len(loaded.records) == 60
    assert loaded.retrieval_labels[0].question_id == "q00"
    assert loaded.retrieval_labels[0].relevant_evidence_ids == ("chunk_000000000000000000000000",)
    assert loaded.retrieval_labels[0].graded_evidence[0].relevance == "relevant"


def test_grounding_benchmark_rejects_unapproved_or_checksum_mismatch() -> None:
    benchmark = _benchmark_bytes()
    with pytest.raises(GroundingLabelError, match="owner-approved") as unapproved:
        load_approved_grounding_benchmark(_manifest_bytes(benchmark, "draft"), benchmark)
    assert unapproved.value.code == "GROUNDING_LABEL_APPROVAL_MISSING"

    with pytest.raises(GroundingLabelError, match="checksum") as mismatch:
        load_approved_grounding_benchmark(_manifest_bytes(benchmark), benchmark + b"\n")
    assert mismatch.value.code == "GROUNDING_LABEL_CHECKSUM_MISMATCH"
