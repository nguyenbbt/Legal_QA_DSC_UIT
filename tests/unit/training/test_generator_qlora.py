from __future__ import annotations

import json

import pytest

from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes, content_json_bytes
from legal_rag.training.generator_qlora import GeneratorDatasetError, load_generator_rows


def _artifacts() -> tuple[bytes, bytes, bytes]:
    target = "Câu trả lời chính thức."
    provenance = content_json_bytes(
        {
            "schema_version": "training.example.v1",
            "example_id": "sft_example",
            "task": "generation",
            "question_id": "q1",
            "split": "train",
            "question_source_checksum": checksum_bytes(b"questions"),
            "evidence_ids": ["chunk_a"],
            "target_source": "official_train_answer",
            "target_checksum": checksum_bytes(target.encode()),
            "contains_generated_text": False,
            "construction_version": "rag-sft.v1",
        }
    )
    material = content_json_bytes(
        {
            "schema_version": "training.rag_sft.material.v1",
            "example_id": "sft_example",
            "question_id": "q1",
            "question": "Câu hỏi?",
            "evidence": [
                {
                    "evidence_id": "chunk_a",
                    "evidence_checksum": checksum_bytes(b"evidence"),
                    "display_text": "Căn cứ chính thức.",
                }
            ],
            "target": target,
        }
    )
    manifest = canonical_json_bytes(
        {
            "schema_version": "training.rag_sft.manifest.v1",
            "accepted_rows": 1,
            "provenance_checksum": checksum_bytes(provenance),
            "material_checksum": checksum_bytes(material),
        }
    )
    return provenance, material, manifest


def test_generator_loader_links_material_to_provenance() -> None:
    provenance, material, manifest = _artifacts()

    rows = load_generator_rows(provenance, material, manifest)

    assert rows[0].target == "Câu trả lời chính thức."
    assert rows[0].evidence == ("Căn cứ chính thức.",)


def test_generator_loader_rejects_changed_target() -> None:
    provenance, material, manifest = _artifacts()
    value = json.loads(material)
    value["target"] = "Câu trả lời đã bị đổi."
    changed = content_json_bytes(value)
    manifest_value = json.loads(manifest)
    manifest_value["material_checksum"] = checksum_bytes(changed)

    with pytest.raises(GeneratorDatasetError) as caught:
        load_generator_rows(provenance, changed, canonical_json_bytes(manifest_value))

    assert caught.value.code == "GENERATOR_TARGET_CHECKSUM_MISMATCH"
