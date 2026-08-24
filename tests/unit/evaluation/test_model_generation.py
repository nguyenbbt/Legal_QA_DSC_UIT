from __future__ import annotations

import json

import pytest

from legal_rag.evaluation.model_generation import run_grounded_generation_experiment


class _Backend:
    model_id = "fixture/generator"
    model_revision = "revision-1"

    def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
        assert system_prompt and question
        return evidence[0]


class _AdapterBackend(_Backend):
    adapter_id = "G3-fixture"
    adapter_checksum = "sha256:" + "a" * 64
    adapter_config_checksum = "sha256:" + "b" * 64


class _IncompleteAdapterBackend(_Backend):
    adapter_id = "G3-incomplete"


def _line(value: object) -> bytes:
    return (json.dumps(value) + "\n").encode()


def test_generation_uses_frozen_retrieval_order() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "question",
            "gold_answer": "answer",
            "candidates": [
                {"evidence_id": "a", "display_text": "first"},
                {"evidence_id": "b", "display_text": "second"},
            ],
        }
    )
    retrieval = _line(
        {"question_id": "q1", "candidates": [{"evidence_id": "b"}, {"evidence_id": "a"}]}
    )

    predictions, references, manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue,
        retrieval_output_data=retrieval,
        backend=_Backend(),
        system_prompt="prompt",
        run_id="G1-fixture",
        evidence_limit=1,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id="G1-fixture-256",
        changed_axes=("maximum_new_tokens",),
    )

    assert json.loads(predictions)["q1"]["answer"] == "second"
    assert json.loads(references) == {"q1": "answer"}
    manifest_value = json.loads(manifest)
    assert manifest_value["question_count"] == 1
    assert manifest_value["decoding"] == {
        "do_sample": False,
        "enable_thinking": False,
        "maximum_input_tokens": 2048,
        "maximum_new_tokens": 512,
    }
    assert manifest_value["comparison"] == {
        "baseline_run_id": "G1-fixture-256",
        "changed_axes": ["maximum_new_tokens"],
    }


def test_generation_fingerprint_records_adapter_provenance() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "question",
            "gold_answer": "answer",
            "candidates": [{"evidence_id": "a", "display_text": "evidence"}],
        }
    )
    retrieval = _line({"question_id": "q1", "candidates": [{"evidence_id": "a"}]})

    _, _, manifest = run_grounded_generation_experiment(
        annotation_queue_data=queue,
        retrieval_output_data=retrieval,
        backend=_AdapterBackend(),
        system_prompt="prompt",
        run_id="G3-fixture",
        evidence_limit=1,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
        do_sample=False,
        enable_thinking=False,
        baseline_run_id="G1A512-fixture",
        changed_axes=("adapter",),
        profile_state="g3_qlora_local_evaluation",
    )

    value = json.loads(manifest)
    assert value["profile_state"] == "g3_qlora_local_evaluation"
    assert value["adapter"] == {
        "adapter_id": "G3-fixture",
        "adapter_checksum": "sha256:" + "a" * 64,
        "adapter_config_checksum": "sha256:" + "b" * 64,
    }


def test_generation_rejects_incomplete_adapter_provenance() -> None:
    queue = _line(
        {
            "question_id": "q1",
            "question": "question",
            "gold_answer": "answer",
            "candidates": [{"evidence_id": "a", "display_text": "evidence"}],
        }
    )
    retrieval = _line({"question_id": "q1", "candidates": [{"evidence_id": "a"}]})

    with pytest.raises(ValueError, match="adapter provenance"):
        run_grounded_generation_experiment(
            annotation_queue_data=queue,
            retrieval_output_data=retrieval,
            backend=_IncompleteAdapterBackend(),
            system_prompt="prompt",
            run_id="G3-incomplete",
            evidence_limit=1,
            maximum_input_tokens=2048,
            maximum_new_tokens=512,
            do_sample=False,
            enable_thinking=False,
        )
