from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.generation.qwen3 import PROMPT_A
from legal_rag.providers.modal_public_generation import (
    ModalPublicGenerationError,
    build_modal_public_requests,
    validate_modal_public_responses,
)


def _public_source() -> bytes:
    return json.dumps(
        {
            "q1": {"question": "Câu hỏi 1?", "answer": None},
            "q2": {"question": "Câu hỏi 2?", "answer": None},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _evidence_queue() -> bytes:
    rows = []
    for position, question_id in enumerate(("q1", "q2"), start=1):
        question = f"Câu hỏi {position}?"
        evidence = [
            {
                "canonical_end": rank,
                "canonical_start": rank - 1,
                "chunk_checksum": checksum_bytes(f"chunk-{position}-{rank}".encode()),
                "context_id": str(position),
                "display_text": f"Căn cứ {position}-{rank}",
                "evidence_id": f"chunk-{position}-{rank}",
                "exact_reference_match": False,
                "hierarchy_path": [f"Điều {position}"],
                "rank": rank,
                "reranker_score": None,
                "sparse_score": float(4 - rank),
            }
            for rank in range(1, 4)
        ]
        rows.append(
            content_json_bytes(
                {
                    "schema_version": "public.evidence.v1",
                    "retrieval_run_id": "R0-public-fixture-v1",
                    "retrieval_fingerprint": checksum_bytes(b"retrieval"),
                    "question_id": question_id,
                    "question_checksum": checksum_bytes(question.encode()),
                    "question": question,
                    "evidence": evidence,
                }
            )
        )
    return b"".join(rows)


def test_modal_requests_expose_only_the_owner_approved_fields() -> None:
    requests = build_modal_public_requests(
        _evidence_queue(),
        public_source_data=_public_source(),
        expected_question_ids=("q1", "q2"),
        system_prompt=PROMPT_A,
    )

    assert tuple(request.question_id for request in requests) == ("q1", "q2")
    assert all(len(request.evidence) <= 3 for request in requests)
    assert all(
        set(request.model_dump()) == {"question_id", "question", "evidence", "system_prompt"}
        for request in requests
    )


def test_modal_requests_reject_non_prompt_a_or_another_question_order() -> None:
    with pytest.raises(ModalPublicGenerationError) as prompt_error:
        build_modal_public_requests(
            _evidence_queue(),
            public_source_data=_public_source(),
            expected_question_ids=("q1", "q2"),
            system_prompt="Another prompt",
        )
    assert prompt_error.value.code == "MODAL_PUBLIC_APPROVAL_SCOPE_VIOLATION"

    with pytest.raises(ModalPublicGenerationError) as identity_error:
        build_modal_public_requests(
            _evidence_queue(),
            public_source_data=_public_source(),
            expected_question_ids=("q2", "q1"),
            system_prompt=PROMPT_A,
        )
    assert identity_error.value.code == "MODAL_PUBLIC_QUESTION_ID_MISMATCH"


def test_modal_requests_reject_a_non_public_question_or_more_than_three_passages() -> None:
    rows = [json.loads(line) for line in _evidence_queue().splitlines()]
    rows[0]["question"] = "Một câu hỏi đã bị thay thế"
    rows[0]["question_checksum"] = checksum_bytes(rows[0]["question"].encode())
    changed_question = b"".join(content_json_bytes(row) for row in rows)

    with pytest.raises(ModalPublicGenerationError) as question_error:
        build_modal_public_requests(
            changed_question,
            public_source_data=_public_source(),
            expected_question_ids=("q1", "q2"),
            system_prompt=PROMPT_A,
        )
    assert question_error.value.code == "MODAL_PUBLIC_QUESTION_SOURCE_MISMATCH"

    rows = [json.loads(line) for line in _evidence_queue().splitlines()]
    fourth = {**rows[0]["evidence"][-1], "evidence_id": "chunk-1-4", "rank": 4}
    rows[0]["evidence"].append(fourth)
    oversized_evidence = b"".join(content_json_bytes(row) for row in rows)

    with pytest.raises(ModalPublicGenerationError) as evidence_error:
        build_modal_public_requests(
            oversized_evidence,
            public_source_data=_public_source(),
            expected_question_ids=("q1", "q2"),
            system_prompt=PROMPT_A,
        )
    assert evidence_error.value.code == "MODAL_PUBLIC_APPROVAL_SCOPE_VIOLATION"


def test_modal_responses_are_strict_and_ordered() -> None:
    responses: tuple[Mapping[str, object], ...] = (
        {
            "question_id": "q1",
            "answer": "Trả lời 1",
            "elapsed_seconds": 1.5,
            "input_tokens": 100,
            "output_tokens": 20,
            "peak_cuda_bytes": 1024,
        },
        {
            "question_id": "q2",
            "answer": "Trả lời 2",
            "elapsed_seconds": 2.5,
            "input_tokens": 120,
            "output_tokens": 30,
            "peak_cuda_bytes": 2048,
        },
    )

    validated = validate_modal_public_responses(
        responses,
        expected_question_ids=("q1", "q2"),
    )

    assert tuple(response.answer for response in validated) == ("Trả lời 1", "Trả lời 2")

    invalid = ({**responses[0], "question": "must not be returned"}, responses[1])
    with pytest.raises(ModalPublicGenerationError) as caught:
        validate_modal_public_responses(invalid, expected_question_ids=("q1", "q2"))
    assert caught.value.code == "MODAL_PUBLIC_RESPONSE_INVALID"
