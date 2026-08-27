from __future__ import annotations

import pytest
from pydantic import ValidationError

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.providers.modal_private_retrieval import (
    ModalPrivateRetrievalChunk,
    ModalPrivateRetrievalConfig,
    ModalPrivateRetrievalError,
    ModalPrivateRetrievalLifecycle,
    ModalPrivateRetrievalQuestion,
    ModalPrivateRetrievalResponse,
    build_modal_private_retrieval_bundle,
    validate_modal_private_retrieval_response,
)

SHA = "sha256:" + "a" * 64


def _chunk(text: str = "Điều 1. Nội dung.") -> dict[str, object]:
    return {
        "evidence_id": "e1",
        "canonical_text": text,
        "context_id": "1",
        "canonical_start": 0,
        "canonical_end": len(text),
        "hierarchy_path": ("Điều 1",),
        "chunk_checksum": checksum_bytes(text.encode("utf-8")),
        "context_checksum": SHA,
        "corpus_checksum": SHA,
        "parent_evidence_id": None,
        "sibling_evidence_ids": (),
    }


def _question(text: str = "Quy định nào áp dụng?") -> dict[str, object]:
    return {
        "question_id": "q1",
        "question": text,
        "question_checksum": checksum_bytes(text.encode("utf-8")),
    }


def _bundle(
    *,
    chunks: tuple[dict[str, object], ...] | None = None,
    questions: tuple[dict[str, object], ...] | None = None,
    expected_chunk_checksums: dict[str, str] | None = None,
    expected_canonical_text_checksums: dict[str, str] | None = None,
):
    chunk_rows = chunks or (_chunk(),)
    question_rows = questions or (_question(),)
    return build_modal_private_retrieval_bundle(
        chunks=chunk_rows,
        questions=question_rows,
        expected_chunk_checksums=expected_chunk_checksums
        or {str(row["evidence_id"]): str(row["chunk_checksum"]) for row in chunk_rows},
        expected_context_checksums={
            str(row["context_id"]): str(row["context_checksum"]) for row in chunk_rows
        },
        expected_canonical_text_checksums=expected_canonical_text_checksums
        or {
            str(row["evidence_id"]): checksum_bytes(str(row["canonical_text"]).encode("utf-8"))
            for row in chunk_rows
        },
        expected_corpus_checksum=str(chunk_rows[0]["corpus_checksum"]),
    )


def test_closed_bundle_accepts_only_approved_fields_and_is_byte_stable() -> None:
    first = _bundle()
    second = _bundle()

    assert first == second
    assert first.schema_version == "modal.private-retrieval.bundle.v1"
    assert first.chunks[0].canonical_text == "Điều 1. Nội dung."


@pytest.mark.parametrize(
    ("field", "value"),
    (("answer", "gold"), ("grounding_label", True), ("absolute_path", "C:\\private")),
)
def test_closed_bundle_rejects_unlisted_fields(field: str, value: object) -> None:
    chunk = _chunk()
    chunk[field] = value
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(chunks=(chunk,))
    assert caught.value.code == "MODAL_PRIVATE_BUNDLE_INVALID"


@pytest.mark.parametrize("text", ("sk-secret-value", "password=hunter2", "C:\\Users\\me"))
def test_preflight_rejects_secrets_and_absolute_paths_before_provider_use(text: str) -> None:
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(chunks=(_chunk(text),))
    assert caught.value.code == "MODAL_PRIVATE_CONTENT_FORBIDDEN"


def test_preflight_scans_identity_and_hierarchy_fields_too() -> None:
    chunk = _chunk()
    chunk["hierarchy_path"] = ("C:\\private",)
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(chunks=(chunk,))
    assert caught.value.code == "MODAL_PRIVATE_CONTENT_FORBIDDEN"


def test_preflight_rejects_content_checksum_mismatch() -> None:
    chunk = _chunk()
    chunk["chunk_checksum"] = SHA
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(chunks=(chunk,), expected_chunk_checksums={"e1": checksum_bytes(b"expected")})
    assert caught.value.code == "MODAL_PRIVATE_CHECKSUM_MISMATCH"

    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(expected_canonical_text_checksums={"e1": checksum_bytes(b"different")})
    assert caught.value.code == "MODAL_PRIVATE_CHECKSUM_MISMATCH"


def test_config_enforces_private_one_worker_no_egress_and_strict_campaign_limit() -> None:
    config = ModalPrivateRetrievalConfig(
        volume_private=True,
        teammate_sharing=False,
        backup_enabled=False,
        egress_enabled=False,
        max_containers=1,
        encrypted_io_retention_days=7,
        maximum_run_cost_usd=10,
        campaign_spend_before_usd=19.99,
        projected_run_cost_usd=10,
    )
    assert config.projected_campaign_cost_usd == pytest.approx(29.99)

    with pytest.raises(ValidationError):
        ModalPrivateRetrievalConfig(
            volume_private=True,
            teammate_sharing=False,
            backup_enabled=False,
            egress_enabled=False,
            max_containers=1,
            encrypted_io_retention_days=7,
            maximum_run_cost_usd=10,
            campaign_spend_before_usd=20,
            projected_run_cost_usd=10,
        )


def test_return_contract_denies_text_and_requires_expected_identity_order() -> None:
    response = {
        "schema_version": "modal.private-retrieval.response.v1",
        "model_id": "approved/model",
        "model_revision": "a" * 40,
        "configuration_checksum": SHA,
        "bundle_checksum": SHA,
        "rows": ({"question_id": "q1", "evidence_id": "e1", "score": 1.0, "rank": 1},),
        "aggregate_telemetry": {"question_count": 1, "candidate_count": 1},
    }
    parsed = validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))
    assert isinstance(parsed, ModalPrivateRetrievalResponse)

    response["corpus_text"] = "forbidden"
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))
    assert caught.value.code == "MODAL_PRIVATE_RESPONSE_INVALID"

    response.pop("corpus_text")
    response["model_id"] = "C:\\private\\model"
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))
    assert caught.value.code == "MODAL_PRIVATE_RESPONSE_INVALID"


def test_lifecycle_is_monotonic_and_absence_requires_deletion_receipt() -> None:
    created = ModalPrivateRetrievalLifecycle.initial("campaign-1").mark_created()
    deleted = created.mark_deleted(deletion_receipt_checksum=SHA)
    verified = deleted.mark_absence_verified()
    assert verified.state == "absence_verified"

    with pytest.raises(ModalPrivateRetrievalError) as caught:
        ModalPrivateRetrievalLifecycle.initial("campaign-1").mark_absence_verified()
    assert caught.value.code == "MODAL_PRIVATE_LIFECYCLE_INVALID"


def test_models_are_strict_and_validate_question_checksum() -> None:
    with pytest.raises(ValidationError):
        ModalPrivateRetrievalQuestion(
            question_id="q1",
            question="text",
            question_checksum=SHA,
            answer="forbidden",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        ModalPrivateRetrievalChunk.model_validate({**_chunk(), "canonical_start": "0"})


def test_bundle_rejects_relationship_and_frozen_corpus_inconsistency() -> None:
    self_sibling = _chunk()
    self_sibling["sibling_evidence_ids"] = ("e1",)
    with pytest.raises(ModalPrivateRetrievalError):
        _bundle(chunks=(self_sibling,))

    second = {**_chunk("Điều 2."), "evidence_id": "e2", "corpus_checksum": "sha256:" + "b" * 64}
    with pytest.raises(ModalPrivateRetrievalError):
        _bundle(chunks=(_chunk(), second))

    duplicate_siblings = _chunk()
    duplicate_siblings["sibling_evidence_ids"] = ("e2", "e2")
    with pytest.raises(ModalPrivateRetrievalError):
        _bundle(chunks=(duplicate_siblings,))

    with pytest.raises(ModalPrivateRetrievalError):
        _bundle(chunks=(_chunk(), _chunk()))


def test_bundle_rejects_question_checksum_mismatch_independently() -> None:
    question = _question()
    question["question_checksum"] = SHA
    with pytest.raises(ModalPrivateRetrievalError) as caught:
        _bundle(questions=(question,))
    assert caught.value.code == "MODAL_PRIVATE_CHECKSUM_MISMATCH"


def test_config_and_return_aggregate_invariants_fail_closed() -> None:
    with pytest.raises(ValidationError):
        ModalPrivateRetrievalConfig(
            volume_private=True,
            teammate_sharing=False,
            backup_enabled=False,
            egress_enabled=False,
            max_containers=1,
            encrypted_io_retention_days=7,
            maximum_run_cost_usd=5,
            campaign_spend_before_usd=0,
            projected_run_cost_usd=6,
        )

    response = {
        "schema_version": "modal.private-retrieval.response.v1",
        "model_id": "approved/model",
        "model_revision": "revision",
        "configuration_checksum": SHA,
        "bundle_checksum": SHA,
        "rows": (
            {"question_id": "q1", "evidence_id": "e1", "score": 1.0, "rank": 1},
            {"question_id": "q1", "evidence_id": "e1", "score": 0.5, "rank": 2},
        ),
        "aggregate_telemetry": {"question_count": 1, "candidate_count": 2},
    }
    with pytest.raises(ModalPrivateRetrievalError):
        validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))

    response["rows"] = ({"question_id": "q1", "evidence_id": "e1", "score": 1.0, "rank": 1},)
    with pytest.raises(ModalPrivateRetrievalError):
        validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))

    response["rows"] = (
        {"question_id": "q1", "evidence_id": "e1", "score": 1.0, "rank": 2},
        {"question_id": "q1", "evidence_id": "e2", "score": 0.5, "rank": 1},
    )
    with pytest.raises(ModalPrivateRetrievalError):
        validate_modal_private_retrieval_response(response, expected_question_ids=("q1",))


def test_lifecycle_rejects_out_of_order_create_and_delete() -> None:
    lifecycle = ModalPrivateRetrievalLifecycle.initial("campaign-1")
    with pytest.raises(ModalPrivateRetrievalError):
        lifecycle.mark_deleted(deletion_receipt_checksum=SHA)
    with pytest.raises(ModalPrivateRetrievalError):
        lifecycle.mark_created().mark_created()
