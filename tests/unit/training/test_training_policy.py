"""CPU-only training-data and authorization governance tests."""

from __future__ import annotations

import pytest

from legal_rag.training.authorization import (
    check_milestone_gate,
    check_training_authorization,
)
from legal_rag.training.dataset_policy import (
    DatasetPolicyError,
    validate_training_dataset,
    validate_training_example,
)
from legal_rag.training.provenance import TrainingExample, parse_training_example


def _payload(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "training.example.v1",
        "example_id": "example-1",
        "task": "generation",
        "question_id": "question-1",
        "split": "train",
        "question_source_checksum": "sha256:" + ("a" * 64),
        "evidence_ids": ["chunk-1"],
        "target_source": "official_train_answer",
        "target_checksum": "sha256:" + ("b" * 64),
        "contains_generated_text": False,
        "construction_version": "training-construction.v1",
    }
    values.update(changes)
    return values


def test_json_compatible_training_example_parses() -> None:
    example = parse_training_example(_payload())

    assert isinstance(example, TrainingExample)
    assert example.evidence_ids == ("chunk-1",)
    validate_training_example(example)


@pytest.mark.parametrize(
    "split", ("development", "local-test", "public", "private", "external", "synthetic")
)
def test_non_train_splits_are_rejected_with_stable_code(split: str) -> None:
    with pytest.raises(DatasetPolicyError) as exc:
        validate_training_dataset((_payload(split=split),))

    assert exc.value.code == "PROVENANCE_SPLIT_REJECTED"


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    (
        ({"contains_generated_text": True}, "PROVENANCE_GENERATED_TEXT"),
        ({"target_source": "external_answer"}, "PROVENANCE_TARGET_SOURCE_REJECTED"),
        ({"evidence_ids": []}, "PROVENANCE_EVIDENCE_MISSING"),
        ({"question_source_checksum": None}, "PROVENANCE_SCHEMA_INVALID"),
        ({"unexpected": "field"}, "PROVENANCE_SCHEMA_INVALID"),
    ),
)
def test_untrusted_rows_fail_closed(changes: dict[str, object], expected_code: str) -> None:
    with pytest.raises(DatasetPolicyError) as exc:
        validate_training_dataset((_payload(**changes),))

    assert exc.value.code == expected_code


@pytest.mark.parametrize(
    ("task", "target_source"),
    (
        ("generation", "deterministic_relevance"),
        ("embedding", "official_train_answer"),
        ("reranking", "official_train_answer"),
    ),
)
def test_task_target_pair_is_rejected(task: str, target_source: str) -> None:
    with pytest.raises(DatasetPolicyError) as exc:
        validate_training_dataset((_payload(task=task, target_source=target_source),))

    assert exc.value.code == "PROVENANCE_TARGET_SOURCE_REJECTED"


def test_dataset_has_no_partial_acceptance() -> None:
    rows = (_payload(), _payload(example_id="example-2", split="development"))

    with pytest.raises(DatasetPolicyError) as exc:
        validate_training_dataset(rows)

    assert exc.value.code == "PROVENANCE_SPLIT_REJECTED"


def test_valid_dataset_report_is_deterministic() -> None:
    rows = (
        _payload(),
        _payload(
            example_id="example-2",
            question_id="question-2",
            task="embedding",
            evidence_ids=["chunk-2", "chunk-3"],
            target_source="deterministic_relevance",
            target_checksum="sha256:" + ("c" * 64),
        ),
    )

    report = validate_training_dataset(rows)

    assert report.candidate_rows == 2
    assert report.accepted_rows == 2
    assert report.rejected_rows == 0
    assert report.unique_question_ids == 2
    assert report.unique_evidence_ids == 3


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(DatasetPolicyError) as exc:
        validate_training_dataset(())

    assert exc.value.code == "DATASET_EMPTY"


def test_mil005_is_still_closed() -> None:
    gate = check_milestone_gate("PRE-MIL-005", "MIL-005", owner_approval=False)

    assert gate.state == "not_started"
    assert gate.blocking_codes == (
        "MILESTONE_MIL-005_NOT_ACTIVE",
        "OWNER_APPROVAL_MIL-005_MISSING",
    )


def test_fine_tuning_requires_every_independent_gate() -> None:
    authorization = check_training_authorization(
        action="FT-EMBED",
        milestone="MIL-005",
        owner_approved=False,
        model_btc_approved=False,
        parameter_gate_passed=False,
        dataset_provenance_valid=False,
        backend_authorized=False,
        oq003_resolved=False,
    )

    assert authorization.can_proceed is False
    assert authorization.blocking_codes == (
        "OWNER_FT_APPROVAL_MISSING",
        "MODEL_BTC_APPROVAL_MISSING",
        "PARAMETER_GATE_FAILED",
        "DATASET_PROVENANCE_INVALID",
        "BACKEND_NOT_AUTHORIZED",
        "OQ003_UNRESOLVED",
    )


def test_no_ft_records_no_workload_decision() -> None:
    authorization = check_training_authorization(
        action="NO_FT",
        milestone="PRE-MIL-005",
        owner_approved=False,
        model_btc_approved=False,
        parameter_gate_passed=False,
        dataset_provenance_valid=False,
        backend_authorized=False,
        oq003_resolved=False,
    )

    assert authorization.can_proceed is True
    assert authorization.blocking_codes == ()


def test_local_training_does_not_require_modal_data_permission() -> None:
    authorization = check_training_authorization(
        action="FT-RERANK",
        milestone="MIL-005",
        owner_approved=True,
        model_btc_approved=True,
        parameter_gate_passed=True,
        dataset_provenance_valid=True,
        backend_authorized=True,
        oq003_resolved=False,
        backend="local",
    )

    assert authorization.can_proceed is True
    assert authorization.blocking_codes == ()
