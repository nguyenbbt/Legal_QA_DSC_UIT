"""CPU-only model governance contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from legal_rag.models.approval import (
    validate_acquisition_mode,
    validate_official_profile,
)
from legal_rag.models.manifest import (
    COMPETITION_PARAMETER_LIMIT,
    ModelComponentManifest,
    ModelManifestError,
    ModelParameterManifest,
    ModelRunFingerprintInputs,
    compute_manifest_checksum,
    compute_model_run_inputs_checksum,
)
from legal_rag.models.parameter_audit import ParameterTensor, audit_parameters

_AUDIT_CHECKSUM = "sha256:" + ("a" * 64)


def _component(**changes: object) -> ModelComponentManifest:
    values: dict[str, object] = {
        "role": "embedding",
        "model_id": "test/model",
        "model_revision": "revision-1",
        "tokenizer_id": "test/tokenizer",
        "tokenizer_revision": "tokenizer-1",
        "license": "Apache-2.0",
        "exact_parameter_count": 600_000_000,
        "trainable_parameter_count": 0,
        "adapter_parameter_count": 0,
        "quantization": None,
        "parameter_audit_checksum": _AUDIT_CHECKSUM,
        "btc_approval_state": "approved",
        "btc_approval_evidence": "D-model-test",
        "local_model_hash": "sha256:" + ("b" * 64),
        "local_tokenizer_hash": "sha256:" + ("c" * 64),
    }
    values.update(changes)
    return ModelComponentManifest.model_validate(values)


def _manifest(*components: ModelComponentManifest) -> ModelParameterManifest:
    models = components or (_component(),)
    total = sum(model.exact_parameter_count + model.adapter_parameter_count for model in models)
    return ModelParameterManifest.model_validate(
        {
            "schema_version": "model.parameter_manifest.v1",
            "models": models,
            "system_parameter_count": total,
            "competition_limit_exclusive": COMPETITION_PARAMETER_LIMIT,
            "passes_parameter_gate": total < COMPETITION_PARAMETER_LIMIT,
        }
    )


def test_exact_numel_and_adapter_accounting() -> None:
    report = audit_parameters(
        (
            ParameterTensor("base.weight", (3, 4), "base", True),
            ParameterTensor("base.bias", (3,), "base", False),
            ParameterTensor("adapter.a", (2, 3), "adapter", True),
        )
    )

    assert report.exact_parameter_count == 15
    assert report.adapter_parameter_count == 6
    assert report.trainable_parameter_count == 18
    assert report.parameter_audit_checksum.startswith("sha256:")


def test_parameter_audit_rejects_duplicate_tensor_names() -> None:
    tensors = (
        ParameterTensor("weight", (2, 2), "base", True),
        ParameterTensor("weight", (2, 2), "adapter", True),
    )

    with pytest.raises(ModelManifestError) as exc:
        audit_parameters(tensors)

    assert exc.value.code == "MODEL_PARAMETER_AUDIT_DUPLICATE_TENSOR"


def test_quantization_does_not_change_parameter_total() -> None:
    plain = _manifest(_component())
    quantized = _manifest(_component(quantization="nf4"))

    assert plain.system_parameter_count == quantized.system_parameter_count


@pytest.mark.parametrize(
    ("total", "passes"),
    ((3_999_999_999, True), (4_000_000_000, False), (4_000_000_001, False)),
)
def test_exclusive_parameter_limit_is_derived(total: int, passes: bool) -> None:
    manifest = _manifest(_component(exact_parameter_count=total))

    assert manifest.passes_parameter_gate is passes
    if passes:
        validate_official_profile(manifest)
    else:
        with pytest.raises(ModelManifestError) as exc:
            validate_official_profile(manifest)
        assert exc.value.code == "MODEL_PARAMETER_LIMIT"


def test_manifest_rejects_incorrect_total_or_gate_result() -> None:
    component = _component()
    base = {
        "schema_version": "model.parameter_manifest.v1",
        "models": (component,),
        "competition_limit_exclusive": COMPETITION_PARAMETER_LIMIT,
    }

    with pytest.raises(ValidationError):
        ModelParameterManifest.model_validate(
            {**base, "system_parameter_count": 1, "passes_parameter_gate": True}
        )
    with pytest.raises(ValidationError):
        ModelParameterManifest.model_validate(
            {
                **base,
                "system_parameter_count": 600_000_000,
                "passes_parameter_gate": False,
            }
        )


def test_manifest_requires_unique_ordered_roles() -> None:
    embedding = _component(role="embedding", model_id="test/embedding")
    reranker = _component(role="reranker", model_id="test/reranker")

    with pytest.raises(ValidationError):
        _manifest(reranker, embedding)
    with pytest.raises(ValidationError):
        _manifest(embedding, embedding.model_copy(update={"model_id": "test/other"}))


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    (
        ({"model_revision": None}, "MODEL_REVISION_UNPINNED"),
        ({"tokenizer_revision": None}, "MODEL_REVISION_UNPINNED"),
        ({"local_model_hash": None}, "MODEL_REVISION_UNPINNED"),
        ({"local_tokenizer_hash": None}, "MODEL_REVISION_UNPINNED"),
        ({"license": None}, "MODEL_LICENSE_MISSING"),
        ({"parameter_audit_checksum": None}, "MODEL_PARAMETER_AUDIT_MISSING"),
        ({"btc_approval_state": "pending"}, "MODEL_COMPETITION_REGISTRATION_MISSING"),
        ({"btc_approval_state": "rejected"}, "MODEL_COMPETITION_REGISTRATION_MISSING"),
        ({"btc_approval_evidence": None}, "MODEL_COMPETITION_REGISTRATION_MISSING"),
    ),
)
def test_official_profile_fails_closed(changes: dict[str, object], expected_code: str) -> None:
    manifest = _manifest(_component(**changes))

    with pytest.raises(ModelManifestError) as exc:
        validate_official_profile(manifest)

    assert exc.value.code == expected_code


def test_official_preflight_error_precedence_is_stable() -> None:
    component = _component(
        model_revision=None,
        license=None,
        parameter_audit_checksum=None,
        btc_approval_state="pending",
    )

    with pytest.raises(ModelManifestError) as exc:
        validate_official_profile(_manifest(component))

    assert exc.value.code == "MODEL_REVISION_UNPINNED"


def test_experiment_profile_accepts_an_unregistered_but_fully_audited_model() -> None:
    from legal_rag.models.approval import validate_experiment_profile

    validate_experiment_profile(
        _manifest(
            _component(
                btc_approval_state="pending",
                btc_approval_evidence=None,
            )
        )
    )


def test_manifest_checksum_is_deterministic_and_material() -> None:
    first = _manifest(_component())
    second = _manifest(_component(model_revision="revision-2"))

    assert compute_manifest_checksum(first) == compute_manifest_checksum(first)
    assert compute_manifest_checksum(first) != compute_manifest_checksum(second)


def test_model_run_identity_binds_all_governance_inputs() -> None:
    base = ModelRunFingerprintInputs.model_validate(
        {
            "schema_version": "model.run_fingerprint_inputs.v1",
            "model_parameter_manifest_checksum": "sha256:" + ("1" * 64),
            "model_hashes": ("sha256:" + ("2" * 64),),
            "tokenizer_hashes": ("sha256:" + ("3" * 64),),
            "adapter_hashes": (),
            "prompt_checksum": "sha256:" + ("4" * 64),
            "training_recipe_checksum": None,
        }
    )
    changed = base.model_copy(update={"prompt_checksum": "sha256:" + ("5" * 64)})

    assert compute_model_run_inputs_checksum(base) != compute_model_run_inputs_checksum(changed)


def test_acquisition_is_prepare_online_only() -> None:
    validate_acquisition_mode("prepare-online")

    for execution_mode in ("local-offline", "private-modal"):
        with pytest.raises(ModelManifestError) as exc:
            validate_acquisition_mode(execution_mode)
        assert exc.value.code == "MODEL_ACQUISITION_MODE_INVALID"
