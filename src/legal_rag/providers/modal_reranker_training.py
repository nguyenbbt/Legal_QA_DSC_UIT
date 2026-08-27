"""Pure closed-schema preflight for the paid D-057 R-008 Modal profile."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Literal, NoReturn, cast

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.training.provenance import ProvenanceError, parse_training_example

_SHA_PREFIX = "sha256:"
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "model_manifest_checksum",
        "recipe_checksum",
        "dataset_manifest_checksum",
        "groups_checksum",
        "provenance_checksum",
        "group_count",
        "pair_count",
        "maximum_length",
        "seed",
        "base_parameter_count",
        "whole_system_base_parameter_count",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "question_id",
        "split",
        "question",
        "question_checksum",
        "positives",
        "negatives",
        "target_checksum",
        "construction_version",
        "contains_generated_text",
    }
)
_PASSAGE_FIELDS = frozenset(
    {
        "evidence_id",
        "context_id",
        "evidence_checksum",
        "hierarchy_path",
        "canonical_start",
        "canonical_end",
        "text",
    }
)
_NEGATIVE_FIELDS = _PASSAGE_FIELDS | {"negative_type"}
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "construction_version",
        "group_count",
        "pair_count",
        "rejected_no_negative_count",
        "unique_question_ids",
        "unique_evidence_ids",
        "maximum_negatives",
        "question_source_checksum",
        "split_manifest_checksum",
        "selection_checksum",
        "chunks_checksum",
        "index_checksum",
        "groups_checksum",
        "provenance_checksum",
        "contains_generated_text",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "adapter_path",
        "adapter_checksum",
        "adapter_parameter_count",
        "whole_system_parameter_count",
        "training_metrics",
        "telemetry",
    }
)
_TRAINING_METRIC_FIELDS = frozenset(
    {
        "epochs",
        "mean_train_loss",
        "optimizer_steps",
        "train_pairs",
        "validation_loss",
        "validation_pair_accuracy",
        "validation_pairs",
    }
)
_TELEMETRY_FIELDS = frozenset({"elapsed_seconds", "gpu_name", "peak_cuda_bytes", "torch_version"})
_NEGATIVE_TYPES = frozenset(
    {
        "SAME_DOCUMENT_WRONG_ARTICLE",
        "SAME_ARTICLE_WRONG_CLAUSE",
        "SAME_CLAUSE_WRONG_POINT",
        "ADJACENT_COORDINATE",
    }
)


class ModalRerankerTrainingError(Exception):
    """Stable error raised before transfer or before accepting remote results."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class R008TrainingPayload:
    run_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    model_manifest_checksum: str
    recipe_checksum: str
    dataset_manifest_checksum: str
    groups_checksum: str
    provenance_checksum: str
    group_count: int
    pair_count: int
    maximum_length: int
    seed: int
    base_parameter_count: int
    whole_system_base_parameter_count: int


@dataclass(frozen=True, slots=True)
class R008TrainingResponse:
    run_id: str
    adapter_path: str
    adapter_checksum: str
    adapter_parameter_count: int
    whole_system_parameter_count: int
    training_metrics: dict[str, float | int]
    telemetry: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class R008TrainingLifecycle:
    state: Literal["planned", "preflighted", "remote_complete", "downloaded", "deleted"] = "planned"
    remote_run_id: str | None = None
    adapter_checksum: str | None = None

    def record_preflight(self) -> R008TrainingLifecycle:
        if self.state != "planned":
            _fail("MODAL_R008_LIFECYCLE_INVALID", "preflight transition is invalid")
        return replace(self, state="preflighted")

    def record_remote_run(self, remote_run_id: str) -> R008TrainingLifecycle:
        if self.state != "preflighted" or not remote_run_id.strip():
            _fail("MODAL_R008_LIFECYCLE_INVALID", "remote-run transition is invalid")
        return replace(self, state="remote_complete", remote_run_id=remote_run_id)

    def record_download(self, adapter_checksum: str) -> R008TrainingLifecycle:
        if self.state != "remote_complete" or not _checksum(adapter_checksum):
            _fail("MODAL_R008_LIFECYCLE_INVALID", "download transition is invalid")
        return replace(self, state="downloaded", adapter_checksum=adapter_checksum)

    def record_volume_deleted(self) -> R008TrainingLifecycle:
        if self.state != "downloaded":
            _fail("MODAL_R008_LIFECYCLE_INVALID", "Volume deletion requires a verified download")
        return replace(self, state="deleted")


def _fail(code: str, message: str) -> NoReturn:
    raise ModalRerankerTrainingError(code, message)


def _checksum(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_SHA_PREFIX):
        return False
    digest = value.removeprefix(_SHA_PREFIX)
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _object(data: bytes, *, code: str) -> dict[str, object]:
    try:
        value: object = json.loads(data, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ModalRerankerTrainingError(code, "Modal R-008 artifact is invalid") from error
    if not isinstance(value, dict):
        _fail(code, "Modal R-008 artifact must be an object")
    return cast(dict[str, object], value)


def _rows(data: bytes, *, code: str) -> tuple[dict[str, object], ...]:
    if not data or not data.endswith(b"\n"):
        _fail(code, "Modal R-008 JSONL must be non-empty and newline terminated")
    return tuple(_object(line, code=code) for line in data.splitlines())


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _required_string(value: dict[str, object], field: str, *, code: str) -> str:
    member = value.get(field)
    if not isinstance(member, str) or not member.strip():
        _fail(code, f"{field} must be a non-empty string")
    return member


def _required_positive_int(value: dict[str, object], field: str, *, code: str) -> int:
    member = value.get(field)
    if not _positive_int(member):
        _fail(code, f"{field} must be a positive integer")
    return cast(int, member)


def _parse_request(request_data: bytes) -> tuple[R008TrainingPayload, dict[str, object]]:
    request = _object(request_data, code="MODAL_R008_REQUEST_SCHEMA_INVALID")
    if set(request) != _REQUEST_FIELDS or request.get("schema_version") != (
        "modal.r008.training-request.v1"
    ):
        _fail("MODAL_R008_REQUEST_SCHEMA_INVALID", "training request schema is not closed")
    code = "MODAL_R008_REQUEST_SCHEMA_INVALID"
    run_id = _required_string(request, "run_id", code=code)
    model_id = _required_string(request, "model_id", code=code)
    model_revision = _required_string(request, "model_revision", code=code)
    tokenizer_revision = _required_string(request, "tokenizer_revision", code=code)
    checksum_fields = (
        "model_manifest_checksum",
        "recipe_checksum",
        "dataset_manifest_checksum",
        "groups_checksum",
        "provenance_checksum",
    )
    if any(not _checksum(request.get(field)) for field in checksum_fields):
        _fail("MODAL_R008_REQUEST_SCHEMA_INVALID", "training request checksum is invalid")
    group_count = _required_positive_int(request, "group_count", code=code)
    pair_count = _required_positive_int(request, "pair_count", code=code)
    maximum_length = _required_positive_int(request, "maximum_length", code=code)
    base_parameter_count = _required_positive_int(request, "base_parameter_count", code=code)
    whole_system_base_parameter_count = _required_positive_int(
        request, "whole_system_base_parameter_count", code=code
    )
    if request.get("seed") != 42 or not 128 <= maximum_length <= 4096:
        _fail("MODAL_R008_REQUEST_SCHEMA_INVALID", "training request recipe bound is invalid")
    if whole_system_base_parameter_count >= 4_000_000_000:
        _fail("MODEL_PARAMETER_LIMIT", "whole-system base parameter count reaches the limit")
    return (
        R008TrainingPayload(
            run_id=run_id,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            model_manifest_checksum=str(request["model_manifest_checksum"]),
            recipe_checksum=str(request["recipe_checksum"]),
            dataset_manifest_checksum=str(request["dataset_manifest_checksum"]),
            groups_checksum=str(request["groups_checksum"]),
            provenance_checksum=str(request["provenance_checksum"]),
            group_count=group_count,
            pair_count=pair_count,
            maximum_length=maximum_length,
            seed=42,
            base_parameter_count=base_parameter_count,
            whole_system_base_parameter_count=whole_system_base_parameter_count,
        ),
        request,
    )


def _validate_passage(value: object, *, negative: bool) -> None:
    expected = _NEGATIVE_FIELDS if negative else _PASSAGE_FIELDS
    if not isinstance(value, dict) or set(value) != expected:
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training passage schema is not closed")
    passage = cast(dict[str, object], value)
    for field in ("evidence_id", "context_id", "text"):
        _required_string(passage, field, code="MODAL_R008_GROUP_SCHEMA_INVALID")
    if not _checksum(passage.get("evidence_checksum")):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training passage checksum is invalid")
    path = passage.get("hierarchy_path")
    if not isinstance(path, list) or not path or any(not isinstance(item, str) for item in path):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training hierarchy is invalid")
    start, end = passage.get("canonical_start"), passage.get("canonical_end")
    if type(start) is not int or type(end) is not int or int(start) < 0 or int(end) <= int(start):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training canonical span is invalid")
    if negative and passage.get("negative_type") not in _NEGATIVE_TYPES:
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training negative type is invalid")


def _validate_group(group: dict[str, object]) -> tuple[tuple[str, ...], int]:
    if set(group) != _GROUP_FIELDS or group.get("schema_version") != ("reranker.training-group.v1"):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training group schema is not closed")
    if group.get("split") != "train":
        _fail("MODAL_R008_NONTRAIN_REJECTED", "only official train groups may leave local")
    if group.get("contains_generated_text") is not False:
        _fail("MODAL_R008_GENERATED_TEXT_REJECTED", "generated training text is forbidden")
    for field in ("group_id", "question_id", "question", "construction_version"):
        _required_string(group, field, code="MODAL_R008_GROUP_SCHEMA_INVALID")
    if checksum_bytes(str(group["question"]).encode()) != group.get("question_checksum"):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training question checksum changed")

    positives = group.get("positives")
    negatives = group.get("negatives")
    if (
        not isinstance(positives, list)
        or not positives
        or not isinstance(negatives, list)
        or not negatives
    ):
        _fail("MODAL_R008_GROUP_SCHEMA_INVALID", "training group requires both labels")
    for passage in positives:
        _validate_passage(passage, negative=False)
    for passage in negatives:
        _validate_passage(passage, negative=True)

    positive_ids = tuple(
        str(cast(dict[str, object], passage)["evidence_id"]) for passage in positives
    )
    negative_ids = tuple(
        str(cast(dict[str, object], passage)["evidence_id"]) for passage in negatives
    )
    evidence_ids = positive_ids + negative_ids
    if len(evidence_ids) != len(set(evidence_ids)):
        _fail("MODAL_R008_IDENTITY_MISMATCH", "training group evidence IDs overlap")
    return evidence_ids, len(positive_ids) * len(negative_ids)


def validate_r008_training_payload(
    *,
    request_data: bytes,
    groups_data: bytes,
    provenance_data: bytes,
    dataset_manifest_data: bytes,
) -> R008TrainingPayload:
    """Validate every private field and checksum before any Modal invocation."""

    payload, _ = _parse_request(request_data)
    groups = _rows(groups_data, code="MODAL_R008_GROUP_SCHEMA_INVALID")
    computed_pair_count = 0
    group_evidence_ids: list[tuple[str, ...]] = []
    for group in groups:
        evidence_ids, pair_count = _validate_group(group)
        group_evidence_ids.append(evidence_ids)
        computed_pair_count += pair_count

    provenance_rows = _rows(provenance_data, code="MODAL_R008_PROVENANCE_INVALID")
    try:
        examples = tuple(parse_training_example(row) for row in provenance_rows)
    except ProvenanceError as error:
        raise ModalRerankerTrainingError(
            "MODAL_R008_PROVENANCE_INVALID", "training provenance is invalid"
        ) from error
    if any(example.task != "reranking" or example.split != "train" for example in examples):
        _fail("MODAL_R008_NONTRAIN_REJECTED", "provenance is not official-train reranking")

    manifest = _object(dataset_manifest_data, code="MODAL_R008_MANIFEST_INVALID")
    if set(manifest) != _MANIFEST_FIELDS or manifest.get("schema_version") != (
        "reranker.training-manifest.v1"
    ):
        _fail("MODAL_R008_MANIFEST_INVALID", "dataset manifest schema is not closed")
    if manifest.get("contains_generated_text") is not False:
        _fail("MODAL_R008_GENERATED_TEXT_REJECTED", "dataset manifest permits generated text")

    checks = (
        (payload.groups_checksum, checksum_bytes(groups_data)),
        (payload.provenance_checksum, checksum_bytes(provenance_data)),
        (payload.dataset_manifest_checksum, checksum_bytes(dataset_manifest_data)),
        (manifest.get("groups_checksum"), checksum_bytes(groups_data)),
        (manifest.get("provenance_checksum"), checksum_bytes(provenance_data)),
    )
    if any(expected != actual for expected, actual in checks):
        _fail("MODAL_R008_CHECKSUM_MISMATCH", "training artifact checksum changed")
    if (
        payload.group_count != len(groups)
        or payload.group_count != len(examples)
        or manifest.get("group_count") != len(groups)
        or payload.pair_count != manifest.get("pair_count")
        or payload.pair_count != computed_pair_count
    ):
        _fail("MODAL_R008_CARDINALITY_MISMATCH", "training artifact counts changed")
    group_ids = tuple(str(group["question_id"]) for group in groups)
    if group_ids != tuple(example.question_id for example in examples):
        _fail("MODAL_R008_IDENTITY_MISMATCH", "training group/provenance order changed")
    if tuple(group_evidence_ids) != tuple(example.evidence_ids for example in examples):
        _fail("MODAL_R008_IDENTITY_MISMATCH", "training evidence/provenance identity changed")
    return payload


def _validate_response_telemetry(
    response: dict[str, object],
) -> tuple[dict[str, float | int], dict[str, float | int | str]]:
    metrics_value = response.get("training_metrics")
    telemetry_value = response.get("telemetry")
    if (
        not isinstance(metrics_value, dict)
        or not metrics_value
        or not isinstance(telemetry_value, dict)
        or not telemetry_value
    ):
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training telemetry is invalid")
    metrics = cast(dict[str, object], metrics_value)
    telemetry = cast(dict[str, object], telemetry_value)
    if set(metrics) != _TRAINING_METRIC_FIELDS or set(telemetry) != _TELEMETRY_FIELDS:
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training telemetry schema is not closed")

    numeric_values = tuple(metrics.values()) + (
        telemetry["elapsed_seconds"],
        telemetry["peak_cuda_bytes"],
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training metric is non-finite")
    if (
        metrics["epochs"] != 2
        or not _positive_int(metrics["optimizer_steps"])
        or not _positive_int(metrics["train_pairs"])
        or not _positive_int(metrics["validation_pairs"])
        or float(cast(int | float, metrics["mean_train_loss"])) < 0.0
        or float(cast(int | float, metrics["validation_loss"])) < 0.0
        or not 0.0 <= float(cast(int | float, metrics["validation_pair_accuracy"])) <= 1.0
        or float(cast(int | float, telemetry["elapsed_seconds"])) <= 0.0
        or not _positive_int(telemetry["peak_cuda_bytes"])
    ):
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training telemetry value is invalid")
    for field in ("gpu_name", "torch_version"):
        value = telemetry[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 100
            or "\n" in value
            or "\r" in value
        ):
            _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training telemetry text is invalid")
    return (
        cast(dict[str, float | int], metrics),
        cast(dict[str, float | int | str], telemetry),
    )


def validate_r008_training_response(
    *, request_data: bytes, response_data: bytes
) -> R008TrainingResponse:
    """Validate the closed adapter and aggregate-telemetry return boundary."""

    request, _ = _parse_request(request_data)
    response = _object(response_data, code="MODAL_R008_RESPONSE_SCHEMA_INVALID")
    if set(response) != _RESPONSE_FIELDS or response.get("schema_version") != (
        "modal.r008.training-response.v1"
    ):
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training response schema is not closed")
    if response.get("run_id") != request.run_id or not _checksum(response.get("adapter_checksum")):
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "training response identity is invalid")
    adapter_path = _required_string(
        response, "adapter_path", code="MODAL_R008_RESPONSE_SCHEMA_INVALID"
    )
    path = PurePosixPath(adapter_path)
    if path.is_absolute() or ".." in path.parts or "\\" in adapter_path:
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "adapter path is unsafe")
    adapter_count = _required_positive_int(
        response, "adapter_parameter_count", code="MODAL_R008_RESPONSE_SCHEMA_INVALID"
    )
    whole_count = _required_positive_int(
        response, "whole_system_parameter_count", code="MODAL_R008_RESPONSE_SCHEMA_INVALID"
    )
    if whole_count >= 4_000_000_000:
        _fail("MODEL_PARAMETER_LIMIT", "whole-system parameters reach the exclusive limit")
    if whole_count != request.whole_system_base_parameter_count + adapter_count:
        _fail("MODAL_R008_RESPONSE_SCHEMA_INVALID", "whole-system parameter count is inconsistent")
    metrics, telemetry = _validate_response_telemetry(response)
    return R008TrainingResponse(
        run_id=request.run_id,
        adapter_path=adapter_path,
        adapter_checksum=str(response["adapter_checksum"]),
        adapter_parameter_count=adapter_count,
        whole_system_parameter_count=whole_count,
        training_metrics=metrics,
        telemetry=telemetry,
    )


__all__ = [
    "ModalRerankerTrainingError",
    "R008TrainingLifecycle",
    "R008TrainingPayload",
    "R008TrainingResponse",
    "validate_r008_training_payload",
    "validate_r008_training_response",
]
