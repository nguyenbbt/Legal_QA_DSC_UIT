"""Train-calibrated, deletion-only answer compaction."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes

_POLICY_ID = "official-train-median-complete-prefix.v1"
_SENTENCE_TERMINAL = re.compile(r"[.!?](?=\s|$)")


class AnswerCompactionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AnswerCompactionPolicy:
    policy_id: str
    source_checksum: str
    source_row_count: int
    maximum_whitespace_tokens: int
    quantile: float
    quantile_method: str
    sentence_terminal_pattern: str


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def derive_answer_compaction_policy(
    source_data: bytes,
    *,
    split: str,
) -> tuple[AnswerCompactionPolicy, bytes]:
    """Freeze the nearest-rank train-median answer length without model output."""

    if split != "train":
        raise AnswerCompactionError(
            "ANSWER_COMPACTION_SPLIT_INVALID",
            "answer compaction may be calibrated only from the official train split",
        )
    try:
        value = json.loads(source_data)
        if not isinstance(value, dict) or not value:
            raise ValueError
        lengths: list[int] = []
        for question_id, row in value.items():
            if (
                not isinstance(question_id, str)
                or not question_id
                or not isinstance(row, dict)
                or set(row) != {"question", "answer"}
                or not isinstance(row["question"], str)
                or not row["question"].strip()
                or not isinstance(row["answer"], str)
                or not row["answer"].strip()
            ):
                raise ValueError
            length = len(_canonical_text(row["answer"]).split())
            if length < 1:
                raise ValueError
            lengths.append(length)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise AnswerCompactionError(
            "ANSWER_COMPACTION_SOURCE_INVALID",
            "official-train answer source is invalid",
        ) from error
    lengths.sort()
    quantile = 0.5
    rank = math.ceil(quantile * len(lengths))
    policy = AnswerCompactionPolicy(
        policy_id=_POLICY_ID,
        source_checksum=checksum_bytes(source_data),
        source_row_count=len(lengths),
        maximum_whitespace_tokens=lengths[rank - 1],
        quantile=quantile,
        quantile_method="nearest-rank",
        sentence_terminal_pattern=_SENTENCE_TERMINAL.pattern,
    )
    manifest = content_json_bytes(
        {
            "schema_version": "answer.compaction.policy.v1",
            **asdict(policy),
            "normalization": "unicode-nfc-collapse-whitespace.v1",
            "transformation": "literal-complete-sentence-prefix-or-identity.v1",
            "contains_generated_calibration_text": False,
        }
    )
    return policy, manifest


def compact_answer(answer: str, policy: AnswerCompactionPolicy) -> str:
    """Return the canonical answer or one complete literal prefix of it."""

    if not isinstance(answer, str) or not answer.strip():
        raise AnswerCompactionError(
            "ANSWER_COMPACTION_INPUT_INVALID", "answer must be a non-empty string"
        )
    canonical = _canonical_text(answer)
    tokens = tuple(re.finditer(r"\S+", canonical))
    if len(tokens) <= policy.maximum_whitespace_tokens:
        return canonical
    prefix = canonical[: tokens[policy.maximum_whitespace_tokens - 1].end()]
    boundaries = tuple(_SENTENCE_TERMINAL.finditer(prefix))
    if not boundaries:
        return canonical
    return prefix[: boundaries[-1].end()].rstrip()


def _prediction_answers(data: bytes) -> tuple[tuple[str, str], ...]:
    try:
        value: Any = json.loads(data)
        if not isinstance(value, dict) or not value:
            raise ValueError
        answers: list[tuple[str, str]] = []
        for question_id, row in value.items():
            if (
                not isinstance(question_id, str)
                or not question_id
                or not isinstance(row, dict)
                or set(row) != {"answer"}
                or not isinstance(row["answer"], str)
                or not row["answer"].strip()
            ):
                raise ValueError
            answers.append((question_id, _canonical_text(row["answer"])))
        return tuple(answers)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise AnswerCompactionError(
            "ANSWER_COMPACTION_PREDICTIONS_INVALID", "predictions are invalid"
        ) from error


def build_deletion_only_grounding_proof(
    baseline_predictions_data: bytes,
    candidate_predictions_data: bytes,
) -> bytes:
    """Prove that the candidate adds or rewrites no generated claim text."""

    baseline = _prediction_answers(baseline_predictions_data)
    candidate = _prediction_answers(candidate_predictions_data)
    if tuple(item[0] for item in baseline) != tuple(item[0] for item in candidate):
        raise AnswerCompactionError(
            "ANSWER_COMPACTION_ID_MISMATCH", "prediction identities or order differ"
        )
    changed = 0
    removed_characters = 0
    for (question_id, original), (_, compacted) in zip(baseline, candidate, strict=True):
        if not original.startswith(compacted) or (
            original != compacted and compacted[-1] not in ".?!"
        ):
            raise AnswerCompactionError(
                "ANSWER_COMPACTION_NOT_PREFIX_ONLY",
                f"candidate answer is not a complete literal prefix: {question_id}",
            )
        if original != compacted:
            changed += 1
            removed_characters += len(original) - len(compacted)
    return content_json_bytes(
        {
            "schema_version": "answer.compaction.deletion-proof.v1",
            "proof_state": "passed",
            "question_count": len(baseline),
            "changed_answer_count": changed,
            "removed_character_count": removed_characters,
            "baseline_predictions_checksum": checksum_bytes(baseline_predictions_data),
            "candidate_predictions_checksum": checksum_bytes(candidate_predictions_data),
            "claims_added": 0,
            "claims_rewritten": 0,
            "grounding_guard": "unsupported-claim-set-cannot-increase-under-prefix-deletion",
        }
    )


__all__ = [
    "AnswerCompactionError",
    "AnswerCompactionPolicy",
    "build_deletion_only_grounding_proof",
    "compact_answer",
    "derive_answer_compaction_policy",
]
