"""Deterministic train-only discovery metrics for the D-066 tournament."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, NoReturn, cast

from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.training.rag_sft import RagSftBuildError, load_gold_questions

_CUTOFFS = (5, 10, 20, 50)
_POSITIVE_CLASSES = frozenset(
    {
        "EXACT_DOC_ARTICLE_POINT",
        "EXACT_DOC_ARTICLE_CLAUSE",
        "EXACT_DOC_ARTICLE",
        "SAME_COORDINATE_MULTICHUNK",
    }
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class DiscoveryTournamentError(Exception):
    """Stable fail-closed error at the D-066 evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class DiscoveryGroup:
    question_id: str
    question_checksum: str
    question: str
    source_answer_checksum: str
    gold_answer: str
    positive_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    chunk_id: str
    display_text: str


@dataclass(frozen=True, slots=True)
class DiscoveryRanking:
    question_id: str
    candidates: tuple[DiscoveryCandidate, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryQuestionMetrics:
    question_id: str
    question_checksum: str
    positive_count: int
    first_positive_rank: int | None
    recall_at: dict[int, float]
    evidence_set_recall_at: dict[int, float]
    answer_bearing_coverage_at: dict[int, float]


@dataclass(frozen=True, slots=True)
class DiscoveryArmEvaluation:
    schema_version: str
    arm_id: str
    question_count: int
    positive_assignment_count: int
    recall_at: dict[int, float]
    evidence_set_recall_at: dict[int, float]
    answer_bearing_coverage_at: dict[int, float]
    mrr_at_50: float
    rows: tuple[DiscoveryQuestionMetrics, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryArmComparison:
    schema_version: str
    baseline_arm_id: str
    candidate_arm_id: str
    metric_deltas: dict[str, float]
    novel_positive_recovery_at_50: tuple[str, ...]
    lost_positive_recovery_at_50: tuple[str, ...]
    standing_winner: str
    decision_reason: str


def _fail(code: str, message: str) -> NoReturn:
    raise DiscoveryTournamentError(code, message)


def _jsonl_values(data: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if not data or data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        _fail("D066_INPUT_INVALID", f"{label} JSONL framing is invalid")
    values: list[dict[str, Any]] = []
    for line in data.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise DiscoveryTournamentError(
                "D066_INPUT_INVALID", f"{label} contains invalid JSON"
            ) from error
        if not isinstance(value, dict):
            _fail("D066_INPUT_INVALID", f"{label} row must be an object")
        values.append(cast(dict[str, Any], value))
    return tuple(values)


def load_discovery_groups(
    *,
    supervision_data: bytes,
    question_source_data: bytes,
    train_question_ids: Sequence[str],
    expected_positive_count: int,
    expected_supervision_checksum: str,
) -> tuple[DiscoveryGroup, ...]:
    """Load exactly the positive D-065 groups and reject split/provenance drift."""

    if checksum_bytes(supervision_data) != expected_supervision_checksum:
        _fail("D066_SUPERVISION_CHECKSUM_MISMATCH", "D-065 supervision checksum drifted")
    train_ids = tuple(train_question_ids)
    train_id_set = set(train_ids)
    if expected_positive_count < 1 or not train_ids or len(train_ids) != len(train_id_set):
        _fail("D066_TRAIN_PARTITION_INVALID", "active train partition is invalid")
    try:
        questions = load_gold_questions(question_source_data)
    except RagSftBuildError as error:
        raise DiscoveryTournamentError(
            "D066_QUESTION_SOURCE_INVALID", "official question source is invalid"
        ) from error
    question_by_id = {item.question_id: item for item in questions}
    if any(question_id not in question_by_id for question_id in train_ids):
        _fail("D066_TRAIN_PARTITION_INVALID", "train identity is absent from question source")

    groups: list[DiscoveryGroup] = []
    seen: set[str] = set()
    for value in _jsonl_values(supervision_data, label="D-065 supervision"):
        mapping_class = value.get("mapping_class")
        if mapping_class not in _POSITIVE_CLASSES:
            continue
        question_id = value.get("question_id")
        question_checksum = value.get("question_checksum")
        answer_checksum = value.get("source_answer_checksum")
        raw_chunk_ids = value.get("canonical_chunk_ids")
        if (
            value.get("schema_version") != "training.retrieval-supervision.group.v2"
            or not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(question_checksum, str)
            or not isinstance(answer_checksum, str)
            or not isinstance(raw_chunk_ids, list)
            or not raw_chunk_ids
            or not all(isinstance(item, str) and item for item in raw_chunk_ids)
            or len(raw_chunk_ids) != len(set(raw_chunk_ids))
        ):
            _fail("D066_SUPERVISION_INVALID", "positive supervision row is invalid")
        if question_id not in train_id_set:
            _fail("D066_SPLIT_LEAKAGE", "positive supervision includes a non-train row")
        question = question_by_id[question_id]
        if question.answer is None:
            _fail("D066_QUESTION_SOURCE_INVALID", "train answer is absent")
        if (
            checksum_bytes(question.question.encode("utf-8")) != question_checksum
            or checksum_bytes(question.answer.encode("utf-8")) != answer_checksum
        ):
            _fail("D066_PROVENANCE_MISMATCH", "question or answer checksum drifted")
        seen.add(question_id)
        groups.append(
            DiscoveryGroup(
                question_id=question_id,
                question_checksum=question_checksum,
                question=question.question,
                source_answer_checksum=answer_checksum,
                gold_answer=question.answer,
                positive_chunk_ids=tuple(cast(list[str], raw_chunk_ids)),
            )
        )
    groups.sort(key=lambda item: item.question_id.encode("utf-8"))
    if len(groups) != expected_positive_count:
        _fail("D066_POSITIVE_COUNT_MISMATCH", "D-065 positive-group count drifted")
    return tuple(groups)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _answer_sentences(answer: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for part in _SENTENCE_BOUNDARY.split(unicodedata.normalize("NFC", answer))
        if (normalized := _normalized(part))
    )


def _answer_bearing(answer: str, candidates: Sequence[DiscoveryCandidate], k: int) -> bool:
    sentences = _answer_sentences(answer)
    displays = tuple(_normalized(candidate.display_text) for candidate in candidates[:k])
    return any(sentence in display for sentence in sentences for display in displays)


def evaluate_discovery_arm(
    arm_id: str,
    groups: Sequence[DiscoveryGroup],
    rankings: Sequence[DiscoveryRanking],
) -> DiscoveryArmEvaluation:
    """Evaluate a complete arm with fixed top-K set-aware discovery metrics."""

    ordered_groups = tuple(sorted(groups, key=lambda item: item.question_id.encode("utf-8")))
    if not arm_id or not ordered_groups:
        _fail("D066_EVALUATION_INVALID", "arm ID and evaluation groups must be non-empty")
    ranking_by_id: dict[str, DiscoveryRanking] = {}
    for ranking in rankings:
        candidate_ids = tuple(item.chunk_id for item in ranking.candidates)
        if (
            not ranking.question_id
            or ranking.question_id in ranking_by_id
            or any(not candidate_id for candidate_id in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
        ):
            _fail("D066_RANKING_INVALID", "ranking identity or candidates are invalid")
        ranking_by_id[ranking.question_id] = ranking
    group_ids = {item.question_id for item in ordered_groups}
    if set(ranking_by_id) != group_ids:
        _fail("D066_RANKING_ID_MISMATCH", "rankings must cover every positive group exactly")

    rows: list[DiscoveryQuestionMetrics] = []
    for group in ordered_groups:
        ranking = ranking_by_id[group.question_id]
        candidate_ids = tuple(item.chunk_id for item in ranking.candidates)
        positive = frozenset(group.positive_chunk_ids)
        first_positive_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(candidate_ids[:50], start=1)
                if chunk_id in positive
            ),
            None,
        )
        rows.append(
            DiscoveryQuestionMetrics(
                question_id=group.question_id,
                question_checksum=group.question_checksum,
                positive_count=len(positive),
                first_positive_rank=first_positive_rank,
                recall_at={
                    cutoff: float(bool(positive.intersection(candidate_ids[:cutoff])))
                    for cutoff in _CUTOFFS
                },
                evidence_set_recall_at={
                    cutoff: float(positive.issubset(candidate_ids[:cutoff])) for cutoff in _CUTOFFS
                },
                answer_bearing_coverage_at={
                    cutoff: float(_answer_bearing(group.gold_answer, ranking.candidates, cutoff))
                    for cutoff in _CUTOFFS
                },
            )
        )
    denominator = float(len(rows))
    return DiscoveryArmEvaluation(
        schema_version="evaluation.discovery-arm.v1",
        arm_id=arm_id,
        question_count=len(rows),
        positive_assignment_count=sum(len(group.positive_chunk_ids) for group in ordered_groups),
        recall_at={
            cutoff: sum(row.recall_at[cutoff] for row in rows) / denominator for cutoff in _CUTOFFS
        },
        evidence_set_recall_at={
            cutoff: sum(row.evidence_set_recall_at[cutoff] for row in rows) / denominator
            for cutoff in _CUTOFFS
        },
        answer_bearing_coverage_at={
            cutoff: sum(row.answer_bearing_coverage_at[cutoff] for row in rows) / denominator
            for cutoff in _CUTOFFS
        },
        mrr_at_50=sum(
            0.0 if row.first_positive_rank is None else 1.0 / float(row.first_positive_rank)
            for row in rows
        )
        / denominator,
        rows=tuple(rows),
    )


def compare_discovery_arms(
    baseline: DiscoveryArmEvaluation,
    candidate: DiscoveryArmEvaluation,
) -> DiscoveryArmComparison:
    """Compare two paired arms without promoting a proxy-only trade-off."""

    if tuple(row.question_id for row in baseline.rows) != tuple(
        row.question_id for row in candidate.rows
    ):
        _fail("D066_COMPARISON_ID_MISMATCH", "paired arm question identities differ")
    baseline_hit = {row.question_id for row in baseline.rows if row.recall_at[50] == 1.0}
    candidate_hit = {row.question_id for row in candidate.rows if row.recall_at[50] == 1.0}
    novel = tuple(sorted(candidate_hit - baseline_hit, key=str.encode))
    lost = tuple(sorted(baseline_hit - candidate_hit, key=str.encode))
    deltas = {
        **{
            f"recall_at_{cutoff}": candidate.recall_at[cutoff] - baseline.recall_at[cutoff]
            for cutoff in _CUTOFFS
        },
        **{
            f"evidence_set_recall_at_{cutoff}": (
                candidate.evidence_set_recall_at[cutoff] - baseline.evidence_set_recall_at[cutoff]
            )
            for cutoff in _CUTOFFS
        },
        **{
            f"answer_bearing_coverage_at_{cutoff}": (
                candidate.answer_bearing_coverage_at[cutoff]
                - baseline.answer_bearing_coverage_at[cutoff]
            )
            for cutoff in _CUTOFFS
        },
        "mrr_at_50": candidate.mrr_at_50 - baseline.mrr_at_50,
    }
    primary = (
        deltas["recall_at_50"],
        deltas["evidence_set_recall_at_50"],
        deltas["answer_bearing_coverage_at_50"],
    )
    candidate_wins = (
        not lost
        and all(delta >= 0.0 for delta in primary)
        and any(delta > 0.0 for delta in primary)
    )
    return DiscoveryArmComparison(
        schema_version="evaluation.discovery-comparison.v1",
        baseline_arm_id=baseline.arm_id,
        candidate_arm_id=candidate.arm_id,
        metric_deltas=deltas,
        novel_positive_recovery_at_50=novel,
        lost_positive_recovery_at_50=lost,
        standing_winner=candidate.arm_id if candidate_wins else baseline.arm_id,
        decision_reason=(
            "candidate-improves-primary-without-top50-loss"
            if candidate_wins
            else "baseline-retained-no-unambiguous-primary-win"
        ),
    )


def serialize_discovery_evaluation(report: DiscoveryArmEvaluation) -> bytes:
    """Return canonical content bytes for deterministic replay/checksums."""

    return content_json_bytes(asdict(report))


def serialize_discovery_comparison(report: DiscoveryArmComparison) -> bytes:
    """Return canonical comparison bytes for deterministic replay/checksums."""

    return content_json_bytes(asdict(report))


def serialize_discovery_rankings(arm_id: str, rankings: Sequence[DiscoveryRanking]) -> bytes:
    """Serialize ordered identities only; train questions, answers, and chunk text stay out."""

    if not arm_id:
        _fail("D066_RANKING_INVALID", "ranking arm ID must be non-empty")
    ordered = tuple(sorted(rankings, key=lambda item: item.question_id.encode("utf-8")))
    question_ids = tuple(item.question_id for item in ordered)
    if any(not item for item in question_ids) or len(question_ids) != len(set(question_ids)):
        _fail("D066_RANKING_INVALID", "ranking question identities are invalid")
    rows: list[bytes] = []
    for ranking in ordered:
        candidate_ids = tuple(item.chunk_id for item in ranking.candidates)
        if any(not item for item in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            _fail("D066_RANKING_INVALID", "ranking candidate identities are invalid")
        rows.append(
            content_json_bytes(
                {
                    "schema_version": "retrieval.discovery-ranking.v1",
                    "arm_id": arm_id,
                    "question_id": ranking.question_id,
                    "candidate_chunk_ids": candidate_ids,
                }
            )
        )
    return b"".join(rows)


__all__ = [
    "DiscoveryArmComparison",
    "DiscoveryArmEvaluation",
    "DiscoveryCandidate",
    "DiscoveryGroup",
    "DiscoveryQuestionMetrics",
    "DiscoveryRanking",
    "DiscoveryTournamentError",
    "compare_discovery_arms",
    "evaluate_discovery_arm",
    "load_discovery_groups",
    "serialize_discovery_comparison",
    "serialize_discovery_evaluation",
    "serialize_discovery_rankings",
]
