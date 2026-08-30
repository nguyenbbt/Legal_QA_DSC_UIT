"""Deterministic train-only feature and ranking contracts for D-067."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

from legal_rag.domain.checksums import content_json_bytes
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryArmEvaluation,
    DiscoveryCandidate,
    DiscoveryRanking,
)
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.exact import document_number_key
from legal_rag.retrieval.legal_citations import ParsedLegalCitation, parse_legal_citations
from legal_rag.retrieval.tokenizer import retrieval_token_values

FusionPartition = Literal["fit", "validation"]
FEATURE_NAMES = (
    "bm25_score",
    "bm25_rank",
    "dense_score",
    "dense_rank",
    "exact_reference_flag",
    "document_id_match",
    "law_title_match",
    "article_number_match",
    "clause_match",
    "point_match",
    "query_length",
    "candidate_length",
    "lexical_overlap",
    "legal_term_overlap",
    "hierarchy_distance",
    "source_retriever_flags",
)
_SPLIT_VERSION = "d067-group-split.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_LEGAL_TERMS = frozenset(
    {
        "bộ",
        "chương",
        "điều",
        "điểm",
        "khoản",
        "luật",
        "mục",
        "nghị",
        "phần",
        "pháp",
        "quyết",
        "thông",
        "tư",
        "định",
    }
)
_PATH_LABELS = (("điều", "article"), ("khoản", "clause"), ("điểm", "point"))


class LearnedFusionError(Exception):
    """Stable fail-closed D-067 feature/ranking error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class FusionGroupSplit:
    split_version: str
    fit_question_ids: tuple[str, ...]
    validation_question_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryLegalSignals:
    question: str
    query_tokens: tuple[str, ...]
    legal_tokens: frozenset[str]
    document_context_ids: frozenset[str]
    law_identity_keys: frozenset[str]
    citations: tuple[ParsedLegalCitation, ...]


@dataclass(frozen=True, slots=True)
class FusionCandidateSignals:
    chunk: ChunkRecord
    sparse_score: float | None
    sparse_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    exact_reference_flag: bool
    candidate_law_key: str | None

    def __post_init__(self) -> None:
        for score in (self.sparse_score, self.dense_score):
            if score is not None and not math.isfinite(score):
                raise LearnedFusionError("D067_FEATURE_INVALID", "component score is non-finite")
        for rank in (self.sparse_rank, self.dense_rank):
            if rank is not None and not 1 <= rank <= 50:
                raise LearnedFusionError("D067_FEATURE_INVALID", "component rank is outside 1..50")


@dataclass(frozen=True, slots=True)
class FusionFeatureRow:
    question_id: str
    question_checksum: str
    chunk_id: str
    partition: FusionPartition
    label: int
    feature_values: tuple[float, ...]
    chunk_checksum: str

    def __post_init__(self) -> None:
        if (
            not self.question_id
            or _SHA256.fullmatch(self.question_checksum) is None
            or not self.chunk_id
            or self.partition not in ("fit", "validation")
            or self.label not in (0, 1)
            or len(self.feature_values) != len(FEATURE_NAMES)
            or not all(math.isfinite(value) for value in self.feature_values)
            or _SHA256.fullmatch(self.chunk_checksum) is None
        ):
            raise LearnedFusionError("D067_FEATURE_ROW_INVALID", "fusion feature row is invalid")


@dataclass(frozen=True, slots=True)
class FusionValidationComparison:
    schema_version: str
    baseline_arm_id: str
    candidate_arm_id: str
    metric_deltas: dict[str, float]
    novel_positive_recovery_at_50: tuple[str, ...]
    lost_positive_recovery_at_50: tuple[str, ...]
    passes_retrieval_gate: bool
    standing_winner: str
    decision_reason: str


def build_group_split(question_ids: Sequence[str]) -> FusionGroupSplit:
    """Apply the frozen SHA-256 modulo-5 group split independent of input order."""

    ordered = tuple(sorted(question_ids, key=str.encode))
    if not ordered or len(ordered) != len(set(ordered)) or any(not value for value in ordered):
        raise LearnedFusionError("D067_SPLIT_INVALID", "question IDs must be unique and non-empty")
    fit: list[str] = []
    validation: list[str] = []
    prefix = (_SPLIT_VERSION + "\0").encode()
    for question_id in ordered:
        digest = hashlib.sha256(prefix + question_id.encode()).digest()
        target = validation if int.from_bytes(digest[:8], "big") % 5 == 0 else fit
        target.append(question_id)
    if not fit or not validation:
        raise LearnedFusionError("D067_SPLIT_INVALID", "both split partitions must be non-empty")
    return FusionGroupSplit(_SPLIT_VERSION, tuple(fit), tuple(validation))


def _fold_legal_identity(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", value).casefold())
    unmarked = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", unmarked.replace("đ", "d")).split())


def law_identity_key_from_context_name(name: str | None) -> str | None:
    """Return the canonical law/code title key used by question citation parsing."""

    if not name:
        return None
    folded = _fold_legal_identity(name)
    match = re.match(
        r"(?P<kind>bo luat|luat) (?P<title>.+?) (?P<year>(?:19|20)[0-9]{2})(?: |$)",
        folded,
    )
    if match is None:
        return None
    return f"{match.group('kind')} {match.group('title')} {match.group('year')}"


def build_query_legal_signals(
    question: str,
    *,
    document_aliases: Mapping[str, Sequence[str]],
) -> QueryLegalSignals:
    """Derive query-only legal identity and coordinate signals without answer data."""

    canonical = unicodedata.normalize("NFC", question)
    tokens = retrieval_token_values(canonical.casefold())
    citations = parse_legal_citations(canonical)
    context_ids: set[str] = set()
    law_keys: set[str] = set()
    for citation in citations:
        if citation.document_number is not None:
            context_ids.update(
                document_aliases.get(document_number_key(citation.document_number), ())
            )
        if citation.law_identity is not None:
            law_keys.add(citation.law_identity)
    return QueryLegalSignals(
        question=canonical,
        query_tokens=tokens,
        legal_tokens=frozenset(tokens) & _LEGAL_TERMS,
        document_context_ids=frozenset(context_ids),
        law_identity_keys=frozenset(law_keys),
        citations=citations,
    )


def _path_coordinates(path: tuple[str, ...]) -> dict[str, str]:
    coordinates: dict[str, str] = {}
    for member in path:
        view = " ".join(unicodedata.normalize("NFC", member).casefold().split())
        for label, kind in _PATH_LABELS:
            match = re.match(rf"^{label}\s+(.+?)\s*$", view)
            if match is not None:
                coordinates[kind] = match.group(1)
                break
    return coordinates


def _coordinate_match(
    citations: tuple[ParsedLegalCitation, ...], coordinates: Mapping[str, str], kind: str
) -> float:
    return float(
        any(
            value is not None and coordinates.get(kind) == value
            for citation in citations
            if (value := getattr(citation, kind if kind != "article" else "article")) is not None
        )
    )


def _hierarchy_distance(
    citations: tuple[ParsedLegalCitation, ...], coordinates: Mapping[str, str]
) -> float:
    distances: list[int] = []
    for citation in citations:
        requested = {
            "article": citation.article,
            "clause": citation.clause,
            "point": citation.point,
        }
        present = {kind: value for kind, value in requested.items() if value is not None}
        if present:
            distances.append(sum(coordinates.get(kind) != value for kind, value in present.items()))
    return float(min(distances, default=3))


def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def build_fusion_feature_values(
    query: QueryLegalSignals, candidate: FusionCandidateSignals
) -> tuple[float, ...]:
    """Return the exact ordered 16-feature vector from answer-independent inputs."""

    candidate_tokens = retrieval_token_values(candidate.chunk.retrieval_text.casefold())
    query_set = frozenset(query.query_tokens)
    candidate_set = frozenset(candidate_tokens)
    coordinates = _path_coordinates(candidate.chunk.hierarchy_path)
    legal_denominator = len(query.legal_tokens)
    legal_overlap = (
        0.0
        if not legal_denominator
        else len(query.legal_tokens & candidate_set) / legal_denominator
    )
    source_flags = (
        int(candidate.sparse_rank is not None)
        | (int(candidate.dense_rank is not None) << 1)
        | (int(candidate.exact_reference_flag) << 2)
    )
    values = (
        0.0 if candidate.sparse_score is None else candidate.sparse_score,
        51.0 if candidate.sparse_rank is None else float(candidate.sparse_rank),
        0.0 if candidate.dense_score is None else candidate.dense_score,
        51.0 if candidate.dense_rank is None else float(candidate.dense_rank),
        float(candidate.exact_reference_flag),
        float(candidate.chunk.context_id in query.document_context_ids),
        float(
            candidate.candidate_law_key is not None
            and candidate.candidate_law_key in query.law_identity_keys
        ),
        _coordinate_match(query.citations, coordinates, "article"),
        _coordinate_match(query.citations, coordinates, "clause"),
        _coordinate_match(query.citations, coordinates, "point"),
        float(len(query.query_tokens)),
        float(len(candidate_tokens)),
        _overlap(query_set, candidate_set),
        legal_overlap,
        _hierarchy_distance(query.citations, coordinates),
        float(source_flags),
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise LearnedFusionError("D067_FEATURE_INVALID", "fusion feature vector is invalid")
    return values


def serialize_feature_rows(rows: Sequence[FusionFeatureRow]) -> bytes:
    """Serialize feature rows in stable question/chunk order with explicit labels."""

    ordered = tuple(sorted(rows, key=lambda row: (row.question_id.encode(), row.chunk_id.encode())))
    identities = tuple((row.question_id, row.chunk_id) for row in ordered)
    if not ordered or len(identities) != len(set(identities)):
        raise LearnedFusionError("D067_FEATURE_ROW_INVALID", "feature identities are invalid")
    return b"".join(
        content_json_bytes(
            {
                "schema_version": "evaluation.d067-feature-row.v1",
                **asdict(row),
            }
        )
        for row in ordered
    )


def deserialize_feature_rows(data: bytes) -> tuple[FusionFeatureRow, ...]:
    """Load canonical D-067 feature JSONL and reject identity/order drift."""

    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise LearnedFusionError("D067_FEATURE_ARTIFACT_INVALID", "feature framing is invalid")
    rows: list[FusionFeatureRow] = []
    try:
        for line in data.splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema_version") != (
                "evaluation.d067-feature-row.v1"
            ):
                raise LearnedFusionError(
                    "D067_FEATURE_ARTIFACT_INVALID", "feature schema is invalid"
                )
            raw_features = value.get("feature_values")
            question_id = value.get("question_id")
            question_checksum = value.get("question_checksum")
            chunk_id = value.get("chunk_id")
            partition = value.get("partition")
            label = value.get("label")
            chunk_checksum = value.get("chunk_checksum")
            if (
                not isinstance(question_id, str)
                or not isinstance(question_checksum, str)
                or not isinstance(chunk_id, str)
                or partition not in ("fit", "validation")
                or type(label) is not int
                or not isinstance(chunk_checksum, str)
                or not isinstance(raw_features, list)
                or len(raw_features) != len(FEATURE_NAMES)
                or any(type(item) not in (int, float) for item in raw_features)
            ):
                raise LearnedFusionError(
                    "D067_FEATURE_ARTIFACT_INVALID", "feature field types are invalid"
                )
            rows.append(
                FusionFeatureRow(
                    question_id=question_id,
                    question_checksum=question_checksum,
                    chunk_id=chunk_id,
                    partition=cast(FusionPartition, partition),
                    label=label,
                    feature_values=tuple(float(item) for item in raw_features),
                    chunk_checksum=chunk_checksum,
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LearnedFusionError(
            "D067_FEATURE_ARTIFACT_INVALID", "feature JSONL is invalid"
        ) from error
    ordered = tuple(sorted(rows, key=lambda row: (row.question_id.encode(), row.chunk_id.encode())))
    identities = tuple((row.question_id, row.chunk_id) for row in ordered)
    if tuple(rows) != ordered or len(identities) != len(set(identities)):
        raise LearnedFusionError(
            "D067_FEATURE_ARTIFACT_INVALID", "feature ordering or identity is invalid"
        )
    return ordered


def rank_learned_fusion(
    rows: Sequence[FusionFeatureRow], scores: Sequence[float], *, limit: int
) -> tuple[DiscoveryRanking, ...]:
    """Rank predicted candidate rows with a stable UTF-8 identity tie-break."""

    if len(rows) != len(scores) or limit < 1 or any(not math.isfinite(score) for score in scores):
        raise LearnedFusionError("D067_PREDICTION_INVALID", "fusion predictions are invalid")
    grouped: dict[str, list[tuple[float, FusionFeatureRow]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[row.question_id].append((float(score), row))
    rankings: list[DiscoveryRanking] = []
    for question_id in sorted(grouped, key=str.encode):
        candidates = sorted(
            grouped[question_id], key=lambda item: (-item[0], item[1].chunk_id.encode())
        )[:limit]
        rankings.append(
            DiscoveryRanking(
                question_id,
                tuple(DiscoveryCandidate(item.chunk_id, "") for _score, item in candidates),
            )
        )
    return tuple(rankings)


def compare_fusion_validation(
    baseline: DiscoveryArmEvaluation, candidate: DiscoveryArmEvaluation
) -> FusionValidationComparison:
    """Apply the frozen D-067 held-out retrieval gate without downstream promotion."""

    baseline_ids = tuple(row.question_id for row in baseline.rows)
    candidate_ids = tuple(row.question_id for row in candidate.rows)
    if baseline_ids != candidate_ids or not baseline_ids:
        raise LearnedFusionError("D067_COMPARISON_ID_MISMATCH", "validation arm identities differ")
    baseline_hit = {row.question_id for row in baseline.rows if row.recall_at[50] == 1.0}
    candidate_hit = {row.question_id for row in candidate.rows if row.recall_at[50] == 1.0}
    deltas = {
        **{
            f"recall_at_{cutoff}": candidate.recall_at[cutoff] - baseline.recall_at[cutoff]
            for cutoff in (5, 10, 20, 50)
        },
        **{
            f"evidence_set_recall_at_{cutoff}": (
                candidate.evidence_set_recall_at[cutoff] - baseline.evidence_set_recall_at[cutoff]
            )
            for cutoff in (5, 10, 20, 50)
        },
        "mrr_at_50": candidate.mrr_at_50 - baseline.mrr_at_50,
    }
    passes = (
        deltas["recall_at_50"] >= 0.0
        and deltas["evidence_set_recall_at_50"] >= 0.0
        and (deltas["recall_at_10"] > 0.0 or deltas["mrr_at_50"] > 0.0)
    )
    return FusionValidationComparison(
        schema_version="evaluation.d067-fusion-comparison.v1",
        baseline_arm_id=baseline.arm_id,
        candidate_arm_id=candidate.arm_id,
        metric_deltas=deltas,
        novel_positive_recovery_at_50=tuple(sorted(candidate_hit - baseline_hit, key=str.encode)),
        lost_positive_recovery_at_50=tuple(sorted(baseline_hit - candidate_hit, key=str.encode)),
        passes_retrieval_gate=passes,
        standing_winner=candidate.arm_id if passes else baseline.arm_id,
        decision_reason=(
            "learned-preserves-top50-and-improves-ranking"
            if passes
            else "fixed-rrf-retained-held-out-gate-failed"
        ),
    )


__all__ = [
    "FEATURE_NAMES",
    "FusionCandidateSignals",
    "FusionFeatureRow",
    "FusionGroupSplit",
    "FusionPartition",
    "FusionValidationComparison",
    "LearnedFusionError",
    "QueryLegalSignals",
    "build_fusion_feature_values",
    "build_group_split",
    "build_query_legal_signals",
    "compare_fusion_validation",
    "deserialize_feature_rows",
    "law_identity_key_from_context_name",
    "rank_learned_fusion",
    "serialize_feature_rows",
]
