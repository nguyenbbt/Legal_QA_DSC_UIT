"""Pre-index deterministic sampling for the private ``grounding.v1`` benchmark."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

from legal_rag.domain.artifacts import ImmutableArtifactError, write_immutable_bytes
from legal_rag.domain.checksums import canonical_json_bytes
from legal_rag.evaluation.split import normalize_split_text
from legal_rag.retrieval.exact import parse_legal_reference

GROUNDING_SAMPLE_VERSION = "grounding-sample.v1"
GROUNDING_SAMPLE_SEED = "dsc2026-grounding-sample-v1"
_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)

FillReason = Literal[
    "stratum_primary",
    "same_tercile_other_branch",
    "global_underfill",
    "not_selected",
]


class GroundingError(Exception):
    """Stable safe failure at a grounding sample boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class GroundingQuestion:
    question_id: str
    question: str


@dataclass(frozen=True, slots=True)
class GroundingSampleRow:
    question_id: str
    exact_reference_syntax: bool
    normalized_length: int
    tercile: int
    sample_digest: str
    selected: bool
    fill_reason: FillReason
    final_position: int | None


@dataclass(frozen=True, slots=True)
class GroundingSampleManifest:
    split_checksum: str
    rows: tuple[GroundingSampleRow, ...]
    selected_question_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "grounding.sample.manifest.v1",
            "sampling_version": GROUNDING_SAMPLE_VERSION,
            "seed": GROUNDING_SAMPLE_SEED,
            "split_checksum": self.split_checksum,
            "annotation_status": "unlabeled",
            "eligible_question_count": len(self.rows),
            "sample_size": len(self.selected_question_ids),
            "selected_question_ids": list(self.selected_question_ids),
            "rows": [
                {
                    "question_id": row.question_id,
                    "exact_reference_syntax": row.exact_reference_syntax,
                    "normalized_length": row.normalized_length,
                    "tercile": row.tercile,
                    "sample_digest": row.sample_digest,
                    "selected": row.selected,
                    "fill_reason": row.fill_reason,
                    "final_position": row.final_position,
                }
                for row in self.rows
            ],
        }

    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


@dataclass(frozen=True, slots=True)
class _Candidate:
    question_id: str
    exact_reference_syntax: bool
    normalized_length: int
    tercile: int
    digest: bytes


def _fail(code: str, message: str) -> NoReturn:
    raise GroundingError(code, message)


def _id_key(question_id: str) -> bytes:
    return question_id.encode("utf-8")


def _digest_key(candidate: _Candidate) -> tuple[bytes, bytes]:
    return candidate.digest, _id_key(candidate.question_id)


def has_exact_reference_syntax(question: str) -> bool:
    """Classify only the exact RET-004 parser shape, without corpus resolution."""

    return parse_legal_reference(question).reference is not None


def _build_candidates(questions: tuple[GroundingQuestion, ...]) -> tuple[_Candidate, ...]:
    ordered = tuple(sorted(questions, key=lambda item: _id_key(item.question_id)))
    if len(ordered) != len({item.question_id for item in ordered}):
        _fail("GROUNDING_QUESTION_ID_DUPLICATE", "development question IDs must be unique")
    if any(not item.question_id for item in ordered):
        _fail("GROUNDING_QUESTION_ID_EMPTY", "development question IDs must be non-empty")

    details = tuple(
        (
            item,
            has_exact_reference_syntax(item.question),
            len(normalize_split_text(item.question)),
        )
        for item in ordered
    )
    terciles: dict[str, int] = {}
    for branch in (False, True):
        branch_rows = sorted(
            (item for item in details if item[1] is branch),
            key=lambda item: (item[2], _id_key(item[0].question_id)),
        )
        count = len(branch_rows)
        for index, item in enumerate(branch_rows):
            terciles[item[0].question_id] = min(2, (3 * index) // count)

    return tuple(
        _Candidate(
            question_id=item.question_id,
            exact_reference_syntax=branch,
            normalized_length=length,
            tercile=terciles[item.question_id],
            digest=hashlib.sha256(f"{GROUNDING_SAMPLE_SEED}\n{item.question_id}".encode()).digest(),
        )
        for item, branch, length in details
    )


def build_grounding_sample(
    questions: tuple[GroundingQuestion, ...], *, split_checksum: str
) -> GroundingSampleManifest:
    """Select and fully describe the immutable 60-question pre-index sample."""

    if _CHECKSUM.fullmatch(split_checksum) is None:
        _fail("GROUNDING_SPLIT_CHECKSUM_INVALID", "grounding requires a typed split checksum")
    if len(questions) < 60:
        _fail("GROUNDING_SAMPLE_UNDERFILLED", "grounding requires 60 unique questions")

    candidates = _build_candidates(questions)
    strata = {
        (branch, tercile): sorted(
            (
                candidate
                for candidate in candidates
                if candidate.exact_reference_syntax is branch and candidate.tercile == tercile
            ),
            key=_digest_key,
        )
        for branch in (False, True)
        for tercile in range(3)
    }
    selected: dict[str, FillReason] = {}
    selected_by_stratum: dict[tuple[bool, int], list[_Candidate]] = {}
    for stratum, members in strata.items():
        primary = members[:10]
        selected_by_stratum[stratum] = list(primary)
        selected.update((candidate.question_id, "stratum_primary") for candidate in primary)

    for stratum in ((branch, tercile) for branch in (False, True) for tercile in range(3)):
        needed = 10 - len(selected_by_stratum[stratum])
        if needed == 0:
            continue
        branch, tercile = stratum
        same_tercile = [
            candidate
            for candidate in strata[(not branch, tercile)]
            if candidate.question_id not in selected
        ]
        borrowed = same_tercile[:needed]
        selected.update(
            (candidate.question_id, "same_tercile_other_branch") for candidate in borrowed
        )
        needed -= len(borrowed)
        if needed:
            available = sorted(
                (candidate for candidate in candidates if candidate.question_id not in selected),
                key=_digest_key,
            )
            selected.update(
                (candidate.question_id, "global_underfill") for candidate in available[:needed]
            )

    if len(selected) != 60:
        _fail("GROUNDING_SAMPLE_UNDERFILLED", "grounding sampling could not select 60 IDs")
    selected_candidates = sorted(
        (candidate for candidate in candidates if candidate.question_id in selected),
        key=_digest_key,
    )
    final_positions = {
        candidate.question_id: position for position, candidate in enumerate(selected_candidates)
    }
    rows = tuple(
        GroundingSampleRow(
            question_id=candidate.question_id,
            exact_reference_syntax=candidate.exact_reference_syntax,
            normalized_length=candidate.normalized_length,
            tercile=candidate.tercile,
            sample_digest=candidate.digest.hex(),
            selected=candidate.question_id in selected,
            fill_reason=selected.get(candidate.question_id, "not_selected"),
            final_position=final_positions.get(candidate.question_id),
        )
        for candidate in sorted(candidates, key=lambda item: _id_key(item.question_id))
    )
    return GroundingSampleManifest(
        split_checksum=split_checksum,
        rows=rows,
        selected_question_ids=tuple(candidate.question_id for candidate in selected_candidates),
    )


def write_grounding_sample(destination: Path, manifest: GroundingSampleManifest) -> str:
    """Create the immutable pre-index sample manifest or accept an identical retry."""

    try:
        return write_immutable_bytes(destination, manifest.json_bytes())
    except ImmutableArtifactError as error:
        code = (
            "GROUNDING_SAMPLE_IMMUTABLE"
            if error.code == "ARTIFACT_IMMUTABLE"
            else "GROUNDING_SAMPLE_WRITE_FAILED"
        )
        raise GroundingError(code, error.message) from error
