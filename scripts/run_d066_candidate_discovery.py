"""Run the local-only D-066 R-DISC-0 sparse candidate-discovery tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.bounded_parallel import ordered_bounded_map
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryCandidate,
    DiscoveryGroup,
    DiscoveryRanking,
    compare_discovery_arms,
    evaluate_discovery_arm,
    load_discovery_groups,
    serialize_discovery_comparison,
    serialize_discovery_evaluation,
    serialize_discovery_rankings,
)
from legal_rag.evaluation.real_retrieval import retrieve_question
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.disk_bm25 import DiskBm25Index, open_disk_bm25_index
from legal_rag.retrieval.exact import AliasIndex, load_frozen_alias_artifact
from legal_rag.retrieval.legal_sparse import LEGAL_SPARSE_VERSION, LegalSparseRetriever
from legal_rag.retrieval.lookup_cache import CachedExpandedSparseIndex
from legal_rag.retrieval.sparse_execution import (
    BoundedTopKDiskBm25Index,
    SparseExecutionContract,
    audit_sparse_execution_contract,
    open_worker_read_only_connection,
)
from legal_rag.retrieval.sparse_preflight import (
    SparsePreflightInput,
    evaluate_sparse_preflight,
    sample_timeout_seconds,
)
from legal_rag.training.rag_sft import load_gold_questions

_QUESTIONS = Path("artifacts/internal/train.questions.jsonl")
_SPLIT = Path("artifacts/splits/train-dev-test.v1.json")
_SUPERVISION = Path("artifacts/training/retrieval-supervision/v2/retrieval-supervision.v2.jsonl")
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_DATABASE = Path("artifacts/indices/bm25.v1.active.sqlite3")
_INDEX_MANIFEST = Path("artifacts/manifests/bm25.index.active.v1.json")
_ALIASES = Path("artifacts/governance/aliases.active.v1.jsonl")
_ALIAS_MANIFEST = Path("artifacts/manifests/aliases.active.v1.json")
_EXPECTED = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "supervision": "sha256:affd0969261243f0718e9faaed5d9cc0617138714cc190171cb9e7bf7253c1d6",
    "chunks": "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143",
    "index_database": "sha256:2b977231a0be77fa2409988ecb1f0955bd22d7175130b08affc49fd04771fdc1",
    "aliases": "sha256:1f213b99cd30fddb0954679245326d83628712fde4f5c59527f49749527118a4",
}
_POSITIVE_COUNT = 2_391
_TRAIN_COUNT = 5_582
_CANDIDATE_LIMIT = 50
_PREFLIGHT_QUESTION_COUNT = 100
_MAXIMUM_RUNTIME_SECONDS = 21_600


class _RetrievalIndex(Protocol):
    index_checksum: str

    def retrieve(self, query: str) -> SparseRetrievalResult: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]: ...


@dataclass(slots=True)
class _WorkerResources:
    index: _RetrievalIndex
    close: Callable[[], None]


class _FixedLimitBm25:
    """Expose the frozen BM25 index with a fixed D-066 top-50 discovery limit."""

    def __init__(self, index: CachedExpandedSparseIndex) -> None:
        self._index = index
        self.index_checksum = index.index_checksum

    def retrieve(self, query: str) -> SparseRetrievalResult:
        return self._index.retrieve(query, candidate_limit=_CANDIDATE_LIMIT)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return self._index.chunks_for_context(context_id)

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return self._index.chunks_for_coordinate(hierarchy_kind, hierarchy_ordinal)


def _rank(
    *,
    arm_id: str,
    groups: Sequence[DiscoveryGroup],
    questions: Sequence[QuestionRecord],
    index_factory: Callable[[], _WorkerResources],
    aliases: AliasIndex,
    max_workers: int,
) -> tuple[tuple[DiscoveryRanking, ...], float]:
    question_by_id = {item.question_id: item for item in questions}
    local = threading.local()
    created: list[_WorkerResources] = []
    created_lock = threading.Lock()

    def rank_one(group: DiscoveryGroup) -> DiscoveryRanking:
        resources = getattr(local, "resources", None)
        if resources is None:
            resources = index_factory()
            local.resources = resources
            with created_lock:
                created.append(resources)
        assert isinstance(resources, _WorkerResources)
        result = retrieve_question(
            question_by_id[group.question_id],
            index=resources.index,
            aliases=aliases,
            candidate_limit=_CANDIDATE_LIMIT,
        )
        return DiscoveryRanking(
            question_id=group.question_id,
            candidates=tuple(
                DiscoveryCandidate(candidate.chunk.chunk_id, candidate.chunk.display_text)
                for candidate in result.candidates
            ),
        )

    started = time.perf_counter()
    try:
        rankings = ordered_bounded_map(
            rank_one,
            groups,
            max_workers=max_workers,
            progress=lambda position, total: (
                print(f"{arm_id}: {position}/{total}", flush=True)
                if position % 100 == 0 or position == total
                else None
            ),
        )
    finally:
        for resources in created:
            resources.close()
    elapsed = time.perf_counter() - started
    return rankings, elapsed


def _check_inputs() -> dict[str, str]:
    paths = {
        "questions": _QUESTIONS,
        "split": _SPLIT,
        "supervision": _SUPERVISION,
        "chunks": _CHUNKS,
        "index_database": _DATABASE,
        "aliases": _ALIASES,
    }
    actual = {name: _streaming_checksum(path) for name, path in paths.items()}
    if actual != _EXPECTED:
        raise SystemExit("D-066 frozen input checksum drift")
    return actual


def _streaming_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record-preflight-timeout", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("reference", "d066-r1-bounded"),
        default="reference",
    )
    arguments = parser.parse_args()

    if arguments.record_preflight_timeout:
        timeout = sample_timeout_seconds(
            total_question_count=_POSITIVE_COUNT,
            sample_question_count=_PREFLIGHT_QUESTION_COUNT,
            maximum_runtime_seconds=_MAXIMUM_RUNTIME_SECONDS,
        )
        timeout_data = content_json_bytes(
            {
                "schema_version": "evaluation.d066-r-disc-0-preflight-timeout.v1",
                "arm_id": "R-DISC-0-BM25",
                "status": "BLOCKED",
                "blocker_codes": ["OPS002_PREFLIGHT_TIMEOUT"],
                "total_question_count": _POSITIVE_COUNT,
                "sample_question_count": _PREFLIGHT_QUESTION_COUNT,
                "sample_complete": False,
                "sample_timeout_seconds": timeout,
                "maximum_runtime_seconds": _MAXIMUM_RUNTIME_SECONDS,
                "worker_count": arguments.workers,
                "input_checksums": _EXPECTED,
                "ranking_artifact_written": False,
                "quality_metrics_computed": False,
                "legal_sparse_sample_started": False,
                "fit_performed": False,
                "gpu_used": False,
                "paid_service_used": False,
                "decision": "DO_NOT_OPEN_FULL_R_DISC_0",
            }
        )
        timeout_checksum = write_immutable_bytes(
            arguments.output_directory / "preflight" / "R-DISC-0.sparse-preflight-timeout.v1.json",
            timeout_data,
        )
        print(f"preflight_timeout={timeout_checksum}", flush=True)
        return 2

    input_checksums = _check_inputs()
    question_data = _QUESTIONS.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        _SPLIT.read_bytes(),
        expected_source_checksum=_EXPECTED["questions"],
        expected_question_ids=tuple(item.question_id for item in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    if len(train_ids) != _TRAIN_COUNT:
        raise SystemExit("D-066 train-fit count drift")
    groups = load_discovery_groups(
        supervision_data=_SUPERVISION.read_bytes(),
        question_source_data=question_data,
        train_question_ids=train_ids,
        expected_positive_count=_POSITIVE_COUNT,
        expected_supervision_checksum=_EXPECTED["supervision"],
    )

    index_manifest_data = _INDEX_MANIFEST.read_bytes()
    alias_manifest_data = _ALIAS_MANIFEST.read_bytes()
    with open_disk_bm25_index(
        database_path=_DATABASE,
        chunks_path=_CHUNKS,
        manifest_data=index_manifest_data,
    ) as disk_index:
        aliases = load_frozen_alias_artifact(
            _ALIASES.read_bytes(),
            manifest_data=alias_manifest_data,
            corpus_checksum=disk_index.manifest.corpus_checksum,
            artifact_path=_ALIASES.name,
        )
        baseline_index_checksum = disk_index.index_checksum
        baseline_contract = SparseExecutionContract.from_index(
            disk_index, candidate_limit=_CANDIDATE_LIMIT
        )
        legal_contract = SparseExecutionContract.from_index(disk_index, candidate_limit=100)
        legal_index_checksum = LegalSparseRetriever(
            CachedExpandedSparseIndex(disk_index), discovery_limit=100
        ).index_checksum

        def clone_disk_index() -> DiskBm25Index:
            connection = open_worker_read_only_connection(_DATABASE)
            return DiskBm25Index(
                connection=connection,
                chunks_path=_CHUNKS,
                manifest=disk_index.manifest,
                manifest_data=index_manifest_data,
            )

        def baseline_factory() -> _WorkerResources:
            clone = clone_disk_index()
            audit_sparse_execution_contract(
                clone, baseline_contract, candidate_limit=_CANDIDATE_LIMIT
            )
            source = (
                BoundedTopKDiskBm25Index(clone)
                if arguments.execution_mode == "d066-r1-bounded"
                else clone
            )
            index = _FixedLimitBm25(CachedExpandedSparseIndex(source))
            return _WorkerResources(index=index, close=clone.close)

        def legal_factory() -> _WorkerResources:
            clone = clone_disk_index()
            audit_sparse_execution_contract(clone, legal_contract, candidate_limit=100)
            source = (
                BoundedTopKDiskBm25Index(clone)
                if arguments.execution_mode == "d066-r1-bounded"
                else clone
            )
            index = LegalSparseRetriever(CachedExpandedSparseIndex(source), discovery_limit=100)
            return _WorkerResources(index=index, close=clone.close)

        ranked_groups = groups[:_PREFLIGHT_QUESTION_COUNT] if arguments.preflight_only else groups
        baseline_rankings, baseline_seconds = _rank(
            arm_id="R-DISC-0-BM25",
            groups=ranked_groups,
            questions=questions,
            index_factory=baseline_factory,
            aliases=aliases,
            max_workers=arguments.workers,
        )
        if arguments.preflight_only:
            baseline_preflight = evaluate_sparse_preflight(
                SparsePreflightInput(
                    arm_id="R-DISC-0-BM25",
                    total_question_count=len(groups),
                    observed_question_count=len(ranked_groups),
                    observed_seconds_millis=max(1, round(baseline_seconds * 1000.0)),
                    maximum_runtime_seconds=_MAXIMUM_RUNTIME_SECONDS,
                    worker_count=arguments.workers,
                )
            )
            if baseline_preflight.status == "BLOCKED":
                preflight_data = content_json_bytes(
                    {
                        "schema_version": "evaluation.d066-r-disc-0-preflight.v2",
                        "execution_mode": arguments.execution_mode,
                        "baseline": asdict(baseline_preflight),
                        "legal_sparse": None,
                        "decision": "BLOCKED_BASELINE_OPS002",
                        "fit_performed": False,
                        "gpu_used": False,
                        "paid_service_used": False,
                    }
                )
                checksum = write_immutable_bytes(
                    arguments.output_directory
                    / "preflight"
                    / "R-DISC-0.sparse-preflight.d066-r1.v1.json",
                    preflight_data,
                )
                print(f"preflight={checksum}", flush=True)
                return 2
        legal_rankings, legal_seconds = _rank(
            arm_id="R-DISC-0-LEGAL-SPARSE",
            groups=ranked_groups,
            questions=questions,
            index_factory=legal_factory,
            aliases=aliases,
            max_workers=arguments.workers,
        )
        if arguments.preflight_only:
            legal_preflight = evaluate_sparse_preflight(
                SparsePreflightInput(
                    arm_id="R-DISC-0-LEGAL-SPARSE",
                    total_question_count=len(groups),
                    observed_question_count=len(ranked_groups),
                    observed_seconds_millis=max(1, round(legal_seconds * 1000.0)),
                    maximum_runtime_seconds=_MAXIMUM_RUNTIME_SECONDS,
                    worker_count=arguments.workers,
                )
            )
            preflight_data = content_json_bytes(
                {
                    "schema_version": "evaluation.d066-r-disc-0-preflight.v2",
                    "execution_mode": arguments.execution_mode,
                    "baseline": asdict(baseline_preflight),
                    "legal_sparse": asdict(legal_preflight),
                    "decision": (
                        "PASS_OPEN_FULL_R_DISC_0"
                        if legal_preflight.status == "PASS"
                        else "BLOCKED_LEGAL_SPARSE_OPS002"
                    ),
                    "fit_performed": False,
                    "gpu_used": False,
                    "paid_service_used": False,
                }
            )
            checksum = write_immutable_bytes(
                arguments.output_directory
                / "preflight"
                / "R-DISC-0.sparse-preflight.d066-r1.v1.json",
                preflight_data,
            )
            print(f"preflight={checksum}", flush=True)
            return 0 if legal_preflight.status == "PASS" else 2

    baseline_evaluation = evaluate_discovery_arm("R-DISC-0-BM25", groups, baseline_rankings)
    legal_evaluation = evaluate_discovery_arm("R-DISC-0-LEGAL-SPARSE", groups, legal_rankings)
    comparison = compare_discovery_arms(baseline_evaluation, legal_evaluation)
    baseline_ranking_data = serialize_discovery_rankings(
        baseline_evaluation.arm_id, baseline_rankings
    )
    legal_ranking_data = serialize_discovery_rankings(legal_evaluation.arm_id, legal_rankings)
    baseline_evaluation_data = serialize_discovery_evaluation(baseline_evaluation)
    legal_evaluation_data = serialize_discovery_evaluation(legal_evaluation)
    comparison_data = serialize_discovery_comparison(comparison)
    replay_passed = all(
        (
            baseline_ranking_data
            == serialize_discovery_rankings(baseline_evaluation.arm_id, baseline_rankings),
            legal_ranking_data
            == serialize_discovery_rankings(legal_evaluation.arm_id, legal_rankings),
            baseline_evaluation_data == serialize_discovery_evaluation(baseline_evaluation),
            legal_evaluation_data == serialize_discovery_evaluation(legal_evaluation),
            comparison_data == serialize_discovery_comparison(comparison),
        )
    )
    if not replay_passed:
        raise SystemExit("D-066 deterministic serialization replay failed")

    output = arguments.output_directory
    artifact_data = {
        "R-DISC-0-BM25.rankings.v1.jsonl": baseline_ranking_data,
        "R-DISC-0-BM25.evaluation.v1.json": baseline_evaluation_data,
        "R-DISC-0-LEGAL-SPARSE.rankings.v1.jsonl": legal_ranking_data,
        "R-DISC-0-LEGAL-SPARSE.evaluation.v1.json": legal_evaluation_data,
        "R-DISC-0.comparison.v1.json": comparison_data,
    }
    artifact_checksums = {
        name: write_immutable_bytes(output / name, data) for name, data in artifact_data.items()
    }
    deterministic_manifest = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r-disc-0.manifest.v1",
            "execution_mode": "local-offline",
            "train_fit_count": _TRAIN_COUNT,
            "positive_group_count": _POSITIVE_COUNT,
            "candidate_limit": _CANDIDATE_LIMIT,
            "sparse_execution_mode": arguments.execution_mode,
            "input_checksums": input_checksums,
            "index_manifest_checksum": checksum_bytes(index_manifest_data),
            "alias_manifest_checksum": checksum_bytes(alias_manifest_data),
            "baseline_index_checksum": baseline_index_checksum,
            "legal_sparse_version": LEGAL_SPARSE_VERSION,
            "legal_sparse_index_checksum": legal_index_checksum,
            "artifact_checksums": artifact_checksums,
            "standing_winner": comparison.standing_winner,
            "decision_reason": comparison.decision_reason,
            "replay_byte_identical": replay_passed,
            "fit_performed": False,
            "modal_used": False,
            "public_data_used": False,
            "development_data_used": False,
        }
    )
    manifest_checksum = write_immutable_bytes(
        output / "R-DISC-0.manifest.v1.json", deterministic_manifest
    )
    telemetry_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r-disc-0.telemetry.v1",
            "question_count_per_arm": _POSITIVE_COUNT,
            "candidate_limit": _CANDIDATE_LIMIT,
            "local_worker_count": arguments.workers,
            "bm25_wall_seconds": baseline_seconds,
            "legal_sparse_wall_seconds": legal_seconds,
            "total_wall_seconds": baseline_seconds + legal_seconds,
            "database_bytes": _DATABASE.stat().st_size,
            "rankings_bytes": len(baseline_ranking_data) + len(legal_ranking_data),
            "gpu_used": False,
            "paid_service_used": False,
            "cost_usd": 0.0,
        }
    )
    telemetry_checksum = write_immutable_bytes(
        output / "R-DISC-0.telemetry.v1.json", telemetry_data
    )
    print(
        json.dumps(
            {
                "baseline": asdict(baseline_evaluation),
                "legal_sparse": asdict(legal_evaluation),
                "comparison": asdict(comparison),
                "manifest_checksum": manifest_checksum,
                "telemetry_checksum": telemetry_checksum,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
