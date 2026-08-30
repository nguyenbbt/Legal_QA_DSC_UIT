"""Build the immutable answer-independent D-067 learned-fusion feature table."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.bounded_parallel import ordered_bounded_map
from legal_rag.evaluation.discovery_tournament import (
    DiscoveryGroup,
    load_discovery_groups,
)
from legal_rag.evaluation.learned_fusion import (
    FEATURE_NAMES,
    FusionCandidateSignals,
    FusionFeatureRow,
    FusionPartition,
    build_fusion_feature_values,
    build_group_split,
    build_query_legal_signals,
    law_identity_key_from_context_name,
    serialize_feature_rows,
)
from legal_rag.evaluation.real_retrieval import retrieve_question
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.dense_store import MemmapDenseIndex
from legal_rag.retrieval.disk_bm25 import (
    DiskBm25Index,
    DiskBm25Manifest,
    open_disk_bm25_index,
)
from legal_rag.retrieval.exact import AliasIndex, load_frozen_alias_artifact
from legal_rag.retrieval.lookup_cache import CachedExpandedSparseIndex
from legal_rag.retrieval.sparse_execution import (
    BoundedTopKDiskBm25Index,
    SparseExecutionContract,
    audit_sparse_execution_contract,
    open_worker_read_only_connection,
)
from legal_rag.training.rag_sft import load_gold_questions

_D066 = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1")
_OUTPUT = Path("artifacts/training/learned-fusion/d067")
_QUESTIONS = Path("artifacts/internal/train.questions.jsonl")
_SPLIT = Path("artifacts/splits/train-dev-test.v1.json")
_SUPERVISION = Path("artifacts/training/retrieval-supervision/v2/retrieval-supervision.v2.jsonl")
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_CONTEXTS = Path("artifacts/internal/contexts.jsonl")
_ALIASES = Path("artifacts/governance/aliases.active.v1.jsonl")
_ALIAS_MANIFEST = Path("artifacts/manifests/aliases.active.v1.json")
_DATABASE = Path("artifacts/indices/bm25.v1.active.sqlite3")
_INDEX_MANIFEST = Path("artifacts/manifests/bm25.index.active.v1.json")
_SPARSE_RANKINGS = _D066 / "R-DISC-0-BM25.rankings.v1.jsonl"
_DENSE_RANKINGS = _D066 / "R-DISC-1.rankings.v1.jsonl"
_QUERY_IDS = _D066 / "R-DISC-1.query-vector-ids.v1.jsonl"
_QUERY_VECTORS = _D066 / "R-DISC-1.query-vectors.f16.npy"
_QUERY_MANIFEST = _D066 / "R-DISC-1.query-vectors.manifest.v1.json"
_DENSE_INDEX = Path("artifacts/indices/dense/d066-r-disc-1-116f6cf195224b12")

_EXPECTED = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "supervision": "sha256:affd0969261243f0718e9faaed5d9cc0617138714cc190171cb9e7bf7253c1d6",
    "chunks": "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143",
    "contexts": "sha256:24650437b0c7ee65fecf8cb5a70028e1c5785bc5ea69cce6534227df652daaf0",
    "aliases": "sha256:1f213b99cd30fddb0954679245326d83628712fde4f5c59527f49749527118a4",
    "database": "sha256:2b977231a0be77fa2409988ecb1f0955bd22d7175130b08affc49fd04771fdc1",
    "sparse_rankings": "sha256:a3c10cd01274efc1cf93b81efd23a9c8756b67acef4119dd165966aa7a866463",
    "dense_rankings": "sha256:001ce3f281bd1db774ff6ab3db551fa3d4a058f2c5c18a5f3b1b8da9acc69d20",
    "query_ids": "sha256:1ad74f81fdc5f6f3cd504e42c50c8c8fe82c11288ebd8b80d0b66549cab5fb7a",
    "query_vectors": "sha256:98a9a13f80ab80ea4c84e7ad83b03c0a83ea2397073479459b855a846f3f0c46",
    "query_manifest": "sha256:2ec06c138a53f7b03ec800f17c706bfccec5a11d1730dd12cd6e07cb283ea870",
    "dense_manifest": "sha256:84cbd768eabb021a2eb472dcede730e25c045af37e145ea8a75b5149410b05e0",
    "dense_vectors": "sha256:f15b803dbae9dc0e2aa9cdf157861017e714c31a6dea65762df1393728dada04",
    "dense_ids": "sha256:218973fe0b4c0589eddfe9e3a9bf2d379a2843e4d327d158ed58e03dc31a8c39",
}
_TRAIN_COUNT = 5_582
_GROUP_COUNT = 2_391
_CANDIDATE_LIMIT = 50


class _RetrievalIndex(Protocol):
    index_checksum: str

    def retrieve(self, query: str) -> SparseRetrievalResult: ...

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]: ...

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]: ...


class _FixedLimitBm25:
    def __init__(self, source: CachedExpandedSparseIndex) -> None:
        self._source = source
        self.index_checksum = source.index_checksum

    def retrieve(self, query: str) -> SparseRetrievalResult:
        return self._source.retrieve(query, candidate_limit=_CANDIDATE_LIMIT)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return self._source.chunks_for_context(context_id)

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return self._source.chunks_for_coordinate(hierarchy_kind, hierarchy_ordinal)


@dataclass(slots=True)
class _WorkerResources:
    raw_index: DiskBm25Index
    retrieval_index: _RetrievalIndex

    def close(self) -> None:
        self.raw_index.close()


@dataclass(frozen=True, slots=True)
class _FrozenRanking:
    question_id: str
    chunk_ids: tuple[str, ...]


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _check_inputs() -> dict[str, str]:
    paths = {
        "questions": _QUESTIONS,
        "split": _SPLIT,
        "supervision": _SUPERVISION,
        "chunks": _CHUNKS,
        "contexts": _CONTEXTS,
        "aliases": _ALIASES,
        "database": _DATABASE,
        "sparse_rankings": _SPARSE_RANKINGS,
        "dense_rankings": _DENSE_RANKINGS,
        "query_ids": _QUERY_IDS,
        "query_vectors": _QUERY_VECTORS,
        "query_manifest": _QUERY_MANIFEST,
        "dense_manifest": _DENSE_INDEX / "manifest.json",
        "dense_vectors": _DENSE_INDEX / "vectors.f16.npy",
        "dense_ids": _DENSE_INDEX / "chunk-ids.jsonl",
    }
    actual = {name: _checksum(path) for name, path in paths.items()}
    if actual != _EXPECTED:
        raise SystemExit("D-067 frozen input checksum drift")
    return actual


def _load_rankings(path: Path, arm_id: str) -> tuple[_FrozenRanking, ...]:
    rankings: list[_FrozenRanking] = []
    for line in path.read_bytes().splitlines():
        row = json.loads(line)
        raw_ids = row.get("candidate_chunk_ids")
        if (
            row.get("schema_version") != "retrieval.discovery-ranking.v1"
            or row.get("arm_id") != arm_id
            or not isinstance(raw_ids, list)
            or len(raw_ids) != _CANDIDATE_LIMIT
            or not all(isinstance(item, str) and item for item in raw_ids)
            or len(raw_ids) != len(set(raw_ids))
        ):
            raise SystemExit("D-067 frozen ranking artifact is invalid")
        rankings.append(_FrozenRanking(str(row["question_id"]), tuple(cast(list[str], raw_ids))))
    rankings.sort(key=lambda item: item.question_id.encode())
    if len(rankings) != _GROUP_COUNT or len({item.question_id for item in rankings}) != len(
        rankings
    ):
        raise SystemExit("D-067 frozen ranking coverage is invalid")
    return tuple(rankings)


def _load_document_aliases() -> dict[str, tuple[str, ...]]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for line in _ALIASES.read_bytes().splitlines():
        row = json.loads(line)
        by_key[str(row["document_number_key"])].append(str(row["context_id"]))
    return {key: tuple(dict.fromkeys(values)) for key, values in by_key.items()}


def _load_context_law_keys() -> dict[str, str]:
    result: dict[str, str] = {}
    with _CONTEXTS.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            context_id = str(row["context_id"])
            name = row.get("name")
            key = law_identity_key_from_context_name(name if isinstance(name, str) else None)
            if key is not None:
                result[context_id] = key
    return result


def _load_query_vectors(
    groups: Sequence[DiscoveryGroup],
) -> tuple[dict[str, int], np.ndarray]:
    rows = tuple(json.loads(line) for line in _QUERY_IDS.read_bytes().splitlines())
    row_by_id: dict[str, int] = {}
    checksum_by_id = {group.question_id: group.question_checksum for group in groups}
    for position, row in enumerate(rows):
        question_id = str(row.get("question_id"))
        if (
            row.get("schema_version") != "retrieval.dense-query-vector-row.v1"
            or row.get("row") != position
            or row.get("question_checksum") != checksum_by_id.get(question_id)
            or question_id in row_by_id
        ):
            raise SystemExit("D-067 dense-query mapping is invalid")
        row_by_id[question_id] = position
    vectors = np.load(_QUERY_VECTORS, mmap_mode="r")
    if set(row_by_id) != set(checksum_by_id) or vectors.shape != (_GROUP_COUNT, 1024):
        raise SystemExit("D-067 dense-query vector shape is invalid")
    return row_by_id, vectors


def _load_groups_and_questions() -> tuple[tuple[DiscoveryGroup, ...], tuple[QuestionRecord, ...]]:
    question_data = _QUESTIONS.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        _SPLIT.read_bytes(),
        expected_source_checksum=_EXPECTED["questions"],
        expected_question_ids=tuple(item.question_id for item in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    if len(train_ids) != _TRAIN_COUNT:
        raise SystemExit("D-067 train-fit count drift")
    groups = load_discovery_groups(
        supervision_data=_SUPERVISION.read_bytes(),
        question_source_data=question_data,
        train_question_ids=train_ids,
        expected_positive_count=_GROUP_COUNT,
        expected_supervision_checksum=_EXPECTED["supervision"],
    )
    return groups, questions


def _build_rows(
    *,
    groups: tuple[DiscoveryGroup, ...],
    questions: tuple[QuestionRecord, ...],
    sparse_rankings: tuple[_FrozenRanking, ...],
    dense_rankings: tuple[_FrozenRanking, ...],
    dense_index: MemmapDenseIndex,
    aliases: AliasIndex,
    sparse_contract: SparseExecutionContract,
    sparse_manifest: DiskBm25Manifest,
    document_aliases: Mapping[str, Sequence[str]],
    context_law_keys: Mapping[str, str],
    query_row_by_id: Mapping[str, int],
    query_vectors: np.ndarray,
    workers: int,
) -> tuple[tuple[FusionFeatureRow, ...], float]:
    group_by_id = {item.question_id: item for item in groups}
    question_by_id = {item.question_id: item for item in questions}
    sparse_by_id = {item.question_id: item.chunk_ids for item in sparse_rankings}
    dense_by_id = {item.question_id: item.chunk_ids for item in dense_rankings}
    dense_row_by_id = {chunk_id: row for row, chunk_id in enumerate(dense_index.chunk_ids)}
    dense_vectors = np.load(_DENSE_INDEX / "vectors.f16.npy", mmap_mode="r")
    split = build_group_split(tuple(group_by_id))
    validation_ids = frozenset(split.validation_question_ids)
    local = threading.local()
    created: list[_WorkerResources] = []
    created_lock = threading.Lock()
    index_manifest_data = _INDEX_MANIFEST.read_bytes()

    def resources() -> _WorkerResources:
        existing = getattr(local, "resources", None)
        if existing is not None:
            return cast(_WorkerResources, existing)
        connection = open_worker_read_only_connection(_DATABASE)
        raw = DiskBm25Index(
            connection=connection,
            chunks_path=_CHUNKS,
            manifest=sparse_manifest,
            manifest_data=index_manifest_data,
        )
        audit_sparse_execution_contract(raw, sparse_contract, candidate_limit=_CANDIDATE_LIMIT)
        bounded = BoundedTopKDiskBm25Index(raw)
        value = _WorkerResources(raw, _FixedLimitBm25(CachedExpandedSparseIndex(bounded)))
        local.resources = value
        with created_lock:
            created.append(value)
        return value

    def one(group: DiscoveryGroup) -> tuple[FusionFeatureRow, ...]:
        worker = resources()
        question = question_by_id[group.question_id]
        sparse_result = retrieve_question(
            question,
            index=worker.retrieval_index,
            aliases=aliases,
            candidate_limit=_CANDIDATE_LIMIT,
        )
        sparse_ids = tuple(item.chunk.chunk_id for item in sparse_result.candidates)
        if sparse_ids != sparse_by_id[group.question_id]:
            raise SystemExit("D-067 sparse reconstruction differs from frozen D-066")
        dense_ids = dense_by_id[group.question_id]
        union_ids = tuple(sorted(set(sparse_ids) | set(dense_ids), key=str.encode))
        by_id = {item.chunk.chunk_id: item for item in sparse_result.candidates}
        missing_ids = tuple(chunk_id for chunk_id in union_ids if chunk_id not in by_id)
        missing_chunks = worker.raw_index.chunks_by_ids(missing_ids)
        chunks = {item.chunk.chunk_id: item.chunk for item in sparse_result.candidates}
        chunks.update({item.chunk_id: item for item in missing_chunks})
        if set(chunks) != set(union_ids):
            raise SystemExit("D-067 candidate chunk resolution is incomplete")
        sparse_rank = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids, start=1)}
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        query_vector = np.asarray(
            query_vectors[query_row_by_id[group.question_id]], dtype=np.float32
        )
        query = build_query_legal_signals(
            question.question,
            document_aliases=document_aliases,
        )
        positives = frozenset(group.positive_chunk_ids)
        partition: FusionPartition = "validation" if group.question_id in validation_ids else "fit"
        output: list[FusionFeatureRow] = []
        for chunk_id in union_ids:
            sparse_candidate = by_id.get(chunk_id)
            rank_dense = dense_rank.get(chunk_id)
            dense_score = None
            if rank_dense is not None:
                vector = np.asarray(dense_vectors[dense_row_by_id[chunk_id]], dtype=np.float32)
                dense_score = float(vector @ query_vector)
            chunk = chunks[chunk_id]
            signals = FusionCandidateSignals(
                chunk=chunk,
                sparse_score=(None if sparse_candidate is None else sparse_candidate.sparse_score),
                sparse_rank=sparse_rank.get(chunk_id),
                dense_score=dense_score,
                dense_rank=rank_dense,
                exact_reference_flag=(
                    False if sparse_candidate is None else sparse_candidate.exact_reference_match
                ),
                candidate_law_key=context_law_keys.get(chunk.context_id),
            )
            output.append(
                FusionFeatureRow(
                    question_id=group.question_id,
                    question_checksum=group.question_checksum,
                    chunk_id=chunk_id,
                    partition=partition,
                    label=int(chunk_id in positives),
                    feature_values=build_fusion_feature_values(query, signals),
                    chunk_checksum=chunk.chunk_checksum,
                )
            )
        return tuple(output)

    started = time.perf_counter()
    try:
        nested = ordered_bounded_map(
            one,
            groups,
            max_workers=workers,
            progress=lambda position, total: (
                print(f"D067-FEATURES: {position}/{total}", flush=True)
                if position % 100 == 0 or position == total
                else None
            ),
        )
    finally:
        for value in created:
            value.close()
    elapsed = time.perf_counter() - started
    return tuple(row for group_rows in nested for row in group_rows), elapsed


def main() -> int:
    inputs = _check_inputs()
    groups, questions = _load_groups_and_questions()
    sparse_rankings = _load_rankings(_SPARSE_RANKINGS, "R-DISC-0-BM25")
    dense_rankings = _load_rankings(_DENSE_RANKINGS, "R-DISC-1-QWEN3-DENSE")
    expected_ids = tuple(item.question_id for item in groups)
    if (
        tuple(item.question_id for item in sparse_rankings) != expected_ids
        or tuple(item.question_id for item in dense_rankings) != expected_ids
    ):
        raise SystemExit("D-067 ranking/group identity drift")
    document_aliases = _load_document_aliases()
    context_law_keys = _load_context_law_keys()
    query_row_by_id, query_vectors = _load_query_vectors(groups)
    split = build_group_split(expected_ids)

    index_manifest_data = _INDEX_MANIFEST.read_bytes()
    alias_manifest_data = _ALIAS_MANIFEST.read_bytes()
    with open_disk_bm25_index(
        database_path=_DATABASE,
        chunks_path=_CHUNKS,
        manifest_data=index_manifest_data,
    ) as sparse_index:
        sparse_contract = SparseExecutionContract.from_index(
            sparse_index, candidate_limit=_CANDIDATE_LIMIT
        )
        aliases = load_frozen_alias_artifact(
            _ALIASES.read_bytes(),
            manifest_data=alias_manifest_data,
            corpus_checksum=sparse_index.manifest.corpus_checksum,
            artifact_path=_ALIASES.name,
        )
        dense_index = MemmapDenseIndex(_DENSE_INDEX, block_rows=32_768)
        rows, elapsed = _build_rows(
            groups=groups,
            questions=questions,
            sparse_rankings=sparse_rankings,
            dense_rankings=dense_rankings,
            dense_index=dense_index,
            aliases=aliases,
            sparse_contract=sparse_contract,
            sparse_manifest=sparse_index.manifest,
            document_aliases=document_aliases,
            context_law_keys=context_law_keys,
            query_row_by_id=query_row_by_id,
            query_vectors=query_vectors,
            workers=4,
        )

    feature_data = serialize_feature_rows(rows)
    if feature_data != serialize_feature_rows(tuple(reversed(rows))):
        raise SystemExit("D-067 feature serialization replay mismatch")
    feature_checksum = write_immutable_bytes(_OUTPUT / "D067.features.v1.jsonl", feature_data)
    split_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-group-split.v1",
            **asdict(split),
        }
    )
    split_checksum = write_immutable_bytes(_OUTPUT / "D067.group-split.v1.json", split_data)
    fit_rows = sum(row.partition == "fit" for row in rows)
    validation_rows = len(rows) - fit_rows
    positive_rows = sum(row.label for row in rows)
    positive_groups = {row.question_id for row in rows if row.label == 1}
    manifest_data = content_json_bytes(
        {
            "schema_version": "evaluation.d067-features.manifest.v1",
            "status": "COMPLETE",
            "input_checksums": inputs,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "question_group_count": len(groups),
            "fit_group_count": len(split.fit_question_ids),
            "validation_group_count": len(split.validation_question_ids),
            "candidate_row_count": len(rows),
            "fit_candidate_row_count": fit_rows,
            "validation_candidate_row_count": validation_rows,
            "positive_candidate_row_count": positive_rows,
            "positive_bearing_group_count": len(positive_groups),
            "all_negative_candidate_group_count": len(groups) - len(positive_groups),
            "feature_checksum": feature_checksum,
            "split_checksum": split_checksum,
            "label_source": "d065-canonical-positive-chunk-membership-only",
            "answer_derived_fields_used_as_features": False,
            "development_or_public_data_used": False,
            "gpu_used": False,
            "modal_used": False,
            "fit_performed": False,
            "construction_wall_seconds": elapsed,
            "worker_count": 4,
        }
    )
    manifest_checksum = write_immutable_bytes(
        _OUTPUT / "D067.features.manifest.v1.json", manifest_data
    )
    print(
        json.dumps(
            {
                "feature_checksum": feature_checksum,
                "split_checksum": split_checksum,
                "manifest_checksum": manifest_checksum,
                "row_count": len(rows),
                "wall_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
