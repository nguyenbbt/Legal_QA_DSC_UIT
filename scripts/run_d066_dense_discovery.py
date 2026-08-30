"""Run D-066 Qwen dense retrieval, paired diagnostics, union, and fixed RRF-60."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.dense_discovery import (
    build_fixed_rrf_60_rankings,
    diagnose_sparse_dense,
    evaluate_sparse_dense_union,
    serialize_dense_diagnostics,
    serialize_union_evaluation,
)
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
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.models.cuda_determinism import configure_cublas_workspace
from legal_rag.models.huggingface_local import Qwen3EmbeddingBackend
from legal_rag.retrieval.dense_batch import (
    compare_dense_hit_replay,
    merge_top_k_blocks,
    select_top_k_rows,
)
from legal_rag.retrieval.dense_store import DenseHit, MemmapDenseIndex
from legal_rag.training.rag_sft import load_gold_questions

_OUTPUT = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1")
_INDEX = Path("artifacts/indices/dense/d066-r-disc-1-116f6cf195224b12")
_QUESTIONS = Path("artifacts/internal/train.questions.jsonl")
_SPLIT = Path("artifacts/splits/train-dev-test.v1.json")
_SUPERVISION = Path("artifacts/training/retrieval-supervision/v2/retrieval-supervision.v2.jsonl")
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_DATABASE = Path("artifacts/indices/bm25.v1.active.sqlite3")
_SPARSE_RANKINGS = _OUTPUT / "R-DISC-0-BM25.rankings.v1.jsonl"
_SPARSE_EVALUATION = _OUTPUT / "R-DISC-0-BM25.evaluation.v1.json"
_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_MODEL_CHECKPOINT = Path(f".local/models/qwen3-embedding-0.6b/{_MODEL_REVISION}")
_QUERY_INSTRUCTION = "Retrieve Vietnamese legal passages that answer the question."
_EXPECTED = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "supervision": "sha256:affd0969261243f0718e9faaed5d9cc0617138714cc190171cb9e7bf7253c1d6",
    "chunks": "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143",
    "database": "sha256:2b977231a0be77fa2409988ecb1f0955bd22d7175130b08affc49fd04771fdc1",
    "sparse_rankings": "sha256:a3c10cd01274efc1cf93b81efd23a9c8756b67acef4119dd165966aa7a866463",
    "sparse_evaluation": "sha256:2795b02443813a9fc0d4ffce92d921bac029d5292a90d63262289f5e106a9358",
}
_TRAIN_COUNT = 5_582
_POSITIVE_COUNT = 2_391
_CHUNK_COUNT = 641_118
_CANDIDATE_LIMIT = 50
_QUERY_BATCH_SIZE = 8
_SEARCH_QUERY_BATCH_SIZE = 64
_SEARCH_BLOCK_ROWS = 32_768
_REPLAY_SCORE_TOLERANCE = 1e-6


def _streaming_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_groups() -> tuple[DiscoveryGroup, ...]:
    paths = {
        "questions": _QUESTIONS,
        "split": _SPLIT,
        "supervision": _SUPERVISION,
        "chunks": _CHUNKS,
        "database": _DATABASE,
        "sparse_rankings": _SPARSE_RANKINGS,
        "sparse_evaluation": _SPARSE_EVALUATION,
    }
    if {name: _streaming_checksum(path) for name, path in paths.items()} != _EXPECTED:
        raise SystemExit("D-066 dense discovery input checksum drift")
    question_data = _QUESTIONS.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        _SPLIT.read_bytes(),
        expected_source_checksum=_EXPECTED["questions"],
        expected_question_ids=tuple(question.question_id for question in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    if len(train_ids) != _TRAIN_COUNT:
        raise SystemExit("D-066 dense discovery train count drift")
    return load_discovery_groups(
        supervision_data=_SUPERVISION.read_bytes(),
        question_source_data=question_data,
        train_question_ids=train_ids,
        expected_positive_count=_POSITIVE_COUNT,
        expected_supervision_checksum=_EXPECTED["supervision"],
    )


def _load_sparse_rankings(groups: tuple[DiscoveryGroup, ...]) -> tuple[DiscoveryRanking, ...]:
    expected_ids = tuple(group.question_id for group in groups)
    rankings: list[DiscoveryRanking] = []
    for line in _SPARSE_RANKINGS.read_bytes().splitlines():
        row = json.loads(line)
        if row.get("schema_version") != "retrieval.discovery-ranking.v1":
            raise SystemExit("D-066 sparse ranking schema drift")
        rankings.append(
            DiscoveryRanking(
                str(row["question_id"]),
                tuple(
                    DiscoveryCandidate(str(chunk_id), "") for chunk_id in row["candidate_chunk_ids"]
                ),
            )
        )
    if tuple(ranking.question_id for ranking in rankings) != expected_ids:
        raise SystemExit("D-066 sparse ranking identity drift")
    return tuple(rankings)


def _query_vector_bytes(vectors: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(vectors, dtype=np.float16), allow_pickle=False)
    return stream.getvalue()


def _encode_queries(groups: tuple[DiscoveryGroup, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    vector_path = _OUTPUT / "R-DISC-1.query-vectors.f16.npy"
    ids_path = _OUTPUT / "R-DISC-1.query-vector-ids.v1.jsonl"
    manifest_path = _OUTPUT / "R-DISC-1.query-vectors.manifest.v1.json"
    expected_ids_data = b"".join(
        content_json_bytes(
            {
                "schema_version": "retrieval.dense-query-vector-row.v1",
                "row": row,
                "question_id": group.question_id,
                "question_checksum": group.question_checksum,
            }
        )
        for row, group in enumerate(groups)
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes())
        if (
            checksum_bytes(ids_path.read_bytes()) != manifest.get("ids_checksum")
            or ids_path.read_bytes() != expected_ids_data
            or _streaming_checksum(vector_path) != manifest.get("vector_checksum")
        ):
            raise SystemExit("D-066 dense query-vector checkpoint drift")
        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.shape != (len(groups), 1_024) or vectors.dtype != np.float16:
            raise SystemExit("D-066 dense query-vector shape drift")
        return vectors, manifest

    import torch

    torch.cuda.reset_peak_memory_stats()
    backend = Qwen3EmbeddingBackend(
        _MODEL_CHECKPOINT,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        device="cuda",
        batch_size=_QUERY_BATCH_SIZE,
        maximum_length=2_048,
        query_instruction=_QUERY_INSTRUCTION,
    )
    started = time.perf_counter()
    vectors = np.asarray(backend.encode_queries(tuple(group.question for group in groups)))
    elapsed = time.perf_counter() - started
    if vectors.shape != (len(groups), 1_024) or not np.isfinite(vectors).all():
        raise SystemExit("D-066 dense query encoding invalid")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise SystemExit("D-066 dense query encoding contains zero")
    vector_data = _query_vector_bytes(vectors / norms)
    vector_checksum = write_immutable_bytes(vector_path, vector_data)
    ids_checksum = write_immutable_bytes(ids_path, expected_ids_data)
    manifest_data = content_json_bytes(
        {
            "schema_version": "retrieval.dense-query-vectors.manifest.v1",
            "model_id": _MODEL_ID,
            "model_revision": _MODEL_REVISION,
            "query_instruction": _QUERY_INSTRUCTION,
            "question_count": len(groups),
            "dimension": 1_024,
            "storage_dtype": "float16",
            "vector_checksum": vector_checksum,
            "ids_checksum": ids_checksum,
            "encode_wall_seconds": elapsed,
            "encode_peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
    )
    write_immutable_bytes(manifest_path, manifest_data)
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(vector_path, allow_pickle=False), json.loads(manifest_data)


def _search_dense(index: MemmapDenseIndex, queries: np.ndarray) -> tuple[tuple[DenseHit, ...], ...]:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output: list[tuple[DenseHit, ...]] = []
    for query_start in range(0, len(queries), _SEARCH_QUERY_BATCH_SIZE):
        query_stop = min(query_start + _SEARCH_QUERY_BATCH_SIZE, len(queries))
        query_values = np.asarray(queries[query_start:query_stop], dtype=np.float32)
        query_values /= np.linalg.norm(query_values, axis=1, keepdims=True)
        query_tensor = torch.tensor(query_values, dtype=torch.float32, device="cuda")
        current: tuple[tuple[DenseHit, ...], ...] = tuple(() for _ in query_values)
        for chunk_ids, vectors in index.vector_blocks():
            document_tensor = torch.tensor(
                np.asarray(vectors, dtype=np.float32), dtype=torch.float32, device="cuda"
            )
            score_block = (query_tensor @ document_tensor.T).cpu().numpy()
            block_hits = select_top_k_rows(score_block, chunk_ids, limit=_CANDIDATE_LIMIT)
            current = merge_top_k_blocks(current, block_hits, limit=_CANDIDATE_LIMIT)
            del document_tensor
        output.extend(current)
        print(f"R-DISC-1-RETRIEVE: {query_stop}/{len(queries)}", flush=True)
    return tuple(output)


def _lookup_display_text(chunk_ids: set[str]) -> dict[str, str]:
    locations: dict[str, tuple[int, int]] = {}
    connection = sqlite3.connect(_DATABASE)
    connection.execute("PRAGMA query_only=ON")
    try:
        ordered = sorted(chunk_ids, key=str.encode)
        for start in range(0, len(ordered), 800):
            batch = ordered[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            for chunk_id, offset, length in connection.execute(
                f"SELECT chunk_id, source_offset, source_length FROM documents "
                f"WHERE chunk_id IN ({placeholders})",
                batch,
            ):
                locations[str(chunk_id)] = (int(offset), int(length))
    finally:
        connection.close()
    if set(locations) != chunk_ids:
        raise SystemExit("D-066 dense candidate display lookup is incomplete")
    displays: dict[str, str] = {}
    with _CHUNKS.open("rb") as stream:
        for chunk_id in sorted(locations, key=str.encode):
            offset, length = locations[chunk_id]
            stream.seek(offset)
            row = json.loads(stream.read(length))
            if row.get("chunk_id") != chunk_id or not isinstance(row.get("display_text"), str):
                raise SystemExit("D-066 dense candidate display identity drift")
            displays[chunk_id] = row["display_text"]
    return displays


def _with_displays(
    rankings: tuple[DiscoveryRanking, ...], displays: dict[str, str]
) -> tuple[DiscoveryRanking, ...]:
    return tuple(
        DiscoveryRanking(
            ranking.question_id,
            tuple(
                DiscoveryCandidate(item.chunk_id, displays[item.chunk_id])
                for item in ranking.candidates
            ),
        )
        for ranking in rankings
    )


def main() -> int:
    cublas_workspace_config = configure_cublas_workspace(os.environ)
    import psutil  # type: ignore[import-untyped]
    import torch

    groups = _load_groups()
    sparse_ids = _load_sparse_rankings(groups)
    index = MemmapDenseIndex(_INDEX, block_rows=_SEARCH_BLOCK_ROWS)
    if (
        index.manifest.chunk_count != _CHUNK_COUNT
        or index.manifest.model_id != _MODEL_ID
        or index.manifest.model_revision != _MODEL_REVISION
        or index.manifest.dimension != 1_024
        or index.manifest.storage_dtype != "float16"
        or index.manifest.source_chunk_checksum != _EXPECTED["chunks"]
    ):
        raise SystemExit("D-066 dense index manifest drift")
    queries, query_manifest = _encode_queries(groups)
    torch.cuda.reset_peak_memory_stats()
    search_started = time.perf_counter()
    dense_hits = _search_dense(index, queries)
    search_seconds = time.perf_counter() - search_started
    search_peak_vram = int(torch.cuda.max_memory_allocated())
    dense_ids = tuple(
        DiscoveryRanking(
            group.question_id,
            tuple(DiscoveryCandidate(hit.chunk_id, "") for hit in hits),
        )
        for group, hits in zip(groups, dense_hits, strict=True)
    )
    replay_hits = _search_dense(index, queries[:2])
    replay_maximum_score_delta = compare_dense_hit_replay(
        dense_hits[:2], replay_hits, score_tolerance=_REPLAY_SCORE_TOLERANCE
    )
    replay_passed = True

    candidate_ids = {
        candidate.chunk_id
        for ranking in (*sparse_ids, *dense_ids)
        for candidate in ranking.candidates
    }
    displays = _lookup_display_text(candidate_ids)
    sparse = _with_displays(sparse_ids, displays)
    dense = _with_displays(dense_ids, displays)
    sparse_evaluation = evaluate_discovery_arm("R-DISC-0-BM25", groups, sparse)
    if (
        checksum_bytes(serialize_discovery_evaluation(sparse_evaluation))
        != _EXPECTED["sparse_evaluation"]
    ):
        raise SystemExit("D-066 frozen sparse evaluation did not reproduce")
    dense_evaluation = evaluate_discovery_arm("R-DISC-1-QWEN3-DENSE", groups, dense)
    diagnostics = diagnose_sparse_dense(groups, sparse, dense)
    union = evaluate_sparse_dense_union(groups, sparse, dense)
    rrf = build_fixed_rrf_60_rankings(sparse, dense, limit=_CANDIDATE_LIMIT)
    rrf_evaluation = evaluate_discovery_arm("R-DISC-4B-FIXED-RRF-60", groups, rrf)
    dense_comparison = compare_discovery_arms(sparse_evaluation, dense_evaluation)
    rrf_comparison = compare_discovery_arms(sparse_evaluation, rrf_evaluation)

    artifact_data = {
        "R-DISC-1.rankings.v1.jsonl": serialize_discovery_rankings(dense_evaluation.arm_id, dense),
        "R-DISC-1.evaluation.v1.json": serialize_discovery_evaluation(dense_evaluation),
        "R-DISC-1.sparse-dense-diagnostics.v1.json": serialize_dense_diagnostics(diagnostics),
        "R-DISC-1.comparison.v1.json": serialize_discovery_comparison(dense_comparison),
        "R-DISC-4A.union-evaluation.v1.json": serialize_union_evaluation(union),
        "R-DISC-4B-RRF60.rankings.v1.jsonl": serialize_discovery_rankings(
            rrf_evaluation.arm_id, rrf
        ),
        "R-DISC-4B-RRF60.evaluation.v1.json": serialize_discovery_evaluation(rrf_evaluation),
        "R-DISC-4B-RRF60.comparison.v1.json": serialize_discovery_comparison(rrf_comparison),
    }
    replay_artifact_data = {
        "R-DISC-1.rankings.v1.jsonl": serialize_discovery_rankings(dense_evaluation.arm_id, dense),
        "R-DISC-1.evaluation.v1.json": serialize_discovery_evaluation(dense_evaluation),
        "R-DISC-1.sparse-dense-diagnostics.v1.json": serialize_dense_diagnostics(diagnostics),
        "R-DISC-1.comparison.v1.json": serialize_discovery_comparison(dense_comparison),
        "R-DISC-4A.union-evaluation.v1.json": serialize_union_evaluation(union),
        "R-DISC-4B-RRF60.rankings.v1.jsonl": serialize_discovery_rankings(
            rrf_evaluation.arm_id, rrf
        ),
        "R-DISC-4B-RRF60.evaluation.v1.json": serialize_discovery_evaluation(rrf_evaluation),
        "R-DISC-4B-RRF60.comparison.v1.json": serialize_discovery_comparison(rrf_comparison),
    }
    if artifact_data != replay_artifact_data:
        raise SystemExit("D-066 dense artifact replay drift")
    artifact_checksums = {
        name: write_immutable_bytes(_OUTPUT / name, data) for name, data in artifact_data.items()
    }
    dense_novel = len(diagnostics.dense_novel_positive_group_ids)
    union_improves = union.recall_at[50] > sparse_evaluation.recall_at[50]
    rrf_captures_union = rrf_evaluation.recall_at[50] >= union.recall_at[50]
    recommended_winner = (
        "R-DISC-4B-FIXED-RRF-60"
        if dense_novel and rrf_captures_union
        else "R-DISC-4A-SPARSE-DENSE-UNION"
        if dense_novel and union_improves
        else "EXACT_PLUS_BM25"
    )
    learned_fusion_recommended = bool(dense_novel and union_improves and not rrf_captures_union)
    telemetry = {
        "schema_version": "evaluation.d066-r-disc-1-r-disc-4.telemetry.v1",
        "execution_mode": "local-gpu-offline",
        "query_encode_wall_seconds": query_manifest["encode_wall_seconds"],
        "dense_search_wall_seconds": search_seconds,
        "query_encode_peak_vram_bytes": query_manifest["encode_peak_vram_bytes"],
        "dense_search_peak_vram_bytes": search_peak_vram,
        "process_rss_bytes": int(psutil.Process(os.getpid()).memory_info().rss),
        "vector_bytes": (_INDEX / "vectors.f16.npy").stat().st_size,
        "ids_bytes": (_INDEX / "chunk-ids.jsonl").stat().st_size,
        "manifest_bytes": (_INDEX / "manifest.json").stat().st_size,
        "search_query_batch_size": _SEARCH_QUERY_BATCH_SIZE,
        "search_block_rows": _SEARCH_BLOCK_ROWS,
        "candidate_limit_per_arm": _CANDIDATE_LIMIT,
        "rrf_constant": 60,
        "cublas_workspace_config": cublas_workspace_config,
        "replay_score_tolerance": _REPLAY_SCORE_TOLERANCE,
        "replay_maximum_score_delta": replay_maximum_score_delta,
        "paid_service_used": False,
        "cost_usd": 0.0,
        "fit_performed": False,
    }
    telemetry_data = content_json_bytes(telemetry)
    telemetry_checksum = write_immutable_bytes(
        _OUTPUT / "R-DISC-1-R-DISC-4.telemetry.v1.json", telemetry_data
    )
    manifest_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r-disc-1-r-disc-4.manifest.v1",
            "status": "COMPLETE",
            "positive_group_count": len(groups),
            "canonical_chunk_count": index.manifest.chunk_count,
            "standing_sparse_winner": "EXACT_PLUS_BM25",
            "model_id": _MODEL_ID,
            "model_revision": _MODEL_REVISION,
            "index_manifest_checksum": _streaming_checksum(_INDEX / "manifest.json"),
            "index_vector_checksum": index.manifest.vector_checksum,
            "index_ids_checksum": index.manifest.ids_checksum,
            "query_vector_manifest_checksum": _streaming_checksum(
                _OUTPUT / "R-DISC-1.query-vectors.manifest.v1.json"
            ),
            "artifact_checksums": artifact_checksums,
            "telemetry_checksum": telemetry_checksum,
            "dense_search_first_two_replay_identical": replay_passed,
            "recommended_d066_standing_winner": recommended_winner,
            "d067_learned_fusion_should_open": learned_fusion_recommended,
            "d067_status": "CLOSED",
            "development_data_used": False,
            "public_data_used": False,
            "fit_performed": False,
            "modal_used": False,
        }
    )
    manifest_checksum = write_immutable_bytes(
        _OUTPUT / "R-DISC-1-R-DISC-4.manifest.v1.json", manifest_data
    )
    print(
        json.dumps(
            {
                "dense": {
                    "recall_at": dense_evaluation.recall_at,
                    "evidence_set_recall_at": dense_evaluation.evidence_set_recall_at,
                    "mrr_at_50": dense_evaluation.mrr_at_50,
                    "answer_bearing_coverage_at": (dense_evaluation.answer_bearing_coverage_at),
                },
                "union": asdict(union),
                "rrf": {
                    "recall_at": rrf_evaluation.recall_at,
                    "evidence_set_recall_at": rrf_evaluation.evidence_set_recall_at,
                    "mrr_at_50": rrf_evaluation.mrr_at_50,
                    "answer_bearing_coverage_at": (rrf_evaluation.answer_bearing_coverage_at),
                },
                "diagnostic_counts": diagnostics.classification_counts,
                "recommended_d066_standing_winner": recommended_winner,
                "d067_learned_fusion_should_open": learned_fusion_recommended,
                "manifest_checksum": manifest_checksum,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
