"""Profile and prove frozen small-set parity for the single D066-R1 corrective."""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_d066_candidate_discovery as d066

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.discovery_tournament import load_discovery_groups
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index
from legal_rag.retrieval.sparse_execution import (
    SCORE_ABSOLUTE_TOLERANCE,
    SparseExecutionContract,
    assert_sparse_result_parity,
    audit_sparse_execution_contract,
)
from legal_rag.training.rag_sft import load_gold_questions

_PARITY_COUNT = 2


def _stat_seconds(stats: pstats.Stats, function_marker: str, *, cumulative: bool) -> float:
    position = 3 if cumulative else 2
    return sum(
        values[position]
        for (_filename, _line, name), values in stats.stats.items()
        if function_marker in name
    )


def _profile(
    function: Callable[[], SparseRetrievalResult],
) -> tuple[SparseRetrievalResult, dict[str, object]]:
    profiler = cProfile.Profile()
    process_started = time.process_time()
    wall_started = time.perf_counter()
    profiler.enable()
    result = function()
    profiler.disable()
    wall_seconds = time.perf_counter() - wall_started
    process_seconds = time.process_time() - process_started
    stats = pstats.Stats(profiler)
    serialization_started = time.perf_counter()
    serialized = result.json_bytes()
    serialization_seconds = time.perf_counter() - serialization_started
    return result, {
        "wall_seconds": wall_seconds,
        "process_seconds": process_seconds,
        "cpu_to_wall_ratio": process_seconds / wall_seconds if wall_seconds else 0.0,
        "query_tokenization_seconds": _stat_seconds(
            stats, "ordered_unique_query_terms", cumulative=True
        ),
        "candidate_lookup_sql_seconds": _stat_seconds(stats, "execute", cumulative=False)
        + _stat_seconds(stats, "fetchall", cumulative=False),
        "bm25_python_or_sql_orchestration_seconds": _stat_seconds(
            stats, "retrieve", cumulative=False
        )
        + _stat_seconds(stats, "retrieve_bounded_top_k", cumulative=False),
        "top_k_selection_seconds": _stat_seconds(stats, "sorted", cumulative=True),
        "chunk_disk_io_and_parse_seconds": _stat_seconds(stats, "_load_chunks", cumulative=True),
        "serialization_seconds": serialization_seconds,
        "serialized_checksum": checksum_bytes(serialized),
    }


def main() -> int:
    input_checksums = d066._check_inputs()
    question_data = d066._QUESTIONS.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        d066._SPLIT.read_bytes(),
        expected_source_checksum=d066._EXPECTED["questions"],
        expected_question_ids=tuple(item.question_id for item in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    groups = load_discovery_groups(
        supervision_data=d066._SUPERVISION.read_bytes(),
        question_source_data=question_data,
        train_question_ids=train_ids,
        expected_positive_count=d066._POSITIVE_COUNT,
        expected_supervision_checksum=d066._EXPECTED["supervision"],
    )
    question_by_id = {question.question_id: question for question in questions}

    index_started = time.perf_counter()
    index = open_disk_bm25_index(
        database_path=d066._DATABASE,
        chunks_path=d066._CHUNKS,
        manifest_data=d066._INDEX_MANIFEST.read_bytes(),
    )
    index_load_seconds = time.perf_counter() - index_started
    try:
        contract = SparseExecutionContract.from_index(index, candidate_limit=d066._CANDIDATE_LIMIT)
        compatibility = audit_sparse_execution_contract(
            index, contract, candidate_limit=d066._CANDIDATE_LIMIT
        )
        parity_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        for position, group in enumerate(groups[:_PARITY_COUNT]):
            question = question_by_id[group.question_id]
            reference, reference_profile = _profile(
                lambda question=question: index.retrieve(
                    question.question, candidate_limit=d066._CANDIDATE_LIMIT
                )
            )
            optimized, optimized_profile = _profile(
                lambda question=question: index.retrieve_bounded_top_k(
                    question.question, candidate_limit=d066._CANDIDATE_LIMIT
                )
            )
            replay = index.retrieve_bounded_top_k(
                question.question, candidate_limit=d066._CANDIDATE_LIMIT
            )
            parity = assert_sparse_result_parity(reference, optimized)
            if optimized.json_bytes() != replay.json_bytes():
                raise RuntimeError("D066-R1 optimized sparse replay differs")
            parity_rows.append(
                {
                    "position": position,
                    "question_id": group.question_id,
                    "question_checksum": checksum_bytes(question.question.encode("utf-8")),
                    **asdict(parity),
                    "optimized_replay_byte_identical": True,
                }
            )
            profile_rows.append(
                {
                    "position": position,
                    "question_id": group.question_id,
                    "reference": reference_profile,
                    "d066_r1_bounded": optimized_profile,
                }
            )
    finally:
        index.close()

    parity_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r1-sparse-parity.v1",
            "status": "PASS",
            "sample_selection": "first-two-d065-positive-groups.v1",
            "sample_count": _PARITY_COUNT,
            "candidate_limit": d066._CANDIDATE_LIMIT,
            "score_absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
            "compatibility": asdict(compatibility),
            "input_checksums": input_checksums,
            "rows": parity_rows,
            "retrieval_semantics_changed": False,
        }
    )
    profile_data = content_json_bytes(
        {
            "schema_version": "evaluation.d066-r1-sparse-profile.v2",
            "sample_selection": "first-two-d065-positive-groups.v1",
            "sample_count": _PARITY_COUNT,
            "index_loading_seconds": index_load_seconds,
            "corpus_tokenization": {
                "seconds": 0.0,
                "state": "reused_immutable_fts_postings",
                "tokenizer_id": contract.tokenizer_id,
                "tokenizer_revision": contract.tokenizer_revision,
            },
            "profiles": profile_rows,
            "disk_io_scope": "read_only_sqlite_and_checksum_bound_chunk_jsonl",
            "worker_process_overhead": "measured_by_exact_100_group_preflight",
            "gpu_used": False,
            "paid_service_used": False,
        }
    )
    output = Path("artifacts/evaluations/post-d062/D066-candidate-discovery-v1/preflight")
    parity_checksum = write_immutable_bytes(output / "D066-R1.sparse-parity.v1.json", parity_data)
    profile_checksum = write_immutable_bytes(
        output / "D066-R1.sparse-profile.v2.json", profile_data
    )
    print(
        json.dumps(
            {
                "parity": parity_checksum,
                "profile": profile_checksum,
                "sample_count": _PARITY_COUNT,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
