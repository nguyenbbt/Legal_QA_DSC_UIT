from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.retrieval.test_disk_bm25 import _build, _chunk_bytes

from legal_rag.retrieval.sparse_execution import (
    SCORE_ABSOLUTE_TOLERANCE,
    BoundedTopKDiskBm25Index,
    SparseExecutionContract,
    SparseExecutionError,
    assert_sparse_result_parity,
    audit_sparse_execution_contract,
    open_worker_read_only_connection,
)


def test_native_profiler_function_names_are_classified_by_marker() -> None:
    from scripts.run_d066_sparse_remediation import _stat_seconds

    stats = SimpleNamespace(
        stats={
            ("~", 0, "{method 'execute' of 'sqlite3.Connection' objects}"): (
                1,
                1,
                0.25,
                0.3,
                {},
            )
        }
    )

    assert _stat_seconds(stats, "execute", cumulative=False) == 0.25


def test_worker_read_only_connection_can_close_on_the_orchestrator_thread(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    source = sqlite3.connect(database)
    source.execute("CREATE TABLE evidence (chunk_id TEXT NOT NULL)")
    source.commit()
    source.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_connection = pool.submit(open_worker_read_only_connection, database).result()

    with pytest.raises(sqlite3.OperationalError):
        worker_connection.execute("INSERT INTO evidence VALUES ('forbidden')")
    worker_connection.close()


def test_worker_read_only_connection_supports_temporary_row_vocabulary(
    tmp_path: Path,
) -> None:
    chunks_data = _chunk_bytes("điều luật mẫu", "điều luật khác")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)

    from legal_rag.retrieval.disk_bm25 import DiskBm25Index, open_disk_bm25_index

    manifest_data = manifest_path.read_bytes()
    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_data,
    ) as validated:
        manifest = validated.manifest
    worker_connection = open_worker_read_only_connection(database_path)
    worker_index = DiskBm25Index(
        connection=worker_connection,
        chunks_path=chunks_path,
        manifest=manifest,
        manifest_data=manifest_data,
    )

    result = worker_index.retrieve_bounded_top_k("điều luật", candidate_limit=2)
    worker_index.close()

    assert len(result.candidates) == 2


def test_bounded_top_k_preserves_ids_order_scores_and_replay(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes(
        "điều 5 luật mẫu luật mẫu",
        "điều 5 luật khác",
        "khoản 2 điều 5 luật mẫu",
        "điểm a khoản 2",
        "văn bản không liên quan",
    )
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)

    from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index

    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    ) as index:
        reference = index.retrieve("điều 5 khoản 2 luật mẫu", candidate_limit=4)
        optimized_index = BoundedTopKDiskBm25Index(index)
        optimized = optimized_index.retrieve("điều 5 khoản 2 luật mẫu", candidate_limit=4)
        replay = optimized_index.retrieve("điều 5 khoản 2 luật mẫu", candidate_limit=4)
        assert optimized_index.index_checksum == index.index_checksum

    parity = assert_sparse_result_parity(reference, optimized)
    assert parity.score_absolute_tolerance == SCORE_ABSOLUTE_TOLERANCE
    assert parity.candidate_count == 4
    assert parity.maximum_absolute_score_delta <= SCORE_ABSOLUTE_TOLERANCE
    assert optimized.json_bytes() == replay.json_bytes()


def test_bounded_top_k_preserves_empty_query_contract(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("điều luật mẫu")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)

    from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index

    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    ) as index:
        reference = index.retrieve("---", candidate_limit=1)
        optimized = index.retrieve_bounded_top_k("---", candidate_limit=1)

    assert_sparse_result_parity(reference, optimized)
    assert reference.json_bytes() == optimized.json_bytes()


def test_sparse_execution_contract_fails_closed_on_candidate_k_drift(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("điều luật mẫu")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)

    from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index

    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    ) as index:
        contract = SparseExecutionContract.from_index(index, candidate_limit=50)
        report = audit_sparse_execution_contract(index, contract, candidate_limit=50)
        with pytest.raises(SparseExecutionError) as mismatch:
            audit_sparse_execution_contract(
                index,
                replace(contract, candidate_limit=100),
                candidate_limit=50,
            )

    assert report.status == "PASS"
    assert report.mismatched_fields == ()
    assert mismatch.value.code == "SPARSE_EXECUTION_CONTRACT_MISMATCH"


def test_sparse_parity_rejects_rank_order_drift(tmp_path: Path) -> None:
    chunks_data = _chunk_bytes("điều luật mẫu", "điều luật khác")
    chunks_path, database_path, manifest_path, _summary = _build(tmp_path, "index", chunks_data)

    from legal_rag.retrieval.disk_bm25 import open_disk_bm25_index

    with open_disk_bm25_index(
        database_path=database_path,
        chunks_path=chunks_path,
        manifest_data=manifest_path.read_bytes(),
    ) as index:
        reference = index.retrieve("điều luật", candidate_limit=2)
        reversed_result = replace(reference, candidates=tuple(reversed(reference.candidates)))

    with pytest.raises(SparseExecutionError) as mismatch:
        assert_sparse_result_parity(reference, reversed_result)

    assert mismatch.value.code == "SPARSE_EXECUTION_PARITY_MISMATCH"
