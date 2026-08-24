from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.public_dry_run import (
    PublicDryRunError,
    build_public_evidence_queue,
    run_checkpointed_public_generation,
)
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.retrieval.bm25 import SparseRetrievalResult
from legal_rag.retrieval.exact import AliasIndex
from legal_rag.retrieval.models import RetrievalCandidate


def _question(question_id: str, position: int, source_checksum: str) -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": question_id,
            "original_id": question_id,
            "original_id_kind": "object_key_string",
            "source_position": position,
            "source_artifact": "fixtures/public.json",
            "source_checksum": source_checksum,
            "question": f"Question {question_id}",
            "answer": None,
            "answer_state": "unlabeled",
        }
    )


def _candidate(evidence_id: str, text: str, sparse_score: float) -> RetrievalCandidate:
    chunk = ChunkRecord(
        evidence_id,
        "1",
        "https://example.invalid",
        ("Điều 1",),
        "ARTICLE",
        "article",
        "1",
        0,
        len(text),
        text,
        text,
        0,
        checksum_bytes(text.encode()),
        checksum_bytes(b"context"),
    )
    return RetrievalCandidate(chunk, False, sparse_score)


class _Index:
    def __init__(self, candidates: tuple[RetrievalCandidate, ...]) -> None:
        self._candidates = candidates

    def retrieve(self, query: str) -> SparseRetrievalResult:
        return SparseRetrievalResult(
            query,
            ("query",),
            self._candidates,
            (),
            checksum_bytes(b"index"),
        )

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return ()

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return ()


class _Reranker:
    model_id = "fixture/reranker"
    model_revision = "revision-1"

    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.calls = 0
        self._fail_if_called = fail_if_called

    def score(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        if self._fail_if_called:
            raise AssertionError("a valid evidence checkpoint must bypass reranking")
        self.calls += 1
        return tuple(float(index) for index, _ in enumerate(documents, start=1))


class _Generator:
    model_id = "fixture/generator"
    model_revision = "revision-1"

    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.calls = 0
        self._fail_if_called = fail_if_called

    def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
        if self._fail_if_called:
            raise AssertionError("a valid checkpoint must bypass generation")
        self.calls += 1
        return f"Answer from {evidence[0]}"


def _fixture() -> tuple[bytes, tuple[QuestionRecord, ...], AliasIndex, _Index]:
    public_value = {
        "q2": {"question": "Question q2", "answer": None},
        "q1": {"question": "Question q1", "answer": None},
    }
    public_source = json.dumps(public_value, ensure_ascii=False, separators=(",", ":")).encode()
    source_checksum = checksum_bytes(public_source)
    questions = (
        _question("q2", 0, source_checksum),
        _question("q1", 1, source_checksum),
    )
    aliases = AliasIndex(
        (), checksum_bytes(b"corpus"), "fixtures/aliases.jsonl", checksum_bytes(b"aliases")
    )
    index = _Index(
        (
            _candidate("chunk-a", "Evidence A", 2.0),
            _candidate("chunk-b", "Evidence B", 1.0),
        )
    )
    return public_source, questions, aliases, index


def test_public_evidence_queue_is_ranked_and_contains_no_answer_text() -> None:
    _, questions, aliases, index = _fixture()

    artifacts = build_public_evidence_queue(
        questions,
        index=index,
        aliases=aliases,
        reranker=_Reranker(),
        retrieval_run_id="public-r2-fixture-v1",
        evidence_limit=1,
        reranker_candidate_limit=2,
    )

    rows = tuple(json.loads(line) for line in artifacts.queue_data.splitlines())
    assert tuple(row["question_id"] for row in rows) == ("q2", "q1")
    assert rows[0]["evidence"][0]["evidence_id"] == "chunk-b"
    assert all("answer" not in row and "gold_answer" not in row for row in rows)
    report = json.loads(artifacts.report_data)
    assert report["question_count"] == 2
    assert report["questions_without_evidence"] == 0


def test_public_evidence_queue_supports_exact_bm25_without_reranking() -> None:
    _, questions, aliases, index = _fixture()

    artifacts = build_public_evidence_queue(
        questions,
        index=index,
        aliases=aliases,
        reranker=None,
        retrieval_run_id="public-r0-fixture-v1",
        evidence_limit=1,
        reranker_candidate_limit=2,
    )

    rows = tuple(json.loads(line) for line in artifacts.queue_data.splitlines())
    assert rows[0]["evidence"][0]["evidence_id"] == "chunk-a"
    assert rows[0]["evidence"][0]["reranker_score"] is None
    report = json.loads(artifacts.report_data)
    assert report["retrieval_mode"] == "exact_bm25"
    assert report["reranker_model_id"] is None


def test_public_evidence_queue_resumes_checksum_bound_checkpoints_without_model_calls(
    tmp_path: Path,
) -> None:
    _, questions, aliases, index = _fixture()
    first_backend = _Reranker()
    arguments = {
        "questions": questions,
        "index": index,
        "aliases": aliases,
        "retrieval_run_id": "public-r2-fixture-v1",
        "evidence_limit": 1,
        "reranker_candidate_limit": 2,
        "checkpoint_directory": tmp_path / "evidence-checkpoints",
        "frozen_inputs": {
            "index": checksum_bytes(b"index"),
            "aliases": checksum_bytes(b"aliases"),
        },
    }

    first = build_public_evidence_queue(reranker=first_backend, **arguments)
    second_backend = _Reranker(fail_if_called=True)
    second = build_public_evidence_queue(reranker=second_backend, **arguments)

    assert first_backend.calls == 2
    assert second_backend.calls == 0
    assert first.queue_data == second.queue_data
    assert json.loads(first.report_data)["generated_question_count"] == 2
    assert json.loads(second.report_data)["resumed_question_count"] == 2
    assert len(tuple((tmp_path / "evidence-checkpoints").glob("*.json"))) == 2


def test_public_generation_resumes_checksum_bound_checkpoints_without_model_calls(
    tmp_path: Path,
) -> None:
    public_source, questions, aliases, index = _fixture()
    queue = build_public_evidence_queue(
        questions,
        index=index,
        aliases=aliases,
        reranker=_Reranker(),
        retrieval_run_id="public-r2-fixture-v1",
        evidence_limit=1,
        reranker_candidate_limit=2,
    )
    first_backend = _Generator()
    arguments = {
        "public_source_data": public_source,
        "evidence_queue_data": queue.queue_data,
        "system_prompt": "Use only the supplied evidence.",
        "run_id": "public-g1a512-fixture-v1",
        "generator_id": "fixture-generator-v1",
        "checkpoint_directory": tmp_path / "checkpoints",
        "maximum_input_tokens": 128,
        "maximum_new_tokens": 32,
        "frozen_inputs": {
            "index": checksum_bytes(b"index"),
            "parameter_manifest": checksum_bytes(b"parameters"),
        },
    }

    first = run_checkpointed_public_generation(backend=first_backend, **arguments)
    second_backend = _Generator(fail_if_called=True)
    second = run_checkpointed_public_generation(backend=second_backend, **arguments)

    assert first_backend.calls == 2
    assert second_backend.calls == 0
    assert first.predictions_data == second.predictions_data
    assert tuple(json.loads(first.predictions_data)) == ("q2", "q1")
    assert first.generated_question_count == 2
    assert first.resumed_question_count == 0
    assert second.generated_question_count == 0
    assert second.resumed_question_count == 2
    assert len(tuple((tmp_path / "checkpoints").glob("*.json"))) == 2


def test_public_generation_rejects_a_queue_question_that_differs_from_the_source(
    tmp_path: Path,
) -> None:
    public_source, questions, aliases, index = _fixture()
    queue = build_public_evidence_queue(
        questions,
        index=index,
        aliases=aliases,
        reranker=_Reranker(),
        retrieval_run_id="public-r2-fixture-v1",
        evidence_limit=1,
        reranker_candidate_limit=2,
    )
    rows = [json.loads(line) for line in queue.queue_data.splitlines()]
    rows[0]["question"] = "A different question"
    rows[0]["question_checksum"] = checksum_bytes(rows[0]["question"].encode())
    changed_queue = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )

    with pytest.raises(PublicDryRunError) as caught:
        run_checkpointed_public_generation(
            public_source_data=public_source,
            evidence_queue_data=changed_queue,
            backend=_Generator(),
            system_prompt="Use only the supplied evidence.",
            run_id="public-g1a512-fixture-v1",
            generator_id="fixture-generator-v1",
            checkpoint_directory=tmp_path / "checkpoints",
            maximum_input_tokens=128,
            maximum_new_tokens=32,
            frozen_inputs={"index": checksum_bytes(b"index")},
        )

    assert caught.value.code == "PUBLIC_QUESTION_SOURCE_MISMATCH"
