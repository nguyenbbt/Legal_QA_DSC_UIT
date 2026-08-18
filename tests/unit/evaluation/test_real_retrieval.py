from __future__ import annotations

import json
from dataclasses import dataclass, replace

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ContextRecord, QuestionRecord
from legal_rag.evaluation.real_retrieval import (
    build_failure_taxonomy,
    build_real_retrieval_artifacts,
    retrieve_question,
)
from legal_rag.ingestion.chunking import ChunkingConfig, ChunkRecord, chunk_context
from legal_rag.retrieval.bm25 import SparseRetrievalResult, build_bm25_index
from legal_rag.retrieval.exact import AliasIndex, load_alias_artifact
from legal_rag.retrieval.models import RetrievalDiagnostic


def _context() -> ContextRecord:
    passage = "Số 08/2022/NQ-HĐND\nĐiều 1. Nội dung thử nghiệm."
    return ContextRecord.model_validate(
        {
            "schema_version": "internal.context.v1",
            "context_id": "740",
            "original_id": "740",
            "original_id_kind": "json_integer",
            "source_position": 0,
            "source_artifact": "fixtures/context_740.json",
            "source_checksum": checksum_bytes(passage.encode()),
            "name": None,
            "source_url": "https://example.invalid/740",
            "passage": passage,
            "indexable": True,
            "quarantine_reason": None,
        }
    )


def _aliases(context: ContextRecord) -> AliasIndex:
    data = (
        '{"schema_version":"legal.reference.alias.v1","document_number":"08/2022/NQ-HĐND",'
        '"document_number_key":"08/2022/nq-hdnd","context_id":"740",'
        '"source_kind":"passage_header","canonical_start":3,"canonical_end":18,'
        '"review_state":"approved"}\n'
    ).encode()
    return load_alias_artifact(
        data,
        contexts=(context,),
        corpus_checksum=checksum_bytes(b"corpus"),
        artifact_path="aliases.active.v1.jsonl",
    )


@dataclass
class _FakeDiskIndex:
    chunks: tuple[ChunkRecord, ...]

    def __post_init__(self) -> None:
        self._bm25 = build_bm25_index(
            self.chunks,
            corpus_checksum=checksum_bytes(b"corpus"),
            alias_manifest_checksum=checksum_bytes(b"aliases"),
            runtime_compatibility_id=(
                "bm25rt.v1-cpython-3.12.7-windows-10.0.26200-x86_64-ucrt-10.0.26100.8875"
            ),
        )

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.context_id == context_id)

    def chunks_for_coordinate(
        self, hierarchy_kind: str, hierarchy_ordinal: str | None
    ) -> tuple[ChunkRecord, ...]:
        return tuple(
            chunk
            for chunk in self.chunks
            if chunk.hierarchy_kind == hierarchy_kind
            and chunk.hierarchy_ordinal == hierarchy_ordinal
        )

    def retrieve(self, query: str) -> SparseRetrievalResult:
        return self._bm25.retrieve(query)


def test_real_retrieval_unions_exact_before_bm25_without_score_mixing() -> None:
    context = _context()
    chunks = chunk_context(context, config=ChunkingConfig(minimum_fragment_tokens=1)).chunks
    question = QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q1",
            "original_id": "q1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "train.json",
            "source_checksum": checksum_bytes(b"question"),
            "question": "Nội dung Điều 1 của 08/2022/NQ-HĐND là gì?",
            "answer": "Nội dung thử nghiệm.",
            "answer_state": "gold",
        }
    )

    result = retrieve_question(question, index=_FakeDiskIndex(chunks), aliases=_aliases(context))

    assert result.candidates[0].exact_reference_match is True
    assert result.candidates[0].chunk.context_id == "740"
    assert len({candidate.chunk.chunk_id for candidate in result.candidates}) == len(
        result.candidates
    )


def test_real_retrieval_artifacts_keep_labels_blocked_and_containment_namespaced() -> None:
    context = _context()
    chunks = chunk_context(context, config=ChunkingConfig(minimum_fragment_tokens=1)).chunks
    base_question = QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q00",
            "original_id": "q00",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "train.json",
            "source_checksum": checksum_bytes(b"question"),
            "question": "Nội dung Điều 1 của 08/2022/NQ-HĐND là gì?",
            "answer": "Nội dung thử nghiệm.",
            "answer_state": "gold",
        }
    )
    index = _FakeDiskIndex(chunks)
    aliases = _aliases(context)
    results = tuple(
        retrieve_question(
            base_question.model_copy(
                update={
                    "question_id": f"q{position:02d}",
                    "original_id": f"q{position:02d}",
                    "source_position": position,
                }
            ),
            index=index,
            aliases=aliases,
        )
        for position in range(60)
    )
    selected_ids = tuple(result.question.question_id for result in results)

    artifacts = build_real_retrieval_artifacts(
        results,
        selected_question_ids=selected_ids,
        split_checksum=checksum_bytes(b"split"),
        index_checksum=checksum_bytes(b"index"),
        chunks_checksum=checksum_bytes(b"chunks"),
        alias_manifest_checksum=checksum_bytes(b"aliases"),
    )

    report = json.loads(artifacts.report)
    queue = [json.loads(line) for line in artifacts.annotation_queue.splitlines()]
    assert report["metrics_status"] == "blocked_pending_owner_approved_labels"
    assert report["containment"]["metric_namespace"] == "diagnostic_answer_containment"
    assert report["retrieval_question_count"] == 60
    assert len(queue) == 60
    assert all(row["annotation_state"] == "pending_primary_annotation" for row in queue)

    absent_is_not_an_error = replace(
        results[0],
        diagnostics=(
            RetrievalDiagnostic(
                code="EXACT_COORDINATE_ABSENT",
                message="question has no exact-reference syntax",
            ),
        ),
    )
    taxonomy = build_failure_taxonomy((absent_is_not_an_error,))
    assert taxonomy[0]["question_count"] == 0
