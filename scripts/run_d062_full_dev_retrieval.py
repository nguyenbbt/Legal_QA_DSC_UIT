"""Freeze D-062 R0 or base-reranker evidence for all development rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, checksum_file, content_json_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.domain.validation import parse_record_json
from legal_rag.evaluation.legal_reranker_contract import LEGAL_EVIDENCE_INSTRUCTION
from legal_rag.evaluation.public_dry_run import (
    PublicEvidenceItem,
    PublicEvidenceRow,
    load_public_evidence_queue,
)
from legal_rag.evaluation.split import load_split_questions_jsonl
from legal_rag.ingestion.chunking import ChunkRecord
from legal_rag.models.huggingface_local import Qwen3RerankerBackend
from legal_rag.retrieval.disk_bm25 import DiskBm25Index, open_disk_bm25_index
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.retrieval.reranker import RerankerBackend, rerank_candidates

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
FROZEN_R0_RETRIEVAL_CHECKSUM = (
    "sha256:d42ff25966e745a8e9eb47a87f1f0477cb8850fe80fcf873cb691cc51f04a54b"
)
QUESTION_COUNT = 716
CANDIDATE_LIMIT = 12
EVIDENCE_LIMIT = 3


class _ReplayReranker:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION

    def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        raise AssertionError("D-062 replay must not invoke the reranker")


def load_gold_development_questions(data: bytes) -> tuple[QuestionRecord, ...]:
    """Validate split framing and retain complete gold records in canonical order."""

    ordered = load_split_questions_jsonl(data, expected_answer_state="gold")
    parsed = tuple(
        parse_record_json(
            line,
            QuestionRecord,
            artifact_path="development.questions.jsonl",
            record_identity=str(line_number),
        )
        for line_number, line in enumerate(data.splitlines(keepends=True), start=1)
    )
    by_id = {question.question_id: question for question in parsed}
    if len(by_id) != len(parsed):
        raise ValueError("D-062 development question IDs must be unique")
    return tuple(by_id[question.question_id] for question in ordered)


def rank_frozen_candidates(
    query: str,
    retrieval_row: Mapping[str, object],
    chunks_by_id: Mapping[str, ChunkRecord],
    *,
    reranker: RerankerBackend | None,
    limit: int,
) -> tuple[RetrievalCandidate, ...]:
    """Select or rerank only the persisted MIL-004 candidate universe."""

    raw_candidates = retrieval_row.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("D-062 frozen retrieval row has no candidates")
    candidates: list[RetrievalCandidate] = []
    for raw in raw_candidates[:CANDIDATE_LIMIT]:
        if not isinstance(raw, dict):
            raise ValueError("D-062 frozen retrieval candidate is invalid")
        evidence_id = raw.get("evidence_id")
        exact = raw.get("exact_reference_match")
        sparse_score = raw.get("sparse_score")
        if (
            not isinstance(evidence_id, str)
            or evidence_id not in chunks_by_id
            or not isinstance(exact, bool)
            or (
                sparse_score is not None
                and (not isinstance(sparse_score, (int, float)) or isinstance(sparse_score, bool))
            )
        ):
            raise ValueError("D-062 frozen retrieval candidate is invalid")
        candidates.append(
            RetrievalCandidate(
                chunk=chunks_by_id[evidence_id],
                exact_reference_match=exact,
                sparse_score=float(sparse_score) if sparse_score is not None else None,
            )
        )
    admitted = tuple(candidates)
    if reranker is None:
        return admitted[:limit]
    return rerank_candidates(
        query,
        admitted,
        reranker,
        limit=limit,
        maximum_candidate_count=CANDIDATE_LIMIT,
    )


def build_development_generation_inputs(
    questions: Sequence[QuestionRecord], evidence_queue_data: bytes
) -> tuple[bytes, bytes]:
    """Bind frozen evidence to immutable local gold answers and retrieval IDs."""

    ordered = tuple(questions)
    evidence = load_public_evidence_queue(evidence_queue_data)
    if len(ordered) != len(evidence) or tuple(q.question_id for q in ordered) != tuple(
        row.question_id for row in evidence
    ):
        raise ValueError("development questions and evidence differ")
    queue_rows: list[bytes] = []
    retrieval_rows: list[bytes] = []
    for question, row in zip(ordered, evidence, strict=True):
        if question.answer_state != "gold" or question.answer is None:
            raise ValueError("D-062 requires immutable gold development answers")
        candidates = [
            {"evidence_id": item.evidence_id, "display_text": item.display_text}
            for item in row.evidence
        ]
        queue_rows.append(
            content_json_bytes(
                {
                    "schema_version": "d062.generation-input.v1",
                    "question_id": question.question_id,
                    "question_checksum": checksum_bytes(question.question.encode("utf-8")),
                    "question": question.question,
                    "gold_answer": question.answer,
                    "candidates": candidates,
                }
            )
        )
        retrieval_rows.append(
            content_json_bytes(
                {
                    "schema_version": "d062.retrieval-output.v1",
                    "question_id": question.question_id,
                    "candidates": [{"evidence_id": item.evidence_id} for item in row.evidence],
                }
            )
        )
    return b"".join(queue_rows), b"".join(retrieval_rows)


def _checkpoint_path(directory: Path, question_id: str) -> Path:
    return directory / f"{hashlib.sha256(question_id.encode('utf-8')).hexdigest()}.json"


def _freeze_evidence(
    *,
    questions: Sequence[QuestionRecord],
    retrieval_rows: Sequence[Mapping[str, object]],
    index: DiskBm25Index,
    reranker: RerankerBackend | None,
    role: str,
    run_id: str,
    fingerprint: str,
    checkpoint_directory: Path,
) -> tuple[bytes, bytes]:
    output: list[PublicEvidenceRow] = []
    generated_count = 0
    resumed_count = 0
    for question, retrieval_row in zip(questions, retrieval_rows, strict=True):
        if retrieval_row.get("question_id") != question.question_id:
            raise ValueError("D-062 frozen retrieval order differs from development")
        checkpoint = _checkpoint_path(checkpoint_directory, question.question_id)
        if checkpoint.exists():
            loaded = load_public_evidence_queue(checkpoint.read_bytes())
            if len(loaded) != 1:
                raise ValueError("D-062 retrieval checkpoint cardinality is invalid")
            row = loaded[0]
            if (
                row.retrieval_run_id != run_id
                or row.retrieval_fingerprint != fingerprint
                or row.question_id != question.question_id
                or row.question != question.question
            ):
                raise ValueError("D-062 retrieval checkpoint differs from frozen inputs")
            output.append(row)
            resumed_count += 1
            continue
        raw_candidates = retrieval_row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("D-062 frozen retrieval candidate list is invalid")
        chunk_ids = tuple(
            candidate["evidence_id"] for candidate in raw_candidates[:CANDIDATE_LIMIT]
        )
        chunks = index.chunks_by_ids(chunk_ids)
        ranked = rank_frozen_candidates(
            question.question,
            retrieval_row,
            {chunk.chunk_id: chunk for chunk in chunks},
            reranker=reranker,
            limit=EVIDENCE_LIMIT,
        )
        evidence = tuple(
            PublicEvidenceItem(
                evidence_id=candidate.chunk.chunk_id,
                context_id=candidate.chunk.context_id,
                hierarchy_path=candidate.chunk.hierarchy_path,
                canonical_start=candidate.chunk.canonical_start,
                canonical_end=candidate.chunk.canonical_end,
                display_text=candidate.chunk.display_text,
                chunk_checksum=candidate.chunk.chunk_checksum,
                exact_reference_match=candidate.exact_reference_match,
                sparse_score=candidate.sparse_score,
                reranker_score=candidate.reranker_score,
                rank=rank,
            )
            for rank, candidate in enumerate(ranked, start=1)
        )
        row = PublicEvidenceRow(
            schema_version="public.evidence.v1",
            retrieval_run_id=run_id,
            retrieval_fingerprint=fingerprint,
            question_id=question.question_id,
            question_checksum=checksum_bytes(question.question.encode("utf-8")),
            question=question.question,
            evidence=evidence,
        )
        write_immutable_bytes(checkpoint, content_json_bytes(row.model_dump(mode="json")))
        output.append(row)
        generated_count += 1
    queue_data = b"".join(content_json_bytes(row.model_dump(mode="json")) for row in output)
    report_data = content_json_bytes(
        {
            "schema_version": "d062.frozen-retrieval.report.v1",
            "role": role,
            "run_id": run_id,
            "question_count": len(output),
            "candidate_limit": CANDIDATE_LIMIT,
            "evidence_limit": EVIDENCE_LIMIT,
            "generated_question_count": generated_count,
            "resumed_question_count": resumed_count,
            "queue_checksum": checksum_bytes(queue_data),
        }
    )
    return queue_data, report_data


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", choices=("r0", "base-reranker"), required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = Path(__file__).resolve().parents[1]
    questions_path = root / "artifacts/evaluations/mil-003/baseline/development/questions.jsonl"
    source_retrieval_path = root / "artifacts/evaluations/mil-004/retrieval.v1.jsonl"
    chunks_path = root / "artifacts/corpus/chunks.v1.jsonl"
    index_path = root / "artifacts/indices/bm25.v1.active.sqlite3"
    index_manifest_path = root / "artifacts/manifests/bm25.index.active.v1.json"
    model_checkpoint = root / ".local/models/qwen3-reranker-0.6b" / MODEL_REVISION
    output = root / "artifacts/evaluations/recovery/R-009/D062-full-development-v1" / arguments.role
    checkpoint_directory = (
        root / ".local/runs/d062-full-development-v1/frozen-retrieval" / arguments.role
    )
    question_data = questions_path.read_bytes()
    questions = load_gold_development_questions(question_data)
    source_retrieval_data = source_retrieval_path.read_bytes()
    if checksum_bytes(source_retrieval_data) != FROZEN_R0_RETRIEVAL_CHECKSUM:
        raise RuntimeError("D-062 source retrieval differs from frozen MIL-004 R0")
    source_rows = tuple(json.loads(line) for line in source_retrieval_data.splitlines())
    if len(questions) != QUESTION_COUNT or len(source_rows) != QUESTION_COUNT:
        raise RuntimeError("D-062 development split must contain exactly 716 rows")
    index_manifest_data = index_manifest_path.read_bytes()
    frozen_inputs = {
        "bm25_database": checksum_file(index_path),
        "bm25_manifest": checksum_bytes(index_manifest_data),
        "development_questions": checksum_bytes(question_data),
        "frozen_r0_retrieval": checksum_bytes(source_retrieval_data),
    }
    reranker: RerankerBackend | None = None
    if arguments.role == "base-reranker":
        reranker = Qwen3RerankerBackend(
            model_checkpoint,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            instruction=LEGAL_EVIDENCE_INSTRUCTION,
            device="cuda",
            batch_size=2,
            maximum_length=1536,
        )
        frozen_inputs["reranker_checkpoint"] = checksum_file(model_checkpoint / "model.safetensors")
    run_id = f"D062-{arguments.role}-development-716-v1"
    fingerprint = checksum_bytes(
        content_json_bytes(
            {
                "schema_version": "d062.frozen-retrieval.fingerprint.v1",
                "role": arguments.role,
                "run_id": run_id,
                "candidate_limit": CANDIDATE_LIMIT,
                "evidence_limit": EVIDENCE_LIMIT,
                "reranker_model_id": MODEL_ID if reranker is not None else None,
                "reranker_model_revision": MODEL_REVISION if reranker is not None else None,
                "frozen_inputs": frozen_inputs,
            }
        )
    )
    started = time.perf_counter()
    with open_disk_bm25_index(
        database_path=index_path,
        chunks_path=chunks_path,
        manifest_data=index_manifest_data,
    ) as index:
        queue_data, report_data = _freeze_evidence(
            questions=questions,
            retrieval_rows=source_rows,
            index=index,
            reranker=reranker,
            role=arguments.role,
            run_id=run_id,
            fingerprint=fingerprint,
            checkpoint_directory=checkpoint_directory,
        )
        replay_queue, replay_report = _freeze_evidence(
            questions=questions,
            retrieval_rows=source_rows,
            index=index,
            reranker=_ReplayReranker() if reranker is not None else None,
            role=arguments.role,
            run_id=run_id,
            fingerprint=fingerprint,
            checkpoint_directory=checkpoint_directory,
        )
    if queue_data != replay_queue:
        raise RuntimeError("D-062 retrieval replay differs")
    generation_queue, retrieval_data = build_development_generation_inputs(questions, queue_data)
    manifest_data = content_json_bytes(
        {
            "schema_version": "d062.retrieval.manifest.v1",
            "role": arguments.role,
            "run_id": run_id,
            "question_count": len(questions),
            "candidate_limit": CANDIDATE_LIMIT,
            "frozen_r0_retrieval_checksum": FROZEN_R0_RETRIEVAL_CHECKSUM,
            "evidence_checksum": checksum_bytes(queue_data),
            "generation_input_checksum": checksum_bytes(generation_queue),
            "retrieval_output_checksum": checksum_bytes(retrieval_data),
            "byte_identical_replay": True,
            "elapsed_seconds": time.perf_counter() - started,
            "execution_mode": "local-offline",
            "frozen_inputs": frozen_inputs,
        }
    )
    outputs = {
        "evidence.v1.jsonl": queue_data,
        "evidence.report.v1.json": report_data,
        "evidence.replay-report.v1.json": replay_report,
        "annotation-queue.v1.jsonl": generation_queue,
        "retrieval.v1.jsonl": retrieval_data,
        "manifest.v1.json": manifest_data,
    }
    checksums = {name: write_immutable_bytes(output / name, data) for name, data in outputs.items()}
    print(json.dumps({"role": arguments.role, "checksums": checksums}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
