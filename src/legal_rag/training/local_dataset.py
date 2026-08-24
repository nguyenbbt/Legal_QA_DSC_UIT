"""Local-only orchestration for evidence mining and RAG-SFT artifact construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import canonical_json_bytes, checksum_bytes
from legal_rag.domain.models import QuestionRecord
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.models.huggingface_local import Qwen3RerankerBackend
from legal_rag.retrieval.disk_bm25 import DiskBm25Index, open_disk_bm25_index
from legal_rag.retrieval.exact import (
    AliasIndex,
    _matches_coordinate,
    document_number_key,
    load_frozen_alias_artifact,
    parse_legal_reference,
)
from legal_rag.retrieval.models import RetrievalCandidate
from legal_rag.training.evidence_mining import EvidenceMiningConfig, mine_evidence_selections
from legal_rag.training.rag_sft import (
    RagSftBuildConfig,
    build_rag_sft_dataset,
    load_gold_questions,
)

SUPPORT_INSTRUCTION = (
    "Given an official legal answer, determine whether the document directly supports its key "
    "legal claims, conditions, and numeric values. Return yes only when the support is explicit."
)


@dataclass(frozen=True, slots=True)
class LocalDatasetPaths:
    questions: Path
    split_manifest: Path
    chunks: Path
    aliases: Path
    alias_manifest: Path
    index_database: Path
    index_manifest: Path
    reranker_checkpoint: Path
    selection_output: Path
    provenance_output: Path
    material_output: Path
    manifest_output: Path
    mining_report_output: Path


@dataclass(frozen=True, slots=True)
class LocalDatasetRunConfig:
    model_id: str
    model_revision: str
    minimum_support_score: float = 0.95
    minimum_answer_token_coverage: float = 0.6
    maximum_candidates: int = 3
    maximum_evidence: int = 3
    maximum_train_questions: int | None = None
    construction_version: str = "rag-sft.v1"
    device: str = "cuda"
    batch_size: int = 8
    maximum_length: int = 4096

    def __post_init__(self) -> None:
        if self.maximum_train_questions is not None and self.maximum_train_questions < 1:
            raise ValueError("maximum train questions must be positive")


@dataclass(frozen=True, slots=True)
class LocalDatasetRunResult:
    candidate_rows: int
    accepted_rows: int
    rejected_rows: int
    selection_checksum: str
    provenance_checksum: str
    material_checksum: str
    manifest_checksum: str
    mining_report_checksum: str


def _support_policy_data(
    config: LocalDatasetRunConfig,
    *,
    index_checksum: str,
    chunks_checksum: str,
    alias_manifest_checksum: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "training.evidence.support-policy.v1",
            "query_source": "official_train_answer",
            "retrieval_source": "official-answer-exact-document-coordinate-fragments.v1",
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "instruction_checksum": checksum_bytes(SUPPORT_INSTRUCTION.encode()),
            "minimum_support_score": format(config.minimum_support_score, ".17g"),
            "minimum_answer_token_coverage": format(config.minimum_answer_token_coverage, ".17g"),
            "maximum_candidates": config.maximum_candidates,
            "maximum_evidence": config.maximum_evidence,
            "index_checksum": index_checksum,
            "chunks_checksum": chunks_checksum,
            "alias_manifest_checksum": alias_manifest_checksum,
        }
    )


def _answer_exact_candidates(
    question: QuestionRecord, *, index: DiskBm25Index, aliases: AliasIndex
) -> tuple[RetrievalCandidate, ...]:
    answer = question.answer
    if answer is None:
        return ()
    reference = parse_legal_reference(answer).reference
    if reference is None or reference.document_number is None:
        return ()
    context_ids = aliases.context_ids_for(document_number_key(reference.document_number))
    if len(context_ids) != 1:
        return ()
    matching = tuple(
        chunk
        for chunk in index.chunks_for_context(context_ids[0])
        if _matches_coordinate(chunk, reference)
    )
    if not matching or len(matching) > 50:
        return ()
    return tuple(
        RetrievalCandidate(chunk=chunk, exact_reference_match=True, sparse_score=None)
        for chunk in matching
    )


def build_local_rag_sft_dataset(
    paths: LocalDatasetPaths, config: LocalDatasetRunConfig
) -> LocalDatasetRunResult:
    """Run model-backed mining and construction without any network or remote workload."""

    question_data = paths.questions.read_bytes()
    split_data = paths.split_manifest.read_bytes()
    alias_data = paths.aliases.read_bytes()
    alias_manifest_data = paths.alias_manifest.read_bytes()
    index_manifest_data = paths.index_manifest.read_bytes()
    questions = load_gold_questions(question_data)
    split_rows = load_split_manifest_rows(
        split_data,
        expected_source_checksum=checksum_bytes(question_data),
        expected_question_ids=tuple(question.question_id for question in questions),
    )
    split_by_question = {row.question_id: row.split for row in split_rows}
    if config.maximum_train_questions is not None:
        train_ids = tuple(row.question_id for row in split_rows if row.split == "train")[
            : config.maximum_train_questions
        ]
        admitted = set(train_ids)
        questions = tuple(question for question in questions if question.question_id in admitted)

    backend = Qwen3RerankerBackend(
        paths.reranker_checkpoint,
        model_id=config.model_id,
        model_revision=config.model_revision,
        instruction=SUPPORT_INSTRUCTION,
        device=config.device,
        batch_size=config.batch_size,
        maximum_length=config.maximum_length,
    )
    with open_disk_bm25_index(
        database_path=paths.index_database,
        chunks_path=paths.chunks,
        manifest_data=index_manifest_data,
    ) as index:
        aliases = load_frozen_alias_artifact(
            alias_data,
            manifest_data=alias_manifest_data,
            corpus_checksum=index.manifest.corpus_checksum,
            artifact_path=paths.aliases.name,
        )
        mining = mine_evidence_selections(
            questions=questions,
            split_by_question=split_by_question,
            retrieve=lambda question: _answer_exact_candidates(
                question, index=index, aliases=aliases
            ),
            backend=backend,
            config=EvidenceMiningConfig(
                minimum_support_score=config.minimum_support_score,
                minimum_answer_token_coverage=config.minimum_answer_token_coverage,
                maximum_candidates=config.maximum_candidates,
                maximum_evidence=config.maximum_evidence,
            ),
        )
        support_policy_data = _support_policy_data(
            config,
            index_checksum=index.index_checksum,
            chunks_checksum=index.manifest.chunks_artifact_checksum,
            alias_manifest_checksum=checksum_bytes(alias_manifest_data),
        )
        artifacts = build_rag_sft_dataset(
            question_data=question_data,
            split_manifest_data=split_data,
            selection_data=mining.selection_data,
            chunks=index,
            config=RagSftBuildConfig(
                construction_version=config.construction_version,
                support_policy_checksum=checksum_bytes(support_policy_data),
                chunks_checksum=index.manifest.chunks_artifact_checksum,
                index_checksum=index.index_checksum,
                minimum_support_score=config.minimum_support_score,
                minimum_answer_token_coverage=config.minimum_answer_token_coverage,
            ),
        )

    report_data = canonical_json_bytes(
        {
            "schema_version": "training.evidence.mining-report.v1",
            "candidate_rows": mining.report.candidate_rows,
            "accepted_rows": mining.report.accepted_rows,
            "rejected_rows": mining.report.rejected_rows,
            "rejected_by_reason": [
                {"reason": reason, "count": count}
                for reason, count in mining.report.rejected_by_reason
            ],
            "support_policy_checksum": checksum_bytes(support_policy_data),
            "support_policy": support_policy_data.decode().strip(),
            "model_id": mining.report.model_id,
            "model_revision": mining.report.model_revision,
            "execution_mode": "local-offline",
            "contains_generated_text": False,
        }
    )
    checksums = {
        "selection": write_immutable_bytes(paths.selection_output, mining.selection_data),
        "provenance": write_immutable_bytes(paths.provenance_output, artifacts.provenance_data),
        "material": write_immutable_bytes(paths.material_output, artifacts.material_data),
        "manifest": write_immutable_bytes(paths.manifest_output, artifacts.manifest_data),
        "report": write_immutable_bytes(paths.mining_report_output, report_data),
    }
    return LocalDatasetRunResult(
        candidate_rows=mining.report.candidate_rows,
        accepted_rows=mining.report.accepted_rows,
        rejected_rows=mining.report.rejected_rows,
        selection_checksum=checksums["selection"],
        provenance_checksum=checksums["provenance"],
        material_checksum=checksums["material"],
        manifest_checksum=checksums["manifest"],
        mining_report_checksum=checksums["report"],
    )


def result_dict(result: LocalDatasetRunResult) -> dict[str, object]:
    """Return a JSON-safe deterministic summary for CLI/notebook callers."""

    return asdict(result)


__all__ = [
    "LocalDatasetPaths",
    "LocalDatasetRunConfig",
    "LocalDatasetRunResult",
    "SUPPORT_INSTRUCTION",
    "build_local_rag_sft_dataset",
    "result_dict",
]
