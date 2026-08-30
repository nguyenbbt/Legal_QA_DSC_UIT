"""Run the bounded D-064 refinement and train-only D-065 supervision build."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.split import load_split_manifest_rows
from legal_rag.evaluation.taxonomy_refinement import build_taxonomy_refinement
from legal_rag.training.rag_sft import load_gold_questions
from legal_rag.training.retrieval_supervision import build_retrieval_supervision_paths

_QUESTIONS = Path("artifacts/internal/train.questions.jsonl")
_SPLIT = Path("artifacts/splits/train-dev-test.v1.json")
_CHUNKS = Path("artifacts/corpus/chunks.v1.jsonl")
_CONTEXTS = Path("artifacts/internal/contexts.jsonl")
_ALIASES = Path("artifacts/governance/aliases.active.v1.jsonl")
_HISTORICAL = Path("artifacts/training/rag-sft/v1/evidence-selection.v1.jsonl")
_EXPECTED = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "chunks": "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143",
    "contexts": "sha256:24650437b0c7ee65fecf8cb5a70028e1c5785bc5ea69cce6534227df652daaf0",
    "aliases": "sha256:1f213b99cd30fddb0954679245326d83628712fde4f5c59527f49749527118a4",
    "historical": "sha256:526906e3efceacd96535be14bb51d1e26bfc23bfef490a6c0224258e73e54011",
}


def _refinement_markdown(report: dict[str, Any], checksum: str) -> bytes:
    lines = [
        "# D-064 Taxonomy Refinement v2",
        "",
        "This versioned aggregate diagnostic separates legal-reference numbers from",
        "semantic numeric signals. It does not overwrite D-064 v1 and is not a",
        "training-label artifact.",
        "",
        f"- Train-fit rows: {report['train_fit_count']}",
        f"- Excluded non-train rows: {report['excluded_non_train_count']}",
        f"- JSON checksum: `{checksum}`",
        "- Model inference / GPU / Modal: none",
        "",
        "## Question signals",
        "",
        *(f"- {name}: {count}" for name, count in report["question_signals"].items()),
        "",
        "## Answer signals",
        "",
        *(f"- {name}: {count}" for name, count in report["answer_signals"].items()),
        "",
        "These aggregate categories remain diagnostic only and were not copied into",
        "retrieval-supervision.v2 as labels.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _completion_markdown(report: dict[str, Any], replay_passed: bool) -> bytes:
    history = report["historical_v1"]
    classes = report["mapping_class_counts"]
    lines = [
        "# D-065 Retrieval Supervision v2 Completion Report",
        "",
        "D-065 is data/supervision infrastructure only. No model inference, training,",
        "GPU, Modal, development generation, or leaderboard feedback was used.",
        "",
        "## Population and coverage",
        "",
        f"- Total train-fit rows: {report['total_train_fit_rows']}",
        f"- Immutable canonical chunks: {report['canonical_chunk_count']}",
        f"- Citation-bearing rows: {report['citation_bearing_rows']}",
        f"- Uniquely resolved groups: {report['uniquely_resolved_groups']}",
        f"- Coverage: {report['coverage_relative_to_train_fit']:.6f}",
        f"- Multi-positive groups: {report['multi_positive_groups']}",
        f"- Total positive chunk assignments: {report['total_positive_chunks']}",
        f"- Distinct positive chunks: {report['distinct_positive_chunks']}",
        "",
        "## Canonical mapping counts",
        "",
        f"- Exact document mappings: {report['exact_document_mappings']}",
        f"- Exact article mappings: {report['exact_article_mappings']}",
        f"- Exact clause mappings: {report['exact_clause_mappings']}",
        f"- Exact point mappings: {report['exact_point_mappings']}",
        f"- Document-only groups: {report['document_only_groups']}",
        f"- Ambiguous groups: {report['ambiguous_groups']}",
        f"- Unresolved groups: {report['unresolved_groups']}",
        "",
        "### Mapping classes",
        "",
        *(f"- {name}: {count}" for name, count in classes.items()),
        "",
        "## Historical-v1 comparison",
        "",
        f"- Historical mappings: {history['mapping_count']}",
        f"- Reproducibly identifiable: {history['reproducibly_identifiable']}",
        f"- Resolved by v2: {history['resolved_by_v2']}",
        f"- Exact positive-set overlap: {history['exact_positive_set_overlap']}",
        f"- Partial positive-set overlap: {history['partial_positive_set_overlap']}",
        f"- Missing positive overlap: {history['missing_positive_overlap']}",
        f"- Any-positive overlap: {history['any_positive_overlap']}",
        f"- Ambiguous under v2: {history['ambiguous_by_v2']}",
        f"- Unresolved under v2: {history['unresolved_by_v2']}",
        "- Historical mappings used for eligibility: false",
        "",
        "## Determinism and policy",
        "",
        f"- Supervision JSONL checksum: `{report['artifact_checksums']['groups']}`",
        f"- Byte-identical replay: {'PASS' if replay_passed else 'FAIL'}",
        "- Ambiguity policy: fail closed with zero positives",
        "- Reranker threshold: none",
        "- Answer-token coverage threshold: none",
        "",
        "## Recommendation",
        "",
        "Open D-066 only after owner review confirms that v2 coverage is materially",
        "broader than the historical 187 mappings and accepts the observed ambiguity/",
        "unresolved profile. D-066 was not started by this run.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=Path("artifacts/training/retrieval-supervision/v2"),
    )
    parser.add_argument(
        "--report-output-dir", type=Path, default=Path("artifacts/reports/post-d062")
    )
    args = parser.parse_args()

    questions_data = _QUESTIONS.read_bytes()
    split_data = _SPLIT.read_bytes()
    if checksum_bytes(questions_data) != _EXPECTED["questions"]:
        raise SystemExit("question checksum drift")
    if checksum_bytes(split_data) != _EXPECTED["split"]:
        raise SystemExit("split checksum drift")
    questions = load_gold_questions(questions_data)
    split_rows = load_split_manifest_rows(
        split_data,
        expected_source_checksum=_EXPECTED["questions"],
        expected_question_ids=tuple(item.question_id for item in questions),
    )
    train_ids = tuple(row.question_id for row in split_rows if row.split == "train")
    if len(train_ids) != 5_582:
        raise SystemExit("active train-fit count drift")

    refinement = build_taxonomy_refinement(
        questions_data=questions_data,
        train_question_ids=train_ids,
        expected_questions_checksum=_EXPECTED["questions"],
    )
    refinement_data = content_json_bytes(refinement)
    refinement_checksum = checksum_bytes(refinement_data)
    refinement_markdown = _refinement_markdown(refinement, refinement_checksum)
    refinement_json_checksum = write_immutable_bytes(
        args.report_output_dir / "D064-taxonomy-refinement.v2.json", refinement_data
    )
    refinement_markdown_checksum = write_immutable_bytes(
        args.report_output_dir / "D064-taxonomy-refinement.v2.md", refinement_markdown
    )
    refinement_checksums = content_json_bytes(
        {
            "schema_version": "evaluation.d064-taxonomy-refinement.checksums.v2",
            "json_checksum": refinement_json_checksum,
            "markdown_checksum": refinement_markdown_checksum,
        }
    )
    write_immutable_bytes(
        args.report_output_dir / "D064-taxonomy-refinement.checksums.v2.json",
        refinement_checksums,
    )

    build_kwargs = {
        "questions_path": _QUESTIONS,
        "train_question_ids": train_ids,
        "chunks_path": _CHUNKS,
        "contexts_path": _CONTEXTS,
        "aliases_path": _ALIASES,
        "historical_path": _HISTORICAL,
        "expected_input_checksums": {
            name: checksum for name, checksum in _EXPECTED.items() if name != "split"
        },
        "expected_train_count": 5_582,
        "expected_chunk_count": 641_118,
        "split_manifest_checksum": _EXPECTED["split"],
    }
    first = build_retrieval_supervision_paths(**build_kwargs)
    second = build_retrieval_supervision_paths(**build_kwargs)
    replay_passed = (
        first.groups_data == second.groups_data and first.report_data == second.report_data
    )
    if not replay_passed:
        raise SystemExit("D-065 replay is not byte-identical")
    history = first.report["historical_v1"]
    if history["mapping_count"] != 187 or history["reproducibly_identifiable"] != 187:
        raise SystemExit("historical 187 mappings are not reproducibly identifiable")

    groups_checksum = write_immutable_bytes(
        args.training_output_dir / "retrieval-supervision.v2.jsonl", first.groups_data
    )
    report_checksum = write_immutable_bytes(
        args.training_output_dir / "retrieval-supervision.report.v2.json", first.report_data
    )
    completion_data = _completion_markdown(first.report, replay_passed)
    completion_checksum = write_immutable_bytes(
        args.report_output_dir / "D065-completion-report.v1.md", completion_data
    )
    checksums_data = content_json_bytes(
        {
            "schema_version": "training.retrieval-supervision.checksums.v2",
            "groups_checksum": groups_checksum,
            "report_checksum": report_checksum,
            "completion_report_checksum": completion_checksum,
            "replay_byte_identical": replay_passed,
        }
    )
    checksums_checksum = write_immutable_bytes(
        args.training_output_dir / "retrieval-supervision.checksums.v2.json",
        checksums_data,
    )
    print(f"train_fit={first.report['total_train_fit_rows']}")
    print(f"resolved={first.report['uniquely_resolved_groups']}")
    print(f"ambiguous={first.report['ambiguous_groups']}")
    print(f"unresolved={first.report['unresolved_groups']}")
    print(f"groups={groups_checksum}")
    print(f"report={report_checksum}")
    print(f"completion={completion_checksum}")
    print(f"checksums={checksums_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
