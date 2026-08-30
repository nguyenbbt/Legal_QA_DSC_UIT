"""Run deterministic D-064 analysis over the authoritative official-train split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes, content_json_bytes
from legal_rag.evaluation.train_forensics import analyze_train_forensics_paths

_INPUTS = {
    "questions": Path("artifacts/internal/train.questions.jsonl"),
    "split": Path("artifacts/splits/train-dev-test.v1.json"),
    "chunks": Path("artifacts/corpus/chunks.v1.jsonl"),
    "selections": Path("artifacts/training/rag-sft/v1/evidence-selection.v1.jsonl"),
}
_EXPECTED = {
    "questions": "sha256:7c553e2252c006e23f7b57d038b45e837b82610b0853c22a279c939e4210b72f",
    "split": "sha256:9e3f7a1cd69b8e983d9c6dbd5b84043057d0ecff3044d041415d0b41232320d8",
    "chunks": "sha256:d8212020059c22f1c303197303362fa03234a3973d202679c9c5ecf6a11da143",
    "selections": "sha256:526906e3efceacd96535be14bb51d1e26bfc23bfef490a6c0224258e73e54011",
}


def _format_counts(values: dict[str, int]) -> list[str]:
    return [f"- {name}: {count}" for name, count in values.items()]


def _markdown(report: dict[str, Any], json_checksum: str) -> bytes:
    token_lengths = report["answer_token_lengths"]
    sentence_counts = report["answer_sentence_counts"]
    overlap = report["corpus_overlap"]
    mapped = report["mapped_evidence"]
    token_shape = "/".join(
        str(token_lengths[key]) for key in ("minimum", "p50", "p90", "p95", "p99", "maximum")
    )
    sentence_shape = "/".join(
        str(sentence_counts[key]) for key in ("minimum", "p50", "p90", "p95", "p99", "maximum")
    )
    lines = [
        "# D-064 Official-Train Forensic Analysis",
        "",
        "This is deterministic aggregate-only analysis. It contains no organizer row text,",
        "uses no generated text, performs no tuning, and runs no model inference.",
        "",
        "## Population",
        "",
        f"- Canonical official rows: {report['source_question_count']}",
        f"- Exact train-fit rows: {report['train_fit_count']}",
        f"- Excluded development/local-test rows: {report['excluded_non_train_count']}",
        f"- Approved mapped-evidence rows: {report['mapped_train_selection_count']}",
        f"- JSON checksum: `{json_checksum}`",
        "",
        "## Question primary types",
        "",
        *_format_counts(report["question_primary_types"]),
        "",
        "## Answer shape",
        "",
        f"- Token length min/p50/p90/p95/p99/max: {token_shape}",
        f"- Sentence count min/p50/p90/p95/p99/max: {sentence_shape}",
        "",
        "## Answer pattern counts",
        "",
        *_format_counts(report["answer_patterns"]),
        "",
        "## Extractive diagnostics",
        "",
        f"- Corpus chunks audited: {overlap['corpus_chunk_count']}",
        f"- Exact full-answer corpus matches: {overlap['exact_full_answer_count']}",
        "- Rows with at least one exact answer-sentence match: "
        f"{overlap['exact_answer_sentence_count']}",
        f"- Mapped selections: {mapped['selection_count']}",
        "",
        "### Potential answer classes",
        "",
        *_format_counts(report["potential_answer_classes"]),
        "",
        "## Governance",
        "",
        "D-065 remains closed. Exact final model registration and OQ-001 packaging",
        "remain separate submission blockers and were not resolved by this analysis.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/post-d062"))
    args = parser.parse_args()

    report = analyze_train_forensics_paths(
        questions_path=_INPUTS["questions"],
        split_path=_INPUTS["split"],
        chunks_path=_INPUTS["chunks"],
        selections_path=_INPUTS["selections"],
        expected_input_checksums=_EXPECTED,
    )
    json_data = content_json_bytes(report)
    json_checksum = checksum_bytes(json_data)
    markdown_data = _markdown(report, json_checksum)
    json_checksum = write_immutable_bytes(
        args.output_dir / "D064-official-train-forensics.v1.json", json_data
    )
    markdown_checksum = write_immutable_bytes(
        args.output_dir / "D064-official-train-forensics.v1.md", markdown_data
    )
    checksum_data = content_json_bytes(
        {
            "schema_version": "training.forensics.checksums.v1",
            "json_checksum": json_checksum,
            "markdown_checksum": markdown_checksum,
        }
    )
    checksum_checksum = write_immutable_bytes(
        args.output_dir / "D064-official-train-forensics.checksums.v1.json", checksum_data
    )
    print(f"train_fit_count={report['train_fit_count']}")
    print(f"json={json_checksum}")
    print(f"markdown={markdown_checksum}")
    print(f"checksums={checksum_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
