"""Streaming deterministic corpus chunks, manifest, and acceptance report."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from legal_rag.domain.artifacts import write_immutable_bytes, write_immutable_chunks
from legal_rag.domain.checksums import canonical_json_bytes
from legal_rag.domain.models import ContextRecord
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.ingestion.chunking import ChunkingConfig, ChunkRecord, chunk_context
from legal_rag.retrieval.tokenizer import RETRIEVAL_TOKENIZER_ID, RETRIEVAL_TOKENIZER_REVISION


class CorpusBuildError(Exception):
    """Stable failure while validating or chunking an internal corpus artifact."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class CorpusBuildSummary:
    context_count: int
    indexable_context_count: int
    quarantined_context_count: int
    chunk_count: int
    chunks_checksum: str
    manifest_checksum: str
    report_checksum: str
    markdown_report_checksum: str


@dataclass(slots=True)
class _CorpusStats:
    context_count: int = 0
    indexable_context_count: int = 0
    quarantined_context_count: int = 0
    chunk_count: int = 0
    fallback_context_count: int = 0
    warning_context_count: int = 0
    hierarchy_counts: Counter[str] = field(default_factory=Counter)
    warning_counts: Counter[str] = field(default_factory=Counter)
    token_counts: list[int] = field(default_factory=list)
    largest_chunks: list[dict[str, int | str]] = field(default_factory=list)


_CONTEXT_ARTIFACT = re.compile(r"context_[0-9]+\.json\Z", re.ASCII)
_SHA256 = re.compile(r"sha256:([0-9a-f]{64})\Z", re.ASCII)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, member in pairs:
        if key in value:
            raise CorpusBuildError(
                "CORPUS_IMPORT_MANIFEST_INVALID",
                "context import manifest contains a duplicate object key",
            )
        value[key] = member
    return value


def corpus_checksum_from_import_manifest(data: bytes) -> str:
    """Reconstruct the raw length-prefixed corpus checksum from a strict import manifest."""

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, CorpusBuildError) as error:
        if isinstance(error, CorpusBuildError):
            raise
        raise CorpusBuildError(
            "CORPUS_IMPORT_MANIFEST_INVALID", "context import manifest is malformed"
        ) from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "entries"}:
        raise CorpusBuildError(
            "CORPUS_IMPORT_MANIFEST_INVALID", "context import manifest has invalid fields"
        )
    if value["schema_version"] != "context.import.v1" or not isinstance(value["entries"], list):
        raise CorpusBuildError(
            "CORPUS_IMPORT_MANIFEST_INVALID", "context import manifest has invalid structure"
        )
    encoded_entries: list[tuple[bytes, bytes]] = []
    seen_paths: set[str] = set()
    for position, row in enumerate(value["entries"]):
        if not isinstance(row, dict):
            raise CorpusBuildError(
                "CORPUS_IMPORT_MANIFEST_INVALID", "context import manifest row is invalid"
            )
        path = row.get("source_artifact")
        checksum = row.get("source_checksum")
        if (
            set(row)
            != {
                "source_artifact",
                "context_id",
                "source_checksum",
                "indexable",
                "quarantine_reason",
                "source_position",
            }
            or not isinstance(path, str)
            or _CONTEXT_ARTIFACT.fullmatch(path) is None
            or path in seen_paths
            or row.get("source_position") != position
            or not isinstance(checksum, str)
        ):
            raise CorpusBuildError(
                "CORPUS_IMPORT_MANIFEST_INVALID", "context import manifest row is invalid"
            )
        checksum_match = _SHA256.fullmatch(checksum)
        if checksum_match is None:
            raise CorpusBuildError(
                "CORPUS_IMPORT_MANIFEST_INVALID", "context import checksum is invalid"
            )
        seen_paths.add(path)
        path_bytes = path.encode("utf-8")
        encoded_entries.append(
            (
                path_bytes,
                struct.pack(">Q", len(path_bytes))
                + path_bytes
                + bytes.fromhex(checksum_match.group(1)),
            )
        )
    encoded_entries.sort(key=lambda item: item[0])
    digest = hashlib.sha256(b"".join(entry for _, entry in encoded_entries)).hexdigest()
    return f"sha256:{digest}"


def _checksum_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _chunk_json_bytes(chunk: ChunkRecord) -> bytes:
    value = {
        "schema_version": "retrieval.chunk.v1",
        "chunk_id": chunk.chunk_id,
        "context_id": chunk.context_id,
        "source_url": chunk.source_url,
        "hierarchy_path": list(chunk.hierarchy_path),
        "hierarchy_rule_id": chunk.hierarchy_rule_id,
        "hierarchy_kind": chunk.hierarchy_kind,
        "hierarchy_ordinal": chunk.hierarchy_ordinal,
        "canonical_start": chunk.canonical_start,
        "canonical_end": chunk.canonical_end,
        "display_text": chunk.display_text,
        "retrieval_text": chunk.retrieval_text,
        "window_index": chunk.window_index,
        "chunk_checksum": chunk.chunk_checksum,
        "context_checksum": chunk.context_checksum,
    }
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _update_largest(stats: _CorpusStats, chunk: ChunkRecord, token_count: int) -> None:
    stats.largest_chunks.append(
        {
            "chunk_id": chunk.chunk_id,
            "context_id": chunk.context_id,
            "token_count": token_count,
            "canonical_start": chunk.canonical_start,
            "canonical_end": chunk.canonical_end,
        }
    )
    stats.largest_chunks.sort(
        key=lambda row: (-int(row["token_count"]), str(row["chunk_id"]).encode("utf-8"))
    )
    del stats.largest_chunks[20:]


def _record_chunks(
    context: ContextRecord,
    stats: _CorpusStats,
    config: ChunkingConfig,
) -> Iterator[bytes]:
    result = chunk_context(context, config=config)
    if result.warnings:
        stats.warning_context_count += 1
        stats.warning_counts.update(warning.code for warning in result.warnings)
    if result.chunks and all(
        chunk.hierarchy_rule_id == "HIERARCHY_FALLBACK" for chunk in result.chunks
    ):
        stats.fallback_context_count += 1
    for chunk in result.chunks:
        token_count = len(chunk.retrieval_text.split())
        stats.chunk_count += 1
        stats.hierarchy_counts[chunk.hierarchy_kind] += 1
        stats.token_counts.append(token_count)
        _update_largest(stats, chunk, token_count)
        yield _chunk_json_bytes(chunk)


def _chunk_stream(
    contexts_path: Path,
    stats: _CorpusStats,
    config: ChunkingConfig,
) -> Iterator[bytes]:
    seen_ids: set[str] = set()
    try:
        with contexts_path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    context = parse_record_json(
                        line,
                        ContextRecord,
                        artifact_path="contexts.jsonl",
                        record_identity=str(line_number),
                    )
                except RecordValidationError as error:
                    issue = error.issues[0]
                    raise CorpusBuildError("CORPUS_CONTEXT_INVALID", issue.message) from error
                expected_position = line_number - 1
                if context.source_position != expected_position:
                    raise CorpusBuildError(
                        "CORPUS_SOURCE_ORDER_INVALID",
                        "context source positions must be consecutive JSONL order",
                    )
                if context.context_id in seen_ids:
                    raise CorpusBuildError(
                        "CORPUS_CONTEXT_DUPLICATE",
                        "context JSONL contains a duplicate context ID",
                    )
                seen_ids.add(context.context_id)
                stats.context_count += 1
                if context.indexable:
                    stats.indexable_context_count += 1
                else:
                    stats.quarantined_context_count += 1
                yield from _record_chunks(context, stats, config)
    except OSError as error:
        raise CorpusBuildError(
            "CORPUS_CONTEXT_SOURCE_INVALID", "context JSONL cannot be read"
        ) from error


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    rank = max(1, (len(values) * percentile + 99) // 100)
    return sorted(values)[rank - 1]


def _manifest_bytes(
    *,
    stats: _CorpusStats,
    config: ChunkingConfig,
    corpus_checksum: str,
    context_import_manifest_checksum: str,
    context_artifact_checksum: str,
    chunks_checksum: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "corpus.chunk.manifest.v1",
            "chunking_version": config.chunking_version,
            "tokenizer_id": RETRIEVAL_TOKENIZER_ID,
            "tokenizer_revision": RETRIEVAL_TOKENIZER_REVISION,
            "unicode_version": unicodedata.unidata_version,
            "corpus_checksum": corpus_checksum,
            "context_import_manifest_checksum": context_import_manifest_checksum,
            "context_artifact_checksum": context_artifact_checksum,
            "chunks_artifact_checksum": chunks_checksum,
            "context_count": stats.context_count,
            "indexable_context_count": stats.indexable_context_count,
            "quarantined_context_count": stats.quarantined_context_count,
            "chunk_count": stats.chunk_count,
        }
    )


def _report_bytes(
    *,
    stats: _CorpusStats,
    config: ChunkingConfig,
    corpus_checksum: str,
    chunks_checksum: str,
) -> bytes:
    denominator = stats.indexable_context_count
    sorted_tokens = sorted(stats.token_counts)
    return canonical_json_bytes(
        {
            "schema_version": "corpus.acceptance.report.v1",
            "chunking_version": config.chunking_version,
            "corpus_checksum": corpus_checksum,
            "chunks_artifact_checksum": chunks_checksum,
            "context_counts": {
                "total": stats.context_count,
                "indexable": stats.indexable_context_count,
                "quarantined": stats.quarantined_context_count,
            },
            "chunk_count": stats.chunk_count,
            "hierarchy_unit_counts": [
                {"kind": kind, "count": count}
                for kind, count in sorted(
                    stats.hierarchy_counts.items(), key=lambda item: item[0].encode("utf-8")
                )
            ],
            "token_percentiles": {
                "method": "nearest_rank_integer",
                "p50": _nearest_rank(sorted_tokens, 50),
                "p95": _nearest_rank(sorted_tokens, 95),
                "p99": _nearest_rank(sorted_tokens, 99),
                "maximum": sorted_tokens[-1] if sorted_tokens else 0,
            },
            "fallback_rate": {
                "numerator": stats.fallback_context_count,
                "denominator": denominator,
            },
            "warning_rate": {
                "numerator": stats.warning_context_count,
                "denominator": denominator,
            },
            "warning_counts": [
                {"code": code, "count": count}
                for code, count in sorted(
                    stats.warning_counts.items(), key=lambda item: item[0].encode("utf-8")
                )
            ],
            "largest_chunks": stats.largest_chunks,
        }
    )


def _report_markdown(data: bytes) -> bytes:
    report = json.loads(data)
    contexts = report["context_counts"]
    percentiles = report["token_percentiles"]
    fallback = report["fallback_rate"]
    warnings = report["warning_rate"]
    lines = [
        "# MIL-004 Corpus Report",
        "",
        f"- Corpus checksum: `{report['corpus_checksum']}`",
        f"- Contexts: {contexts['total']} total; {contexts['indexable']} indexable; "
        f"{contexts['quarantined']} quarantined.",
        f"- Chunks: {report['chunk_count']}.",
        f"- Token percentiles ({percentiles['method']}): p50={percentiles['p50']}, "
        f"p95={percentiles['p95']}, p99={percentiles['p99']}, "
        f"max={percentiles['maximum']}.",
        f"- Fallback contexts: {fallback['numerator']}/{fallback['denominator']}.",
        f"- Warning contexts: {warnings['numerator']}/{warnings['denominator']}.",
        "",
        "No private passage text or absolute local path is included in this report.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_corpus_artifacts(
    *,
    contexts_path: Path,
    chunks_path: Path,
    manifest_path: Path,
    report_path: Path,
    markdown_path: Path,
    corpus_checksum: str,
    context_import_manifest_checksum: str,
    config: ChunkingConfig | None = None,
) -> CorpusBuildSummary:
    """Stream validated contexts into immutable chunk, manifest, and report artifacts."""

    active_config = config or ChunkingConfig()
    stats = _CorpusStats()
    try:
        context_artifact_checksum = _checksum_path(contexts_path)
    except OSError as error:
        raise CorpusBuildError(
            "CORPUS_CONTEXT_SOURCE_INVALID", "context JSONL cannot be read"
        ) from error
    chunks_checksum = write_immutable_chunks(
        chunks_path,
        _chunk_stream(contexts_path, stats, active_config),
    )
    manifest_checksum = write_immutable_bytes(
        manifest_path,
        _manifest_bytes(
            stats=stats,
            config=active_config,
            corpus_checksum=corpus_checksum,
            context_import_manifest_checksum=context_import_manifest_checksum,
            context_artifact_checksum=context_artifact_checksum,
            chunks_checksum=chunks_checksum,
        ),
    )
    report_data = _report_bytes(
        stats=stats,
        config=active_config,
        corpus_checksum=corpus_checksum,
        chunks_checksum=chunks_checksum,
    )
    report_checksum = write_immutable_bytes(report_path, report_data)
    markdown_report_checksum = write_immutable_bytes(markdown_path, _report_markdown(report_data))
    return CorpusBuildSummary(
        context_count=stats.context_count,
        indexable_context_count=stats.indexable_context_count,
        quarantined_context_count=stats.quarantined_context_count,
        chunk_count=stats.chunk_count,
        chunks_checksum=chunks_checksum,
        manifest_checksum=manifest_checksum,
        report_checksum=report_checksum,
        markdown_report_checksum=markdown_report_checksum,
    )
