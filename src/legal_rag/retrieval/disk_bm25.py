"""Deterministic disk-backed bm25.v1 index for the real MIL-004 corpus."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import tempfile
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, NoReturn

from legal_rag.domain.artifacts import write_immutable_bytes
from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import (
    AbsoluteHttpUrl,
    CanonicalIntegerString,
    FiniteScore,
    FrozenStrictModel,
    NfcString,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    SafeRelativePath,
    Sha256,
)
from legal_rag.domain.validation import RecordValidationError, parse_record_json
from legal_rag.ingestion.chunking import ChunkHierarchyKind, ChunkRecord
from legal_rag.retrieval.bm25 import (
    APPROVED_BM25_RUNTIME_ID,
    BM25_B,
    BM25_K1,
    BM25_VERSION,
    SparseRetrievalResult,
    ordered_unique_query_terms,
)
from legal_rag.retrieval.exact import DOCUMENT_KEY_VERSION, REFERENCE_PARSER_VERSION
from legal_rag.retrieval.models import RetrievalCandidate, RetrievalDiagnostic
from legal_rag.retrieval.tokenizer import (
    RETRIEVAL_TOKENIZER_ID,
    RETRIEVAL_TOKENIZER_REVISION,
    retrieval_token_values,
)

DISK_INDEX_SCHEMA_VERSION = "bm25.disk-index.v1"


class DiskBm25Error(Exception):
    """Stable failure at the serialized sparse-index boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise DiskBm25Error(code, message)


class _StoredChunk(FrozenStrictModel, frozen=True):
    schema_version: Literal["retrieval.chunk.v1"]
    chunk_id: NonEmptyString
    context_id: CanonicalIntegerString
    source_url: AbsoluteHttpUrl
    hierarchy_path: tuple[NonEmptyString, ...]
    hierarchy_rule_id: NonEmptyString
    hierarchy_kind: ChunkHierarchyKind
    hierarchy_ordinal: NfcString | None
    canonical_start: NonNegativeInt
    canonical_end: PositiveInt
    display_text: NonEmptyString
    retrieval_text: NonEmptyString
    window_index: NonNegativeInt
    chunk_checksum: Sha256
    context_checksum: Sha256

    def to_chunk(self) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=self.chunk_id,
            context_id=self.context_id,
            source_url=self.source_url,
            hierarchy_path=self.hierarchy_path,
            hierarchy_rule_id=self.hierarchy_rule_id,
            hierarchy_kind=self.hierarchy_kind,
            hierarchy_ordinal=self.hierarchy_ordinal,
            canonical_start=self.canonical_start,
            canonical_end=self.canonical_end,
            display_text=self.display_text,
            retrieval_text=self.retrieval_text,
            window_index=self.window_index,
            chunk_checksum=self.chunk_checksum,
            context_checksum=self.context_checksum,
        )


class _IndexFile(FrozenStrictModel, frozen=True):
    path: SafeRelativePath
    checksum: Sha256


class DiskBm25Manifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["bm25.disk-index.manifest.v1"]
    retrieval_version: Literal["bm25.v1"]
    k1: FiniteScore
    b: FiniteScore
    tokenizer_id: Literal["legal-retrieval-unicode-v1"]
    tokenizer_revision: NonEmptyString
    corpus_checksum: Sha256
    chunking_version: Literal["chunking.v1"]
    chunks_artifact_checksum: Sha256
    document_count: NonNegativeInt
    total_document_length: NonNegativeInt
    average_length: float
    legal_reference_parser: Literal["legal-reference-parser.v1"]
    document_key_version: Literal["legal-document-number-key.v1"]
    unicode_version: NonEmptyString
    alias_manifest_checksum: Sha256
    runtime_compatibility_id: NonEmptyString
    sqlite_version: NonEmptyString
    ordered_files: tuple[_IndexFile, ...]


class _CorpusChunkManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["corpus.chunk.manifest.v1"]
    chunking_version: Literal["chunking.v1"]
    tokenizer_id: Literal["legal-retrieval-unicode-v1"]
    tokenizer_revision: NonEmptyString
    unicode_version: NonEmptyString
    corpus_checksum: Sha256
    context_import_manifest_checksum: Sha256
    context_artifact_checksum: Sha256
    chunks_artifact_checksum: Sha256
    context_count: NonNegativeInt
    indexable_context_count: NonNegativeInt
    quarantined_context_count: NonNegativeInt
    chunk_count: NonNegativeInt


class _AliasManifest(FrozenStrictModel, frozen=True):
    schema_version: Literal["legal.reference.alias.manifest.v1"]
    document_key_version: Literal["legal-document-number-key.v1"]
    unicode_version: NonEmptyString
    corpus_checksum: Sha256
    ordered_files: tuple[_IndexFile, ...]
    record_count: NonNegativeInt
    aggregate_checksum: Sha256


@dataclass(frozen=True, slots=True)
class DiskBm25BuildSummary:
    document_count: int
    total_document_length: int
    database_checksum: str
    manifest_checksum: str
    index_checksum: str


def _checksum_path(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise DiskBm25Error("SPARSE_INDEX_SOURCE_INVALID", "index source cannot be read") from error
    return f"sha256:{digest.hexdigest()}"


def _encoded_term(term: str) -> str:
    return "x" + term.encode("utf-8").hex()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _parse_chunk(line: bytes, line_number: int) -> ChunkRecord:
    try:
        return parse_record_json(
            line,
            _StoredChunk,
            artifact_path="chunks.v1.jsonl",
            record_identity=str(line_number),
        ).to_chunk()
    except RecordValidationError as error:
        message = error.issues[0].message if error.issues else "chunk schema is invalid"
        raise DiskBm25Error("SPARSE_CHUNK_INVALID", message) from error


def _configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA page_size=4096")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA auto_vacuum=NONE")
    connection.execute("PRAGMA secure_delete=OFF")
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            context_id TEXT NOT NULL,
            hierarchy_kind TEXT NOT NULL,
            hierarchy_ordinal TEXT,
            chunk_checksum TEXT NOT NULL,
            document_length INTEGER NOT NULL,
            source_offset INTEGER NOT NULL,
            source_length INTEGER NOT NULL
        );
        CREATE INDEX documents_context_id ON documents(context_id, doc_id);
        CREATE INDEX documents_coordinate
            ON documents(hierarchy_kind, hierarchy_ordinal, doc_id);
        CREATE VIRTUAL TABLE chunk_terms USING fts5(
            term_text,
            content='',
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE VIRTUAL TABLE chunk_vocab USING fts5vocab(chunk_terms, 'instance');
        """
    )


def _populate_database(connection: sqlite3.Connection, chunks_path: Path) -> tuple[int, int]:
    document_count = 0
    total_length = 0
    seen_chunk_ids: set[str] = set()
    try:
        with chunks_path.open("rb") as stream:
            while line := stream.readline():
                offset = stream.tell() - len(line)
                chunk = _parse_chunk(line, document_count + 1)
                if chunk.chunk_id in seen_chunk_ids:
                    _fail("SPARSE_CHUNK_DUPLICATE", "BM25 input contains a duplicate chunk ID")
                seen_chunk_ids.add(chunk.chunk_id)
                terms = retrieval_token_values(
                    unicodedata.normalize("NFC", chunk.retrieval_text).casefold()
                )
                document_count += 1
                document_length = len(terms)
                total_length += document_length
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_count,
                        chunk.chunk_id,
                        chunk.context_id,
                        chunk.hierarchy_kind,
                        chunk.hierarchy_ordinal,
                        chunk.chunk_checksum,
                        document_length,
                        offset,
                        len(line),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_terms(rowid, term_text) VALUES (?, ?)",
                    (document_count, " ".join(_encoded_term(term) for term in terms)),
                )
    except OSError as error:
        raise DiskBm25Error(
            "SPARSE_CHUNK_SOURCE_INVALID", "chunk artifact cannot be read"
        ) from error
    if document_count and total_length == 0:
        _fail(
            "SPARSE_INDEX_ZERO_AVGDL",
            "non-empty BM25 index cannot have zero average document length",
        )
    return document_count, total_length


def _finalize_database(
    connection: sqlite3.Connection,
    *,
    document_count: int,
    total_length: int,
    chunks_checksum: str,
    corpus_checksum: str,
    alias_manifest_checksum: str,
    runtime_compatibility_id: str,
) -> None:
    metadata = {
        "schema_version": DISK_INDEX_SCHEMA_VERSION,
        "retrieval_version": BM25_VERSION,
        "chunks_artifact_checksum": chunks_checksum,
        "corpus_checksum": corpus_checksum,
        "alias_manifest_checksum": alias_manifest_checksum,
        "runtime_compatibility_id": runtime_compatibility_id,
        "document_count": str(document_count),
        "total_document_length": str(total_length),
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)", sorted(metadata.items())
    )
    connection.execute("INSERT INTO chunk_terms(chunk_terms) VALUES ('optimize')")
    connection.commit()
    connection.execute("VACUUM")


def _publish_database(temporary_path: Path, database_path: Path) -> str:
    database_checksum = _checksum_path(temporary_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        if _checksum_path(database_path) != database_checksum:
            _fail("ARTIFACT_IMMUTABLE", "an existing sparse index cannot be replaced")
        return database_checksum
    try:
        os.link(temporary_path, database_path)
    except OSError as error:
        raise DiskBm25Error("ARTIFACT_WRITE_FAILED", "sparse index cannot be published") from error
    return database_checksum


def build_disk_bm25_index(
    *,
    chunks_path: Path,
    chunks_checksum: str,
    corpus_checksum: str,
    alias_manifest_checksum: str,
    database_path: Path,
    manifest_path: Path,
    runtime_compatibility_id: str,
    expected_document_count: int | None = None,
) -> DiskBm25BuildSummary:
    """Build a deterministic contentless FTS posting store and exact BM25 manifest."""

    if runtime_compatibility_id != APPROVED_BM25_RUNTIME_ID:
        _fail("BM25_RUNTIME_UNAPPROVED", "BM25 runtime compatibility ID is not owner-approved")
    if _checksum_path(chunks_path) != chunks_checksum:
        _fail("SPARSE_CHUNK_CHECKSUM_MISMATCH", "chunk artifact checksum does not match")
    temporary_path: Path | None = None
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=database_path.parent,
            prefix=f".{database_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        connection = sqlite3.connect(temporary_path)
        try:
            _configure_database(connection)
            document_count, total_length = _populate_database(connection, chunks_path)
            if expected_document_count is not None and document_count != expected_document_count:
                _fail(
                    "SPARSE_DOCUMENT_COUNT_MISMATCH",
                    "chunk count differs from the approved corpus manifest",
                )
            _finalize_database(
                connection,
                document_count=document_count,
                total_length=total_length,
                chunks_checksum=chunks_checksum,
                corpus_checksum=corpus_checksum,
                alias_manifest_checksum=alias_manifest_checksum,
                runtime_compatibility_id=runtime_compatibility_id,
            )
        finally:
            connection.close()
        database_checksum = _publish_database(temporary_path, database_path)
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    average_length = float(total_length) / float(document_count) if document_count else 0.0
    manifest_data = _json_bytes(
        {
            "schema_version": "bm25.disk-index.manifest.v1",
            "retrieval_version": BM25_VERSION,
            "k1": BM25_K1,
            "b": BM25_B,
            "tokenizer_id": RETRIEVAL_TOKENIZER_ID,
            "tokenizer_revision": RETRIEVAL_TOKENIZER_REVISION,
            "corpus_checksum": corpus_checksum,
            "chunking_version": "chunking.v1",
            "chunks_artifact_checksum": chunks_checksum,
            "document_count": document_count,
            "total_document_length": total_length,
            "average_length": average_length,
            "legal_reference_parser": REFERENCE_PARSER_VERSION,
            "document_key_version": DOCUMENT_KEY_VERSION,
            "unicode_version": unicodedata.unidata_version,
            "alias_manifest_checksum": alias_manifest_checksum,
            "runtime_compatibility_id": runtime_compatibility_id,
            "sqlite_version": sqlite3.sqlite_version,
            "ordered_files": [{"path": database_path.name, "checksum": database_checksum}],
        }
    )
    manifest_checksum = write_immutable_bytes(manifest_path, manifest_data)
    return DiskBm25BuildSummary(
        document_count=document_count,
        total_document_length=total_length,
        database_checksum=database_checksum,
        manifest_checksum=manifest_checksum,
        index_checksum=checksum_bytes(manifest_data),
    )


def _parse_manifest[ManifestT: FrozenStrictModel](
    data: bytes,
    model: type[ManifestT],
    *,
    artifact_path: str,
) -> ManifestT:
    try:
        return parse_record_json(
            data,
            model,
            artifact_path=artifact_path,
            record_identity="manifest",
        )
    except RecordValidationError as error:
        message = error.issues[0].message if error.issues else "manifest is invalid"
        raise DiskBm25Error("SPARSE_INDEX_MANIFEST_INVALID", message) from error


def build_disk_bm25_from_manifests(
    *,
    chunks_path: Path,
    chunk_manifest_data: bytes,
    aliases_path: Path,
    alias_manifest_data: bytes,
    database_path: Path,
    manifest_path: Path,
    runtime_compatibility_id: str,
) -> DiskBm25BuildSummary:
    """Build only from checksum-linked active corpus and alias manifests."""

    chunk_manifest = _parse_manifest(
        chunk_manifest_data,
        _CorpusChunkManifest,
        artifact_path="corpus.chunks.v1.json",
    )
    alias_manifest = _parse_manifest(
        alias_manifest_data,
        _AliasManifest,
        artifact_path="aliases.active.v1.json",
    )
    if chunk_manifest.corpus_checksum != alias_manifest.corpus_checksum:
        _fail("ALIAS_CORPUS_MISMATCH", "alias and chunk manifests name different corpora")
    if (
        chunk_manifest.tokenizer_revision != RETRIEVAL_TOKENIZER_REVISION
        or chunk_manifest.unicode_version != unicodedata.unidata_version
        or alias_manifest.unicode_version != unicodedata.unidata_version
    ):
        _fail("SPARSE_INDEX_MANIFEST_INVALID", "manifest runtime identities are incompatible")
    if (
        len(alias_manifest.ordered_files) != 1
        or alias_manifest.ordered_files[0].path != aliases_path.name
        or _checksum_path(aliases_path) != alias_manifest.ordered_files[0].checksum
        or alias_manifest.aggregate_checksum != alias_manifest.ordered_files[0].checksum
    ):
        _fail("ALIAS_MANIFEST_MISMATCH", "active alias file differs from its manifest")
    return build_disk_bm25_index(
        chunks_path=chunks_path,
        chunks_checksum=chunk_manifest.chunks_artifact_checksum,
        corpus_checksum=chunk_manifest.corpus_checksum,
        alias_manifest_checksum=checksum_bytes(alias_manifest_data),
        database_path=database_path,
        manifest_path=manifest_path,
        runtime_compatibility_id=runtime_compatibility_id,
        expected_document_count=chunk_manifest.chunk_count,
    )


class DiskBm25Index:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        chunks_path: Path,
        manifest: DiskBm25Manifest,
        manifest_data: bytes,
    ) -> None:
        self._connection = connection
        self._chunks_path = chunks_path
        self.manifest = manifest
        self.document_count = manifest.document_count
        self.average_length = manifest.average_length
        self.index_checksum = checksum_bytes(manifest_data)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DiskBm25Index:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _load_chunks(self, doc_ids: tuple[int, ...]) -> tuple[ChunkRecord, ...]:
        if not doc_ids:
            return ()
        chunks: list[ChunkRecord] = []
        try:
            with self._chunks_path.open("rb") as stream:
                for doc_id in doc_ids:
                    row = self._connection.execute(
                        """
                        SELECT chunk_checksum, source_offset, source_length
                        FROM documents WHERE doc_id=?
                        """,
                        (doc_id,),
                    ).fetchone()
                    if row is None:
                        _fail(
                            "SPARSE_DOCUMENT_MISSING",
                            "ranked document is absent from the index",
                        )
                    expected_checksum, offset, length = row
                    stream.seek(int(offset))
                    chunk = _parse_chunk(stream.read(int(length)), doc_id)
                    if chunk.chunk_checksum != expected_checksum:
                        _fail(
                            "SPARSE_DOCUMENT_CHECKSUM_MISMATCH",
                            "ranked chunk differs from the index",
                        )
                    chunks.append(chunk)
        except OSError as error:
            raise DiskBm25Error(
                "SPARSE_CHUNK_SOURCE_INVALID", "chunk artifact cannot be read"
            ) from error
        return tuple(chunks)

    def chunks_for_context(self, context_id: str) -> tuple[ChunkRecord, ...]:
        """Load only active chunks belonging to one canonical context."""

        doc_ids = tuple(
            int(row[0])
            for row in self._connection.execute(
                "SELECT doc_id FROM documents WHERE context_id=? ORDER BY doc_id",
                (context_id,),
            )
        )
        return self._load_chunks(doc_ids)

    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]:
        """Resolve an explicit bounded ID list in caller order."""

        if len(chunk_ids) != len(set(chunk_ids)):
            _fail("SPARSE_CHUNK_ID_DUPLICATE", "requested chunk IDs must be unique")
        doc_ids: list[int] = []
        for chunk_id in chunk_ids:
            row = self._connection.execute(
                "SELECT doc_id FROM documents WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                _fail("SPARSE_DOCUMENT_MISSING", "requested chunk is absent from the index")
            doc_ids.append(int(row[0]))
        return self._load_chunks(tuple(doc_ids))

    def chunks_for_coordinate(
        self,
        hierarchy_kind: str,
        hierarchy_ordinal: str | None,
    ) -> tuple[ChunkRecord, ...]:
        """Load the narrowed corpus candidates for one hierarchy leaf coordinate."""

        doc_ids = tuple(
            int(row[0])
            for row in self._connection.execute(
                """
                SELECT doc_id FROM documents
                WHERE hierarchy_kind=? AND hierarchy_ordinal IS ?
                ORDER BY doc_id
                """,
                (hierarchy_kind, hierarchy_ordinal),
            )
        )
        return self._load_chunks(doc_ids)

    def retrieve(self, query: str) -> SparseRetrievalResult:
        canonical_query = unicodedata.normalize("NFC", query)
        if self.document_count == 0:
            return SparseRetrievalResult(
                query=canonical_query,
                query_terms=(),
                candidates=(),
                diagnostics=(
                    RetrievalDiagnostic("SPARSE_INDEX_EMPTY", "BM25 index has no documents"),
                ),
                index_checksum=self.index_checksum,
            )
        query_terms = ordered_unique_query_terms(canonical_query)
        if not query_terms:
            return SparseRetrievalResult(
                query=canonical_query,
                query_terms=(),
                candidates=(),
                diagnostics=(
                    RetrievalDiagnostic("SPARSE_QUERY_EMPTY", "query contains no retrieval tokens"),
                ),
                index_checksum=self.index_checksum,
            )
        scores: dict[int, float] = {}
        chunk_ids: dict[int, str] = {}
        for term in query_terms:
            rows = self._connection.execute(
                """
                SELECT chunk_vocab.doc, COUNT(*), documents.document_length, documents.chunk_id
                FROM chunk_vocab
                JOIN documents ON documents.doc_id = chunk_vocab.doc
                WHERE chunk_vocab.term = ?
                GROUP BY chunk_vocab.doc
                ORDER BY chunk_vocab.doc
                """,
                (_encoded_term(term),),
            ).fetchall()
            df_value = float(len(rows))
            ratio = (float(self.document_count) - df_value + 0.5) / (df_value + 0.5)
            idf = math.log1p(ratio)
            for doc_id, term_frequency, document_length, chunk_id in rows:
                numeric_doc_id = int(doc_id)
                tf_value = float(term_frequency)
                length_ratio = float(document_length) / self.average_length
                length_norm = (1.0 - BM25_B) + (BM25_B * length_ratio)
                numerator = tf_value * (BM25_K1 + 1.0)
                denominator = tf_value + (BM25_K1 * length_norm)
                term_score = idf * (numerator / denominator)
                score = scores.get(numeric_doc_id, 0.0) + term_score
                if not math.isfinite(score):
                    _fail("SPARSE_SCORE_NONFINITE", "BM25 produced a non-finite value")
                scores[numeric_doc_id] = 0.0 if score == 0.0 else score
                chunk_ids[numeric_doc_id] = str(chunk_id)
        ranked = sorted(
            ((doc_id, score) for doc_id, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], chunk_ids[item[0]]),
        )[:12]
        ranked_chunks = self._load_chunks(tuple(doc_id for doc_id, _score in ranked))
        return SparseRetrievalResult(
            query=canonical_query,
            query_terms=query_terms,
            candidates=tuple(
                RetrievalCandidate(
                    chunk=chunk,
                    exact_reference_match=False,
                    sparse_score=score,
                )
                for chunk, (_doc_id, score) in zip(ranked_chunks, ranked, strict=True)
            ),
            diagnostics=(),
            index_checksum=self.index_checksum,
        )


def open_disk_bm25_index(
    *,
    database_path: Path,
    chunks_path: Path,
    manifest_data: bytes,
) -> DiskBm25Index:
    """Open a checksum-bound disk index only after all immutable identities validate."""

    try:
        manifest = parse_record_json(
            manifest_data,
            DiskBm25Manifest,
            artifact_path="bm25.index.manifest.json",
            record_identity="manifest",
        )
    except RecordValidationError as error:
        message = error.issues[0].message if error.issues else "index manifest is invalid"
        raise DiskBm25Error("SPARSE_INDEX_MANIFEST_INVALID", message) from error
    if manifest.runtime_compatibility_id != APPROVED_BM25_RUNTIME_ID:
        _fail("BM25_RUNTIME_UNAPPROVED", "BM25 runtime compatibility ID is not owner-approved")
    if manifest.k1 != BM25_K1 or manifest.b != BM25_B:
        _fail("SPARSE_INDEX_MANIFEST_INVALID", "BM25 parameters differ from bm25.v1")
    if manifest.sqlite_version != sqlite3.sqlite_version:
        _fail("BM25_RUNTIME_UNAPPROVED", "SQLite runtime differs from the index manifest")
    if len(manifest.ordered_files) != 1 or manifest.ordered_files[0].path != database_path.name:
        _fail("SPARSE_INDEX_FILE_MISMATCH", "index database is absent from the approved file list")
    if _checksum_path(database_path) != manifest.ordered_files[0].checksum:
        _fail("SPARSE_INDEX_CHECKSUM_MISMATCH", "index database checksum does not match")
    if _checksum_path(chunks_path) != manifest.chunks_artifact_checksum:
        _fail("SPARSE_CHUNK_CHECKSUM_MISMATCH", "chunk artifact checksum does not match")
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA query_only=ON")
    metadata = dict(connection.execute("SELECT key, value FROM metadata ORDER BY key"))
    expected = {
        "schema_version": DISK_INDEX_SCHEMA_VERSION,
        "retrieval_version": manifest.retrieval_version,
        "chunks_artifact_checksum": manifest.chunks_artifact_checksum,
        "corpus_checksum": manifest.corpus_checksum,
        "alias_manifest_checksum": manifest.alias_manifest_checksum,
        "runtime_compatibility_id": manifest.runtime_compatibility_id,
        "document_count": str(manifest.document_count),
        "total_document_length": str(manifest.total_document_length),
    }
    if metadata != expected:
        connection.close()
        _fail("SPARSE_INDEX_METADATA_MISMATCH", "index metadata differs from its manifest")
    return DiskBm25Index(
        connection=connection,
        chunks_path=chunks_path,
        manifest=manifest,
        manifest_data=manifest_data,
    )
