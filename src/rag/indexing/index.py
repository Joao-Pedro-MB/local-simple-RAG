"""Index: every Chunk and its vector, searched by brute-force cosine similarity (ADR 0002)
or by BM25 keyword search over the Chunks' text.

Saved as four files in the index directory:
    vectors.npy     one row per Chunk (float32, unit length)
    chunks.jsonl    one Chunk per line, in the same order as the rows
    documents.json  each admitted Document's title, Layer and frontmatter metadata, by path
    metadata.json   the id of the embedding model that made the vectors
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.indexing.bm25 import BM25
from rag.documents.chunker import Chunk, FixedWindows, HeadingAwareChunking
from rag.documents.corpus import Document

__all__ = [
    "ContextChunk",
    "DocumentDiff",
    "EmbeddingModelMismatch",
    "Index",
    "IndexStats",
    "RETRIEVAL_MODES",
]

RRF_CONSTANT = 60


def _reciprocal_rank_fusion(rankings):
    """Merge rankings of Index positions, each best first, into (position, fused score) pairs.

    The result is best first. Equal fused scores keep Index order, so the result never
    depends on the order of a dict or a set.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] += 1 / (RRF_CONSTANT + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class EmbeddingModelMismatch(Exception):
    """A Query vector from one embedding model compared against Chunk vectors from another."""


def _chunk_limit_warning(chunk, number, tokens, embedder):
    return (
        f"{chunk.document}, chunk {number} ({chunk.label}): {tokens} tokens with its label, "
        f"over {embedder.model_id}'s limit of {embedder.max_input_tokens}, "
        f"so the model reads only the first {embedder.max_input_tokens}."
    )


def _embed_chunks(chunks, embedder):
    dim = getattr(embedder, "dimension", 384)
    if chunks:
        started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text_to_embed for chunk in chunks])
        seconds = time.perf_counter() - started
        return vectors, seconds
    return np.empty((0, dim), dtype=np.float32), 0.0


def _normalize_chunking(chunking):
    if chunking is None or chunking == "fixed":
        return FixedWindows()
    if chunking in ("headings", "heading-aware"):
        return HeadingAwareChunking()
    if isinstance(chunking, str):
        raise ValueError(f"Unknown chunking strategy {chunking!r}; choose 'fixed' or 'headings'.")
    return chunking


RETRIEVAL_MODES = ("embeddings", "hybrid")


@dataclass(frozen=True)
class ContextChunk:
    """A Chunk of the Context at its rank. In hybrid retrieval, `rank` and `score` are the fused
    ones and the Chunk's place in each of the two rankings that were fused is kept too."""

    rank: int
    score: float
    chunk: Chunk
    embedding_rank: int | None = None
    embedding_score: float | None = None
    bm25_rank: int | None = None  # None also when keyword search did not find the Chunk
    bm25_score: float | None = None


@dataclass(frozen=True)
class DocumentDiff:
    """Documents classified by how they differ from the Index."""

    added: list[Document]
    changed: list[Document]
    removed: list[str]
    unchanged: list[Document]


@dataclass(frozen=True)
class IndexStats:
    """Counts and measurements from building or incrementally updating an Index."""

    documents_added: int = 0
    documents_changed: int = 0
    documents_removed: int = 0
    documents_unchanged: int = 0
    chunks_reembedded: int = 0
    embedding_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedding_model: str
    # {path: {"title": ..., "layer": ..., "metadata": {...}}} for every admitted Document,
    # including any too short to yield a Chunk.
    documents: dict = field(default_factory=dict)
    chunking_strategy: str = "fixed"
    chunk_tokenizer: str = "words"
    content_hashes: dict = field(default_factory=dict)
    vision_descriptions: dict = field(default_factory=dict)
    notes_location: str = ""
    build_time: str = ""
    _bm25: BM25 | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([chunk.text for chunk in self.chunks])
        return self._bm25

    def search(
        self,
        query,
        query_vector,
        embedding_model,
        k=5,
        mode="embeddings",
    ):
        """The k Chunks most relevant to the Query, best first, as ranked ContextChunk instances.

        Supports both 'embeddings' and 'hybrid' retrieval modes.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose one of {', '.join(RETRIEVAL_MODES)}.")

        if query_vector is None:
            raise ValueError("query_vector is required for search")

        if not self.chunks:
            return []

        if mode == "embeddings":
            best = self.similarity_ranking(query_vector, embedding_model)[:k]
            return [
                ContextChunk(rank=rank, score=score, chunk=self.chunks[position])
                for rank, (position, score) in enumerate(best, start=1)
            ]

        if not query:
            raise ValueError("A query string is required for hybrid retrieval.")

        embedding = {
            position: (rank, score)
            for rank, (position, score) in enumerate(
                self.similarity_ranking(query_vector, embedding_model), start=1
            )
        }
        keyword = {
            position: (rank, score)
            for rank, (position, score) in enumerate(self.keyword_ranking(query), start=1)
        }
        fused = _reciprocal_rank_fusion([list(embedding), list(keyword)])[:k]
        context = []
        for rank, (position, score) in enumerate(fused, start=1):
            embedding_rank, embedding_score = embedding[position]
            bm25_rank, bm25_score = keyword.get(position, (None, None))
            chunk = self.chunks[position]
            context.append(
                ContextChunk(
                    rank=rank,
                    score=score,
                    chunk=chunk,
                    embedding_rank=embedding_rank,
                    embedding_score=embedding_score,
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                )
            )
        return context

    def similarity_ranking(self, query_vector, embedding_model):
        """Every Chunk's position in the Index and cosine similarity to the Query, best first.

        `embedding_model` names the model that embedded the Query. Vectors from different
        models live in unrelated spaces, so comparing them would give meaningless scores.
        """
        if embedding_model != self.embedding_model:
            raise EmbeddingModelMismatch(
                f"The Index was built with {self.embedding_model} but the Query was embedded "
                f"with {embedding_model}. Rebuild the Index with `rag index` using {embedding_model}."
            )
        # Every vector is unit length, so the dot product *is* the cosine similarity.
        scores = self.vectors @ query_vector
        # A stable sort keeps ties in Index order, so the same Query always ranks the same way.
        order = np.argsort(-scores, kind="stable")
        return [(int(position), float(scores[position])) for position in order]

    def keyword_ranking(self, query):
        """Position and BM25 score of every Chunk containing a Query word, best first.

        BM25 corpus statistics are cached with the Index and reused across queries.
        """
        if not self.chunks:
            return []
        return self.bm25.rank(query)

    def diff(self, documents: list[Document]) -> DocumentDiff:
        """Diff a collection of current Documents against this Index.

        Identifies added, changed, removed and unchanged Documents using
        paths and content hashes.
        """
        existing_hashes = self.content_hashes
        current_doc_paths = {d.path for d in documents}
        existing_doc_paths = set(self.documents) | set(existing_hashes)

        added = [d for d in documents if d.path not in existing_doc_paths]
        changed = [
            d for d in documents
            if d.path in existing_doc_paths and existing_hashes.get(d.path) != d.content_hash
        ]
        unchanged = [
            d for d in documents
            if d.path in existing_doc_paths and existing_hashes.get(d.path) == d.content_hash
        ]
        removed = [path for path in sorted(existing_doc_paths) if path not in current_doc_paths]
        return DocumentDiff(added=added, changed=changed, removed=removed, unchanged=unchanged)

    def get_cached_vision_description(self, image_path: Path | str, raw_bytes: bytes) -> str | None:
        """Return the cached vision description for an image if its content hasn't changed.

        Returns the description string if cached and unchanged, or None if new/modified/uncached.
        """
        image_path_str = Path(image_path).as_posix()
        old_hash = self.content_hashes.get(image_path_str)
        existing_chunks = [c for c in self.chunks if c.document == image_path_str]
        old_doc_text = existing_chunks[0].text if existing_chunks else None
        if (
            old_hash
            and old_doc_text is not None
            and hashlib.sha256(raw_bytes + old_doc_text.encode("utf-8")).hexdigest() == old_hash
        ):
            return self.vision_descriptions.get(image_path_str, "")
        return None

    @classmethod
    def build(
        cls,
        documents: list[Document],
        embedder,
        chunk_tokenizer,
        chunking=None,
        notes_location="",
    ) -> tuple["Index", IndexStats]:
        """Build a new Index from scratch from a collection of Documents."""
        chunking = _normalize_chunking(chunking)
        strategy = getattr(chunking, "strategy", "fixed")
        tokenizer_id = getattr(chunk_tokenizer, "name", getattr(chunk_tokenizer, "model_id", chunk_tokenizer.__class__.__name__))
        dim = getattr(embedder, "dimension", 384)
        index = cls(
            chunks=[],
            vectors=np.empty((0, dim), dtype=np.float32),
            embedding_model=embedder.model_id,
            chunking_strategy=strategy,
            chunk_tokenizer=tokenizer_id,
            notes_location=str(notes_location),
        )
        stats = index.update(
            documents,
            embedder,
            chunk_tokenizer,
            chunking=chunking,
            notes_location=notes_location,
        )
        return index, stats

    def update(
        self,
        documents: list[Document],
        embedder,
        chunk_tokenizer,
        chunking=None,
        notes_location="",
    ) -> IndexStats:
        """Incrementally update this Index from a collection of Documents."""
        chunking = _normalize_chunking(chunking)
        strategy = getattr(chunking, "strategy", "fixed")
        tokenizer_id = getattr(chunk_tokenizer, "name", getattr(chunk_tokenizer, "model_id", chunk_tokenizer.__class__.__name__))

        diff = self.diff(documents)

        existing_chunks_by_doc = {}
        existing_vectors_by_doc = {}
        for i, chunk in enumerate(self.chunks):
            existing_chunks_by_doc.setdefault(chunk.document, []).append(chunk)
            existing_vectors_by_doc.setdefault(chunk.document, []).append(self.vectors[i])

        new_chunks_by_doc = {}
        chunks_to_embed = []
        for doc in diff.added + diff.changed:
            doc_chunks = chunking.chunk(doc, chunk_tokenizer)
            new_chunks_by_doc[doc.path] = doc_chunks
            chunks_to_embed.extend(doc_chunks)

        new_vectors, embedding_seconds = _embed_chunks(chunks_to_embed, embedder)

        new_vectors_by_doc = {}
        offset = 0
        for doc in diff.added + diff.changed:
            count = len(new_chunks_by_doc[doc.path])
            new_vectors_by_doc[doc.path] = new_vectors[offset : offset + count]
            offset += count

        final_chunks = []
        final_vectors_list = []
        unchanged_paths = {d.path for d in diff.unchanged}
        for doc in documents:
            if doc.path in unchanged_paths:
                final_chunks.extend(existing_chunks_by_doc.get(doc.path, []))
                final_vectors_list.extend(existing_vectors_by_doc.get(doc.path, []))
            else:
                c_list = new_chunks_by_doc.get(doc.path, [])
                final_chunks.extend(c_list)
                v_arr = new_vectors_by_doc.get(doc.path, [])
                for row in v_arr:
                    final_vectors_list.append(row)

        warnings = []
        doc_chunk_counters = {}
        for chunk in final_chunks:
            num = doc_chunk_counters.get(chunk.document, 0) + 1
            doc_chunk_counters[chunk.document] = num
            tokens = len(chunk_tokenizer.spans(chunk.text_to_embed))
            if tokens > embedder.max_input_tokens:
                warnings.append(_chunk_limit_warning(chunk, num, tokens, embedder))

        self.chunks = final_chunks
        self._bm25 = None
        if final_vectors_list:
            self.vectors = np.stack(final_vectors_list).astype(np.float32)
        else:
            self.vectors = np.empty((0, dim), dtype=np.float32)

        self.documents = {d.path: {"title": d.title, "layer": d.layer, "metadata": d.metadata} for d in documents}
        self.content_hashes = {d.path: d.content_hash for d in documents if d.content_hash}

        new_vision = dict(self.vision_descriptions)
        for d in documents:
            if getattr(d, "vision_description", None):
                new_vision[d.path] = d.vision_description
        current_paths = {d.path for d in documents}
        self.vision_descriptions = {p: v for p, v in new_vision.items() if p in current_paths}

        if notes_location:
            self.notes_location = str(notes_location)
        self.chunking_strategy = strategy
        self.chunk_tokenizer = tokenizer_id
        self.build_time = datetime.now(timezone.utc).isoformat()

        return IndexStats(
            documents_added=len(diff.added),
            documents_changed=len(diff.changed),
            documents_removed=len(diff.removed),
            documents_unchanged=len(diff.unchanged),
            chunks_reembedded=len(chunks_to_embed),
            embedding_seconds=embedding_seconds,
            warnings=warnings,
        )

    @classmethod
    def build_or_update(
        cls,
        documents: list[Document],
        embedder,
        chunk_tokenizer,
        chunking=None,
        existing_index=None,
        notes_location="",
    ) -> tuple["Index", IndexStats]:
        """Build or incrementally update an Index from documents and an embedder."""
        if existing_index is not None:
            stats = existing_index.update(
                documents,
                embedder,
                chunk_tokenizer,
                chunking=chunking,
                notes_location=notes_location,
            )
            return existing_index, stats
        return cls.build(
            documents,
            embedder,
            chunk_tokenizer,
            chunking=chunking,
            notes_location=notes_location,
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors.astype(np.float32))
        with open(directory / "chunks.jsonl", "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")
        documents = json.dumps(self.documents, indent=2, ensure_ascii=False)
        (directory / "documents.json").write_text(documents + "\n")
        hashes = json.dumps(self.content_hashes, indent=2, ensure_ascii=False)
        (directory / "hashes.json").write_text(hashes + "\n")
        if self.vision_descriptions:
            vision = json.dumps(self.vision_descriptions, indent=2, ensure_ascii=False)
            (directory / "vision_descriptions.json").write_text(vision + "\n")
        if not self.build_time:
            self.build_time = datetime.now(timezone.utc).isoformat()
        metadata = {
            "embedding_model": self.embedding_model,
            "chunking_strategy": self.chunking_strategy,
            "chunk_tokenizer": self.chunk_tokenizer,
            "notes_location": str(self.notes_location),
            "build_time": self.build_time,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        with open(directory / "chunks.jsonl") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        vectors = np.load(directory / "vectors.npy")
        documents = json.loads((directory / "documents.json").read_text())
        hashes_file = directory / "hashes.json"
        content_hashes = json.loads(hashes_file.read_text()) if hashes_file.is_file() else {}
        vision_file = directory / "vision_descriptions.json"
        vision_descriptions = json.loads(vision_file.read_text()) if vision_file.is_file() else {}
        return cls(
            chunks,
            vectors,
            metadata["embedding_model"],
            documents,
            chunking_strategy=metadata.get("chunking_strategy", "fixed"),
            chunk_tokenizer=metadata.get("chunk_tokenizer", "words"),
            content_hashes=content_hashes,
            vision_descriptions=vision_descriptions,
            notes_location=metadata.get("notes_location", ""),
            build_time=metadata.get("build_time", ""),
        )

    @classmethod
    def try_load(cls, directory, embedding_model, chunking_strategy, chunk_tokenizer=""):
        """Load an Index from directory if it exists and matches the model, strategy and tokenizer."""
        directory = Path(directory)
        try:
            if not (directory / "metadata.json").is_file():
                return None
            loaded = cls.load(directory)
            if (
                loaded.embedding_model == embedding_model
                and loaded.chunking_strategy == chunking_strategy
                and getattr(loaded, "chunk_tokenizer", "") == chunk_tokenizer
            ):
                return loaded
        except Exception:
            return None
        return None
