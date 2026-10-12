"""Index: every Chunk and its vector, searched by brute-force cosine similarity (ADR 0002)
or by BM25 keyword search over the Chunks' text.

Saved as four files in the index directory:
    vectors.npy     one row per Chunk (float32, unit length)
    chunks.jsonl    one Chunk per line, in the same order as the rows
    documents.json  each admitted Document's title, Layer and frontmatter metadata, by path
    metadata.json   the id of the embedding model that made the vectors
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rag.bm25 import BM25
from rag.chunker import Chunk


class EmbeddingModelMismatch(Exception):
    """A Query vector from one embedding model compared against Chunk vectors from another."""


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

    def search(self, query_vector, embedding_model, k):
        """The k Chunks most similar to the Query, best first, as (Chunk, score) pairs."""
        best = self.similarity_ranking(query_vector, embedding_model)[:k]
        return [(self.chunks[position], score) for position, score in best]

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

        BM25's corpus statistics (how many Chunks contain each word, the average Chunk length)
        are recomputed from the Index's current Chunks on every call and never saved, so they
        always describe exactly the Chunks being searched, however the Index was built.
        """
        return BM25([chunk.text for chunk in self.chunks]).rank(query)

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
            from datetime import datetime, timezone

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

