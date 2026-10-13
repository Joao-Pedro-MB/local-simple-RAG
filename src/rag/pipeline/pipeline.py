"""Pipeline: the two operations the CLI and the tests share.

    build_index  Notes folder -> Documents -> Chunks -> vectors -> Index saved to a directory,
                 plus a report of what was admitted, excluded, created and warned about
    ask          Question -> Query vector -> the most similar Chunks (the Context), optionally
                 merged with a BM25 keyword ranking by reciprocal rank fusion
                 -> prompt -> the Generator's answer, with Citations

Models arrive through their ports (an Embedder, a chunk tokenizer, a Generator), so this
module never loads one itself.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.documents.chunker import Chunk, FixedWindows, HeadingAwareChunking
from rag.documents.corpus import RAW_SOURCE, WIKI_PAGE, load_documents
from rag.adapters.generator import GenerationStats
from rag.indexing.index import ContextChunk, EmbeddingModelMismatch, Index, RETRIEVAL_MODES
from rag.pipeline.prompt import build_prompt

__all__ = [
    "ANSWER_TOKENS",
    "DEFAULT_SEED",
    "Answer",
    "BuildReport",
    "ContextChunk",
    "EmbeddingModelMismatch",
    "FixedWindows",
    "HeadingAwareChunking",
    "RETRIEVAL_MODES",
    "ask",
    "build_index",
]

# At temperature 0 the model always picks its most likely next token, so the same prompt
# always gives the same answer. Above 0 it samples, and a fixed seed keeps even that
# repeatable: any change in the answer then comes from the prompt, not from chance.
DEFAULT_SEED = 1234

# The longest answer the model may write; the prompt must leave this much of the context
# window free.
ANSWER_TOKENS = 512


@dataclass(frozen=True)
class BuildReport:
    """What a build admitted, left out and produced, and what embedding cost."""

    admitted: dict[str, int]  # Documents admitted per Layer
    excluded: list[str]  # paths left out; an excluded folder appears once, as "name/"
    chunks_created: int
    warnings: list[str]
    embedding_seconds: float
    chunking_strategy: str = "fixed"
    image_description_seconds: float = 0.0
    documents_added: int = 0
    documents_changed: int = 0
    documents_removed: int = 0
    documents_unchanged: int = 0
    chunks_reembedded: int = 0


@dataclass(frozen=True)
class Answer:
    question: str
    context: list[ContextChunk]
    text: str
    citations: list[str]
    prompt: str  # exactly what the model read, after its chat template
    stats: GenerationStats
    warnings: list[str]
    chunking_strategy: str = "fixed"


def _resolve_chunking(chunking):
    if isinstance(chunking, str):
        if chunking == "fixed":
            return FixedWindows()
        elif chunking in ("headings", "heading-aware"):
            return HeadingAwareChunking()
        else:
            raise ValueError(f"Unknown chunking strategy {chunking!r}; choose 'fixed' or 'headings'.")
    return chunking


def build_index(
    notes_dir,
    index_dir,
    embedder,
    chunk_tokenizer,
    chunking=FixedWindows(),
    image_describer=None,
    force=False,
):
    """Build an Index from every Document in the Notes folder and save it to index_dir.

    Returns the Index and a BuildReport.
    """
    notes_dir = Path(notes_dir)
    index_dir = Path(index_dir)
    if index_dir.resolve().is_relative_to(notes_dir.resolve()):
        raise ValueError(f"Refusing to save the Index inside the Notes folder ({notes_dir}): the Notes are read-only.")

    chunking = _resolve_chunking(chunking)
    tokenizer_id = getattr(chunk_tokenizer, "name", getattr(chunk_tokenizer, "model_id", chunk_tokenizer.__class__.__name__))

    existing_index = None
    if not force:
        existing_index = Index.try_load(index_dir, embedder.model_id, chunking.strategy, tokenizer_id)

    describer_to_use = image_describer
    timed_describer = None
    if image_describer is not None:
        class _TimedDescriber:
            def __init__(self, inner):
                self.inner = inner
                self.elapsed = 0.0

            def describe(self, path):
                t0 = time.perf_counter()
                res = self.inner.describe(path)
                self.elapsed += time.perf_counter() - t0
                return res

        timed_describer = _TimedDescriber(image_describer)
        describer_to_use = timed_describer

    cached_descriptions = existing_index.get_cached_vision_description if existing_index is not None else None
    documents, excluded, corpus_warnings = load_documents(
        notes_dir, image_describer=describer_to_use, cached_descriptions=cached_descriptions
    )
    image_description_seconds = timed_describer.elapsed if timed_describer else 0.0

    index, stats = Index.build_or_update(
        documents,
        embedder,
        chunk_tokenizer,
        chunking=chunking,
        existing_index=existing_index,
        notes_location=str(notes_dir),
    )
    index.save(index_dir)

    report = BuildReport(
        admitted={layer: sum(d.layer == layer for d in documents) for layer in (WIKI_PAGE, RAW_SOURCE)},
        excluded=excluded,
        chunks_created=len(index.chunks),
        warnings=list(corpus_warnings) + stats.warnings,
        embedding_seconds=stats.embedding_seconds,
        chunking_strategy=chunking.strategy,
        image_description_seconds=image_description_seconds,
        documents_added=stats.documents_added,
        documents_changed=stats.documents_changed,
        documents_removed=stats.documents_removed,
        documents_unchanged=stats.documents_unchanged,
        chunks_reembedded=stats.chunks_reembedded,
    )
    return index, report


def ask(question, index_dir, embedder, generator, k=5, retrieval="embeddings", temperature=0.0, seed=DEFAULT_SEED):
    """Retrieve the k best Chunks for the Question as its Context, and have the Generator
    answer the Question from that Context.

    retrieval="embeddings" ranks Chunks by cosine similarity to the Query; "hybrid" also ranks
    them with BM25 and merges the two rankings by reciprocal rank fusion.
    """
    if retrieval not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {retrieval!r}; choose one of {', '.join(RETRIEVAL_MODES)}.")
    index = Index.load(index_dir)
    query = question  # the Query is the Question until something rewrites it
    query_vector = embedder.embed_query(query)
    retrieved = index.search(query, query_vector, embedder.model_id, k=k, mode=retrieval)
    prompt = build_prompt(question, retrieved, generator, ANSWER_TOKENS)
    warnings = []
    if prompt.dropped:
        chunks = "Chunk" if prompt.dropped == 1 else "Chunks"
        warnings.append(f"Dropped the {prompt.dropped} lowest-ranked {chunks} so the prompt fits the context window.")
    generation = generator.generate(prompt.text, temperature, seed, max_tokens=ANSWER_TOKENS)
    for w in getattr(generation, "warnings", []):
        if w not in warnings:
            warnings.append(w)
    # The Context is what the model was actually given, so dropped Chunks are not cited.
    context = prompt.context
    return Answer(
        question,
        context,
        generation.text,
        citations(context),
        prompt.text,
        generation.stats,
        warnings,
        chunking_strategy=index.chunking_strategy,
    )


def citations(context):
    """Where the Context came from, as `Title › Section`: each once, in rank order.

    Built from the Context by the code, never parsed from the answer: small models cite
    unreliably, but this list always names exactly what the model was given.
    """
    return list(dict.fromkeys(item.chunk.label for item in context))
