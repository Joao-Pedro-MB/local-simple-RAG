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

from rag.chunker import Chunk, FixedWindows, HeadingAwareChunking
from rag.corpus import RAW_SOURCE, WIKI_PAGE, load_documents
from rag.fusion import reciprocal_rank_fusion
from rag.generator import GenerationStats
from rag.index import EmbeddingModelMismatch, Index
from rag.prompt import GEMMA_THOUGHT_TAGS, build_prompt

__all__ = [
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

# Embeddings-only is the baseline; hybrid adds BM25 keyword search (step 3).
RETRIEVAL_MODES = ("embeddings", "hybrid")

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
class Answer:
    question: str
    context: list[ContextChunk]
    text: str
    citations: list[str]
    prompt: str  # exactly what the model read, after its chat template
    stats: GenerationStats
    warnings: list[str]
    chunking_strategy: str = "fixed"


def _chunk_limit_warning(chunk, number, tokens, embedder):
    return (
        f"{chunk.document}, chunk {number} ({chunk.label}): {tokens} tokens with its label, "
        f"over {embedder.model_id}'s limit of {embedder.max_input_tokens}, "
        f"so the model reads only the first {embedder.max_input_tokens}."
    )


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
    if isinstance(chunking, str):
        if chunking == "fixed":
            chunking = FixedWindows()
        elif chunking in ("headings", "heading-aware"):
            chunking = HeadingAwareChunking()
        else:
            raise ValueError(f"Unknown chunking strategy {chunking!r}; choose 'fixed' or 'headings'.")
    strategy = getattr(chunking, "strategy", "fixed")
    tokenizer_id = getattr(chunk_tokenizer, "name", getattr(chunk_tokenizer, "model_id", chunk_tokenizer.__class__.__name__))

    existing_index = None
    if not force:
        try:
            if (index_dir / "metadata.json").is_file():
                loaded = Index.load(index_dir)
                if (
                    loaded.embedding_model == embedder.model_id
                    and loaded.chunking_strategy == strategy
                    and getattr(loaded, "chunk_tokenizer", "") == tokenizer_id
                ):
                    existing_index = loaded
        except Exception:
            existing_index = None

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

    documents, excluded, corpus_warnings = load_documents(
        notes_dir, image_describer=describer_to_use, existing_index=existing_index
    )
    image_description_seconds = timed_describer.elapsed if timed_describer else 0.0

    if existing_index is None:
        docs_added = documents
        docs_changed = []
        docs_removed = []
        docs_unchanged = []

        chunks, warnings = [], list(corpus_warnings)
        for document in documents:
            for number, chunk in enumerate(chunking.chunk(document, chunk_tokenizer), start=1):
                chunks.append(chunk)
                tokens = len(chunk_tokenizer.spans(chunk.text_to_embed))
                if tokens > embedder.max_input_tokens:
                    warnings.append(_chunk_limit_warning(chunk, number, tokens, embedder))
        started = time.perf_counter()
        dim = getattr(embedder, "dimension", 384)
        vectors = embedder.embed_documents([chunk.text_to_embed for chunk in chunks]) if chunks else np.empty((0, dim), dtype=np.float32)
        embedding_seconds = time.perf_counter() - started if chunks else 0.0
        chunks_reembedded = len(chunks)
    else:
        existing_hashes = existing_index.content_hashes
        current_doc_paths = {d.path for d in documents}
        existing_doc_paths = set(existing_index.documents) | set(existing_hashes)

        docs_added = [d for d in documents if d.path not in existing_doc_paths]
        docs_changed = [
            d for d in documents
            if d.path in existing_doc_paths and existing_hashes.get(d.path) != d.content_hash
        ]
        docs_unchanged = [
            d for d in documents
            if d.path in existing_doc_paths and existing_hashes.get(d.path) == d.content_hash
        ]
        docs_removed = [path for path in existing_doc_paths if path not in current_doc_paths]

        existing_chunks_by_doc = {}
        existing_vectors_by_doc = {}
        for i, chunk in enumerate(existing_index.chunks):
            existing_chunks_by_doc.setdefault(chunk.document, []).append(chunk)
            existing_vectors_by_doc.setdefault(chunk.document, []).append(existing_index.vectors[i])

        new_chunks_by_doc = {}
        chunks_to_embed = []
        for doc in docs_added + docs_changed:
            doc_chunks = chunking.chunk(doc, chunk_tokenizer)
            new_chunks_by_doc[doc.path] = doc_chunks
            chunks_to_embed.extend(doc_chunks)

        dim = getattr(embedder, "dimension", 384)
        if chunks_to_embed:
            started = time.perf_counter()
            new_vectors = embedder.embed_documents([c.text_to_embed for c in chunks_to_embed])
            embedding_seconds = time.perf_counter() - started
        else:
            new_vectors = np.empty((0, dim), dtype=np.float32)
            embedding_seconds = 0.0

        chunks_reembedded = len(chunks_to_embed)

        new_vectors_by_doc = {}
        offset = 0
        for doc in docs_added + docs_changed:
            count = len(new_chunks_by_doc[doc.path])
            new_vectors_by_doc[doc.path] = new_vectors[offset : offset + count]
            offset += count

        final_chunks = []
        final_vectors_list = []
        unchanged_paths = {d.path for d in docs_unchanged}
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

        warnings = list(corpus_warnings)
        doc_chunk_counters = {}
        for chunk in final_chunks:
            num = doc_chunk_counters.get(chunk.document, 0) + 1
            doc_chunk_counters[chunk.document] = num
            tokens = len(chunk_tokenizer.spans(chunk.text_to_embed))
            if tokens > embedder.max_input_tokens:
                warnings.append(_chunk_limit_warning(chunk, num, tokens, embedder))

        chunks = final_chunks
        if final_vectors_list:
            vectors = np.stack(final_vectors_list).astype(np.float32)
        else:
            vectors = np.empty((0, dim), dtype=np.float32)

    records = {d.path: {"title": d.title, "layer": d.layer, "metadata": d.metadata} for d in documents}
    content_hashes = {d.path: d.content_hash for d in documents if d.content_hash}
    vision_descriptions = {}
    if existing_index is not None:
        vision_descriptions.update(existing_index.vision_descriptions)
    for d in documents:
        if getattr(d, "vision_description", None):
            vision_descriptions[d.path] = d.vision_description
    current_paths = {d.path for d in documents}
    vision_descriptions = {p: v for p, v in vision_descriptions.items() if p in current_paths}

    index = Index(
        chunks,
        vectors,
        embedder.model_id,
        records,
        chunking_strategy=strategy,
        chunk_tokenizer=tokenizer_id,
        content_hashes=content_hashes,
        vision_descriptions=vision_descriptions,
        notes_location=str(notes_dir),
    )
    index.save(index_dir)
    report = BuildReport(
        admitted={layer: sum(d.layer == layer for d in documents) for layer in (WIKI_PAGE, RAW_SOURCE)},
        excluded=excluded,
        chunks_created=len(chunks),
        warnings=warnings,
        embedding_seconds=embedding_seconds,
        chunking_strategy=strategy,
        image_description_seconds=image_description_seconds,
        documents_added=len(docs_added),
        documents_changed=len(docs_changed),
        documents_removed=len(docs_removed),
        documents_unchanged=len(docs_unchanged),
        chunks_reembedded=chunks_reembedded,
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
    if retrieval == "hybrid":
        # BM25 gets the Query itself, never the instruction the Embedder puts in front of it.
        retrieved = _hybrid_context(index, query, query_vector, embedder.model_id, k)
    else:
        results = index.search(query_vector, embedder.model_id, k)
        retrieved = [ContextChunk(rank, score, chunk) for rank, (chunk, score) in enumerate(results, start=1)]
    prompt = build_prompt(question, retrieved, generator, ANSWER_TOKENS)
    warnings = []
    if prompt.dropped:
        chunks = "Chunk" if prompt.dropped == 1 else "Chunks"
        warnings.append(f"Dropped the {prompt.dropped} lowest-ranked {chunks} so the prompt fits the context window.")
    generation = generator.generate(prompt.text, temperature, seed, max_tokens=ANSWER_TOKENS)
    for w in getattr(generation, "warnings", []):
        if w not in warnings:
            warnings.append(w)
    if any(tag in generation.text for tag in GEMMA_THOUGHT_TAGS):
        tag_warning = "The answer holds thought tags (<|channel>…<channel|>): thinking mode was on by accident."
        if tag_warning not in warnings:
            warnings.append(tag_warning)
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


def _hybrid_context(index, query, query_vector, embedding_model, k):
    """The top k of the embedding and BM25 rankings fused, each Chunk keeping its place in both.

    The embedding ranking holds every Chunk; the BM25 ranking only those containing a Query
    word. Both are fused whole and only then cut to k, so a Chunk embeddings rank 23rd can
    still make the Context if keyword search ranks it first.
    """
    # {Index position: (rank, score)}, in rank order.
    embedding = {
        position: (rank, score)
        for rank, (position, score) in enumerate(index.similarity_ranking(query_vector, embedding_model), start=1)
    }
    keyword = {position: (rank, score) for rank, (position, score) in enumerate(index.keyword_ranking(query), start=1)}
    fused = reciprocal_rank_fusion([list(embedding), list(keyword)])[:k]
    context = []
    for rank, (position, score) in enumerate(fused, start=1):
        embedding_rank, embedding_score = embedding[position]
        bm25_rank, bm25_score = keyword.get(position, (None, None))
        chunk = index.chunks[position]
        context.append(ContextChunk(rank, score, chunk, embedding_rank, embedding_score, bm25_rank, bm25_score))
    return context
