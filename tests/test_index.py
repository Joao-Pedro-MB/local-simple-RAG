import hashlib
import numpy as np
import pytest

from fakes import FakeEmbedder, WordTokenizer
from rag.documents.chunker import FixedWindows, HeadingAwareChunking
from rag.documents.corpus import Document, Heading
from rag.indexing.index import DocumentDiff, Index, IndexStats


def make_doc(path, text, title=None, content_hash=None, headings=None, vision_description=None):
    if title is None:
        title = path.rsplit(".", 1)[0]
    if content_hash is None:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(
        path=path,
        layer="Wiki page",
        title=title,
        text=text,
        headings=headings or [],
        metadata={"title": title},
        content_hash=content_hash,
        vision_description=vision_description,
    )


def test_index_build_from_scratch():
    docs = [
        make_doc("concepts/rag.md", "Retrieval augmented generation connects search to generation."),
        make_doc("concepts/chunking.md", "Chunking cuts documents into retrieval units."),
    ]
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()

    index, stats = Index.build(docs, embedder, tokenizer)

    assert isinstance(index, Index)
    assert isinstance(stats, IndexStats)
    assert len(index.chunks) == 2
    assert index.vectors.shape == (2, FakeEmbedder.dimension)
    assert stats.documents_added == 2
    assert stats.documents_changed == 0
    assert stats.documents_removed == 0
    assert stats.documents_unchanged == 0
    assert stats.chunks_reembedded == 2
    assert len(embedder.embedded_documents) == 2
    assert "concepts/rag.md" in index.documents
    assert "concepts/chunking.md" in index.documents
    assert index.content_hashes["concepts/rag.md"] == docs[0].content_hash
    assert index.content_hashes["concepts/chunking.md"] == docs[1].content_hash


def test_index_diff_identifies_added_changed_removed_unchanged():
    doc_a = make_doc("doc_a.md", "Alpha note original.")
    doc_b = make_doc("doc_b.md", "Beta note original.")
    doc_c = make_doc("doc_c.md", "Gamma note original.")

    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build([doc_a, doc_b, doc_c], embedder, tokenizer)

    # Current docs:
    # doc_a: untouched
    # doc_b: changed text and hash
    # doc_c: removed (not in current list)
    # doc_d: newly added
    doc_b_mod = make_doc("doc_b.md", "Beta note revised and edited.")
    doc_d = make_doc("doc_d.md", "Delta note new.")

    diff = index.diff([doc_a, doc_b_mod, doc_d])

    assert isinstance(diff, DocumentDiff)
    assert [d.path for d in diff.added] == ["doc_d.md"]
    assert [d.path for d in diff.changed] == ["doc_b.md"]
    assert diff.removed == ["doc_c.md"]
    assert [d.path for d in diff.unchanged] == ["doc_a.md"]


def test_index_incremental_update_reembeds_only_added_and_changed():
    doc_a = make_doc("doc_a.md", "Alpha note content.")
    doc_b = make_doc("doc_b.md", "Beta note content.")
    doc_c = make_doc("doc_c.md", "Gamma note content.")

    embedder1 = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, stats1 = Index.build([doc_a, doc_b, doc_c], embedder1, tokenizer)

    alpha_vector_initial = index.vectors[0].copy()

    # Modify doc_b, add doc_d, drop doc_c
    doc_b_mod = make_doc("doc_b.md", "Beta note updated with new concepts.")
    doc_d = make_doc("doc_d.md", "Delta note added.")

    embedder2 = FakeEmbedder()
    stats2 = index.update([doc_a, doc_b_mod, doc_d], embedder2, tokenizer)

    assert stats2.documents_added == 1
    assert stats2.documents_changed == 1
    assert stats2.documents_removed == 1
    assert stats2.documents_unchanged == 1
    assert stats2.chunks_reembedded == 2

    # Only doc_b and doc_d were embedded by embedder2! doc_a was NOT embedded!
    assert len(embedder2.embedded_documents) == 2
    embedded_texts = embedder2.embedded_documents
    assert any("Beta note updated" in t for t in embedded_texts)
    assert any("Delta note added" in t for t in embedded_texts)
    assert not any("Alpha note content" in t for t in embedded_texts)

    # Chunks in index reflect current docs: [doc_a, doc_b, doc_d]
    doc_paths = [c.document for c in index.chunks]
    assert doc_paths == ["doc_a.md", "doc_b.md", "doc_d.md"]

    # doc_a vector is untouched
    np.testing.assert_array_equal(index.vectors[0], alpha_vector_initial)
    assert "doc_c.md" not in index.documents
    assert "doc_c.md" not in index.content_hashes


def test_index_build_or_update_dispatches_correctly():
    doc_a = make_doc("doc_a.md", "Alpha note.")
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()

    # Without existing_index: builds from scratch
    index1, stats1 = Index.build_or_update([doc_a], embedder, tokenizer, existing_index=None)
    assert stats1.documents_added == 1

    # With existing_index: updates incrementally
    doc_a_mod = make_doc("doc_a.md", "Alpha note changed.")
    embedder2 = FakeEmbedder()
    index2, stats2 = Index.build_or_update([doc_a_mod], embedder2, tokenizer, existing_index=index1)
    assert index2 is index1
    assert stats2.documents_changed == 1
    assert stats2.documents_added == 0


def test_index_try_load_verifies_compatibility(tmp_path):
    index_dir = tmp_path / "test_idx"
    doc = make_doc("note.md", "Some note.")
    embedder = FakeEmbedder(model_id="embed-v1")
    tokenizer = WordTokenizer()

    index, _ = Index.build([doc], embedder, tokenizer, chunking=FixedWindows())
    index.save(index_dir)

    # Compatible load
    loaded = Index.try_load(index_dir, "embed-v1", "fixed", "words")
    assert loaded is not None
    assert loaded.embedding_model == "embed-v1"

    # Incompatible embedding model
    assert Index.try_load(index_dir, "other-model", "fixed", "words") is None

    # Incompatible chunking strategy
    assert Index.try_load(index_dir, "embed-v1", "headings", "words") is None

    # Incompatible tokenizer
    assert Index.try_load(index_dir, "embed-v1", "fixed", "other-tokenizer") is None

    # Non-existent directory
    assert Index.try_load(tmp_path / "non_existent", "embed-v1", "fixed", "words") is None


def test_index_cached_vision_description_lookup():
    raw_img_bytes = b"fake-png-image-bytes"
    doc_text = "diagram\n\nArchitecture Diagram"
    full_content_hash = hashlib.sha256(raw_img_bytes + doc_text.encode("utf-8")).hexdigest()

    img_doc = Document(
        path="assets/diagram.png",
        layer="Wiki page",
        title="diagram",
        text=doc_text,
        headings=[],
        metadata={},
        content_hash=full_content_hash,
        vision_description="A neural network architecture diagram.",
    )
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build([img_doc], embedder, tokenizer)

    # When raw bytes match what produced the hash, returns cached description
    cached = index.get_cached_vision_description("assets/diagram.png", raw_img_bytes)
    assert cached == "A neural network architecture diagram."

    # When raw bytes are modified, returns None
    assert index.get_cached_vision_description("assets/diagram.png", b"modified-bytes") is None

    # When path is not in index, returns None
    assert index.get_cached_vision_description("assets/other.png", raw_img_bytes) is None


def test_chunk_limit_warnings_emitted_during_build_and_update():
    long_text = "word " * 100
    doc = make_doc("long.md", long_text)
    embedder = FakeEmbedder(max_input_tokens=10)
    tokenizer = WordTokenizer()

    # Build emits warnings
    index, stats = Index.build([doc], embedder, tokenizer, chunking=FixedWindows(size=50, overlap=0))
    assert len(stats.warnings) > 0
    assert any("over fake-embedder's limit of 10" in w for w in stats.warnings)

    # Update also retains/emits warnings
    doc_mod = make_doc("long.md", long_text + " extra")
    stats_update = index.update([doc_mod], embedder, tokenizer, chunking=FixedWindows(size=50, overlap=0))
    assert len(stats_update.warnings) > 0
    assert any("over fake-embedder's limit of 10" in w for w in stats_update.warnings)


def test_cli_index_reports_diff_counts_accurately(notes, tmp_path, monkeypatch, capsys):
    from rag.cli import main

    notes_dir = notes({
        "doc1.md": "Content of doc one.",
        "doc2.md": "Content of doc two.",
    })
    index_dir = tmp_path / "cli_diff_idx"

    monkeypatch.setattr("rag.adapters.embedder.BgeSmallEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag.documents.chunker.BgeTokenizer", WordTokenizer)

    # Initial build: 2 added
    main(["index", "--notes", str(notes_dir), "--index-dir", str(index_dir)])
    out1 = capsys.readouterr().out
    assert "Documents: 2 added, 0 changed, 0 removed, 0 unchanged." in out1
    assert "2 re-embedded" in out1

    # Modify doc1, add doc3, delete doc2
    (notes_dir / "doc1.md").write_text("Modified content of doc one.")
    (notes_dir / "doc3.md").write_text("New doc three.")
    (notes_dir / "doc2.md").unlink()

    # Re-index: 1 added, 1 changed, 1 removed, 0 unchanged; 2 re-embedded
    main(["index", "--notes", str(notes_dir), "--index-dir", str(index_dir)])
    out2 = capsys.readouterr().out
    assert "Documents: 1 added, 1 changed, 1 removed, 0 unchanged." in out2
    assert "2 re-embedded" in out2


def test_index_search_returns_context_chunks_for_embeddings_mode():
    from rag.indexing.index import ContextChunk

    doc_a = make_doc("doc_a.md", "Cats purr when content.")
    doc_b = make_doc("doc_b.md", "Dogs bark loudly.")
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build([doc_a, doc_b], embedder, tokenizer)

    query = "cats purr"
    query_vector = embedder.embed_query(query)
    results = index.search(query, query_vector, embedder.model_id, k=2, mode="embeddings")

    assert len(results) == 2
    first = results[0]
    assert isinstance(first, ContextChunk)
    assert first.chunk.document == "doc_a.md"
    assert first.score > 0
    assert not hasattr(first, "document")

    # Verify ContextChunk is a pure dataclass: no tuple unpacking or indexing
    with pytest.raises(TypeError):
        chunk, score = first

    with pytest.raises(TypeError):
        _ = first[0]


def test_index_search_validates_query_vector():
    doc_a = make_doc("doc_a.md", "Cats purr when content.")
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build([doc_a], embedder, tokenizer)

    with pytest.raises(ValueError, match="query_vector is required for search"):
        index.search("cats", None, embedder.model_id)


def test_index_search_returns_fused_context_chunks_for_hybrid_mode():
    docs = [
        make_doc("attention.md", "What is attention?"),
        make_doc("cats.md", "Cats purr."),
        make_doc("mamba.md", "Mamba is a state space model."),
        make_doc("recurrent-network.md", "What is a recurrent network?"),
    ]
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build(docs, embedder, tokenizer)

    query = "What is Mamba?"
    query_vector = embedder.embed_query(query)
    results = index.search(query, query_vector, embedder.model_id, k=4, mode="hybrid")

    assert len(results) == 4
    assert [(c.chunk.document, c.embedding_rank, c.bm25_rank, c.rank, c.score) for c in results] == [
        ("attention.md", 1, 2, 1, pytest.approx(1 / 61 + 1 / 62)),
        ("mamba.md", 3, 1, 2, pytest.approx(1 / 63 + 1 / 61)),
        ("recurrent-network.md", 2, 3, 3, pytest.approx(1 / 62 + 1 / 63)),
        ("cats.md", 4, None, 4, pytest.approx(1 / 64)),
    ]


def test_index_caches_bm25_instance_across_searches():
    from rag.indexing.bm25 import BM25

    doc_a = make_doc("doc_a.md", "Cats purr when content.")
    doc_b = make_doc("doc_b.md", "Dogs bark loudly.")
    embedder = FakeEmbedder()
    tokenizer = WordTokenizer()
    index, _ = Index.build([doc_a, doc_b], embedder, tokenizer)

    # First access to keyword ranking computes and caches BM25
    kw1 = index.keyword_ranking("cats")
    bm25_first = index.bm25
    assert isinstance(bm25_first, BM25)

    # Subsequent access uses the cached instance
    kw2 = index.keyword_ranking("dogs")
    assert index.bm25 is bm25_first

    # Search in hybrid mode also uses the cached instance
    qv = embedder.embed_query("cats")
    index.search("cats", qv, embedder.model_id, k=2, mode="hybrid")
    assert index.bm25 is bm25_first

    # Incrementally updating the index clears/invalidates the cached BM25 instance
    doc_c = make_doc("doc_c.md", "Birds sing happily.")
    index.update([doc_a, doc_b, doc_c], embedder, tokenizer)
    assert index._bm25 is None
    kw3 = index.keyword_ranking("birds")
    assert index.bm25 is not bm25_first

