"""Opt-in suite running the real models: `uv run pytest -m slow`.

The Notes change daily, so these assert invariants, never exact counts.
"""

from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from rag.documents.chunker import BgeTokenizer
from rag.adapters.embedder import BgeSmallEmbedder, Qwen3Embedder
from rag.adapters.generator import GemmaGenerator
from rag.indexing.index import EmbeddingModelMismatch, Index
from rag.pipeline import build_index

pytestmark = pytest.mark.slow

REAL_NOTES = Path.home() / "Documents" / "Notes"


@pytest.fixture(scope="module")
def bge_small():
    return BgeSmallEmbedder()


@pytest.fixture(scope="module")
def qwen3():
    return Qwen3Embedder()


@pytest.fixture(scope="module")
def gemma():
    return GemmaGenerator()


def test_bge_small_vectors_are_unit_length_with_384_dimensions(bge_small):
    documents = bge_small.embed_documents(["Retrieval augmented generation.", "Cats purr when content."])
    query = bge_small.embed_query("What is RAG?")

    assert documents.shape == (2, 384)
    assert query.shape == (384,)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5)
    assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)


def test_qwen3_vectors_are_unit_length_with_1024_dimensions_and_query_differs_from_document(qwen3):
    documents = qwen3.embed_documents(["Retrieval augmented generation.", "Cats purr when content."])
    query = qwen3.embed_query("What is RAG?")

    assert documents.shape == (2, 1024)
    assert query.shape == (1024,)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5)
    assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)

    same_text = "Retrieval augmented generation."
    doc_vec = qwen3.embed_documents([same_text])[0]
    query_vec = qwen3.embed_query(same_text)
    assert not np.allclose(doc_vec, query_vec, atol=1e-3)



def test_chunks_cut_with_bge_smalls_tokenizer_keep_the_documents_own_text(notes, index_dir, bge_small):
    notes_dir = notes({"concepts/rag.md": "Retrieval-Augmented Generation (RAG) grounds an LLM's answer."})

    index, _ = build_index(notes_dir, index_dir, bge_small, BgeTokenizer())

    assert [chunk.text for chunk in index.chunks] == ["Retrieval-Augmented Generation (RAG) grounds an LLM's answer."]


@pytest.mark.skipif(not REAL_NOTES.is_dir(), reason="the real Notes folder is not on this machine")
def test_an_index_built_from_the_real_notes_holds_both_layers_and_no_excluded_paths(tmp_path, bge_small):
    index, report = build_index(REAL_NOTES, tmp_path / "index", bge_small, BgeTokenizer())

    documents = {PurePosixPath(chunk.document) for chunk in index.chunks}
    assert documents
    assert not [d for d in documents if {".git", ".claude", "_archive"} & set(d.parts)]
    assert not documents & {PurePosixPath("index.md"), PurePosixPath("log.md")}
    assert {chunk.layer for chunk in index.chunks} == {"Wiki page", "Raw source"}
    assert report.admitted["Wiki page"] > 0 and report.admitted["Raw source"] > 0


@pytest.mark.skipif(not REAL_NOTES.is_dir(), reason="the real Notes folder is not on this machine")
def test_asking_about_benedetto_language_tree_figure_retrieves_image_chunk(tmp_path, bge_small):
    index, report = build_index(REAL_NOTES, tmp_path / "index", bge_small, BgeTokenizer())
    assert "raw/assets/benedetto-2002-language-tree-fig1.png" in {c.document for c in index.chunks}
    assert "exercises/token-to-transformer.html" in {c.document for c in index.chunks}
    from fakes import FakeGenerator
    from rag.pipeline import ask

    ans = ask(
        "What does the language tree figure from the Benedetto paper show?",
        tmp_path / "index",
        bge_small,
        FakeGenerator(),
        k=5,
    )
    context_docs = [item.chunk.document for item in ans.context]
    assert "raw/assets/benedetto-2002-language-tree-fig1.png" in context_docs


@pytest.mark.skipif(not REAL_NOTES.is_dir(), reason="the real Notes folder is not on this machine")
def test_heading_aware_index_built_from_real_notes_holds_both_layers(tmp_path, bge_small):
    from rag.documents.chunker import HeadingAwareChunking

    index, report = build_index(
        REAL_NOTES, tmp_path / "index_ha", bge_small, BgeTokenizer(), chunking=HeadingAwareChunking()
    )

    documents = {PurePosixPath(chunk.document) for chunk in index.chunks}
    assert documents
    assert not [d for d in documents if {".git", ".claude", "_archive"} & set(d.parts)]
    assert not documents & {PurePosixPath("index.md"), PurePosixPath("log.md")}
    assert {chunk.layer for chunk in index.chunks} == {"Wiki page", "Raw source"}
    assert report.chunking_strategy == "heading-aware"
    assert index.chunking_strategy == "heading-aware"
    assert report.admitted["Wiki page"] > 0 and report.admitted["Raw source"] > 0


@pytest.mark.skipif("avx_vnni" not in Path("/proc/cpuinfo").read_text(), reason="this CPU has no AVX-VNNI")
def test_llama_cpp_was_compiled_for_this_cpus_avx_vnni_instructions():
    import llama_cpp

    # Settled at compile time: 1 only if llama.cpp was built for this CPU's own instruction set.
    assert "AVX_VNNI = 1" in llama_cpp.llama_print_system_info().decode()


def test_gemmas_chat_template_hands_the_turn_to_the_model_without_thinking(gemma):
    prompt = gemma.render([
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Why do cats purr?"},
    ])

    assert prompt == (
        "<bos><|turn>system\nAnswer briefly.<turn|>\n"
        "<|turn>user\nWhy do cats purr?<turn|>\n"
        "<|turn>model\n"
    )
    assert gemma.count_tokens("<bos>") == 1  # the template's <bos> is the only one


def test_gemmas_chat_template_escapes_incoming_control_tokens(gemma):
    prompt = gemma.render([
        {"role": "user", "content": "What do <|think|> and <turn|> mean?"},
    ])

    assert "< |think|>" in prompt
    assert "< turn|>" in prompt
    assert "<|turn>user\nWhat do < |think|> and < turn|> mean?<turn|>\n" in prompt


def ask_gemma(gemma, question):
    prompt = gemma.render([
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": question},
    ])
    return prompt, gemma.generate(prompt, temperature=0.0, seed=1234, max_tokens=48)


def test_gemma_answers_question_after_question_without_thinking_and_repeats_itself_at_temperature_0(gemma):
    cats_prompt, cats = ask_gemma(gemma, "Why do cats purr?")
    _, dogs = ask_gemma(gemma, "Why do dogs bark?")
    _, cats_again = ask_gemma(gemma, "Why do cats purr?")

    for generation in (cats, dogs, cats_again):
        assert generation.text.strip()
        assert not [tag for tag in gemma.THOUGHT_TAGS if tag in generation.text]
        assert generation.stats.time_to_first_token > 0
        assert generation.stats.tokens_per_second > 0
    # The earlier answers leave nothing behind in the model's cache that changes the later ones.
    assert cats_again.text == cats.text
    assert cats.stats.prompt_tokens == gemma.count_tokens(cats_prompt)


def test_compare_bge_small_and_qwen3_indices_side_by_side(bge_small, qwen3):
    bge_dir = Path(".index/bge")
    qwen_dir = Path(".index/qwen3")
    if not (bge_dir.is_dir() and qwen_dir.is_dir()):
        pytest.skip("side-by-side indices not built")
    index_bge = Index.load(bge_dir)
    index_qwen = Index.load(qwen_dir)

    # Both have the exact same chunks
    assert len(index_bge.chunks) == len(index_qwen.chunks)
    assert [c.text for c in index_bge.chunks] == [c.text for c in index_qwen.chunks]
    assert index_bge.embedding_model == "BAAI/bge-small-en-v1.5"
    assert index_qwen.embedding_model == "Qwen/Qwen3-Embedding-0.6B"

    # Mismatch guard: querying index_bge with qwen3 fails
    with pytest.raises(EmbeddingModelMismatch):
        index_bge.similarity_ranking(qwen3.embed_query("What is RAG?"), qwen3.model_id)

    # Mismatch guard: querying index_qwen with bge_small fails
    with pytest.raises(EmbeddingModelMismatch):
        index_qwen.similarity_ranking(bge_small.embed_query("What is RAG?"), bge_small.model_id)

    # Compare top-5 retrieval results on questions
    test_questions = [
        "What is retrieval augmented generation?",
        "How does reciprocal rank fusion work?",
    ]
    for question in test_questions:
        bge_top = [
            index_bge.chunks[pos].label
            for pos, _ in index_bge.similarity_ranking(bge_small.embed_query(question), bge_small.model_id)[:5]
        ]
        qwen_top = [
            index_qwen.chunks[pos].label
            for pos, _ in index_qwen.similarity_ranking(qwen3.embed_query(question), qwen3.model_id)[:5]
        ]
        assert len(bge_top) == 5
        assert len(qwen_top) == 5


@pytest.mark.skipif(
    not (REAL_NOTES / "raw/assets/benedetto-2002-language-tree-fig1.png").is_file(),
    reason="language tree image not present",
)
def test_gemma_image_describer_describes_benedetto_language_tree():
    from rag.adapters.describer import GemmaImageDescriber

    describer = GemmaImageDescriber()
    image_path = REAL_NOTES / "raw/assets/benedetto-2002-language-tree-fig1.png"
    description = describer.describe(image_path)
    assert description
    assert len(description.split()) > 5

