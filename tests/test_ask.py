import pytest

from fakes import FakeEmbedder, FakeGenerator, FakeImageDescriber, WordTokenizer
from rag.documents.chunker import HeadingAwareChunking
from rag.pipeline import EmbeddingModelMismatch, ask, build_index


def build(notes_dir, index_dir, embedder=None):
    index, _report = build_index(notes_dir, index_dir, embedder or FakeEmbedder(), WordTokenizer())
    return index


def test_the_chunk_sharing_most_words_with_the_question_ranks_first(notes, index_dir):
    notes_dir = notes({
        "cats.md": "Cats purr when they are content.",
        "dogs.md": "Dogs bark at the mail carrier.",
    })
    build(notes_dir, index_dir)

    answer = ask("Why do cats purr?", index_dir, FakeEmbedder(), FakeGenerator())

    assert [c.chunk.document for c in answer.context] == ["cats.md", "dogs.md"]
    assert [c.rank for c in answer.context] == [1, 2]


# Seven Documents of six words, sharing 0 to 6 of them with the Question below.
GRADED_NOTES = {
    "shares-0.md": "one two three four five six",
    "shares-1.md": "alpha two three four five six",
    "shares-2.md": "alpha beta three four five six",
    "shares-3.md": "alpha beta gamma four five six",
    "shares-4.md": "alpha beta gamma delta five six",
    "shares-5.md": "alpha beta gamma delta epsilon six",
    "shares-6.md": "alpha beta gamma delta epsilon zeta",
}
GRADED_QUESTION = "alpha beta gamma delta epsilon zeta"


def test_the_five_most_similar_chunks_are_the_context_by_default(notes, index_dir):
    build(notes(GRADED_NOTES), index_dir)

    answer = ask(GRADED_QUESTION, index_dir, FakeEmbedder(), FakeGenerator())

    assert [c.chunk.document for c in answer.context] == [
        "shares-6.md",
        "shares-5.md",
        "shares-4.md",
        "shares-3.md",
        "shares-2.md",
    ]


def test_k_sets_how_many_chunks_are_kept_each_with_its_cosine_similarity(notes, index_dir):
    build(notes(GRADED_NOTES), index_dir)

    answer = ask(GRADED_QUESTION, index_dir, FakeEmbedder(), FakeGenerator(), k=2)

    # Each Chunk is embedded behind its title, here its filename ("shares-6"), so it holds 8
    # distinct words to the Question's 6: cosine = shared / sqrt(8 * 6).
    assert [(c.chunk.document, c.score) for c in answer.context] == [
        ("shares-6.md", pytest.approx(6 / 48**0.5)),
        ("shares-5.md", pytest.approx(5 / 48**0.5)),
    ]


def test_asking_reloads_the_saved_index_with_the_chunks_it_was_built_with(notes, index_dir):
    notes_dir = notes({
        "cats.md": "---\ntitle: Cats\n---\nCats purr when they are content.",
        "long.md": " ".join(f"w{i}" for i in range(400)),
    })
    built = build(notes_dir, index_dir)

    answer = ask("Why do cats purr?", index_dir, FakeEmbedder(), FakeGenerator(), k=10)

    assert len(built.chunks) == 3
    assert {c.chunk for c in answer.context} == set(built.chunks)


def test_rebuilding_the_index_drops_documents_deleted_from_the_notes(notes, index_dir):
    notes_dir = notes({"cats.md": "Cats purr.", "dogs.md": "Dogs bark."})
    build(notes_dir, index_dir)
    (notes_dir / "dogs.md").unlink()

    build(notes_dir, index_dir)
    answer = ask("Do dogs bark?", index_dir, FakeEmbedder(), FakeGenerator())

    assert [c.chunk.document for c in answer.context] == ["cats.md"]


def test_an_index_is_not_searched_with_a_different_embedding_model_than_built_it(notes, index_dir):
    build(notes({"cats.md": "Cats purr."}), index_dir, FakeEmbedder(model_id="model-a"))

    with pytest.raises(EmbeddingModelMismatch) as refusal:
        ask("Why do cats purr?", index_dir, FakeEmbedder(model_id="model-b"), FakeGenerator())

    message = str(refusal.value)
    assert "model-a" in message and "model-b" in message and "rebuild" in message.lower()


def test_an_index_built_with_qwen3_is_searched_with_qwen3_and_refuses_bge_small(notes, index_dir):
    build(notes({"cats.md": "Cats purr."}), index_dir, FakeEmbedder(model_id="Qwen/Qwen3-Embedding-0.6B"))

    answer = ask("Why do cats purr?", index_dir, FakeEmbedder(model_id="Qwen/Qwen3-Embedding-0.6B"), FakeGenerator())
    assert [c.chunk.document for c in answer.context] == ["cats.md"]

    with pytest.raises(EmbeddingModelMismatch) as refusal:
        ask("Why do cats purr?", index_dir, FakeEmbedder(model_id="BAAI/bge-small-en-v1.5"), FakeGenerator())

    message = str(refusal.value)
    assert "Qwen/Qwen3-Embedding-0.6B" in message and "BAAI/bge-small-en-v1.5" in message


def test_answer_records_index_chunking_strategy(notes, index_dir):
    notes_dir = notes({"cats.md": "Cats purr when they are content."})
    build_index(
        notes_dir,
        index_dir,
        FakeEmbedder(),
        WordTokenizer(),
        chunking=HeadingAwareChunking(),
    )

    answer = ask("Why do cats purr?", index_dir, FakeEmbedder(), FakeGenerator())

    assert answer.chunking_strategy == "heading-aware"


def test_asking_retrieves_image_chunk_based_on_vision_description_with_vision_flag(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Zipping measures how similar two texts are.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
        "other.md": "Unrelated notes about something else.",
    })
    describer = FakeImageDescriber({
        "benedetto-2002-language-tree-fig1.png": "phylogenetic dendrogram clustering Indo-European dialect branches"
    })
    embedder = FakeEmbedder()
    build_index(notes_dir, index_dir, embedder, WordTokenizer(), image_describer=describer)

    answer = ask("phylogenetic dendrogram dialect branches", index_dir, embedder, FakeGenerator(), k=1)

    assert len(answer.context) == 1
    top_chunk = answer.context[0].chunk
    assert top_chunk.document == "raw/assets/benedetto-2002-language-tree-fig1.png"
    assert top_chunk.has_vision_description is True
    assert top_chunk.is_image is True


def test_asking_retrieves_image_chunk_without_vision_describer_marked_as_borrowed_text(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Figure 1: Language Tree illustrates the evolutionary tree.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
    })
    embedder = FakeEmbedder()
    build_index(notes_dir, index_dir, embedder, WordTokenizer(), image_describer=None)

    answer = ask("language tree evolutionary tree", index_dir, embedder, FakeGenerator(), k=2)

    image_chunk = next(c.chunk for c in answer.context if c.chunk.document == "raw/assets/benedetto-2002-language-tree-fig1.png")
    assert image_chunk.has_vision_description is False
    assert image_chunk.is_image is True


def test_cli_ask_detects_embedding_model_from_index_metadata(notes, tmp_path, monkeypatch, capsys):
    from rag.cli import main

    notes_dir = notes({"cats.md": "Cats purr when content."})
    index_dir = tmp_path / "cli_ask_index"
    build_index(notes_dir, index_dir, FakeEmbedder(model_id="BAAI/bge-small-en-v1.5"), WordTokenizer())

    monkeypatch.setattr("rag.cli.resolve_embedder", lambda mid: FakeEmbedder(model_id=mid))
    monkeypatch.setattr("rag.adapters.generator.GemmaGenerator", lambda: FakeGenerator(text="Cats purr."))

    main(["ask", "Why do cats purr?", "--index-dir", str(index_dir)])
    captured = capsys.readouterr().out
    assert "Cats purr." in captured


def test_cli_ask_rejects_embedding_model_flag(notes, tmp_path):
    from rag.cli import main

    notes_dir = notes({"cats.md": "Cats purr."})
    index_dir = tmp_path / "cli_ask_index"
    build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    with pytest.raises(SystemExit):
        main(["ask", "Why do cats purr?", "--index-dir", str(index_dir), "--embedding-model", "bge-small"])


