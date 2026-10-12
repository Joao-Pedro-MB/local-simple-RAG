"""Hybrid retrieval: BM25 keyword search merged with the embedding ranking by reciprocal rank fusion.

BM25 is checked by hand with k1 = 1.2 and b = 0.75. When every Chunk has the same length and a
Query word occurs once, a word's score is exactly its IDF, ln(1 + (N - n + 0.5) / (n + 0.5)),
for N Chunks of which n hold the word.
"""

import pytest
from pytest import approx

from fakes import FakeEmbedder, FakeGenerator, WordTokenizer
from rag.pipeline import ask, build_index


def ask_hybrid(notes_dir, index_dir, question, k=10):
    build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())
    return ask(question, index_dir, FakeEmbedder(), FakeGenerator(), k=k, retrieval="hybrid")


def bm25_by_document(answer):
    return {c.chunk.document: (c.bm25_rank, c.bm25_score) for c in answer.context}


def test_bm25_weighs_each_query_word_by_how_rare_it_is_across_the_chunks(notes, index_dir):
    notes_dir = notes({
        "attention.md": "attention is everywhere",
        "mamba.md": "mamba replaces attention",
        "cats.md": "cats purr loudly",
    })

    answer = ask_hybrid(notes_dir, index_dir, "mamba attention")

    # N = 3. "attention" is in 2 Chunks: ln(1 + 1.5 / 2.5) = ln 1.6   = 0.4700
    #        "mamba" is in 1 Chunk:      ln(1 + 2.5 / 1.5) = ln 8/3  = 0.9808
    assert bm25_by_document(answer) == {
        "mamba.md": (1, approx(0.9808 + 0.4700, abs=1e-4)),
        "attention.md": (2, approx(0.4700, abs=1e-4)),
        "cats.md": (None, None),  # holds no Query word, so keyword search did not find it
    }


def test_bm25_rewards_repeating_a_query_word_with_diminishing_returns(notes, index_dir):
    notes_dir = notes({  # four Chunks of four words, so length plays no part
        "once.md": "mamba one two three",
        "twice.md": "mamba mamba two three",
        "four-times.md": "mamba mamba mamba mamba",
        "cats.md": "cats purr very loudly",
    })

    answer = ask_hybrid(notes_dir, index_dir, "mamba")

    # idf("mamba") = ln(1 + 1.5 / 3.5) = 0.3567, times f * 2.2 / (f + 1.2) for f occurrences:
    # f = 1 -> 1, f = 2 -> 1.375, f = 4 -> 1.692, creeping towards k1 + 1 = 2.2.
    assert bm25_by_document(answer) == {
        "four-times.md": (1, approx(0.3567 * 8.8 / 5.2, abs=1e-4)),
        "twice.md": (2, approx(0.3567 * 4.4 / 3.2, abs=1e-4)),
        "once.md": (3, approx(0.3567, abs=1e-4)),
        "cats.md": (None, None),
    }


def test_bm25_favours_the_shorter_of_two_chunks_that_match_equally(notes, index_dir):
    notes_dir = notes({
        "short.md": "mamba scans",
        "long.md": "mamba scans tokens in linear time",
    })

    answer = ask_hybrid(notes_dir, index_dir, "mamba")

    # "mamba" is in both Chunks: idf = ln(1 + 0.5 / 2.5) = 0.1823, still positive. The average
    # length is 4 tokens, so with b = 0.75 the short Chunk's 2 tokens count as 0.625 of the usual
    # length and the long Chunk's 6 tokens as 1.375:
    #   short: 0.1823 * 2.2 / (1 + 1.2 * 0.625) = 0.2292
    #   long:  0.1823 * 2.2 / (1 + 1.2 * 1.375) = 0.1514
    assert bm25_by_document(answer) == {
        "short.md": (1, approx(0.2292, abs=1e-4)),
        "long.md": (2, approx(0.1514, abs=1e-4)),
    }


def bm25_ranks(answer):
    return {c.chunk.document: c.bm25_rank for c in answer.context}


def test_bm25_matches_a_jargon_term_like_pass_at_k_as_a_whole(notes, index_dir):
    notes_dir = notes({
        "pass-at-k.md": "pass@k counts the problems solved by any of k samples.",
        "samples.md": "k samples pass.",
        "cats.md": "Cats purr.",
    })

    answer = ask_hybrid(notes_dir, index_dir, "What is pass@k?")

    # Both Chunks hold "pass" and "k", and the shorter samples.md would win on those alone.
    # Only pass-at-k.md holds the rarer whole term "pass@k".
    assert bm25_ranks(answer) == {"pass-at-k.md": 1, "samples.md": 2, "cats.md": None}


def test_bm25_still_finds_a_joined_word_by_its_parts(notes, index_dir):
    notes_dir = notes({
        "sampling.md": "Top-k sampling keeps the k likeliest tokens.",
        "cats.md": "Cats purr.",
    })

    answer = ask_hybrid(notes_dir, index_dir, "top k")

    assert bm25_ranks(answer) == {"sampling.md": 1, "cats.md": None}


def test_an_unknown_retrieval_mode_is_refused(notes, index_dir):
    build_index(notes({"cats.md": "Cats purr."}), index_dir, FakeEmbedder(), WordTokenizer())

    with pytest.raises(ValueError, match="hybird"):
        ask("Why do cats purr?", index_dir, FakeEmbedder(), FakeGenerator(), retrieval="hybird")


# "mamba" is in one Chunk, "what" and "is" in most, so BM25 puts the Mamba Chunk first while
# the fake Embedder, which only counts shared words, puts it third. Each Chunk is embedded
# behind its title, here its filename, so each filename only repeats words of its own text.
MAMBA_NOTES = {
    "attention.md": "What is attention?",
    "cats.md": "Cats purr.",
    "mamba.md": "Mamba is a state space model.",
    "recurrent-network.md": "What is a recurrent network?",
}


def test_reciprocal_rank_fusion_orders_chunks_by_the_sum_of_1_over_60_plus_each_rank(notes, index_dir):
    answer = ask_hybrid(notes(MAMBA_NOTES), index_dir, "What is Mamba?")

    assert [(c.chunk.document, c.embedding_rank, c.bm25_rank, c.rank, c.score) for c in answer.context] == [
        ("attention.md", 1, 2, 1, approx(1 / 61 + 1 / 62)),
        ("mamba.md", 3, 1, 2, approx(1 / 63 + 1 / 61)),
        ("recurrent-network.md", 2, 3, 3, approx(1 / 62 + 1 / 63)),
        ("cats.md", 4, None, 4, approx(1 / 64)),  # a ranking that misses a Chunk adds nothing
    ]


def test_hybrid_retrieval_keeps_the_top_k_of_the_fused_ranking(notes, index_dir):
    notes_dir = notes(MAMBA_NOTES)
    build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    embeddings_only = ask("What is Mamba?", index_dir, FakeEmbedder(), FakeGenerator(), k=2)
    hybrid = ask("What is Mamba?", index_dir, FakeEmbedder(), FakeGenerator(), k=2, retrieval="hybrid")

    assert [c.chunk.document for c in embeddings_only.context] == ["attention.md", "recurrent-network.md"]
    assert [c.chunk.document for c in hybrid.context] == ["attention.md", "mamba.md"]
