import pytest

from fakes import FakeEmbedder, FakeGenerator, WordTokenizer
from rag.pipeline import ANSWER_TOKENS, ask, build_index


def build(notes_dir, index_dir):
    return build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())


def answer(question, index_dir, generator, **options):
    return ask(question, index_dir, FakeEmbedder(), generator, **options)


CATS_AND_DOGS = {
    "cats.md": "---\ntitle: Cats\n---\nCats purr when they are content.",
    "dogs.md": "---\ntitle: Dogs\n---\nDogs bark at the mail carrier.",
}


def test_the_model_answers_from_the_instruction_the_numbered_context_and_the_question(notes, index_dir):
    build(notes(CATS_AND_DOGS), index_dir)
    generator = FakeGenerator(text="Cats purr when they are content [1].")

    result = answer("Why do cats purr?", index_dir, generator)

    assert result.text == "Cats purr when they are content [1]."
    prompt = generator.prompt
    assert "only from the Context" in prompt
    assert "isn't in your notes" in prompt
    assert "[1] Cats\nCats purr when they are content." in prompt
    assert "[2] Dogs\nDogs bark at the mail carrier." in prompt
    assert prompt.index("Why do cats purr?") > prompt.index("[2] Dogs")
    assert result.prompt == prompt
    assert result.warnings == []


def test_special_token_strings_in_the_notes_and_the_question_reach_the_model_as_plain_text(notes, index_dir):
    # Gemma thinks before answering when <|think|> starts its system prompt, and reads <turn|> as
    # the end of a turn. Quoted in a note about Gemma, they must stay plain text.
    build(notes({"gemma.md": "Thinking is on when <|think|> starts the system prompt; <turn|> ends a turn."}), index_dir)
    generator = FakeGenerator()

    answer("What do <|think|> and <eos> mean?", index_dir, generator)

    assert not [token for token in ("<|think|>", "<turn|>", "<eos>") if token in generator.prompt]
    assert "Thinking is on when < |think|> starts the system prompt; < turn|> ends a turn." in generator.prompt
    assert "What do < |think|> and < eos> mean?" in generator.prompt


def test_an_answer_holding_thought_tags_comes_with_a_warning(notes, index_dir):
    build(notes(CATS_AND_DOGS), index_dir)
    # What Gemma writes when thinking mode is on: its reasoning between thought tags, then the answer.
    generator = FakeGenerator(text="<|channel>thought\nThe notes say cats purr.<channel|>Cats purr when content.")

    result = answer("Why do cats purr?", index_dir, generator)

    assert len(result.warnings) == 1 and "thought tags" in result.warnings[0]


def test_generator_flags_thought_tags_in_generation_warnings():
    gen_with_thoughts = FakeGenerator(text="<|channel>reasoning<channel|>answer").generate("prompt", 0, 1234, 100)
    assert len(gen_with_thoughts.warnings) == 1
    assert "thought tags" in gen_with_thoughts.warnings[0]

    gen_without_thoughts = FakeGenerator(text="clean answer").generate("prompt", 0, 1234, 100)
    assert gen_without_thoughts.warnings == []


def test_answers_default_to_temperature_0_with_the_same_seed_every_time(notes, index_dir):
    build(notes(CATS_AND_DOGS), index_dir)
    generator = FakeGenerator()

    answer("Why do cats purr?", index_dir, generator)
    answer("Why do dogs bark?", index_dir, generator)

    first, second = generator.calls
    assert first["temperature"] == second["temperature"] == 0
    assert first["seed"] == second["seed"]


def test_a_higher_temperature_reaches_the_model(notes, index_dir):
    build(notes(CATS_AND_DOGS), index_dir)
    generator = FakeGenerator()

    answer("Why do cats purr?", index_dir, generator, temperature=0.8)

    assert generator.calls[-1]["temperature"] == 0.8


def test_citations_name_each_document_and_section_in_the_context_once_in_rank_order(notes, index_dir):
    # a.md is long enough to be cut into two Chunks, both under its "Greek" heading; they rank
    # first and third.
    a = "---\ntitle: Letters\n---\n## Greek\n" + "alpha beta gamma " + "x " * 345 + "alpha beta " + "x " * 10
    build(notes({
        "a.md": a,
        "b.md": "---\ntitle: Greek letters\n---\nalpha beta gamma y z",
        "c.md": "---\ntitle: First letter\n---\nalpha",
    }), index_dir)
    # Small models cite unreliably, so whatever the answer claims is ignored.
    generator = FakeGenerator(text="As zebra.md explains [9], they are the first three letters.")

    result = answer("alpha beta gamma", index_dir, generator)

    assert [c.chunk.document for c in result.context] == ["a.md", "b.md", "a.md", "c.md"]
    assert result.citations == ["Letters › Greek", "Greek letters", "First letter"]


# Five 200-word Documents sharing 5, 4, 3, 2 and 1 words with LONG_QUESTION.
LONG_QUESTION = "alpha beta gamma delta epsilon"
LONG_NOTES = {
    f"shares-{n}.md": " ".join(LONG_QUESTION.split()[:n] + ["x"] * (200 - n)) for n in range(1, 6)
}


def test_the_lowest_ranked_chunks_are_dropped_so_the_prompt_leaves_room_for_the_answer(notes, index_dir):
    build(notes(LONG_NOTES), index_dir)
    # Room for the answer plus 700 words of prompt: three 200-word Chunks fit, four don't.
    generator = FakeGenerator(context_window=ANSWER_TOKENS + 700)

    result = answer(LONG_QUESTION, index_dir, generator)

    assert [c.chunk.document for c in result.context] == ["shares-5.md", "shares-4.md", "shares-3.md"]
    assert result.citations == ["shares-5", "shares-4", "shares-3"]  # titled by filename
    assert "shares-2" not in generator.prompt
    prompt_tokens = generator.count_tokens(generator.prompt)
    assert prompt_tokens + generator.calls[-1]["max_tokens"] <= generator.context_window
    assert len(result.warnings) == 1 and "2 lowest-ranked Chunks" in result.warnings[0]


def test_a_question_too_long_for_the_context_window_on_its_own_is_refused(notes, index_dir):
    build(notes(CATS_AND_DOGS), index_dir)
    generator = FakeGenerator(context_window=ANSWER_TOKENS + 20)

    with pytest.raises(ValueError, match="too long"):
        answer("Why " + "really " * 50 + "do cats purr?", index_dir, generator)

    assert generator.calls == []
