import pytest

from fakes import FakeGenerator, FakeEmbedder, WordTokenizer
from rag.adapters.generator import Generation, GenerationStats, GemmaGenerator
from rag.pipeline import ask, build_index
import rag.pipeline.prompt as prompt_module


def test_prompt_module_has_no_model_specific_tokens_or_thought_tags():
    """src/rag/pipeline/prompt.py must not expose model-specific constants or plain_text helper."""
    assert not hasattr(prompt_module, "GEMMA_SPECIAL_TOKENS")
    assert not hasattr(prompt_module, "GEMMA_THOUGHT_TAGS")
    assert not hasattr(prompt_module, "plain_text")


def test_prompt_messages_preserves_clean_text():
    """Prompt messages builder leaves incoming text clean and unescaped."""
    question = "What do <|think|> and <turn|> do?"
    msgs = prompt_module.messages(question, [])
    assert len(msgs) == 2
    assert msgs[1]["content"] == f"Context:\n\n\n\nQuestion: {question}"
    assert "<|think|>" in msgs[1]["content"]
    assert "<turn|>" in msgs[1]["content"]


def test_gemma_generator_escapes_control_tokens():
    """GemmaGenerator escapes control tokens by adding a space after '<'."""
    text = "Tokens: <|think|>, <turn|>, <eos>, <pad>, <bos>, <|channel>, <channel|>"
    escaped = GemmaGenerator.escape(text)
    assert "< |think|>" in escaped
    assert "< turn|>" in escaped
    assert "< eos>" in escaped
    assert "< pad>" in escaped
    assert "< bos>" in escaped
    assert "< |channel>" in escaped
    assert "< channel|>" in escaped
    assert "<|think|>" not in escaped
    assert "<turn|>" not in escaped


def test_gemma_generator_render_escapes_incoming_message_content():
    """GemmaGenerator.render escapes control tokens in messages before template formatting."""
    # Instantiate without calling __init__ (avoids downloading or loading llama.cpp weights)
    gen = object.__new__(GemmaGenerator)
    # Mock _template to return rendered string
    gen._template = lambda messages, enable_thinking=False: type(
        "ChatTemplateResult", (), {"prompt": "\n".join(f"{m['role']}: {m['content']}" for m in messages)}
    )()

    rendered = gen.render([
        {"role": "system", "content": "Instruction with <|think|>"},
        {"role": "user", "content": "Question with <turn|> and <eos>"},
    ])

    assert "< |think|>" in rendered
    assert "< turn|>" in rendered
    assert "< eos>" in rendered
    assert "<|think|>" not in rendered
    assert "<turn|>" not in rendered
    assert "<eos>" not in rendered


def test_gemma_generator_filters_thought_tags_and_adds_warning():
    """GemmaGenerator filters thought tags internally and produces a warning."""
    raw = "<|channel>thought reasoning here<channel|>The answer is 42."
    clean, warnings = GemmaGenerator.filter_thought_tags(raw)

    assert "<|channel>" not in clean
    assert "<channel|>" not in clean
    assert "The answer is 42." in clean
    assert len(warnings) == 1
    assert "thought tags" in warnings[0]


def test_gemma_generator_filter_thought_tags_when_no_tags_present():
    """Clean text produces no warnings and remains unchanged."""
    raw = "A clean answer without tags."
    clean, warnings = GemmaGenerator.filter_thought_tags(raw)
    assert clean == raw
    assert warnings == []


def test_fake_generator_has_no_gemma_thought_tags_import_and_supports_warnings():
    """FakeGenerator does not import GEMMA_THOUGHT_TAGS and passes configured warnings."""
    fake = FakeGenerator(text="Cats purr.", warnings=["A test warning."])
    gen = fake.generate("test prompt", 0.0, 1234, 100)
    assert gen.text == "Cats purr."
    assert gen.warnings == ["A test warning."]


def test_swapping_generators_works_cleanly_across_generator_port(notes, index_dir):
    """The pipeline interact with generators purely through the Generator port."""
    notes_dir = notes({"cats.md": "Cats purr when content."})
    build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    # 1. FakeGenerator works cleanly
    fake_gen = FakeGenerator(text="Clean fake answer.")
    ans1 = ask("Why do cats purr?", index_dir, FakeEmbedder(), fake_gen)
    assert ans1.text == "Clean fake answer."
    assert ans1.warnings == []

    # 2. FakeGenerator with warning propagates warning cleanly
    fake_with_warn = FakeGenerator(text="Warned answer.", warnings=["Custom warning."])
    ans2 = ask("Why do cats purr?", index_dir, FakeEmbedder(), fake_with_warn)
    assert ans2.text == "Warned answer."
    assert "Custom warning." in ans2.warnings

    # 3. Dummy adapter adhering to the Generator port
    class MockAdapter:
        context_window = 4096

        def count_tokens(self, text):
            return len(text.split())

        def render(self, messages):
            return "\n".join(m["content"] for m in messages)

        def generate(self, prompt, temperature, seed, max_tokens):
            return Generation(
                "Adapter answer.",
                GenerationStats(prompt_tokens=self.count_tokens(prompt), time_to_first_token=0.1, tokens_per_second=10.0),
                warnings=["Adapter warning."],
            )

    ans3 = ask("Why do cats purr?", index_dir, FakeEmbedder(), MockAdapter())
    assert ans3.text == "Adapter answer."
    assert "Adapter warning." in ans3.warnings
