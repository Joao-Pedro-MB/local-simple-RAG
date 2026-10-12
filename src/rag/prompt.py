"""Prompt builder: turns a Question and its Context into the exact prompt the model reads."""

from dataclasses import dataclass

INSTRUCTIONS = (
    "Answer the Question only from the Context below. "
    "If the Context does not contain the answer, say that the answer isn't in your notes."
)


# Strings Gemma 4's tokenizer turns into control tokens rather than text: every CONTROL and
# USER_DEFINED token in its vocabulary. The rendered prompt is tokenized with special tokens
# recognised (that is how the chat template's own <|turn> markers work), so any of these
# quoted in a Document or the Question would act on the model: <turn|> would end the user's turn
# early, and <|think|> could switch thinking mode on.
GEMMA_SPECIAL_TOKENS = (
    "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
    "<|tool>", "<tool|>", "<|think|>", "<|turn>", "<turn|>",
    "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>",
    "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>',
    "<|channel>", "<channel|>",
)

# With thinking on, Gemma writes its reasoning between these before the answer. E2B writes
# none at all when thinking is off, so finding either one means thinking mode was on.
GEMMA_THOUGHT_TAGS = ("<|channel>", "<channel|>")


@dataclass(frozen=True)
class Prompt:
    text: str  # exactly what the model reads, after its chat template
    context: list  # the Context Chunks that fit, best first
    dropped: int  # how many lowest-ranked Chunks were left out to make it fit


def build_prompt(question, context, generator, answer_tokens):
    """The prompt for the Question, holding as much of the Context as fits.

    The model's context window holds the prompt and the answer together, so the prompt may
    use the window minus `answer_tokens`. While it is longer than that, the lowest-ranked
    Chunk is dropped and the prompt rendered again. It is measured after the chat template
    is applied, so the template's own tokens count too.
    """
    budget = generator.context_window - answer_tokens
    kept = list(context)
    while True:
        text = generator.render(messages(question, kept))
        if generator.count_tokens(text) <= budget:
            return Prompt(text, kept, dropped=len(context) - len(kept))
        if not kept:
            raise ValueError(
                f"The Question is too long: even with no Context the prompt exceeds the "
                f"{budget} tokens left in the context window after room for the answer."
            )
        kept.pop()


def messages(question, context):
    """The instructions as the system message; the numbered Context and the Question as the
    user's, the Question last so it is the freshest thing the model reads before answering."""
    # Each Chunk is named by its label, `Title › Section`, the same name its Citation gets.
    chunk_texts = "\n\n".join(f"[{item.rank}] {item.chunk.label}\n{item.chunk.text}" for item in context)
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": plain_text(f"Context:\n\n{chunk_texts}\n\nQuestion: {question}")},
    ]


def plain_text(text):
    """Break up every special-token string with a space after its `<`, so the model reads it
    as the words it is made of: `<|think|>` becomes `< |think|>`."""
    for token in GEMMA_SPECIAL_TOKENS:
        text = text.replace(token, "< " + token[1:])
    return text
