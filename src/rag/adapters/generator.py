"""Generator: the language model that writes the answer.

Every Generator has the same port, so the Pipeline never knows which model it is using:

    context_window       most tokens the model can attend to: prompt and answer together
    count_tokens(text)   length of text in the model's own tokens
    render(messages)     the exact prompt string the model reads, made by its own chat template
    generate(prompt, temperature, seed, max_tokens) -> Generation: the text plus its stats
"""

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    time_to_first_token: float  # seconds: reading the whole prompt, then picking one token
    tokens_per_second: float  # writing speed once the first token is out


@dataclass(frozen=True)
class Generation:
    text: str
    stats: GenerationStats
    warnings: list[str] = field(default_factory=list)


class GemmaGenerator:
    """Gemma 4 E2B, as Google's quantization-aware-trained Q4_0 GGUF, run in-process on the CPU
    by llama.cpp (through llama-cpp-python). The weights download once into the Hugging Face
    cache, outside the repository."""

    REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"  # public: no account approval needed
    FILE = "gemma-4-E2B_q4_0-it.gguf"
    REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"  # pinned, so the weights never change underneath

    # Google's recommended sampling for Gemma 4 (also stored in the GGUF). Used only above
    # temperature 0; at 0 the likeliest token is always taken.
    TOP_K, TOP_P = 64, 0.95

    # Strings Gemma 4's tokenizer turns into control tokens rather than text: every CONTROL and
    # USER_DEFINED token in its vocabulary. The rendered prompt is tokenized with special tokens
    # recognised (that is how the chat template's own <|turn> markers work), so any of these
    # quoted in a Document or the Question would act on the model: <turn|> would end the user's turn
    # early, and <|think|> could switch thinking mode on.
    SPECIAL_TOKENS = (
        "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
        "<|tool>", "<tool|>", "<|think|>", "<|turn>", "<turn|>",
        "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>",
        "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>',
        "<|channel>", "<channel|>",
    )

    # With thinking on, Gemma writes its reasoning between these before the answer. E2B writes
    # none at all when thinking is off, so finding either one means thinking mode was on.
    THOUGHT_TAGS = ("<|channel>", "<channel|>")

    @classmethod
    def escape(cls, text):
        """Break up every special-token string with a space after its `<`, so the model reads it
        as the words it is made of: `<|think|>` becomes `< |think|>`."""
        for token in cls.SPECIAL_TOKENS:
            text = text.replace(token, "< " + token[1:])
        return text

    @classmethod
    def filter_thought_tags(cls, text):
        """Filter out thinking-mode tags from generated text and return (clean_text, warnings)."""
        warnings = []
        if any(tag in text for tag in cls.THOUGHT_TAGS):
            warnings.append("The answer holds thought tags (<|channel>…<channel|>): thinking mode was on by accident.")
            for tag in cls.THOUGHT_TAGS:
                text = text.replace(tag, "")
        return text, warnings

    def __init__(self, context_window=8192):
        # Imported here, not at the top, so code that never runs Gemma stays fast.
        import llama_cpp
        from huggingface_hub import hf_hub_download
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._llama_cpp = llama_cpp
        self.context_window = context_window
        path = hf_hub_download(self.REPO, self.FILE, revision=self.REVISION)
        self._llm = llama_cpp.Llama(model_path=path, n_ctx=context_window, verbose=False)
        # The chat template ships inside the GGUF, written by Google: rendering with it gives
        # exactly the turn markers the model was trained on.
        self._template = Jinja2ChatFormatter(
            template=self._llm.metadata["tokenizer.chat_template"],
            bos_token=self._special_text(self._llm.token_bos()),
            eos_token=self._special_text(self._llm.token_eos()),
        )

    def count_tokens(self, text):
        return len(self._tokenize(text))

    def render(self, messages):
        escaped_messages = [
            {**msg, "content": self.escape(msg["content"])} if "content" in msg and isinstance(msg["content"], str) else msg
            for msg in messages
        ]
        # Thinking mode is on only when the template is asked for it, which puts <|think|> at the
        # start of the system prompt. Off is the template's default; said here so it is visible.
        return self._template(messages=escaped_messages, enable_thinking=False).prompt

    def generate(self, prompt, temperature, seed, max_tokens):
        llama_cpp, llm = self._llama_cpp, self._llm
        tokens = self._tokenize(prompt)
        # Start every answer from an empty KV cache. llama-cpp-python would otherwise reuse the
        # cached prefix shared with the previous prompt, which Gemma's sliding-window layers
        # cannot always restore, and which would make an answer depend on the question before it.
        llm.reset()
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(llm.ctx), True)
        llama_cpp.llama_perf_context_reset(llm.ctx)

        started = time.perf_counter()
        first_token_at = None
        pieces = []
        stream = llm.create_completion(
            tokens,  # already tokenized: given a string, llama-cpp-python would add a second <bos>
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_k=self.TOP_K,
            top_p=self.TOP_P,
            min_p=0.0,
            repeat_penalty=1.0,
            stream=True,
        )
        for piece in stream:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(piece["choices"][0]["text"])

        # llama.cpp's own counters: tokens written after the first, and the time spent on them.
        perf = llama_cpp.llama_perf_context(llm.ctx)
        tokens_per_second = perf.n_eval / (perf.t_eval_ms / 1000) if perf.t_eval_ms else 0.0
        stats = GenerationStats(len(tokens), (first_token_at or time.perf_counter()) - started, tokens_per_second)
        raw_text = "".join(pieces)
        text, warnings = self.filter_thought_tags(raw_text)
        return Generation(text, stats, warnings=warnings)

    def _tokenize(self, text):
        # special=True turns the template's markers (<|turn>, <turn|>, ...) into their control
        # tokens. add_bos=False because the template already starts with <bos>.
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def _special_text(self, token):
        return self._llm.detokenize([token], special=True).decode("utf-8")
