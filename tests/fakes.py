"""Test adapters for the Pipeline's ports: no models are downloaded or loaded."""

import re
import zlib

import numpy as np

from rag.adapters.generator import Generation, GenerationStats


class FakeEmbedder:
    """Embeds text as a bag of words, so cosine similarity is high exactly when words are shared.

    Each lowercased word is hashed (crc32, stable across runs) into its own dimension, and the
    vector is L2-normalized like a real Embedder's. Query and Document paths are identical.
    Every text given to the Document path is kept in `embedded_documents`.
    """

    dimension = 1 << 14

    def __init__(self, model_id="fake-embedder", max_input_tokens=512):
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.embedded_documents = []

    def embed_documents(self, texts):
        self.embedded_documents.extend(texts)
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimension] = 1.0
        return vector / max(np.linalg.norm(vector), 1e-12)


class WordTokenizer:
    """A chunk tokenizer where every whitespace-separated word is one token."""

    name = "words"

    def spans(self, text):
        return [match.span() for match in re.finditer(r"\S+", text)]


FAKE_STATS = {"time_to_first_token": 1.5, "tokens_per_second": 25.0}


class FakeGenerator:
    """Counts every word as a token, records each prompt it is given and answers with fixed text.

    Its chat template simply writes each message under its role, with no special tokens.
    """

    def __init__(self, text="Cats purr when they are content.", context_window=8192, warnings=None):
        self.text = text
        self.context_window = context_window
        self.warnings = list(warnings) if warnings is not None else []
        self.calls = []  # one dict per generate(): prompt, temperature, seed, max_tokens

    @property
    def prompt(self):
        """The prompt of the latest call."""
        return self.calls[-1]["prompt"]

    def count_tokens(self, text):
        return len(text.split())

    def render(self, messages):
        return "".join(f"{message['role']}:\n{message['content']}\n\n" for message in messages) + "model:\n"

    def generate(self, prompt, temperature, seed, max_tokens):
        self.calls.append({"prompt": prompt, "temperature": temperature, "seed": seed, "max_tokens": max_tokens})
        return Generation(
            self.text,
            GenerationStats(prompt_tokens=self.count_tokens(prompt), **FAKE_STATS),
            warnings=list(self.warnings),
        )


class FakeImageDescriber:
    """Returns fixed text per image, recording each path it was asked to describe."""

    def __init__(self, descriptions=None, default="A generated vision description."):
        self.descriptions = descriptions if isinstance(descriptions, dict) else {}
        if isinstance(descriptions, str):
            self.default = descriptions
        else:
            self.default = default
        self.calls = []

    def describe(self, image_path):
        from pathlib import Path

        path_str = str(image_path)
        self.calls.append(path_str)
        if path_str in self.descriptions:
            return self.descriptions[path_str]
        p = Path(image_path)
        if p.name in self.descriptions:
            return self.descriptions[p.name]
        for key, text in self.descriptions.items():
            if path_str.endswith(key):
                return text
        return self.default

