"""Model adapters for embedding, vision description, and answer generation."""

from rag.adapters.describer import GemmaImageDescriber
from rag.adapters.embedder import BgeSmallEmbedder, Qwen3Embedder
from rag.adapters.generator import GemmaGenerator, Generation, GenerationStats

__all__ = [
    "BgeSmallEmbedder",
    "GemmaGenerator",
    "GemmaImageDescriber",
    "Generation",
    "GenerationStats",
    "Qwen3Embedder",
]
