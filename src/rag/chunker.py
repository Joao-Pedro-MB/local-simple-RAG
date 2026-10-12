"""Chunker: cuts Documents into Chunks, measuring size with the chunk tokenizer.

A chunk tokenizer has one method, `spans(text) -> [(start, end), ...]`: the character range of
every token. Counting spans measures a Chunk; slicing the text by them cuts one.
"""

from dataclasses import dataclass
from pathlib import Path

from rag.corpus import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Chunk:
    document: str  # the Document's path relative to the Notes folder
    text: str  # the Chunk text as it stands in the Document's clean text, without the label
    token_count: int
    layer: str = ""  # the Document's Layer: Wiki page or Raw source
    title: str = ""  # the Document's title
    section: str = ""  # the heading in effect where the Chunk starts; the title when there is none
    has_vision_description: bool = False

    @property
    def is_image(self):
        return Path(self.document).suffix.lower() in IMAGE_EXTENSIONS

    @property
    def label(self):
        """`Title › Section`, naming where the Chunk came from in the Document; the title
        alone when the section is the title itself (no heading yet, or a heading equal to it)."""
        return self.title if self.section == self.title else f"{self.title} › {self.section}"

    @property
    def text_to_embed(self):
        """What the Embedder reads: the label, then the text. Text that never names its
        own topic still carries where it came from into its vector."""
        return f"{self.label}\n\n{self.text}"


@dataclass(frozen=True)
class FixedWindows:
    """Fixed windows of `size` tokens, each starting `size - overlap` tokens after the last.

    The overlap means a sentence cut at one window's end still appears whole in the next.
    """

    size: int = 350
    overlap: int = 50
    strategy: str = "fixed"

    def chunk(self, document, tokenizer):
        spans = tokenizer.spans(document.text)
        chunks = []
        start = 0
        while start < len(spans):
            end = min(start + self.size, len(spans))
            # Slice the original text by character offsets instead of decoding the tokens:
            # bge's tokenizer is uncased, so decoding would lowercase the whole Chunk.
            text = document.text[spans[start][0] : spans[end - 1][1]]
            section = section_at(document, spans[start][0])
            chunks.append(
                Chunk(
                    document.path,
                    text,
                    end - start,
                    document.layer,
                    document.title,
                    section,
                    has_vision_description=getattr(document, "has_vision_description", False),
                )
            )
            if end == len(spans):
                break
            start = end - self.overlap
        return chunks



@dataclass(frozen=True)
class HeadingAwareChunking:
    """Cuts Documents at headings.

    A section longer than `size` tokens falls back to fixed windows with the same `size`
    and `overlap`. A section below `min_size` tokens is merged into a neighbouring section.
    """

    size: int = 350
    overlap: int = 50
    min_size: int = 100
    strategy: str = "heading-aware"

    def chunk(self, document, tokenizer):
        spans = tokenizer.spans(document.text)
        if not spans:
            return []

        heading_offsets = sorted(
            {h.offset for h in document.headings if 0 < h.offset < len(document.text)}
        )
        raw_sections = []
        current_start = 0
        current_heading_idx = 0

        for token_idx, (t_start, t_end) in enumerate(spans):
            if (
                current_heading_idx < len(heading_offsets)
                and t_start >= heading_offsets[current_heading_idx]
            ):
                if token_idx > current_start:
                    raw_sections.append((current_start, token_idx))
                    current_start = token_idx
                while (
                    current_heading_idx < len(heading_offsets)
                    and t_start >= heading_offsets[current_heading_idx]
                ):
                    current_heading_idx += 1

        if current_start < len(spans):
            raw_sections.append((current_start, len(spans)))

        sections = []
        for start_idx, end_idx in raw_sections:
            count = end_idx - start_idx
            if not sections:
                sections.append((start_idx, end_idx))
            elif sections[-1][1] - sections[-1][0] < self.min_size:
                prev_start, _ = sections.pop()
                sections.append((prev_start, end_idx))
            elif count < self.min_size:
                prev_start, _ = sections.pop()
                sections.append((prev_start, end_idx))
            else:
                sections.append((start_idx, end_idx))

        if len(sections) > 1 and (sections[-1][1] - sections[-1][0] < self.min_size):
            last_start, last_end = sections.pop()
            prev_start, _ = sections.pop()
            sections.append((prev_start, last_end))

        chunks = []
        for start_idx, end_idx in sections:
            count = end_idx - start_idx
            if count <= self.size:
                text = document.text[spans[start_idx][0] : spans[end_idx - 1][1]]
                section = section_at(document, spans[start_idx][0])
                chunks.append(
                    Chunk(
                        document.path,
                        text,
                        count,
                        document.layer,
                        document.title,
                        section,
                        has_vision_description=getattr(document, "has_vision_description", False),
                    )
                )
            else:
                pos = start_idx
                while pos < end_idx:
                    window_end = min(pos + self.size, end_idx)
                    text = document.text[spans[pos][0] : spans[window_end - 1][1]]
                    section = section_at(document, spans[pos][0])
                    chunks.append(
                        Chunk(
                            document.path,
                            text,
                            window_end - pos,
                            document.layer,
                            document.title,
                            section,
                            has_vision_description=getattr(document, "has_vision_description", False),
                        )
                    )
                    if window_end == end_idx:
                        break
                    pos = max(pos + 1, window_end - self.overlap)

        return chunks



def section_at(document, offset):
    """The heading in effect at a character offset: the last one starting at or before it.
    Before the Document's first heading there is none, and the title stands in for it."""
    section = document.title
    for heading in document.headings:
        if heading.offset > offset:
            break
        section = heading.text
    return section


class BgeTokenizer:
    """bge-small's own tokenizer, always used to measure Chunks whichever Embedder is active,
    so swapping embedding models changes only the vectors, never the Chunks."""

    name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        from transformers import AutoTokenizer  # imported here so the fast tests never load it

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)

    def spans(self, text):
        # No [CLS]/[SEP]: only the text's own tokens count. verbose=False silences the
        # "longer than 512 tokens" warning, since whole Documents are tokenized before cutting.
        encoding = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return encoding["offset_mapping"]
