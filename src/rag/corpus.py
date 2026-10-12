import hashlib
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

import yaml

# Folders whose contents are never Documents, wherever they appear: version control,
# tool configuration and the wiki's archive.
EXCLUDED_FOLDERS = {".git", ".claude", "_archive"}

# The wiki's own bookkeeping, relative to the Notes folder: its catalog and its activity log.
EXCLUDED_FILES = {"index.md", "log.md"}

MARKDOWN_EXTENSIONS = {".md"}
TEXT_EXTENSIONS = {".txt"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}
ALL_ADMITTED_EXTENSIONS = MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS | HTML_EXTENSIONS | IMAGE_EXTENSIONS

# YAML frontmatter: a block fenced by `---` lines at the very start of a Markdown file.
# Anchored with \A so a `---` horizontal rule later in the body is left alone.
FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)^---[ \t]*$\n?", re.DOTALL | re.MULTILINE)

# An ATX heading: one to six `#`, a space, the heading text, optionally closed by more `#`.
# Setext headings (text underlined with `---`) are not recognised: a stray `---` under a line
# of prose is far more common in the Notes than a real setext heading.
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?$")

# A code fence: three or more backticks or tildes, indented by at most three spaces.
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

# A wikilink: [[target]] or [[target|alias]].
WIKILINK = re.compile(r"\[\[([^\[\]|\n]+)(?:\|([^\[\]\n]+))?\]\]")

# The two Layers. Raw sources live in the wiki's top-level raw folder; every other admitted
# Document was written within the wiki.
RAW_SOURCE = "Raw source"
WIKI_PAGE = "Wiki page"


class Heading(NamedTuple):
    offset: int  # where the heading's line starts in the Document's text
    level: int  # 1 for `#`, 2 for `##`, ...
    text: str


@dataclass(frozen=True)
class Document:
    path: str  # relative to the Notes folder, e.g. "concepts/rag.md"
    layer: str  # RAW_SOURCE or WIKI_PAGE
    title: str
    text: str
    headings: list[Heading]
    metadata: dict  # the frontmatter's fields (title, tags, sources, ...), JSON-safe
    has_vision_description: bool = False
    content_hash: str = ""
    vision_description: str | None = None






class HTMLContentExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "section",
        "article", "header", "footer", "main", "aside", "nav",
        "li", "blockquote", "pre", "tr", "hr",
    }
    IGNORE_TAGS = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_text = ""
        self._in_title = False
        self._in_head = False
        self._ignore_depth = 0
        self._in_pre = False
        self.headings = []
        self._current_heading_level = None
        self._current_heading_parts = []
        self._current_block_parts = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "head":
            self._in_head = True
        elif tag in self.IGNORE_TAGS:
            self._ignore_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "pre":
            self._in_pre = True
            self._flush_block()
        elif tag in self.BLOCK_TAGS:
            self._flush_block()
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self._current_heading_level = int(tag[1])
                self._current_heading_parts = []
        elif tag == "br":
            if not self._in_head and self._ignore_depth == 0:
                self._current_block_parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "head":
            self._in_head = False
        elif tag in self.IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "pre":
            self._in_pre = False
            self._flush_block()
        elif tag in self.BLOCK_TAGS:
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                heading_text = "".join(self._current_heading_parts).strip()
                if heading_text:
                    offset = self._current_offset()
                    self.headings.append(
                        Heading(offset, self._current_heading_level, heading_text)
                    )
                self._current_heading_level = None
                self._current_heading_parts = []
            self._flush_block()

    def handle_data(self, data):
        if self._in_title:
            self.title_text += data
        if self._in_title or self._in_head or self._ignore_depth > 0:
            return
        if self._current_heading_level is not None:
            self._current_heading_parts.append(data)
        self._current_block_parts.append(data)

    def _current_offset(self):
        if not self.blocks:
            return 0
        return sum(len(b) for b in self.blocks) + 2 * len(self.blocks)

    def _flush_block(self):
        if not self._current_block_parts:
            return
        raw = "".join(self._current_block_parts)
        self._current_block_parts = []
        if self._in_pre:
            cleaned = raw.strip("\r\n")
        else:
            cleaned = re.sub(r"[ \t\r\n]+", " ", raw).strip()
        if cleaned:
            self.blocks.append(cleaned)

    def finish(self):
        self._flush_block()
        text = "\n\n".join(self.blocks)
        return text, self.headings, self.title_text.strip()


def humanize_filename(stem):
    """A filename stem turned into human words by turning hyphens and underscores into spaces."""
    return re.sub(r"[-_]+", " ", stem).strip()


def matches_image_path(ref_path_str, image_path_str, doc_path_str):
    """Whether a path string found in a referencing Document refers to an image's path."""
    if not ref_path_str:
        return False
    ref_path_str = ref_path_str.strip()
    if ref_path_str == image_path_str:
        return True
    try:
        doc_dir = Path(doc_path_str).parent
        normalized = Path(os.path.normpath(str(doc_dir / ref_path_str))).as_posix()
        if normalized == image_path_str:
            return True
    except Exception:
        pass
    return False


def clean_label(raw_label):
    """A reference label with any hash or bookkeeping stripped."""
    if not raw_label:
        return ""
    label = raw_label.split(";")[0].strip()
    label = re.sub(r"\b(?:sha256|sha1|md5|hash)[:\s]+[0-9a-fA-F]+\b", "", label).strip()
    return label


def extract_frontmatter_references(metadata, image_path_str, doc_path_str):
    """Labels from any frontmatter field referencing the image's path."""
    refs = []
    for key, value in metadata.items():
        if isinstance(value, str):
            _check_frontmatter_value(value, image_path_str, doc_path_str, refs)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _check_frontmatter_value(item, image_path_str, doc_path_str, refs)
    return refs


def _check_frontmatter_value(val_str, image_path_str, doc_path_str, refs):
    match = re.search(r"(\S+?)(?:\s*\((.*?)\))?(?:\s|$)", val_str)
    if match:
        candidate_path = match.group(1).strip()
        if matches_image_path(candidate_path, image_path_str, doc_path_str):
            raw_label = match.group(2) or ""
            refs.append(clean_label(raw_label))
            return
    if image_path_str in val_str:
        paren_match = re.search(rf"{re.escape(image_path_str)}\s*\((.*?)\)", val_str)
        if paren_match:
            refs.append(clean_label(paren_match.group(1)))
        else:
            refs.append("")


def extract_body_references(raw_text, image_path_str, doc_path_str):
    """Labels from any markdown image, link, wikilink or HTML img referencing the image's path."""
    refs = []
    for m in re.finditer(r"!\[(.*?)\]\((.*?)\)", raw_text):
        target = m.group(2).split()[0].strip("<>")
        if matches_image_path(target, image_path_str, doc_path_str):
            refs.append(clean_label(m.group(1)))
    for m in re.finditer(r"\[\[([^\[\]|\n]+)(?:\|([^\[\]\n]+))?\]\]", raw_text):
        target = m.group(1).strip()
        if matches_image_path(target, image_path_str, doc_path_str):
            label = m.group(2) or ""
            refs.append(clean_label(label))
    for m in re.finditer(r"<img\s+[^>]*src=[\"'](.*?)[\"'][^>]*>", raw_text, re.IGNORECASE):
        src = m.group(1).strip()
        if matches_image_path(src, image_path_str, doc_path_str):
            alt_match = re.search(r"alt=[\"'](.*?)[\"']", m.group(0), re.IGNORECASE)
            label = alt_match.group(1) if alt_match else ""
            refs.append(clean_label(label))
    for m in re.finditer(r"(?<!!)\[(.*?)\]\((.*?)\)", raw_text):
        target = m.group(2).split()[0].strip("<>")
        if matches_image_path(target, image_path_str, doc_path_str):
            refs.append(clean_label(m.group(1)))
    if not refs and image_path_str in raw_text:
        refs.append("")
    return refs


def extract_caption_paragraph(doc_text, label):
    """The matching 'Figure N:' caption paragraph when the label names a figure number."""
    if not label:
        return None
    fig_match = re.search(r"\b(?:fig\.?|figure)\s*(\d+)\b", label, re.IGNORECASE)
    if not fig_match:
        return None
    fig_num = fig_match.group(1)
    pattern = re.compile(rf"^\s*(?:figure|fig\.?)\s*{fig_num}\s*:", re.IGNORECASE)
    paragraphs = re.split(r"\n\s*\n", doc_text)
    for para in paragraphs:
        if pattern.match(para):
            return para.strip()
    return None


def build_image_document(
    relative_path,
    text_docs_with_raw,
    image_describer=None,
    notes_dir=None,
    content_hash="",
    cached_description=None,
    raw_bytes=b"",
):
    """An image Document: represented by its humanized filename plus, for each Document that
    references it, that Document's title, reference label (bookkeeping stripped) and matching
    caption paragraph. If unreferenced, flagged as weakly described.
    When an image describer is supplied, its generated description is included alongside
    the borrowed text."""
    image_path_str = relative_path.as_posix()
    humanized = humanize_filename(relative_path.stem)
    pieces = [humanized]
    has_reference = False

    for doc, raw_text in sorted(text_docs_with_raw, key=lambda pair: pair[0].path):
        fm_refs = extract_frontmatter_references(doc.metadata, image_path_str, doc.path)
        body_refs = extract_body_references(raw_text, image_path_str, doc.path)
        all_refs = fm_refs + body_refs
        if not all_refs:
            continue
        has_reference = True
        label = next((r for r in all_refs if r), "")
        caption = extract_caption_paragraph(doc.text, label)

        pieces.append(doc.title)
        if label and label != doc.title:
            pieces.append(label)
        if caption and caption != label:
            pieces.append(caption)

    has_vision = False
    vision_desc = None
    if cached_description is not None:
        if cached_description.strip():
            pieces.append(cached_description.strip())
            has_vision = True
            vision_desc = cached_description.strip()
    elif image_describer is not None and notes_dir is not None:
        full_path = Path(notes_dir) / relative_path
        desc = image_describer.describe(full_path)
        if desc and desc.strip():
            pieces.append(desc.strip())
            has_vision = True
            vision_desc = desc.strip()

    if not raw_bytes and notes_dir is not None:
        try:
            raw_bytes = (Path(notes_dir) / relative_path).read_bytes()
        except Exception:
            raw_bytes = b""

    text = "\n\n".join(pieces)
    content_hash = hashlib.sha256(raw_bytes + text.encode("utf-8")).hexdigest()
    doc = Document(
        path=image_path_str,
        layer=layer_of(relative_path),
        title=relative_path.stem,
        text=text,
        headings=[],
        metadata={},
        has_vision_description=has_vision,
        content_hash=content_hash,
        vision_description=vision_desc,
    )
    warning = None
    if not has_reference:
        warning = f"{image_path_str}: no Document references this image, so it is indexed only by its filename."
    return doc, warning


def load_documents(notes_dir, image_describer=None, existing_index=None):
    """Every admitted Document in the Notes folder, every path left out, and warnings for weakly
    described images, all sorted.

    An excluded folder is listed once (as "name/") and never walked into, so the thousands
    of files under .git neither slow the walk nor swamp the list.
    """
    notes_dir = Path(notes_dir)
    documents, excluded, warnings = [], [], []
    text_docs_with_raw = []
    image_entries = []

    for folder, subfolders, files in notes_dir.walk():
        for name in [name for name in subfolders if name in EXCLUDED_FOLDERS]:
            subfolders.remove(name)  # pruning the list in place stops the walk going in
            excluded.append((folder / name).relative_to(notes_dir).as_posix() + "/")
        for name in files:
            relative_path = (folder / name).relative_to(notes_dir)
            if is_admitted(relative_path):
                file_path = folder / name
                raw_bytes = file_path.read_bytes()
                content_hash = hashlib.sha256(raw_bytes).hexdigest()
                suffix = relative_path.suffix.lower()
                if suffix in MARKDOWN_EXTENSIONS:
                    raw_text = raw_bytes.decode("utf-8", errors="replace")
                    doc = read_markdown(relative_path, raw_text, content_hash=content_hash)
                    documents.append(doc)
                    text_docs_with_raw.append((doc, raw_text))
                elif suffix in TEXT_EXTENSIONS:
                    raw_text = raw_bytes.decode("utf-8", errors="replace")
                    doc = read_plain_text(relative_path, raw_text, content_hash=content_hash)
                    documents.append(doc)
                    text_docs_with_raw.append((doc, raw_text))
                elif suffix in HTML_EXTENSIONS:
                    raw_text = raw_bytes.decode("utf-8", errors="replace")
                    doc = read_html(relative_path, raw_text, content_hash=content_hash)
                    documents.append(doc)
                    text_docs_with_raw.append((doc, raw_text))
                elif suffix in IMAGE_EXTENSIONS:
                    image_entries.append((relative_path, raw_bytes))
            else:
                excluded.append(relative_path.as_posix())

    for image_path, raw_bytes in sorted(image_entries, key=lambda pair: pair[0]):
        cached_desc = None
        if existing_index is not None:
            image_path_str = image_path.as_posix()
            old_hash = existing_index.content_hashes.get(image_path_str)
            existing_chunks = [c for c in existing_index.chunks if c.document == image_path_str]
            old_doc_text = existing_chunks[0].text if existing_chunks else None
            if (
                old_hash
                and old_doc_text is not None
                and hashlib.sha256(raw_bytes + old_doc_text.encode("utf-8")).hexdigest() == old_hash
            ):
                cached_desc = existing_index.vision_descriptions.get(image_path_str, "")
        img_doc, warning = build_image_document(
            image_path,
            text_docs_with_raw,
            image_describer=image_describer,
            notes_dir=notes_dir,
            raw_bytes=raw_bytes,
            cached_description=cached_desc,
        )
        documents.append(img_doc)
        if warning:
            warnings.append(warning)

    return sorted(documents, key=lambda document: document.path), sorted(excluded), sorted(warnings)


def read_html(relative_path, html_text, content_hash=""):
    """An HTML Document: visible text only (scripts and styles dropped), titled by its title
    element, its first top-level heading or its filename."""
    parser = HTMLContentExtractor()
    parser.feed(html_text)
    text, headings, parsed_title = parser.finish()
    first_top_level_heading = next((h.text for h in headings if h.level == 1), None)
    title = parsed_title or first_top_level_heading or relative_path.stem
    return Document(
        path=relative_path.as_posix(),
        layer=layer_of(relative_path),
        title=title,
        text=text,
        headings=headings,
        metadata={},
        content_hash=content_hash,
    )


def read_plain_text(relative_path, text, content_hash=""):
    """A plain-text Document: admitted as-is, titled by its filename."""
    return Document(
        path=relative_path.as_posix(),
        layer=layer_of(relative_path),
        title=relative_path.stem,
        text=text,
        headings=[],
        metadata={},
        content_hash=content_hash,
    )


def read_markdown(relative_path, markdown, content_hash=""):
    """A Markdown Document: frontmatter lifted out of its text into metadata (left in, it would
    pollute the embeddings), wikilinks rendered, and titled by the first of its frontmatter
    title, its first top-level heading and its filename."""
    frontmatter = FRONTMATTER.match(markdown)
    metadata = lift_frontmatter(frontmatter.group(1)) if frontmatter else {}
    body = markdown[frontmatter.end() :] if frontmatter else markdown
    text, headings = render_body(body)
    frontmatter_title = str(metadata.get("title") or "").strip()
    first_top_level_heading = next((h.text for h in headings if h.level == 1), None)
    return Document(
        path=relative_path.as_posix(),
        layer=layer_of(relative_path),
        title=frontmatter_title or first_top_level_heading or relative_path.stem,
        text=text,
        headings=headings,
        metadata=metadata,
        content_hash=content_hash,
    )



def lift_frontmatter(block):
    """The frontmatter's fields as a dict, even when strict YAML rejects the block.

    Some Raw sources quote their title and then keep writing
    (`title: "Prediction and Entropy of Printed English" — C. E. Shannon`), which makes YAML
    reject the whole block. Rather than lose every field, each top-level `key: value` line is
    then parsed on its own, and a line YAML still rejects keeps its value as plain text.
    """
    try:
        fields = yaml.safe_load(block)
    except yaml.YAMLError:
        fields = None
    if not isinstance(fields, dict):
        fields = {}
        for line in block.splitlines():
            key, colon, value = line.partition(":")
            if not colon or line[:1].isspace():  # not a top-level field
                continue
            try:
                parsed = yaml.safe_load(line)
            except yaml.YAMLError:
                parsed = {key.strip(): value.strip()}
            if isinstance(parsed, dict):
                fields.update(parsed)
    return json_safe(fields)


def json_safe(value):
    """The value with every date YAML parsed (`created: 2026-09-27`) turned back into text,
    so the metadata can be saved as JSON and reads the same after the Index is reloaded."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def render_body(body):
    """The body as clean text, and the headings found in it.

    Outside fenced code, each wikilink is rendered as the words a reader sees: its alias if
    it has one, otherwise its target, so `[[deep-modules|deep module]]` reads "deep module".
    Heading offsets are taken in the rendered text, the text the Chunker will cut.
    """
    lines, headings, offset = [], [], 0
    for line, is_code in lines_with_fences(body):
        if not is_code:
            line = WIKILINK.sub(lambda link: (link.group(2) or link.group(1)).strip(), line)
            heading = HEADING.match(line.rstrip())
            if heading:
                headings.append(Heading(offset, len(heading.group(1)), heading.group(2)))
        lines.append(line)
        offset += len(line)
    return "".join(lines), headings


def lines_with_fences(text):
    """Each line of the text (line ending kept) and whether it belongs to fenced code.

    Inside a fence, a `#` line is a shell comment or a quoted Markdown file, not a heading.
    A fence closes only on a line of the same character, at least as long as the opening
    fence and with nothing after it, so a four-backtick fence can quote three-backtick ones.
    """
    fence = None  # the opening fence, e.g. "````", while inside fenced code
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence is None:
            if marker:
                fence = marker.group(1)
            yield line, fence is not None
        else:
            closing = marker and line.strip() == marker.group(1)
            if closing and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            yield line, True


def is_admitted(relative_path):
    suffix = relative_path.suffix.lower()
    return (
        suffix in ALL_ADMITTED_EXTENSIONS
        and relative_path.as_posix() not in EXCLUDED_FILES
    )


def layer_of(relative_path):
    return RAW_SOURCE if relative_path.parts[0] == "raw" else WIKI_PAGE
