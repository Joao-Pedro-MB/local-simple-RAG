"""Image Document synthesis: resolves cross-references to images, extracts captions,
and optionally invokes vision description.
"""

import hashlib
import os
import re
from pathlib import Path

from rag.documents.corpus import Document, layer_of


def humanize_filename(stem: str) -> str:
    """A filename stem turned into human words by turning hyphens and underscores into spaces."""
    return re.sub(r"[-_]+", " ", stem).strip()


def matches_image_path(ref_path_str: str, image_path_str: str, doc_path_str: str) -> bool:
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


def clean_label(raw_label: str) -> str:
    """A reference label with any hash or bookkeeping stripped."""
    if not raw_label:
        return ""
    label = raw_label.split(";")[0].strip()
    label = re.sub(r"\b(?:sha256|sha1|md5|hash)[:\s]+[0-9a-fA-F]+\b", "", label).strip()
    return label


def extract_frontmatter_references(metadata: dict, image_path_str: str, doc_path_str: str) -> list[str]:
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


def _check_frontmatter_value(val_str: str, image_path_str: str, doc_path_str: str, refs: list[str]) -> None:
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


def extract_body_references(raw_text: str, image_path_str: str, doc_path_str: str) -> list[str]:
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


def extract_caption_paragraph(doc_text: str, label: str) -> str | None:
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
    relative_path: Path | str,
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
    relative_path = Path(relative_path)
    image_path_str = relative_path.as_posix()
    humanized = humanize_filename(relative_path.stem)
    pieces = [humanized]
    has_reference = False

    normalized_docs = []
    for item in text_docs_with_raw:
        if isinstance(item, tuple):
            normalized_docs.append(item)
        else:
            normalized_docs.append((item, getattr(item, "text", "")))

    for doc, raw_text in sorted(normalized_docs, key=lambda pair: pair[0].path):
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


def synthesize_image_documents(
    image_entries: list[Path | str | tuple[Path | str, bytes]],
    text_docs_with_raw,
    notes_dir=None,
    image_describer=None,
    cached_descriptions=None,
) -> tuple[list[Document], list[str]]:
    """Synthesize image Documents for all admitted image entries, borrowing cross-reference text
    from parsed text documents and generating vision descriptions when an adapter is present."""
    documents, warnings = [], []

    def _sort_key(entry):
        p = entry[0] if isinstance(entry, tuple) else entry
        return Path(p).as_posix()

    for entry in sorted(image_entries, key=_sort_key):
        if isinstance(entry, tuple):
            image_path, raw_bytes = entry
            image_path = Path(image_path)
        else:
            image_path = Path(entry)
            raw_bytes = b""
            if notes_dir is not None:
                try:
                    raw_bytes = (Path(notes_dir) / image_path).read_bytes()
                except Exception:
                    raw_bytes = b""

        cached_desc = (
            cached_descriptions(image_path, raw_bytes)
            if cached_descriptions is not None
            else None
        )

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

    return documents, warnings
