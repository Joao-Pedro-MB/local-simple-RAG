from pathlib import Path

from rag.documents.corpus import Document
from rag.documents.image_synthesis import (
    clean_label,
    extract_body_references,
    extract_caption_paragraph,
    extract_frontmatter_references,
    humanize_filename,
    matches_image_path,
    build_image_document,
    synthesize_image_documents,
)
from fakes import FakeImageDescriber


def test_humanize_filename_converts_hyphens_and_underscores_to_spaces():
    assert humanize_filename("benedetto-2002-language_tree-fig1") == "benedetto 2002 language tree fig1"
    assert humanize_filename("standalone-diagram") == "standalone diagram"


def test_clean_label_strips_hashes_and_bookkeeping():
    assert clean_label("Fig. 1, the language tree; sha256 7909cd5e") == "Fig. 1, the language tree"
    assert clean_label("Diagram; md5: abc12345") == "Diagram"
    assert clean_label("Just a label") == "Just a label"
    assert clean_label("") == ""


def test_matches_image_path_resolves_exact_and_relative_paths():
    assert matches_image_path("assets/arch.png", "assets/arch.png", "notes/doc.md")
    # Relative path from doc location:
    # doc is in "concepts/sub/note.md", image is "concepts/sub/arch.png"
    assert matches_image_path("arch.png", "concepts/sub/arch.png", "concepts/sub/note.md")
    assert matches_image_path("../arch.png", "concepts/arch.png", "concepts/sub/note.md")
    # Non matching
    assert not matches_image_path("other.png", "assets/arch.png", "notes/doc.md")
    assert not matches_image_path("", "assets/arch.png", "notes/doc.md")


def test_extract_frontmatter_references():
    metadata = {
        "figure": "raw/assets/tree.png (Fig. 1, the tree; sha256 abc)",
        "figures": ["raw/assets/other.png", "raw/assets/tree.png (Figure 2)"],
        "unrelated": "something else",
    }
    refs = extract_frontmatter_references(metadata, "raw/assets/tree.png", "paper.md")
    assert refs == ["Fig. 1, the tree", "Figure 2"]


def test_extract_body_references_handles_markdown_wikilinks_and_html():
    body = (
        "Here is an image: ![Overview Diagram](assets/overview.png)\n"
        "And a wikilink: [[assets/overview.png|Wikilink Alt]]\n"
        'And html: <img src="assets/overview.png" alt="HTML Alt" />\n'
        "And regular link: [Link Alt](assets/overview.png)\n"
    )
    refs = extract_body_references(body, "assets/overview.png", "notes/index.md")
    assert refs == ["Overview Diagram", "Wikilink Alt", "HTML Alt", "Link Alt"]


def test_extract_caption_paragraph_matches_figure_number():
    doc_text = (
        "Some introduction text.\n\n"
        "Figure 1: Overview of system architecture and data flows.\n\n"
        "Figure 2: Benchmark evaluation results.\n\n"
        "Conclusion text."
    )
    assert extract_caption_paragraph(doc_text, "Fig. 1, system overview") == (
        "Figure 1: Overview of system architecture and data flows."
    )
    assert extract_caption_paragraph(doc_text, "Figure 2") == (
        "Figure 2: Benchmark evaluation results."
    )
    assert extract_caption_paragraph(doc_text, "Figure 3") is None
    assert extract_caption_paragraph(doc_text, "Unrelated label") is None


def test_build_image_document_assembles_borrowed_text():
    referencing_doc = Document(
        path="concepts/rag.md",
        layer="Wiki page",
        title="RAG Architecture",
        text=(
            "An overview of RAG.\n\n"
            "Figure 1: Detailed pipeline flow of RAG components.\n"
        ),
        headings=[],
        metadata={"figure": "assets/pipeline.png (Fig. 1: Architecture)"},
    )
    raw_text = (
        "---\n"
        "figure: assets/pipeline.png (Fig. 1: Architecture)\n"
        "---\n"
        "An overview of RAG.\n\n"
        "Figure 1: Detailed pipeline flow of RAG components.\n"
    )

    doc, warning = build_image_document(
        Path("assets/pipeline.png"),
        [(referencing_doc, raw_text)],
        raw_bytes=b"fake-bytes",
    )

    assert warning is None
    assert doc.path == "assets/pipeline.png"
    assert doc.title == "pipeline"
    assert doc.text == (
        "pipeline\n\n"
        "RAG Architecture\n\n"
        "Fig. 1: Architecture\n\n"
        "Figure 1: Detailed pipeline flow of RAG components."
    )
    assert doc.has_vision_description is False
    assert doc.content_hash != ""


def test_build_image_document_unreferenced_produces_warning_and_filename_only():
    doc, warning = build_image_document(
        Path("assets/standalone.png"),
        [],
        raw_bytes=b"fake-bytes",
    )

    assert warning == "assets/standalone.png: no Document references this image, so it is indexed only by its filename."
    assert doc.path == "assets/standalone.png"
    assert doc.title == "standalone"
    assert doc.text == "standalone"
    assert doc.has_vision_description is False


def test_build_image_document_invokes_vision_describer(tmp_path):
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(b"image-content")

    describer = FakeImageDescriber({"diagram.png": "A diagram with boxes and arrows."})
    doc, warning = build_image_document(
        Path("diagram.png"),
        [],
        image_describer=describer,
        notes_dir=tmp_path,
        raw_bytes=b"image-content",
    )

    assert doc.text == "diagram\n\nA diagram with boxes and arrows."
    assert doc.has_vision_description is True
    assert doc.vision_description == "A diagram with boxes and arrows."
    assert len(describer.calls) == 1


def test_build_image_document_uses_cached_description_without_calling_describer():
    describer = FakeImageDescriber()
    doc, _ = build_image_document(
        Path("diagram.png"),
        [],
        image_describer=describer,
        cached_description="Pre-existing cached description.",
        raw_bytes=b"image-content",
    )

    assert "Pre-existing cached description." in doc.text
    assert doc.has_vision_description is True
    assert len(describer.calls) == 0


def test_synthesize_image_documents_processes_entries_and_collects_warnings():
    image_entries = [
        (Path("assets/unreferenced.png"), b"unref-bytes"),
        (Path("assets/referenced.png"), b"ref-bytes"),
    ]
    ref_doc = Document(
        path="notes.md",
        layer="Wiki page",
        title="Notes",
        text="A note.",
        headings=[],
        metadata={},
    )
    raw_text = "![Referenced](assets/referenced.png)"

    docs, warnings = synthesize_image_documents(
        image_entries,
        [(ref_doc, raw_text)],
    )

    assert len(docs) == 2
    paths = [d.path for d in docs]
    assert "assets/unreferenced.png" in paths
    assert "assets/referenced.png" in paths
    assert warnings == [
        "assets/unreferenced.png: no Document references this image, so it is indexed only by its filename."
    ]


def test_synthesize_image_documents_uses_callable_cached_descriptions():
    image_entries = [
        (Path("assets/diagram.png"), b"diagram-bytes"),
    ]

    def mock_lookup(image_path, raw_bytes):
        assert image_path == Path("assets/diagram.png")
        assert raw_bytes == b"diagram-bytes"
        return "Cached description from callable."

    docs, _ = synthesize_image_documents(
        image_entries,
        [],
        cached_descriptions=mock_lookup,
    )

    assert len(docs) == 1
    assert "Cached description from callable." in docs[0].text
    assert docs[0].has_vision_description is True
    assert docs[0].vision_description == "Cached description from callable."

