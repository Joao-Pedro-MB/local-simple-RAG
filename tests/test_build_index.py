import pytest

from fakes import FakeEmbedder, FakeGenerator, FakeImageDescriber, WordTokenizer
from rag.chunker import FixedWindows, HeadingAwareChunking
from rag.index import Index
from rag.pipeline import ask, build_index


def build(notes_dir, index_dir, **options):
    index, _report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), **options)
    return index


def documents_in(index):
    return {chunk.document for chunk in index.chunks}


def test_the_wikis_own_index_and_log_are_not_documents(notes, index_dir):
    notes_dir = notes({
        "index.md": "Catalog of every page in the wiki.",
        "log.md": "2026-10-11 ingested a new source.",
        "concepts/rag.md": "Retrieval augmented generation.",
        # Only the wiki's own index and log are bookkeeping; a clipped web page may well be
        # saved as index.md further down and is a real Document.
        "raw/articles/index.md": "A clipped article about embeddings.",
    })

    index = build(notes_dir, index_dir)

    assert documents_in(index) == {"concepts/rag.md", "raw/articles/index.md"}


def test_version_control_tool_configuration_and_archive_folders_are_not_read(notes, index_dir):
    notes_dir = notes({
        ".git/description.md": "Unnamed repository.",
        ".claude/commands/ingest.md": "Ingest a new source into the wiki.",
        "_archive/old-summary.md": "A superseded chapter summary.",
        "learning/_archive/draft.md": "An archived draft.",
        "learning/chapter-1.md": "What I learned in chapter one.",
    })

    index = build(notes_dir, index_dir)

    assert documents_in(index) == {"learning/chapter-1.md"}


def test_only_markdown_documents_are_admitted(notes, index_dir):
    notes_dir = notes({
        "SCHEMA.md": "How the wiki is organised.",
        "raw/papers/attention.pdf": "%PDF-1.7 binary content",
        "raw/data.json": '{"not": "a document"}',
    })

    index = build(notes_dir, index_dir)

    assert documents_in(index) == {"SCHEMA.md"}


def test_plain_text_documents_are_admitted_as_they_are(notes, index_dir):
    notes_dir = notes({
        "notes/plain.txt": "Some plain notes.",
        "raw/transcripts/talk.txt": "A captured transcript of a talk.",
    })

    index = build(notes_dir, index_dir)

    assert documents_in(index) == {"notes/plain.txt", "raw/transcripts/talk.txt"}
    chunks_by_doc = {chunk.document: chunk for chunk in index.chunks}

    plain_chunk = chunks_by_doc["notes/plain.txt"]
    assert plain_chunk.text == "Some plain notes."
    assert plain_chunk.title == "plain"
    assert plain_chunk.section == "plain"
    assert plain_chunk.layer == "Wiki page"

    talk_chunk = chunks_by_doc["raw/transcripts/talk.txt"]
    assert talk_chunk.text == "A captured transcript of a talk."
    assert talk_chunk.title == "talk"
    assert talk_chunk.section == "talk"
    assert talk_chunk.layer == "Raw source"


def test_html_documents_contribute_only_visible_text_dropping_scripts_and_styles(notes, index_dir):
    notes_dir = notes({
        "exercises/tutorial.html": (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "  <title>Tutorial Page</title>\n"
            "  <style>\n"
            "    body { color: red; font-family: sans-serif; }\n"
            "  </style>\n"
            "  <script>\n"
            "    console.log('Ignore script');\n"
            "  </script>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>Getting Started</h1>\n"
            "  <p>This is the first paragraph with <b>bold</b> text.</p>\n"
            "  <script>alert('inline script');</script>\n"
            "  <h2>Next Steps</h2>\n"
            "  <p>Follow along with the steps.</p>\n"
            "</body>\n"
            "</html>\n"
        ),
        "raw/articles/page.html": (
            "<html><body>"
            "<p>Raw HTML article without a title tag.</p>"
            "</body></html>"
        ),
    })

    index = build(notes_dir, index_dir)

    assert documents_in(index) == {"exercises/tutorial.html", "raw/articles/page.html"}

    tutorial_chunk = next(c for c in index.chunks if c.document == "exercises/tutorial.html")
    assert "body {" not in tutorial_chunk.text
    assert "Ignore script" not in tutorial_chunk.text
    assert "inline script" not in tutorial_chunk.text
    assert "Getting Started" in tutorial_chunk.text
    assert "first paragraph with bold text." in tutorial_chunk.text
    assert tutorial_chunk.title == "Tutorial Page"
    assert tutorial_chunk.layer == "Wiki page"

    raw_chunk = next(c for c in index.chunks if c.document == "raw/articles/page.html")
    assert raw_chunk.title == "page"
    assert raw_chunk.text == "Raw HTML article without a title tag."
    assert raw_chunk.layer == "Raw source"


def test_html_title_prefers_title_tag_then_first_top_level_heading_then_filename(notes, index_dir):
    notes_dir = notes({
        "with-title.html": "<title>Page Title</title><h1>Heading Title</h1><p>Content</p>",
        "with-h1.html": "<h1>Heading Title</h1><p>Content</p>",
        "no-title-or-h1.html": "<h2>Section Title</h2><p>Content</p>",
    })

    index = build(notes_dir, index_dir)

    assert titles_in(index) == {
        "with-title.html": "Page Title",
        "with-h1.html": "Heading Title",
        "no-title-or-h1.html": "no-title-or-h1",
    }


def test_html_headings_track_sections_for_chunk_labels(notes, index_dir):
    notes_dir = notes({
        "guide.html": (
            "<title>Guide</title>\n"
            "<h1>Introduction</h1>\n"
            "<p>Introductory words.</p>\n"
            "<h2>Details</h2>\n"
            "<p>Detailed explanations.</p>\n"
        ),
    })

    index = build(notes_dir, index_dir, chunking=FixedWindows(size=5, overlap=0))

    labels = [chunk.label for chunk in index.chunks]
    assert "Guide › Introduction" in labels
    assert "Guide › Details" in labels


def test_unreferenced_images_are_indexed_by_humanized_filename_and_flagged_in_build_report(notes, index_dir):
    notes_dir = notes({
        "raw/assets/unreferenced-tree.png": b"",
        "assets/standalone-diagram.jpg": b"",
        "notes/general.md": "A note that does not mention any images.",
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    assert "raw/assets/unreferenced-tree.png" in documents_in(index)
    assert "assets/standalone-diagram.jpg" in documents_in(index)

    chunks = {chunk.document: chunk for chunk in index.chunks}
    tree_chunk = chunks["raw/assets/unreferenced-tree.png"]
    assert tree_chunk.text == "unreferenced tree"
    assert tree_chunk.title == "unreferenced-tree"
    assert tree_chunk.layer == "Raw source"

    diag_chunk = chunks["assets/standalone-diagram.jpg"]
    assert diag_chunk.text == "standalone diagram"
    assert diag_chunk.title == "standalone-diagram"
    assert diag_chunk.layer == "Wiki page"

    assert report.warnings == [
        "assets/standalone-diagram.jpg: no Document references this image, so it is indexed only by its filename.",
        "raw/assets/unreferenced-tree.png: no Document references this image, so it is indexed only by its filename.",
    ]


def test_image_borrowed_text_assembles_humanized_filename_referencing_title_label_and_caption(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Zipping measures how similar two texts are.\n\n"
            "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    assert "raw/assets/benedetto-2002-language-tree-fig1.png" in documents_in(index)
    image_chunk = next(c for c in index.chunks if c.document == "raw/assets/benedetto-2002-language-tree-fig1.png")

    assert image_chunk.text == (
        "benedetto 2002 language tree fig1\n\n"
        '"Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n\n'
        "Fig. 1, the language tree\n\n"
        "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree."
    )
    assert image_chunk.layer == "Raw source"
    assert image_chunk.title == "benedetto-2002-language-tree-fig1"
    assert report.warnings == []


def test_image_referenced_in_body_without_figure_number_borrows_title_and_label(notes, index_dir):
    notes_dir = notes({
        "concepts/architecture.md": (
            "# Architecture\n\n"
            "![System Overview Diagram](assets/system.png)\n\n"
            "Details of the system.\n"
        ),
        "assets/system.png": b"",
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    assert "assets/system.png" in documents_in(index)
    image_chunk = next(c for c in index.chunks if c.document == "assets/system.png")

    assert image_chunk.text == (
        "system\n\n"
        "Architecture\n\n"
        "System Overview Diagram"
    )
    assert image_chunk.layer == "Wiki page"
    assert report.warnings == []


def test_asking_about_benedetto_language_tree_retrieves_the_image_chunk(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Zipping measures how similar two texts are.\n\n"
            "Figure 1: Language Tree: This figure illustrates the phylogenetic tree.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
        "concepts/rag.md": "Retrieval augmented generation uses embeddings.",
    })
    embedder = FakeEmbedder()
    build_index(notes_dir, index_dir, embedder, WordTokenizer())

    from fakes import FakeGenerator
    from rag.pipeline import ask

    ans = ask("benedetto language tree figure", index_dir, embedder, FakeGenerator(), k=1)

    assert ans.context[0].chunk.document == "raw/assets/benedetto-2002-language-tree-fig1.png"


def test_yaml_frontmatter_is_left_out_of_chunk_text(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": "---\ntitle: RAG\ntags: [retrieval]\n---\n\nRetrieval augmented generation.\n",
        "concepts/stub.md": "---\ntitle: A page with nothing written yet\n---\n",
    })

    index = build(notes_dir, index_dir)

    assert [chunk.text for chunk in index.chunks] == ["Retrieval augmented generation."]


def test_frontmatter_is_lifted_into_the_documents_metadata_with_dates_stored_as_text(notes, index_dir):
    notes_dir = notes({
        "concepts/deep-modules.md": (
            "---\n"
            "title: Deep Modules\n"
            "created: 2026-09-27\n"
            "tags: [system-design, practice]\n"
            'sources: [raw/notes/transcript.md, "Ousterhout (2018), ch. 4"]\n'
            "contested: false\n"
            "---\n"
            "A deep module hides a lot behind a small interface.\n"
        ),
    })

    built = build(notes_dir, index_dir)
    reloaded = Index.load(index_dir)

    expected = {
        "concepts/deep-modules.md": {
            "title": "Deep Modules",
            "layer": "Wiki page",
            "metadata": {
                "title": "Deep Modules",
                "created": "2026-09-27",
                "tags": ["system-design", "practice"],
                "sources": ["raw/notes/transcript.md", "Ousterhout (2018), ch. 4"],
                "contested": False,
            },
        },
    }
    assert built.documents == expected
    assert reloaded.documents == expected


def test_frontmatter_that_strict_yaml_rejects_still_gives_its_title_and_other_fields(notes, index_dir):
    # Six Raw sources in the real Notes quote their title and then keep writing, which YAML
    # rejects for the whole block. Each line is then read on its own.
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "ingested: 2026-10-04\n"
            'duration: "32:19"\n'
            "figure: raw/assets/tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "note: Converted from HTML: math kept as LaTeX.\n"
            "---\n"
            "# Language Trees and Zipping\n\nZipping measures how similar two texts are.\n"
        ),
    })

    index = build(notes_dir, index_dir)

    assert index.documents["raw/papers/benedetto.md"]["metadata"] == {
        "title": '"Language Trees and Zipping" — Benedetto, Caglioti & Loreto',
        "ingested": "2026-10-04",
        "duration": "32:19",
        "figure": "raw/assets/tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)",
        "note": "Converted from HTML: math kept as LaTeX.",
    }
    assert titles_in(index) == {
        "raw/papers/benedetto.md": '"Language Trees and Zipping" — Benedetto, Caglioti & Loreto'
    }


def test_frontmatter_that_is_not_valid_yaml_is_left_out_all_the_same(notes, index_dir):
    # Six Raw sources in the real Notes have a quoted title followed by more text.
    notes_dir = notes({
        "raw/papers/shannon.md": (
            '---\ntitle: "Prediction and Entropy of Printed English" — C. E. Shannon (excerpts)\n'
            "ingested: 2026-09-20\n---\nPrinted English is about 75% redundant.\n"
        ),
    })

    index = build(notes_dir, index_dir)

    assert [chunk.text for chunk in index.chunks] == ["Printed English is about 75% redundant."]


def test_a_horizontal_rule_in_the_body_is_not_mistaken_for_frontmatter(notes, index_dir):
    notes_dir = notes({"essay.md": "First part.\n\n---\n\nSecond part.\n"})

    index = build(notes_dir, index_dir)

    assert [chunk.text for chunk in index.chunks] == ["First part.\n\n---\n\nSecond part."]


def snapshot(folder):
    return {
        path.relative_to(folder): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in folder.rglob("*")
    }


def test_building_an_index_leaves_the_notes_folder_untouched(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": "---\ntitle: RAG\n---\nRetrieval augmented generation.",
        "index.md": "Catalog.",
        "_archive/old.md": "Old.",
    })
    before = snapshot(notes_dir)

    build(notes_dir, index_dir)

    assert snapshot(notes_dir) == before


def test_an_index_is_never_saved_inside_the_notes_folder(notes):
    notes_dir = notes({"concepts/rag.md": "Retrieval augmented generation."})
    before = snapshot(notes_dir)

    with pytest.raises(ValueError, match="inside the Notes"):
        build(notes_dir, notes_dir / ".index")

    assert snapshot(notes_dir) == before


def first_last_and_count(chunk):
    words = chunk.text.split()
    return words[0], words[-1], chunk.token_count


def test_documents_are_cut_into_350_token_windows_overlapping_by_50(notes, index_dir):
    notes_dir = notes({"long.md": " ".join(f"w{i}" for i in range(700))})

    index = build(notes_dir, index_dir)

    assert [first_last_and_count(chunk) for chunk in index.chunks] == [
        ("w0", "w349", 350),
        ("w300", "w649", 350),
        ("w600", "w699", 100),
    ]


def test_the_window_that_reaches_the_end_of_a_document_is_its_last(notes, index_dir):
    notes_dir = notes({"long.md": " ".join(f"w{i}" for i in range(650))})

    index = build(notes_dir, index_dir)

    assert [first_last_and_count(chunk) for chunk in index.chunks] == [
        ("w0", "w349", 350),
        ("w300", "w649", 350),
    ]


def test_each_chunk_records_the_heading_in_effect_where_it_starts_or_else_the_title(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": (
            "---\ntitle: RAG\n---\n"
            "Intro words before any heading.\n\n"
            "## Retrieval\n\none two three four five six\n\n"
            "## Generation\n\nseven eight nine ten\n"
        ),
    })

    index = build(notes_dir, index_dir, chunking=FixedWindows(size=5, overlap=0))

    assert [(chunk.text.split()[0], chunk.section) for chunk in index.chunks] == [
        ("Intro", "RAG"),  # no heading yet: the title alone
        ("##", "Retrieval"),  # a Chunk starting on a heading is in that heading's section
        ("four", "Retrieval"),
        ("seven", "Generation"),
    ]


def test_each_chunk_is_embedded_with_a_title_and_section_prefix(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": "---\ntitle: RAG\n---\nIntro.\n\n## Retrieval\n\nFind the passages.\n",
        # Titled by its top-level heading, so that heading's section is the title itself,
        # which is not repeated.
        "concepts/bm25.md": "# BM25\n\nKeywords.\n",
    })
    embedder = FakeEmbedder()

    index, _ = build_index(notes_dir, index_dir, embedder, WordTokenizer(), chunking=FixedWindows(size=3, overlap=0))

    assert [chunk.label for chunk in index.chunks] == ["BM25", "RAG", "RAG › Retrieval"]
    assert embedder.embedded_documents == [
        "BM25\n\n# BM25\n\nKeywords.",
        "RAG\n\nIntro.\n\n## Retrieval",
        "RAG › Retrieval\n\nFind the passages.",
    ]


def test_documents_in_the_raw_folder_are_raw_sources_and_every_other_document_a_wiki_page(notes, index_dir):
    notes_dir = notes({
        "raw/articles/rag-intro.md": "A clipped article about retrieval.",
        "concepts/rag.md": "Retrieval augmented generation.",
        "SCHEMA.md": "How the wiki is organised.",
        # Only the wiki's own top-level raw folder holds Raw sources.
        "learning/raw/drills.md": "Drills I wrote myself.",
    })

    index = build(notes_dir, index_dir)

    assert {(chunk.document, chunk.layer) for chunk in index.chunks} == {
        ("raw/articles/rag-intro.md", "Raw source"),
        ("concepts/rag.md", "Wiki page"),
        ("SCHEMA.md", "Wiki page"),
        ("learning/raw/drills.md", "Wiki page"),
    }


def titles_in(index):
    return {chunk.document: chunk.title for chunk in index.chunks}


def test_a_documents_title_is_its_frontmatter_title_else_its_first_top_level_heading_else_its_filename(notes, index_dir):
    notes_dir = notes({
        "raw/articles/installed.md": "---\ntitle: Software Installed\n---\n# What I installed\n\nAn agent CLI.\n",
        "raw/articles/chapter-2.md": (
            "---\ncreated: 2026-09-20\n---\n**Sourcing note:** checked twice.\n\n"
            "# Understanding Foundation Models\n\n## Training data\n\nMostly web text.\n"
        ),
        "raw/notes/kirsanov-transcript.md": "## Description\n\nA video about probability.\n",
    })

    index = build(notes_dir, index_dir)

    assert titles_in(index) == {
        "raw/articles/installed.md": "Software Installed",
        "raw/articles/chapter-2.md": "Understanding Foundation Models",
        "raw/notes/kirsanov-transcript.md": "kirsanov-transcript",
    }


def test_wikilinks_read_as_their_alias_or_else_their_target_but_fenced_code_is_left_as_written(notes, index_dir):
    notes_dir = notes({
        "concepts/sampling.md": (
            "Greedy decoding is [[model-sampling|top-k sampling]] with k = 1.\n"
            "Long prompts drift into the [[smart-zone-dumb-zone]].\n\n"
            '```bash\nif [[ "$result" == *done* ]]; then exit; fi\n```\n'
        ),
    })

    index = build(notes_dir, index_dir)

    assert [chunk.text for chunk in index.chunks] == [
        "Greedy decoding is top-k sampling with k = 1.\n"
        "Long prompts drift into the smart-zone-dumb-zone.\n\n"
        '```bash\nif [[ "$result" == *done* ]]; then exit; fi\n```'
    ]


def test_a_heading_inside_fenced_code_is_not_a_heading(notes, index_dir):
    # A four-backtick fence can hold three-backtick fences; only a closing fence at least as
    # long as the opening one (and of the same character) ends it.
    notes_dir = notes({
        "raw/articles/workshop.md": (
            "## `client-brief.md`\n\n"
            "````markdown\n# Slack Message from Sarah Chen\n```bash\necho hi\n```\n# Still inside\n````\n\n"
            "~~~\n# Inside a tilde fence\n```\n~~~\n\n"
            "Text after the fences.\n"
        ),
    })

    index = build(notes_dir, index_dir)

    assert titles_in(index) == {"raw/articles/workshop.md": "workshop"}


def test_a_chunk_the_embedder_cannot_read_whole_is_warned_about_and_kept_whole(notes, index_dir):
    notes_dir = notes({
        "concepts/long.md": "---\ntitle: Long\n---\na b c\n\n## A very long section name\n\nd e f g\n",
    })
    embedder = FakeEmbedder(max_input_tokens=10)

    index, report = build_index(
        notes_dir, index_dir, embedder, WordTokenizer(), chunking=FixedWindows(size=5, overlap=0)
    )

    # Five tokens each, plus the label the Embedder reads with them:
    #   chunk 1  "Long" + "a b c ## A"                                 1 + 5 = 6
    #   chunk 2  "Long › A very long section name" + "very long section name d"   7 + 5 = 12
    #   chunk 3  "Long › A very long section name" + "e f g"           7 + 3 = 10, just fits
    assert report.warnings == [
        "concepts/long.md, chunk 2 (Long › A very long section name): 12 tokens with its label, "
        "over fake-embedder's limit of 10, so the model reads only the first 10."
    ]
    assert [chunk.text for chunk in index.chunks] == [
        "a b c\n\n## A",
        "very long section name\n\nd",
        "e f g",
    ]


def test_the_build_report_counts_documents_per_layer_what_was_excluded_and_the_chunks(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": "Retrieval augmented generation.",
        "concepts/long.md": " ".join(f"w{i}" for i in range(400)),  # two windows
        "raw/articles/rag-intro.md": "A clipped article about retrieval.",
        "index.md": "Catalog.",
        "log.md": "2026-10-11 ingested a new source.",
        # An excluded folder is reported once and never walked into.
        ".git/objects/ab/cdef0123": "a git object",
        ".git/HEAD": "ref: refs/heads/main",
        "_archive/old-summary.md": "Superseded.",
        "learning/_archive/draft.md": "An archived draft.",
        "raw/papers/attention.pdf": "%PDF-1.7 binary content",
    })

    _, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    assert report.admitted == {"Wiki page": 2, "Raw source": 1}
    assert report.excluded == [
        ".git/",
        "_archive/",
        "index.md",
        "learning/_archive/",
        "log.md",
        "raw/papers/attention.pdf",
    ]
    assert report.chunks_created == 4
    assert report.warnings == []
    assert report.embedding_seconds >= 0


def test_chunks_are_identical_whichever_embedding_model_builds_the_index(notes, tmp_path):
    notes_dir = notes({
        "concepts/rag.md": "---\ntitle: RAG\n---\nRetrieval augmented generation grounds answers in documents.",
        "concepts/long.md": "---\ntitle: Long\n---\n" + " ".join(f"word{i}" for i in range(400)),
        "raw/papers/overview.md": "An overview of modern language models.",
    })
    tokenizer = WordTokenizer()
    index_bge, _ = build_index(
        notes_dir, tmp_path / "index_bge",
        FakeEmbedder(model_id="BAAI/bge-small-en-v1.5", max_input_tokens=510),
        tokenizer,
    )
    index_qwen, _ = build_index(
        notes_dir, tmp_path / "index_qwen",
        FakeEmbedder(model_id="Qwen/Qwen3-Embedding-0.6B", max_input_tokens=32768),
        tokenizer,
    )

    assert index_bge.embedding_model == "BAAI/bge-small-en-v1.5"
    assert index_qwen.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert index_bge.chunks == index_qwen.chunks


def test_heading_aware_chunking_splits_documents_at_headings(notes, index_dir):
    notes_dir = notes({
        "concepts/rag.md": (
            "---\ntitle: RAG\n---\n"
            "## Retrieval\n\nFind the passages.\n\n"
            "## Generation\n\nAnswer the question.\n"
        ),
    })

    index = build(
        notes_dir,
        index_dir,
        chunking=HeadingAwareChunking(size=20, overlap=5, min_size=3),
    )

    assert [(chunk.section, chunk.text, chunk.label) for chunk in index.chunks] == [
        ("Retrieval", "## Retrieval\n\nFind the passages.", "RAG › Retrieval"),
        ("Generation", "## Generation\n\nAnswer the question.", "RAG › Generation"),
    ]


def test_heading_aware_chunking_handles_documents_with_no_headings(notes, index_dir):
    notes_dir = notes({
        "short.md": "A brief document without any headings at all.",
        "long.md": " ".join(f"w{i}" for i in range(25)),
    })

    index = build(
        notes_dir,
        index_dir,
        chunking=HeadingAwareChunking(size=10, overlap=2, min_size=3),
    )

    short_chunks = [c for c in index.chunks if c.document == "short.md"]
    assert len(short_chunks) == 1
    assert short_chunks[0].section == "short"
    assert short_chunks[0].label == "short"
    assert short_chunks[0].text == "A brief document without any headings at all."

    long_chunks = [c for c in index.chunks if c.document == "long.md"]
    assert [first_last_and_count(c) for c in long_chunks] == [
        ("w0", "w9", 10),
        ("w8", "w17", 10),
        ("w16", "w24", 9),
    ]
    assert all(c.section == "long" for c in long_chunks)
    assert all(c.label == "long" for c in long_chunks)


def test_heading_aware_chunking_falls_back_to_fixed_windows_for_long_sections(notes, index_dir):
    notes_dir = notes({
        "concepts/doc.md": (
            "---\ntitle: Doc\n---\n"
            "## Short Section\n\none two three four\n\n"
            "## Long Section\n\n" + " ".join(f"w{i}" for i in range(25)) + "\n\n"
            "## Another Short\n\nfive six seven eight\n"
        ),
    })

    index = build(
        notes_dir,
        index_dir,
        chunking=HeadingAwareChunking(size=10, overlap=2, min_size=3),
    )

    chunks = index.chunks
    assert len(chunks) == 6
    assert chunks[0].section == "Short Section"
    assert chunks[0].label == "Doc › Short Section"
    assert chunks[0].text == "## Short Section\n\none two three four"

    long_chunks = chunks[1:5]
    assert all(c.section == "Long Section" for c in long_chunks)
    assert all(c.label == "Doc › Long Section" for c in long_chunks)
    # Long Section has "##", "Long", "Section" + 25 words = 28 tokens
    # Window 1: tokens 0..9 (10)
    # Window 2: tokens 8..17 (10)
    # Window 3: tokens 16..25 (10)
    # Window 4: tokens 24..27 (4)
    assert [first_last_and_count(c) for c in long_chunks] == [
        ("##", "w6", 10),
        ("w5", "w14", 10),
        ("w13", "w22", 10),
        ("w21", "w24", 4),
    ]

    assert chunks[5].section == "Another Short"
    assert chunks[5].label == "Doc › Another Short"
    assert chunks[5].text == "## Another Short\n\nfive six seven eight"


def test_heading_aware_chunking_merges_sections_below_minimum_size_into_neighbour(notes, index_dir):
    notes_dir = notes({
        "concepts/intro-merge.md": (
            "---\ntitle: Intro Merge\n---\n"
            "Short intro.\n\n"
            "## First\n\none two three four five\n"
        ),
        "concepts/trailing-merge.md": (
            "---\ntitle: Trailing Merge\n---\n"
            "## Main\n\n"
            "alpha beta gamma delta epsilon\n\n"
            "## Note\n\ntiny note\n"
        ),
    })

    index = build(
        notes_dir,
        index_dir,
        chunking=HeadingAwareChunking(size=20, overlap=5, min_size=5),
    )

    # Short intro is merged with the first heading section
    intro_chunks = [c for c in index.chunks if c.document == "concepts/intro-merge.md"]
    assert len(intro_chunks) == 1
    assert intro_chunks[0].text == "Short intro.\n\n## First\n\none two three four five"
    assert intro_chunks[0].section == "Intro Merge"
    assert intro_chunks[0].label == "Intro Merge"
    assert intro_chunks[0].token_count == 9

    # Short trailing section is merged into the preceding heading section
    trailing_chunks = [c for c in index.chunks if c.document == "concepts/trailing-merge.md"]
    assert len(trailing_chunks) == 1
    assert trailing_chunks[0].text == "## Main\n\nalpha beta gamma delta epsilon\n\n## Note\n\ntiny note"
    assert trailing_chunks[0].section == "Main"
    assert trailing_chunks[0].label == "Trailing Merge › Main"
    assert trailing_chunks[0].token_count == 11


def test_heading_aware_chunking_merges_middle_small_section_and_consecutive_small_sections(notes, index_dir):
    notes_dir = notes({
        "concepts/doc.md": (
            "---\ntitle: Middle Merge\n---\n"
            "## Sec1\n\none two three four five six seven\n\n"
            "## Sec2\n\ntiny\n\n"
            "## Sec3\n\neight nine ten eleven twelve thirteen fourteen\n"
        ),
        "concepts/consecutive.md": (
            "---\ntitle: Consecutive\n---\n"
            "## Part1\n\na b\n\n"
            "## Part2\n\nc d\n\n"
            "## Part3\n\ne f g h i j k\n"
        ),
    })

    index = build(
        notes_dir,
        index_dir,
        chunking=HeadingAwareChunking(size=20, overlap=5, min_size=5),
    )

    doc_chunks = [c for c in index.chunks if c.document == "concepts/doc.md"]
    assert len(doc_chunks) == 2
    assert doc_chunks[0].section == "Sec1"
    assert doc_chunks[0].text == "## Sec1\n\none two three four five six seven\n\n## Sec2\n\ntiny"
    assert doc_chunks[0].token_count == 12
    assert doc_chunks[1].section == "Sec3"
    assert doc_chunks[1].text == "## Sec3\n\neight nine ten eleven twelve thirteen fourteen"
    assert doc_chunks[1].token_count == 9

    consec_chunks = [c for c in index.chunks if c.document == "concepts/consecutive.md"]
    assert len(consec_chunks) == 2
    assert consec_chunks[0].section == "Part1"
    assert consec_chunks[0].text == "## Part1\n\na b\n\n## Part2\n\nc d"
    assert consec_chunks[0].token_count == 8
    assert consec_chunks[1].section == "Part3"


def test_index_and_build_report_record_chunking_strategy(notes, index_dir):
    notes_dir = notes({"concepts/rag.md": "Retrieval augmented generation."})

    # Heading-aware strategy
    index_ha, report_ha = build_index(
        notes_dir,
        index_dir / "ha",
        FakeEmbedder(),
        WordTokenizer(),
        chunking=HeadingAwareChunking(),
    )
    assert report_ha.chunking_strategy == "heading-aware"
    assert index_ha.chunking_strategy == "heading-aware"
    reloaded_ha = Index.load(index_dir / "ha")
    assert reloaded_ha.chunking_strategy == "heading-aware"

    # Fixed strategy
    index_fix, report_fix = build_index(
        notes_dir,
        index_dir / "fix",
        FakeEmbedder(),
        WordTokenizer(),
        chunking=FixedWindows(),
    )
    assert report_fix.chunking_strategy == "fixed"
    assert index_fix.chunking_strategy == "fixed"
    reloaded_fix = Index.load(index_dir / "fix")
    assert reloaded_fix.chunking_strategy == "fixed"


def test_build_index_accepts_chunking_strategy_by_name(notes, index_dir):
    notes_dir = notes({"concepts/rag.md": "Retrieval augmented generation."})

    index_ha, report_ha = build_index(
        notes_dir,
        index_dir / "ha",
        FakeEmbedder(),
        WordTokenizer(),
        chunking="headings",
    )
    assert report_ha.chunking_strategy == "heading-aware"
    assert index_ha.chunking_strategy == "heading-aware"

    index_fix, report_fix = build_index(
        notes_dir,
        index_dir / "fix",
        FakeEmbedder(),
        WordTokenizer(),
        chunking="fixed",
    )
    assert report_fix.chunking_strategy == "fixed"
    assert index_fix.chunking_strategy == "fixed"


def test_building_index_without_image_describer_leaves_chunks_with_borrowed_text_only(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Zipping measures how similar two texts are.\n\n"
            "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=None)

    image_chunk = next(c for c in index.chunks if c.document == "raw/assets/benedetto-2002-language-tree-fig1.png")
    assert image_chunk.text == (
        "benedetto 2002 language tree fig1\n\n"
        '"Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n\n'
        "Fig. 1, the language tree\n\n"
        "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree."
    )
    assert image_chunk.has_vision_description is False
    assert report.image_description_seconds == pytest.approx(0.0)


def test_building_index_with_image_describer_includes_description_alongside_borrowed_text(notes, index_dir):
    notes_dir = notes({
        "raw/papers/benedetto.md": (
            "---\n"
            'title: "Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n'
            "figure: raw/assets/benedetto-2002-language-tree-fig1.png (Fig. 1, the language tree; sha256 7909cd5e)\n"
            "---\n"
            "# Language Trees and Zipping\n\n"
            "Zipping measures how similar two texts are.\n\n"
            "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree.\n"
        ),
        "raw/assets/benedetto-2002-language-tree-fig1.png": b"",
    })
    describer = FakeImageDescriber({
        "benedetto-2002-language-tree-fig1.png": "Phylogenetic tree diagram of 52 languages derived from gzip compression distances."
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=describer)

    image_chunk = next(c for c in index.chunks if c.document == "raw/assets/benedetto-2002-language-tree-fig1.png")
    assert image_chunk.text == (
        "benedetto 2002 language tree fig1\n\n"
        '"Language Trees and Zipping" — Benedetto, Caglioti & Loreto\n\n'
        "Fig. 1, the language tree\n\n"
        "Figure 1: Language Tree: This figure illustrates the phylogenetic-like tree.\n\n"
        "Phylogenetic tree diagram of 52 languages derived from gzip compression distances."
    )
    assert image_chunk.has_vision_description is True
    assert report.image_description_seconds >= 0.0
    assert len(describer.calls) == 1
    assert "benedetto-2002-language-tree-fig1.png" in describer.calls[0]


def test_unreferenced_image_with_describer_includes_description_alongside_humanized_filename(notes, index_dir):
    notes_dir = notes({
        "raw/assets/unreferenced-tree.png": b"",
        "notes/general.md": "A note that does not mention any images.",
    })
    describer = FakeImageDescriber({
        "unreferenced-tree.png": "A diagram depicting a taxonomic hierarchy."
    })

    index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=describer)

    chunks = {chunk.document: chunk for chunk in index.chunks}
    tree_chunk = chunks["raw/assets/unreferenced-tree.png"]
    assert tree_chunk.text == "unreferenced tree\n\nA diagram depicting a taxonomic hierarchy."
    assert tree_chunk.has_vision_description is True
    assert tree_chunk.is_image is True
    assert report.image_description_seconds >= 0.0


def test_index_stores_content_hash_for_each_document(notes, index_dir):
    import hashlib

    text_content = "Retrieval augmented generation."
    png_content = b"\x89PNG\r\n\x1a\nfake-png-data"
    notes_dir = notes({
        "concepts/rag.md": text_content,
        "raw/assets/diag.png": png_content,
    })

    index = build(notes_dir, index_dir)

    expected_rag_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    expected_img_hash = hashlib.sha256(png_content + b"diag").hexdigest()

    assert index.content_hashes["concepts/rag.md"] == expected_rag_hash
    assert index.content_hashes["raw/assets/diag.png"] == expected_img_hash

    reloaded = Index.load(index_dir)
    assert reloaded.content_hashes["concepts/rag.md"] == expected_rag_hash
    assert reloaded.content_hashes["raw/assets/diag.png"] == expected_img_hash


def test_build_report_initial_build_records_counts_and_reembedded(notes, index_dir):
    notes_dir = notes({
        "doc1.md": "First document with some content.",
        "doc2.md": "Second document with other content.",
    })

    _index, report = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer())

    assert report.documents_added == 2
    assert report.documents_changed == 0
    assert report.documents_removed == 0
    assert report.documents_unchanged == 0
    assert report.chunks_reembedded == report.chunks_created
    assert report.chunks_reembedded > 0


def test_reindex_reembeds_only_new_or_changed_and_drops_deleted(notes, index_dir):
    notes_dir = notes({
        "doc_a.md": "Alpha note stays untouched throughout.",
        "doc_b.md": "Beta note will be changed later.",
        "doc_c.md": "Gamma note will be deleted later.",
    })

    initial_embedder = FakeEmbedder()
    initial_index, initial_report = build_index(notes_dir, index_dir, initial_embedder, WordTokenizer())
    assert initial_report.chunks_created == 3
    assert len(initial_embedder.embedded_documents) == 3

    # Remember Alpha's vector from initial build
    alpha_idx_initial = next(i for i, c in enumerate(initial_index.chunks) if c.document == "doc_a.md")
    alpha_vector_initial = initial_index.vectors[alpha_idx_initial].copy()

    # Modify doc_b.md, add doc_d.md, delete doc_c.md
    (notes_dir / "doc_b.md").write_text("Beta note with updated and revised content.")
    (notes_dir / "doc_d.md").write_text("Delta note newly added to the wiki.")
    (notes_dir / "doc_c.md").unlink()

    reindex_embedder = FakeEmbedder()
    new_index, report = build_index(notes_dir, index_dir, reindex_embedder, WordTokenizer())

    assert report.documents_added == 1
    assert report.documents_changed == 1
    assert report.documents_removed == 1
    assert report.documents_unchanged == 1

    # Documents present in new index
    docs_in_new_index = {chunk.document for chunk in new_index.chunks}
    assert docs_in_new_index == {"doc_a.md", "doc_b.md", "doc_d.md"}
    assert "doc_c.md" not in docs_in_new_index

    # Only doc_b (changed) and doc_d (added) were embedded! doc_a was NOT embedded!
    embedded_labels = [text.split("\n\n")[0] for text in reindex_embedder.embedded_documents]
    assert "doc_a" not in embedded_labels
    assert "doc_b" in embedded_labels
    assert "doc_d" in embedded_labels
    assert len(reindex_embedder.embedded_documents) == 2
    assert report.chunks_reembedded == 2

    # Alpha's vector is untouched
    alpha_idx_new = next(i for i, c in enumerate(new_index.chunks) if c.document == "doc_a.md")
    import numpy as np
    np.testing.assert_array_equal(new_index.vectors[alpha_idx_new], alpha_vector_initial)


def test_reindex_with_no_changes_reembeds_zero_chunks(notes, index_dir):
    notes_dir = notes({
        "doc_a.md": "Alpha note untouched.",
        "doc_b.md": "Beta note untouched.",
    })

    initial_embedder = FakeEmbedder()
    _idx1, report1 = build_index(notes_dir, index_dir, initial_embedder, WordTokenizer())
    assert report1.chunks_created == 2
    assert report1.chunks_reembedded == 2

    # Second build with same untouched files
    second_embedder = FakeEmbedder()
    _idx2, report2 = build_index(notes_dir, index_dir, second_embedder, WordTokenizer())

    assert report2.documents_added == 0
    assert report2.documents_changed == 0
    assert report2.documents_removed == 0
    assert report2.documents_unchanged == 2
    assert report2.chunks_reembedded == 0
    assert len(second_embedder.embedded_documents) == 0


def test_incremental_update_produces_identical_search_and_ask_results_to_full_rebuild(notes, tmp_path):
    notes_dir = notes({
        "concepts/alpha.md": "Alpha concepts about deep retrieval architectures and attention mechanisms.",
        "concepts/beta.md": "Beta concepts covering indexing and tokenization strategies.",
        "concepts/gamma.md": "Gamma concepts discussing prompt caching and generation latency.",
    })

    incremental_dir = tmp_path / "incremental_index"
    full_rebuild_dir = tmp_path / "full_index"

    # Step 1: Initial build for incremental index
    build_index(notes_dir, incremental_dir, FakeEmbedder(), WordTokenizer())

    # Step 2: Modify beta, add delta, delete gamma
    (notes_dir / "concepts/beta.md").write_text("Beta concepts covering indexing, BM25 keyword fusion and reciprocal rank fusion.")
    (notes_dir / "concepts/delta.md").write_text("Delta concepts exploring vector search and dense representations.")
    (notes_dir / "concepts/gamma.md").unlink()

    # Step 3: Run incremental update
    incremental_index, _ = build_index(notes_dir, incremental_dir, FakeEmbedder(), WordTokenizer())

    # Step 4: Run full rebuild from scratch on the modified notes
    full_index, _ = build_index(notes_dir, full_rebuild_dir, FakeEmbedder(), WordTokenizer(), force=True)

    # Chunks must match exactly in order, content and metadata
    assert len(incremental_index.chunks) == len(full_index.chunks)
    for inc_chunk, full_chunk in zip(incremental_index.chunks, full_index.chunks):
        assert inc_chunk == full_chunk

    # Vectors must match exactly
    import numpy as np
    np.testing.assert_allclose(incremental_index.vectors, full_index.vectors, atol=1e-6)

    # Embedding search rankings must match
    query = "deep retrieval indexing"
    embedder = FakeEmbedder()
    query_vector = embedder.embed_query(query)
    inc_search = incremental_index.search(query_vector, embedder.model_id, k=5)
    full_search = full_index.search(query_vector, embedder.model_id, k=5)
    assert [(c.document, round(s, 5)) for c, s in inc_search] == [(c.document, round(s, 5)) for c, s in full_search]

    # BM25 keyword rankings must match identically
    inc_kw = incremental_index.keyword_ranking(query)
    full_kw = full_index.keyword_ranking(query)
    assert inc_kw == full_kw

    # End-to-end ask() with both retrieval modes must produce identical results
    generator = FakeGenerator()
    for mode in ("embeddings", "hybrid"):
        ans_inc = ask(query, incremental_dir, FakeEmbedder(), generator, retrieval=mode)
        ans_full = ask(query, full_rebuild_dir, FakeEmbedder(), generator, retrieval=mode)
        assert ans_inc.text == ans_full.text
        assert ans_inc.citations == ans_full.citations
        assert len(ans_inc.context) == len(ans_full.context)
        for c1, c2 in zip(ans_inc.context, ans_full.context):
            assert c1.chunk == c2.chunk
            assert pytest.approx(c1.score) == c2.score
            assert c1.embedding_rank == c2.embedding_rank
            assert c1.bm25_rank == c2.bm25_rank


def test_full_rebuild_triggers_on_force_or_model_or_chunking_or_tokenizer_mismatch(notes, tmp_path):
    notes_dir = notes({
        "doc1.md": "Content of doc one.",
        "doc2.md": "Content of doc two.",
    })

    # Case 1: force=True
    idx_dir_force = tmp_path / "idx_force"
    build_index(notes_dir, idx_dir_force, FakeEmbedder(model_id="m1"), WordTokenizer(), chunking="fixed")
    embedder_forced = FakeEmbedder(model_id="m1")
    _idx, report_forced = build_index(notes_dir, idx_dir_force, embedder_forced, WordTokenizer(), chunking="fixed", force=True)
    assert report_forced.chunks_reembedded == report_forced.chunks_created == 2
    assert len(embedder_forced.embedded_documents) == 2
    assert report_forced.documents_unchanged == 0

    # Case 2: embedding model mismatch triggers full rebuild
    idx_dir_model = tmp_path / "idx_model"
    build_index(notes_dir, idx_dir_model, FakeEmbedder(model_id="m1"), WordTokenizer(), chunking="fixed")
    embedder_new_model = FakeEmbedder(model_id="m2")
    _idx, report_model = build_index(notes_dir, idx_dir_model, embedder_new_model, WordTokenizer(), chunking="fixed")
    assert report_model.chunks_reembedded == report_model.chunks_created == 2
    assert len(embedder_new_model.embedded_documents) == 2
    assert report_model.documents_unchanged == 0

    # Case 3: chunking strategy mismatch triggers full rebuild
    idx_dir_chunking = tmp_path / "idx_chunking"
    build_index(notes_dir, idx_dir_chunking, FakeEmbedder(model_id="m1"), WordTokenizer(), chunking="fixed")
    embedder_chunking = FakeEmbedder(model_id="m1")
    _idx, report_chunking = build_index(notes_dir, idx_dir_chunking, embedder_chunking, WordTokenizer(), chunking="headings")
    assert report_chunking.chunks_reembedded == report_chunking.chunks_created == 2
    assert len(embedder_chunking.embedded_documents) == 2
    assert report_chunking.documents_unchanged == 0
    assert report_chunking.chunking_strategy == "heading-aware"

    # Case 4: chunk tokenizer mismatch triggers full rebuild
    class OtherTokenizer(WordTokenizer):
        name = "other-words"

    idx_dir_tok = tmp_path / "idx_tok"
    build_index(notes_dir, idx_dir_tok, FakeEmbedder(model_id="m1"), WordTokenizer(), chunking="fixed")
    embedder_tok = FakeEmbedder(model_id="m1")
    _idx, report_tok = build_index(notes_dir, idx_dir_tok, embedder_tok, OtherTokenizer(), chunking="fixed")
    assert report_tok.chunks_reembedded == report_tok.chunks_created == 2
    assert len(embedder_tok.embedded_documents) == 2
    assert report_tok.documents_unchanged == 0


def test_incremental_reindexing_describes_only_new_or_changed_images(notes, index_dir):
    notes_dir = notes({
        "raw/assets/fig1.png": b"image-1-bytes",
        "raw/assets/fig2.png": b"image-2-bytes",
        "raw/papers/paper.md": (
            "# A Paper\n\n"
            "Here is figure 1: ![Fig 1](raw/assets/fig1.png)\n"
        ),
    })

    describer1 = FakeImageDescriber({
        "fig1.png": "Description of figure 1.",
        "fig2.png": "Description of figure 2.",
    })
    idx1, report1 = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=describer1)
    assert len(describer1.calls) == 2

    # Verify fig1 and fig2 chunks have vision description
    chunks1 = {c.document: c for c in idx1.chunks}
    assert chunks1["raw/assets/fig1.png"].has_vision_description is True
    assert "Description of figure 1." in chunks1["raw/assets/fig1.png"].text
    assert chunks1["raw/assets/fig2.png"].has_vision_description is True
    assert "Description of figure 2." in chunks1["raw/assets/fig2.png"].text

    # Step 2: Add fig3.png, leave fig1 and fig2 untouched
    (notes_dir / "raw/assets/fig3.png").write_bytes(b"image-3-bytes")

    describer2 = FakeImageDescriber({
        "fig3.png": "Description of figure 3.",
    })
    idx2, report2 = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=describer2)

    # Only fig3 was described!
    assert len(describer2.calls) == 1
    assert "fig3.png" in describer2.calls[0]

    # fig1 and fig2 still retain their vision descriptions in idx2
    chunks2 = {c.document: c for c in idx2.chunks}
    assert chunks2["raw/assets/fig1.png"].has_vision_description is True
    assert "Description of figure 1." in chunks2["raw/assets/fig1.png"].text
    assert chunks2["raw/assets/fig2.png"].has_vision_description is True
    assert "Description of figure 2." in chunks2["raw/assets/fig2.png"].text
    assert chunks2["raw/assets/fig3.png"].has_vision_description is True
    assert "Description of figure 3." in chunks2["raw/assets/fig3.png"].text

    # Step 3: Modify fig1.png bytes
    (notes_dir / "raw/assets/fig1.png").write_bytes(b"image-1-MODIFIED-bytes")
    describer3 = FakeImageDescriber({
        "fig1.png": "Updated description of figure 1.",
    })
    idx3, report3 = build_index(notes_dir, index_dir, FakeEmbedder(), WordTokenizer(), image_describer=describer3)

    # Only fig1 was described!
    assert len(describer3.calls) == 1
    assert "fig1.png" in describer3.calls[0]
    chunks3 = {c.document: c for c in idx3.chunks}
    assert "Updated description of figure 1." in chunks3["raw/assets/fig1.png"].text


def test_cli_index_force_flag_accepted(notes, tmp_path, monkeypatch, capsys):
    from rag.cli import main

    notes_dir = notes({"note.md": "A note."})
    index_dir = tmp_path / "cli_index"

    monkeypatch.setattr("rag.embedder.BgeSmallEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag.chunker.BgeTokenizer", WordTokenizer)
    main(["index", "--notes", str(notes_dir), "--index-dir", str(index_dir), "--force"])
    captured = capsys.readouterr().out
    assert "full rebuild forced" in captured
    assert "1 added" in captured
    assert (index_dir / "metadata.json").is_file()


def test_manual_check_add_note_to_notes_and_reindex_only_that_doc_embedded(notes, index_dir):
    # Simulates: user has existing Notes, adds a new note, runs re-index
    notes_dir = notes({
        "concepts/existing1.md": "Existing knowledge about transformers.",
        "concepts/existing2.md": "Existing knowledge about vector search.",
    })

    embedder1 = FakeEmbedder()
    _idx1, report1 = build_index(notes_dir, index_dir, embedder1, WordTokenizer())
    assert report1.documents_added == 2
    assert report1.chunks_reembedded == 2

    # User adds a note to the Notes
    (notes_dir / "concepts/new_note.md").write_text("New note about incremental re-indexing in RAG systems.")

    # Re-index
    embedder2 = FakeEmbedder()
    _idx2, report2 = build_index(notes_dir, index_dir, embedder2, WordTokenizer())

    assert report2.documents_added == 1
    assert report2.documents_changed == 0
    assert report2.documents_removed == 0
    assert report2.documents_unchanged == 2
    assert report2.chunks_reembedded == 1

    # Only that document was embedded
    assert len(embedder2.embedded_documents) == 1
    assert "new_note" in embedder2.embedded_documents[0]


def test_index_saves_and_loads_notes_location_and_build_time_in_metadata(notes, index_dir):
    import json
    from datetime import datetime

    notes_dir = notes({"note.md": "A single note."})
    index = build(notes_dir, index_dir)

    assert index.notes_location == str(notes_dir)
    assert index.build_time
    # Validate ISO-8601 format
    parsed_time = datetime.fromisoformat(index.build_time)
    assert parsed_time is not None

    metadata_path = index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["notes_location"] == str(notes_dir)
    assert metadata["build_time"] == index.build_time

    reloaded = Index.load(index_dir)
    assert reloaded.notes_location == str(notes_dir)
    assert reloaded.build_time == index.build_time


def test_incremental_reindex_reembeds_image_when_referencing_doc_caption_changes(notes, index_dir):
    notes_dir = notes({
        "paper.md": (
            "---\n"
            "title: Paper Title\n"
            "figure: raw/assets/fig.png (Fig. 1, architecture diagram)\n"
            "---\n"
            "# Paper Title\n\n"
            "Figure 1: Initial caption describing architecture.\n"
        ),
        "raw/assets/fig.png": b"fake-png-bytes",
    })

    embedder1 = FakeEmbedder()
    idx1, report1 = build_index(notes_dir, index_dir, embedder1, WordTokenizer())
    assert report1.documents_added == 2
    assert report1.chunks_created == 2
    assert report1.chunks_reembedded == 2
    initial_img_hash = idx1.content_hashes["raw/assets/fig.png"]

    # Now change only the figure caption in paper.md; the png bytes stay identical
    (notes_dir / "paper.md").write_text(
        "---\n"
        "title: Paper Title\n"
        "figure: raw/assets/fig.png (Fig. 1, architecture diagram)\n"
        "---\n"
        "# Paper Title\n\n"
        "Figure 1: Updated caption with new details.\n"
    )

    embedder2 = FakeEmbedder()
    idx2, report2 = build_index(notes_dir, index_dir, embedder2, WordTokenizer())

    # Both paper.md and fig.png must be marked as changed and re-embedded
    assert report2.documents_changed == 2
    assert report2.chunks_reembedded == 2
    assert idx2.content_hashes["raw/assets/fig.png"] != initial_img_hash


