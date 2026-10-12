# Agentic RAG

A local question-answering system that answers from the user's Notes by retrieving relevant passages and handing them to a language model. Built to learn how retrieval-augmented generation works.

## Language

### Corpus

**Notes**:
The user's personal knowledge wiki; the only place Documents come from.
_Avoid_: knowledge base, vault, dataset

**Document**:
A single Markdown, HTML, plain-text or image file from the Notes that answers can be drawn from. The wiki's own bookkeeping (its index, its log, its archive) is not a Document.
_Avoid_: note, file

**Layer**:
Which half of the wiki a Document belongs to: Wiki page or Raw source.

**Wiki page**:
A Document written within the wiki rather than captured from elsewhere; mostly syntheses of Raw sources, but also the wiki's own reference material and exercises.
_Avoid_: page (on its own), synthesized note

**Raw source**:
A Document holding original material as it was captured: an article, a transcript, a paper excerpt.
_Avoid_: original, raw note

### Retrieval

**Chunk**:
A contiguous passage of one Document's text; the unit that is searched for and handed to the model.
_Avoid_: passage, snippet, segment, split

**Index**:
The searchable collection of every Chunk, built from all Documents with a single embedding model.
_Avoid_: vector store, database

**Question**:
What the user asks, in their own words.
_Avoid_: prompt, query (when meaning the user's words)

**Query**:
The text the retriever searches the Index with. Identical to the Question unless something rewrites it first.
_Avoid_: search string, question (when meaning the search text)

**Context**:
The Chunks retrieved for a Question and handed to the model to answer from. The model's token limit is the context window, never "context".
_Avoid_: retrieved documents, evidence

### Answering

**Citation**:
A pointer from an answer to a Chunk it was given, naming the Document and its section.
_Avoid_: source, sources list, reference
