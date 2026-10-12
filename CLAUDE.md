# Agentic RAG

A fully local RAG system over the user's Notes (`~/Documents/Notes`, read-only), built to learn how RAG works. The user is learning AI engineering: favour inspectable, hand-rolled code with visible intermediate steps over abstractions.

## Constraints

- **Local, CPU-only.** No provider APIs, no GPU. torch comes from the CPU index; llama-cpp-python compiles from source (see the comments in `pyproject.toml`). Rationale: `.scratch/docs/adr/0001-local-cpu-only-inference.md`.
- **No frameworks, no vector DB.** The pipeline is hand-rolled (numpy matrix + JSONL Index).

## Vocabulary

Read `GLOSSARY.md` before naming anything. Use its terms (Chunk, Context, Citation, Question vs Query, ...) in code, tests and messages, never the synonyms it lists under _Avoid_.

## Layout

- `src/rag/pipeline.py`: `build_index` and `ask`, the two operations the CLI and tests share. Models arrive through ports (Embedder, chunk tokenizer, Generator); pipeline never loads one itself.
- `tests/fakes.py`: test adapters for those ports.
- `.scratch/`: gitignored working notes (`stage-1-classic-rag/spec.md`, issues, ADRs, `docs/agents/`).

## Commands

- `uv run pytest`: fast suite, models faked. `uv run pytest -m slow` loads real models (`tests/test_real_models.py`).
- `uv run rag index` / `uv run rag ask "..." --show-context`. The Index lands in `.index/`.

## Conventions

- Tests first for deterministic parts; exercise the Pipeline through fakes, so no test downloads a model.
- Chunk size is always counted with bge-small's tokenizer, whichever embedding model is active.
- The Index records its embedding model id and refuses queries from a different one.
- Commit only when the user asks.
