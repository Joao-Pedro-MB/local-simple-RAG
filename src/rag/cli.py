"""`rag index` and `rag ask`: a thin command-line layer over the Pipeline."""

import argparse
import sys
from pathlib import Path

from rag.pipeline import ANSWER_TOKENS, RETRIEVAL_MODES, EmbeddingModelMismatch, ask, build_index

DEFAULT_NOTES = Path.home() / "Documents" / "Notes"
# The project root (this file is src/rag/cli.py), so the Index is found from any directory.
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / ".index"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rag", description="Answer Questions from the Notes.")
    commands = parser.add_subparsers(dest="command", required=True)

    index_cmd = commands.add_parser("index", help="build the Index from the Notes (re-indexes only changed Documents by default)")
    index_cmd.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help=f"Notes folder (default: {DEFAULT_NOTES})")
    index_cmd.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help=f"where to save the Index (default: {DEFAULT_INDEX_DIR})")
    index_cmd.add_argument("--embedding-model", choices=["bge-small", "qwen3"], default="bge-small", help="embedding model to use: bge-small (default) or qwen3")
    index_cmd.add_argument(
        "--chunking",
        choices=["fixed", "headings", "heading-aware"],
        default="fixed",
        help="chunking strategy: fixed (default) or headings (heading-aware)",
    )
    index_cmd.add_argument(
        "--describe-images",
        action="store_true",
        help="use Gemma 4 E2B's vision capability to describe each image Document",
    )
    index_cmd.add_argument(
        "--force",
        action="store_true",
        help="force a full rebuild even if documents haven't changed",
    )

    ask_cmd = commands.add_parser("ask", help="answer a Question from the Notes, with Citations")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("-k", type=int, default=5, help="how many Chunks to retrieve (default: 5)")
    ask_cmd.add_argument("--retrieval", choices=RETRIEVAL_MODES, default="embeddings", help="embeddings only (default), or hybrid: embeddings and BM25 keyword search fused by RRF")
    ask_cmd.add_argument("--temperature", type=float, default=0.0, help="0 always picks the likeliest next token; higher samples more freely (default: 0)")
    ask_cmd.add_argument("--show-context", action="store_true", help="also print the Context, the exact prompt, its size and the generation speed")
    ask_cmd.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help=f"where the Index is (default: {DEFAULT_INDEX_DIR})")

    args = parser.parse_args(argv)
    if args.command == "index":
        run_index(args)
    else:
        run_ask(args)


def resolve_embedder(model_name_or_id):
    if model_name_or_id in ("bge-small", "BAAI/bge-small-en-v1.5"):
        from rag.adapters.embedder import BgeSmallEmbedder
        return BgeSmallEmbedder()
    elif model_name_or_id in ("qwen3", "Qwen/Qwen3-Embedding-0.6B"):
        from rag.adapters.embedder import Qwen3Embedder
        return Qwen3Embedder()
    raise ValueError(f"unknown embedding model {model_name_or_id!r}; choose 'bge-small' or 'qwen3'")


def run_index(args):
    if not args.notes.is_dir():
        sys.exit(f"rag: no Notes folder at {args.notes}")
    from rag.documents.chunker import BgeTokenizer, FixedWindows, HeadingAwareChunking

    chunking = HeadingAwareChunking() if args.chunking in ("headings", "heading-aware") else FixedWindows()
    embedder = resolve_embedder(args.embedding_model)
    tokenizer = BgeTokenizer()
    describer = None
    if args.describe_images:
        from rag.adapters.describer import GemmaImageDescriber

        describer = GemmaImageDescriber()
    mode_str = " (full rebuild forced)" if getattr(args, "force", False) else ""
    print(f"Building the Index from {args.notes} with {embedder.model_id} ({chunking.strategy}){mode_str} ...")
    try:
        _, report = build_index(
            args.notes,
            args.index_dir,
            embedder,
            tokenizer,
            chunking=chunking,
            image_describer=describer,
            force=getattr(args, "force", False),
        )
    except ValueError as refusal:  # the Index would land inside the Notes
        sys.exit(f"rag: {refusal}")
    per_layer = ", ".join(f"{layer}: {count}" for layer, count in report.admitted.items())
    print(f"Admitted {sum(report.admitted.values())} Documents ({per_layer}).")
    print(f"Documents: {report.documents_added} added, {report.documents_changed} changed, {report.documents_removed} removed, {report.documents_unchanged} unchanged.")
    print(f"Excluded {len(report.excluded)}: {', '.join(report.excluded) or 'nothing'}")
    desc_msg = f" (describing images took {report.image_description_seconds:.1f} s)" if report.image_description_seconds > 0 else ""
    print(f"Created {report.chunks_created} Chunks ({report.chunking_strategy}), {report.chunks_reembedded} re-embedded; embedding took {report.embedding_seconds:.1f} s{desc_msg}.")
    print(f"Warnings: {len(report.warnings)}")
    for warning in report.warnings:
        print(f"  {warning}")
    print(f"Saved to {args.index_dir}")


def run_ask(args):
    if not args.index_dir.is_dir():
        sys.exit(f"rag: no Index at {args.index_dir}; build one with `rag index`")
    import json
    from rag.adapters.generator import GemmaGenerator
    from rag.indexing.index import Index

    try:
        metadata_file = args.index_dir / "metadata.json"
        if metadata_file.exists():
            model_id = json.loads(metadata_file.read_text()).get("embedding_model")
        else:
            model_id = Index.load(args.index_dir).embedding_model
        embedder = resolve_embedder(model_id)
    except Exception as e:
        sys.exit(f"rag: could not determine embedding model for Index at {args.index_dir}: {e}")

    generator = GemmaGenerator()
    try:
        answer = ask(
            args.question, args.index_dir, embedder, generator,
            k=args.k, retrieval=args.retrieval, temperature=args.temperature,
        )
    except (EmbeddingModelMismatch, ValueError) as refusal:  # ValueError: the Question alone is too long
        sys.exit(f"rag: {refusal}")
    if args.show_context:
        print(f"Context for: {answer.question} ({answer.chunking_strategy})\n")
        for item in answer.context:
            chunk = item.chunk
            source_tag = ""
            if chunk.is_image:
                source_tag = f" · {'vision model' if chunk.has_vision_description else 'borrowed text'}"
            print(
                f"  [{item.rank}] {scores(item)} · {chunk.layer} · {chunk.label}"
                f" ({chunk.document}){source_tag} · {chunk.token_count} tokens"
            )
        print("\nThe exact prompt the model reads:")
        print("-" * 72)
        print(answer.prompt)
        print("-" * 72)
        window = generator.context_window
        print(f"Prompt: {answer.stats.prompt_tokens} tokens of the {window}-token context window "
              f"({ANSWER_TOKENS} kept free for the answer)\n")
    for warning in answer.warnings:
        print(f"rag: warning: {warning}", file=sys.stderr)
    print(answer.text.strip())
    print("\nCitations:")
    for number, citation in enumerate(answer.citations, start=1):
        print(f"  {number}. {citation}")
    if args.show_context:
        stats = answer.stats
        print(f"\nTime to first token {stats.time_to_first_token:.2f} s · {stats.tokens_per_second:.1f} tokens/s")


def scores(item):
    """A Context Chunk's score; in hybrid retrieval, its fused score and its place in both rankings."""
    if item.embedding_rank is None:
        return f"score {item.score:.4f}"
    bm25 = f"#{item.bm25_rank} {item.bm25_score:.2f}" if item.bm25_rank else "no match"
    return f"fused {item.score:.4f} · embedding #{item.embedding_rank} {item.embedding_score:.4f} · bm25 {bm25}"
