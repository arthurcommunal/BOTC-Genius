"""Build the retrieval index: chunk the clean corpus, embed every chunk, save to disk.

Each (model, chunk size) pair gets its own directory, e.g. data/index/bge-small-en-v1.5_200w,
so several indexes can be compared side by side with the eval set.
"""

import argparse
import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from botc.corpus.chunk import DEFAULT_MAX_WORDS, DEFAULT_PROCESSED, Chunk, chunk_corpus
from botc.retrieval.embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from botc.retrieval.index import save_index

DEFAULT_INDEX_ROOT = Path("data/index")


def content_hash(chunks: list[Chunk]) -> str:
    """Fingerprint of what was embedded: same hash = same texts, in the same order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.embedding_text().encode("utf-8"))
    return digest.hexdigest()


def build_index(processed_dir: Path, out_dir: Path, model_name: str, max_words: int) -> None:
    chunks = chunk_corpus(processed_dir, max_words)
    print(f"{len(chunks)} chunks, embedding them with {model_name}...")

    embedder = SentenceTransformerEmbedder(model_name)
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks], show_progress=True)

    manifest = {
        "model": model_name,
        "query_prefix": embedder.query_prefix,
        "passage_prefix": embedder.passage_prefix,
        "max_words": max_words,
        "content_sha256": content_hash(chunks),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_index(out_dir, [asdict(c) for c in chunks], vectors, manifest)
    print(f"Index written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED, help="directory written by clean.py")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--out", type=Path, help="index directory (default: data/index/<model>_<max-words>w)")
    args = parser.parse_args()

    out = args.out or DEFAULT_INDEX_ROOT / f"{args.model.split('/')[-1]}_{args.max_words}w"
    build_index(args.processed, out, args.model, args.max_words)


if __name__ == "__main__":
    main()
