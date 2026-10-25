"""Ask a question to an index from the command line and print the closest chunks.

The same path as the server: index on disk -> vector store, question -> query embedding -> search."""

import argparse
from pathlib import Path

from botc.retrieval.embedder import SentenceTransformerEmbedder
from botc.retrieval.index import build_store, load_index

DEFAULT_INDEX = Path("data/index/bge-small-en-v1.5_200w")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    index = load_index(args.index)
    store = build_store(index)
    chunks = index.chunks_by_id()
    # The question must be embedded by the model that embedded the corpus.
    embedder = SentenceTransformerEmbedder(index.manifest["model"])

    query = embedder.embed([args.question])[0]
    for hit in store.search(query, args.k):
        chunk = chunks[hit.id]
        print(f"--- {hit.score:.3f}  {hit.id}  {chunk['url']}")
        print(chunk["header"].splitlines()[0])
        print(chunk["text"][:300], end="\n\n")


if __name__ == "__main__":
    main()
