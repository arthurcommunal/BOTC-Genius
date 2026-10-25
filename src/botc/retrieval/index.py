"""On-disk index: the vectors and the chunks they come from, aligned row by row.

    data/index/<name>/
        vectors.npy     float32 matrix (n, dim): row i is the embedding of chunk i
        chunks.jsonl    line i = chunk i (id, text, header, metadata)
        manifest.json   how the index was built (model, prefixes, chunking, size, content hash)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from botc.retrieval.embedder import DTYPE
from botc.retrieval.store import ShardedVectorStore

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class Index:
    chunks: list[dict]
    vectors: np.ndarray
    manifest: dict

    def chunks_by_id(self) -> dict[str, dict]:
        return {chunk["id"]: chunk for chunk in self.chunks}


def save_index(directory: Path, chunks: list[dict], vectors: np.ndarray, manifest: dict) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).unlink(missing_ok=True)  # the index is incomplete until the end

    np.save(directory / VECTORS_FILE, vectors.astype(DTYPE))
    with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {**manifest, "count": len(chunks), "dim": int(vectors.shape[1])}
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")


def load_index(directory: Path) -> Index:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"No {MANIFEST_FILE} in {directory}: missing or interrupted build.")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    vectors = np.load(directory / VECTORS_FILE)
    lines = (directory / CHUNKS_FILE).read_text("utf-8").splitlines()
    chunks = [json.loads(line) for line in lines]

    expected = (manifest["count"], manifest["dim"])
    if vectors.shape != expected or len(chunks) != manifest["count"]:
        raise ValueError(f"Corrupted index in {directory}: vectors {vectors.shape}, "
                         f"{len(chunks)} chunks, manifest says {expected}")
    return Index(chunks=chunks, vectors=vectors, manifest=manifest)


def build_store(index: Index, num_shards: int = 2) -> ShardedVectorStore:
    """Load the index into the vector store used by the server."""
    store = ShardedVectorStore.create(dim=index.manifest["dim"], num_shards=num_shards)
    store.add(ids=[chunk["id"] for chunk in index.chunks], vectors=index.vectors)
    return store
