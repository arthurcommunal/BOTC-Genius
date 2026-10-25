import numpy as np
import pytest

from botc.retrieval.embedder import HashEmbedder
from botc.retrieval.index import MANIFEST_FILE, VECTORS_FILE, build_store, load_index, save_index

CHUNKS = [{"id": f"Page#{i}", "text": f"chunk number {i}"} for i in range(10)]


@pytest.fixture
def embedder():
    return HashEmbedder(dim=16, fixed_cost_s=0, per_item_cost=0)


@pytest.fixture
def saved(tmp_path, embedder):
    vectors = embedder.embed([c["text"] for c in CHUNKS])
    save_index(tmp_path, CHUNKS, vectors, {"model": "hash"})
    return tmp_path, vectors


def test_round_trip_keeps_rows_aligned(saved):
    directory, vectors = saved
    index = load_index(directory)
    assert index.chunks == CHUNKS
    np.testing.assert_array_equal(index.vectors, vectors)
    assert index.manifest == {"model": "hash", "count": 10, "dim": 16}


def test_save_refuses_misaligned_inputs(tmp_path, embedder):
    with pytest.raises(ValueError):
        save_index(tmp_path, CHUNKS, embedder.embed(["only one"]), {})


def test_missing_manifest_means_incomplete_index(saved):
    directory, _ = saved
    (directory / MANIFEST_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        load_index(directory)


def test_load_detects_a_corrupted_index(saved):
    directory, vectors = saved
    np.save(directory / VECTORS_FILE, vectors[:5])
    with pytest.raises(ValueError):
        load_index(directory)


def test_build_store_finds_each_chunk_from_its_own_vector(saved, embedder):
    directory, _ = saved
    store = build_store(load_index(directory), num_shards=2)
    assert len(store) == 10
    query = embedder.embed(["chunk number 7"])[0]
    assert store.search(query, k=1)[0].id == "Page#7"
