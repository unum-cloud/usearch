import os
from time import time

import pytest
import numpy as np

import usearch
from usearch.eval import random_vectors, self_recall, SearchStats
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
    Match,
    Matches,
    BatchMatches,
    Clustering,
)
from usearch.index import (
    DEFAULT_CONNECTIVITY,
)


ndims = [3, 97, 256]
batch_sizes = [1, 11, 77]
quantizations = [
    ScalarKind.F32,
    ScalarKind.F64,
    ScalarKind.F16,
    ScalarKind.BF16,
    ScalarKind.I8,
]
dtypes = [np.float32, np.float64, np.float16]
threads = 2

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
continuous_metrics = [MetricKind.Cos, MetricKind.L2sq]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


def reset_randomness():
    np.random.seed(int(time()))


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_initialization_and_addition(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.F16, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_retrieval(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    vectors_retrieved = np.vstack(index.get(keys, dtype))
    assert np.allclose(vectors_retrieved, vectors, atol=0.1)

    # Try retrieving all the keys
    keys_retrieved = index.keys
    keys_retrieved = np.array(keys_retrieved)
    assert np.all(np.sort(keys_retrieved) == keys)

    # Try retrieving all of them
    if quantization != ScalarKind.I8:
        # The returned vectors can be in a different order
        vectors_batch_retrieved = index.vectors
        vectors_reordering = np.argsort(keys_retrieved)
        vectors_batch_retrieved = vectors_batch_retrieved[vectors_reordering]
        assert np.allclose(vectors_batch_retrieved, vectors, atol=0.1)

    if quantization != ScalarKind.I8 and batch_size > 1:
        # When dealing with non-continuous data, it's important to check that
        # the native bindings access them with correct strides or normalize
        # similar to `np.ascontiguousarray`:
        index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
        vectors = random_vectors(count=batch_size, ndim=ndim + 1, dtype=dtype)
        # Let's skip the first dimension of each vector:
        vectors = vectors[:, 1:]
        index.add(keys, vectors, threads=threads)
        vectors_retrieved = np.vstack(index.get(keys, dtype))
        assert np.allclose(vectors_retrieved, vectors, atol=0.1)

        # Try a transposed version of the same vectors, that is not C-contiguous
        # and should raise an exception!
        index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
        vectors = random_vectors(count=ndim, ndim=batch_size, dtype=dtype)  #! reversed dims
        assert vectors.strides == (batch_size * dtype().itemsize, dtype().itemsize)
        assert vectors.T.strides == (dtype().itemsize, batch_size * dtype().itemsize)
        with pytest.raises(Exception):
            index.add(keys, vectors.T, threads=threads)


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_search(ndim, metric, quantization, dtype, batch_size):
    reset_randomness()

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    if batch_size == 1:
        matches: Matches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, Matches)
        assert isinstance(matches[0], Match)
        assert matches.keys.ndim == 1
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))

    else:
        matches: BatchMatches = index.search(vectors, 10, threads=threads)
        assert isinstance(matches, BatchMatches)
        assert isinstance(matches[0], Matches)
        assert isinstance(matches[0][0], Match)
        assert matches.keys.ndim == 2
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_self_recall(ndim: int, batch_size: int):
    """
    Test self-recall evaluation scripts.
    """
    reset_randomness()

    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    stats_all: SearchStats = self_recall(index, keys=keys)
    stats_quarter: SearchStats = self_recall(index, sample=0.25, count=10)

    assert stats_all.computed_distances > 0
    assert stats_quarter.computed_distances > 0


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_duplicates(batch_size):
    reset_randomness()

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    with pytest.raises(Exception):
        index.add(keys, vectors, threads=threads)

    index = Index(ndim=ndim, multi=True)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size * 2

    two_per_key = index.get(keys)
    assert np.vstack(two_per_key).shape == (2 * batch_size, ndim)


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_stats(batch_size):
    reset_randomness()

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    assert index.max_level >= 0
    assert index.stats.nodes >= batch_size
    assert index.levels_stats[0].nodes == batch_size
    assert index.level_stats(0).nodes == batch_size

    assert index.levels_stats[index.max_level].nodes > 0


@pytest.mark.parametrize("ndim", [1, 3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [0, 1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
def test_index_save_load_restore_copy(ndim, quantization, batch_size):
    reset_randomness()
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)
        index.add(keys, vectors, threads=threads)

    # Try copying the original
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0
    assert os.path.exists("tmp.usearch")

    index.load("tmp.usearch")
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    index_meta = Index.metadata("tmp.usearch")
    assert index_meta is not None

    index = Index.restore("tmp.usearch", view=False)
    assert len(index) == batch_size
    if batch_size > 0:
        assert len(index[0].flatten()) == ndim

    # Try copying the restored index
    copied_index = index.copy()
    assert len(copied_index) == len(index)
    if batch_size > 0:
        assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    # Perform the same operations in RAM, without touching the filesystem
    serialized_index = index.save()
    deserialized_metadata = Index.metadata(serialized_index)
    assert deserialized_metadata is not None

    deserialized_index = Index.restore(serialized_index)
    assert len(deserialized_index) == len(index)
    assert set(np.array(deserialized_index.keys)) == set(np.array(index.keys))
    if batch_size > 0:
        assert np.allclose(np.vstack(deserialized_index.get(keys)), np.vstack(index.get(keys)))

    deserialized_index.reset()
    index.reset()
    os.remove("tmp.usearch")


@pytest.mark.parametrize("ndim", [1, 3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [0, 1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
def test_index_restore_multithread_search(
    ndim, quantization, batch_size, tmp_path
):
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "tmp.usearch"

    reset_randomness()
    index = Index(ndim=ndim, dtype=quantization, multi=False)

    if batch_size > 0:
        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)
        index.add(keys, vectors, threads=threads)

    index.save(index_path)

    query = random_vectors(count=batch_size, ndim=ndim)

    # When restoring from disk, search *must not* fail if using multiple
    # threads.

    index_restore = Index.restore(index_path, view=False)
    result = index_restore.search(query, count=10, threads=threads)

    index_restore_view = Index.restore(index_path, view=True)
    result_view = index_restore_view.search(query, count=10, threads=threads)

    assert np.allclose(result.distances, result_view.distances)


@pytest.mark.parametrize("batch_size", [32])
def test_index_contains_remove_rename(batch_size):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    removed_keys = keys[: batch_size // 2]
    remaining_keys = keys[batch_size // 2 :]
    index.remove(removed_keys)
    del index[removed_keys] # ! This will trigger the `__delitem__` dunder method
    assert len(index) == (len(keys) - len(removed_keys))
    assert np.sum(index.contains(keys)) == len(remaining_keys)
    assert np.sum(index.count(keys)) == len(remaining_keys)
    assert np.sum(index.count(removed_keys)) == 0

    assert keys[0] not in index
    assert keys[-1] in index

    renamed_counts = index.rename(removed_keys, removed_keys)
    assert np.sum(index.count(renamed_counts)) == 0

    renamed_counts = index.rename(remaining_keys, removed_keys)
    assert np.sum(index.count(removed_keys)) == len(index)


@pytest.mark.skip(reason="Not guaranteed")
@pytest.mark.parametrize("batch_size", [3, 17, 33])
@pytest.mark.parametrize("threads", [1, 4])
def test_index_oversubscribed_search(batch_size: int, threads: int):
    reset_randomness()
    if batch_size <= 1:
        return

    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors, threads=threads)
    assert np.all(index.contains(keys))
    assert np.all(index.count(keys) == np.ones(batch_size))

    batch_matches: BatchMatches = index.search(vectors, batch_size * 10, threads=threads)
    for i, match in enumerate(batch_matches):
        assert i == match.keys[0]
        assert len(match.keys) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [500, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_clustering(ndim, metric, quantization, dtype, batch_size):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    clusters: Clustering = index.cluster(vectors=vectors, threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(threads=threads)
    assert len(clusters.matches.keys) == batch_size

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(keys=keys[:50], threads=threads)
    assert len(clusters.matches.keys) == 50

    # If no argument is provided, we cluster the present entries
    clusters: Clustering = index.cluster(min_count=3, max_count=10, threads=threads)
    unique_clusters = set(clusters.matches.keys.flatten().tolist())
    assert len(unique_clusters) >= 3 and len(unique_clusters) <= 10
