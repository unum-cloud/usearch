import os

import pytest
import numpy as np

from usearch.io import load_matrix, save_matrix
from usearch.eval import random_vectors
from usearch.index import search

from usearch.index import (
    Index,
    Indexes,
    MetricKind,
    ScalarKind,
    Match,
    Matches,
    BatchMatches,
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


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_initialization_and_addition(
    ndim, metric, quantization, dtype, batch_size
):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    assert len(index) == batch_size


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_retrieval(ndim, metric, quantization, dtype, batch_size):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    vectors_retrived = np.vstack(index.get(keys))
    assert np.allclose(vectors_retrived.astype(dtype), vectors, atol=0.1)


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_search(ndim, metric, quantization, dtype, batch_size):
    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)

    if batch_size == 1:
        matches: Matches = index.search(vectors, 10, threads=threads)
        assert matches.keys.ndim == 1
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))

    else:
        matches: BatchMatches = index.search(vectors, 10, threads=threads)
        assert matches.keys.ndim == 2
        assert matches.keys.shape[0] == matches.distances.shape[0]
        assert len(matches) == batch_size
        assert np.all(np.sort(index.keys) == np.sort(keys))


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_index_duplicates(batch_size):
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
    ndim = 8
    index = Index(ndim=ndim, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    assert index.max_level >= 0
    assert index.levels_stats.nodes >= batch_size
    assert index.level_stats(0).nodes == batch_size


@pytest.mark.parametrize("ndim", [1, 3, 8, 32, 256, 4096])
@pytest.mark.parametrize("batch_size", [1, 7, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
def test_index_save_load_restore_copy(ndim, quantization, batch_size):
    index = Index(ndim=ndim, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    index.add(keys, vectors, threads=threads)

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0

    index.load("tmp.usearch")
    assert len(index) == batch_size
    assert len(index[0].flatten()) == ndim

    index = Index.restore("tmp.usearch", view=True)
    assert len(index) == batch_size
    assert len(index[0].flatten()) == ndim

    copied_index = index.copy()
    assert len(copied_index) == len(index)
    assert np.allclose(np.vstack(copied_index.get(keys)), np.vstack(index.get(keys)))

    os.remove("tmp.usearch")


@pytest.mark.parametrize("batch_size", [32])
def test_index_contains_remove_rename(batch_size):
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


@pytest.mark.parametrize("ndim", [3, 97, 256])
@pytest.mark.parametrize("metric", [MetricKind.Cos, MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", [10, 1024])
@pytest.mark.parametrize("quantization", [ScalarKind.F32, ScalarKind.I8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_index_clustering(ndim, metric, quantization, dtype, batch_size):
    if batch_size <= 1:
        return

    index = Index(ndim=ndim, metric=metric, dtype=quantization, multi=False)
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=dtype)
    index.add(keys, vectors, threads=threads)
    clusters: BatchMatches = index.cluster(vectors, threads=threads)
    assert len(clusters.keys) == batch_size
