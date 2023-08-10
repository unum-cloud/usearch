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
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)

# When running the tests concurrently in different VMs, one can get access violation errors.
# To avoid that, we embed the OS name into the temporary files name.
# https://github.com/unum-cloud/usearch/actions/runs/5705359698/job/15459945738
temporary_filename = f"tmp-{os.name}"
temporary_usearch_filename = temporary_filename + ".usearch"

dimensions = [3, 97, 256]
batch_sizes = [1, 77]
index_types = [
    ScalarKind.F32,
    ScalarKind.F64,
    ScalarKind.F16,
    ScalarKind.I8,
]
numpy_types = [np.float32, np.float64, np.float16]

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
continuous_metrics = [
    MetricKind.Cos,
    MetricKind.L2sq,
]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_fbin_matrix(rows: int, cols: int):
    """
    Test the serialization of floating point binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.rand(rows, cols).astype(np.float32)
    save_matrix(original, temporary_filename + ".fbin")
    reconstructed = load_matrix(temporary_filename + ".fbin")
    assert np.allclose(original, reconstructed)
    os.remove(temporary_filename + ".fbin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_ibin_matrix(rows: int, cols: int):
    """
    Test the serialization of integer binary matrix.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.randint(0, rows + 1, size=(rows, cols)).astype(np.int32)
    save_matrix(original, temporary_filename + ".ibin")
    reconstructed = load_matrix(temporary_filename + ".ibin")
    assert np.allclose(original, reconstructed)
    os.remove(temporary_filename + ".ibin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_exact_search(rows: int, cols: int):
    """
    Test exact search.

    :param int rows: The number of rows in the matrix.
    :param int cols: The number of columns in the matrix.
    """
    original = np.random.rand(rows, cols)
    matches: BatchMatches = search(original, original, 10, exact=True)
    top_matches = (
        [int(m.keys[0]) for m in matches] if rows > 1 else int(matches.keys[0])
    )
    assert np.all(top_matches == np.arange(rows))

    matches: Matches = search(original, original[0], 10, exact=True)
    top_match = int(matches.keys[0])
    assert top_match == 0


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("metric", continuous_metrics)
@pytest.mark.parametrize("index_type", index_types)
@pytest.mark.parametrize("numpy_type", numpy_types)
@pytest.mark.parametrize("connectivity", connectivity_options)
def test_minimal_index(
    ndim: int,
    metric: MetricKind,
    index_type: ScalarKind,
    numpy_type: str,
    connectivity: int,
):
    index = Index(
        metric=metric,
        ndim=ndim,
        dtype=index_type,
        connectivity=connectivity,
        expansion_add=DEFAULT_EXPANSION_ADD,
        expansion_search=DEFAULT_EXPANSION_SEARCH,
    )
    assert index.ndim == ndim
    assert index.connectivity == connectivity

    vector = random_vectors(count=1, ndim=ndim, dtype=numpy_type).flatten()
    index.add(42, vector)

    # Ban vectors with a wrong number of dimensions
    with pytest.raises(Exception):
        index.add(
            42, random_vectors(count=1, ndim=(ndim * 2), dtype=numpy_type).flatten()
        )

    # Ban duplicates unless explicitly allowed
    with pytest.raises(Exception):
        index.add(42, vector)

    assert len(index) == 1, "Size after addition"
    assert 42 in index, "Presence in the index"
    assert 42 in index.keys, "Presence among keys"
    assert 43 not in index, "Presence in the index, false positive"
    assert index[42] is not None, "Vector recovery"
    assert index[43] is None, "Vector recovery, false positive"
    assert len(index[42]) == ndim
    if numpy_type != np.byte:
        assert np.allclose(index[42], vector, atol=0.1)

    matches: Matches = index.search(vector, 10)
    assert len(matches.keys) == 1, "Number of matches"
    assert len(matches.keys) == len(matches.distances), "Symmetric match sub-arrays"
    assert len({match.key for match in matches}) == 1, "Iteration over matches"
    assert matches[0].key == 42
    assert matches[0].distance == pytest.approx(0, abs=1e-3)
    assert matches.computed_distances != 0
    assert matches.visited_members != 0

    # Validating the index structure and metadata:
    assert index.max_level >= 0
    assert index.levels_stats.nodes >= 1
    assert index.level_stats(0).nodes == 1
    assert str(index).startswith("usearch.")

    # Try removals
    other_vector = random_vectors(count=1, ndim=ndim, dtype=numpy_type).flatten()
    index.add(43, other_vector)
    assert len(index) == 2
    index.remove(43)
    assert len(index) == 1

    # Try inserting back
    index.add(43, other_vector)
    assert len(index) == 2
    index.remove(43)
    assert len(index) == 1

    index.save(temporary_usearch_filename)

    # Re-populate cleared index
    index.clear()
    assert len(index) == 0
    index.add(42, vector)
    assert len(index) == 1
    matches: Matches = index.search(vector, 10)
    assert len(matches) == 1

    index_copy = index.copy()
    assert len(index_copy) == 1
    assert len(index_copy[42]) == ndim
    matches_copy: Matches = index_copy.search(vector, 10)
    assert np.all(matches_copy.keys == matches.keys)

    index.load(temporary_usearch_filename)
    assert len(index) == 1
    assert len(index[42]) == ndim

    matches_loaded: Matches = index.search(vector, 10)
    assert np.all(matches_loaded.keys == matches.keys)

    index = Index.restore(temporary_usearch_filename, view=True)
    assert len(index) == 1
    assert len(index[42]) == ndim

    matches_viewed: Matches = index.search(vector, 10)
    assert np.all(matches_viewed.keys == matches.keys)

    # Cleanup
    index.reset()
    os.remove(temporary_usearch_filename)

    # Try opening a missing file
    meta = Index.metadata(temporary_usearch_filename)
    assert meta is None
    index = Index.restore(temporary_usearch_filename)
    assert index is None

    # Try opening a corrupt file
    with open(temporary_usearch_filename, "w") as file:
        file.write("Some random string")
    meta = Index.metadata(temporary_usearch_filename)
    assert meta is None
    index = Index.restore(temporary_usearch_filename)

    # Try saving and opening and empty index
    index_copy.reset()
    index_copy.save(temporary_usearch_filename)
    assert Index.restore(temporary_usearch_filename, view=False) is not None
    assert Index.restore(temporary_usearch_filename, view=True) is not None

    assert index is None
    os.remove(temporary_usearch_filename)


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("metric", continuous_metrics)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("index_type", index_types)
@pytest.mark.parametrize("numpy_type", numpy_types)
def test_index_batch(
    ndim: int,
    metric: MetricKind,
    batch_size: int,
    index_type: ScalarKind,
    numpy_type: str,
):
    index = Index(ndim=ndim, metric=metric, dtype=index_type)

    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=numpy_type)

    index.add(keys, vectors, threads=2)
    assert len(index) == batch_size
    assert np.allclose(index.get_vectors(keys).astype(numpy_type), vectors, atol=0.1)

    # Ban duplicates unless explicitly allowed
    with pytest.raises(Exception):
        index.add(keys, vectors, threads=2)

    matches: BatchMatches = index.search(vectors, 10, threads=2)
    assert matches.keys.shape[0] == matches.distances.shape[0]
    assert len(matches) == batch_size
    assert np.all(np.sort(index.keys) == np.sort(keys))

    if batch_size > 1:
        assert index.max_level >= 1
    else:
        assert index.max_level >= 0
    assert index.levels_stats.nodes >= batch_size
    assert index.level_stats(0).nodes == batch_size

    index.save(temporary_usearch_filename)
    index.clear()
    assert len(index) == 0

    index.load(temporary_usearch_filename)
    assert len(index) == batch_size
    assert len(index[0]) == ndim

    if batch_size > 1:
        matches_loaded: BatchMatches = index.search(vectors, 10, threads=2)
        for idx in range(len(matches_loaded)):
            assert np.all(matches_loaded[idx].keys == matches[idx].keys)

    index = Index.restore(temporary_usearch_filename, view=True)
    assert len(index) == batch_size
    assert len(index[0]) == ndim

    if batch_size > 1:
        matches_viewed: BatchMatches = index.search(vectors, 10, threads=2)
        for idx in range(len(matches_viewed)):
            assert np.all(matches_viewed[idx].keys == matches[idx].keys)

    # Test clustering
    if batch_size > 1:
        clusters: BatchMatches = index.cluster(vectors, 1, threads=2)
        assert len(clusters.keys) == batch_size

    # Cleanup
    index.reset()
    os.remove(temporary_usearch_filename)


@pytest.mark.parametrize("metric", [MetricKind.L2sq])
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("index_type", index_types)
@pytest.mark.parametrize("numpy_type", numpy_types)
def test_exact_recall(
    metric: MetricKind,
    batch_size: int,
    index_type: ScalarKind,
    numpy_type: str,
):
    ndim: int = batch_size
    index = Index(ndim=ndim, metric=metric, dtype=index_type)
    assert index.ndim == ndim

    vectors = np.zeros((batch_size, ndim), dtype=numpy_type)
    for i in range(batch_size):
        vectors[i, i] = 1
    keys = np.arange(batch_size)
    index.add(keys, vectors)
    assert len(index) == batch_size

    # Search one at a time
    for i in range(batch_size):
        matches: Matches = index.search(vectors[i], 10, exact=True)
        found_labels = matches.keys
        assert found_labels[0] == i
        assert matches.computed_distances == len(index)
        assert matches.visited_members == 0, "Exact search won't traverse the graph"

    # Search the whole batch
    if batch_size > 1:
        matches: BatchMatches = index.search(vectors, 10, exact=True)
        assert matches.computed_distances == len(index) * len(vectors)
        assert matches.visited_members == 0, "Exact search won't traverse the graph"

        found_labels = matches.keys
        for i in range(batch_size):
            assert found_labels[i, 0] == i

    # Match entries against themselves
    index_copy: Index = index.copy()
    mapping: dict = index.join(index_copy, exact=True)
    for man, woman in mapping.items():
        assert man == woman, "Stable marriage failed"


def test_indexes():
    ndim = 10
    index_a = Index(ndim=ndim)
    index_b = Index(ndim=ndim)

    vectors = random_vectors(count=3, ndim=ndim)
    index_a.add(42, vectors[0])
    index_b.add(43, vectors[1])

    indexes = Indexes([index_a, index_b])
    matches = indexes.search(vectors[2], 10)
    assert len(matches) == 2


@pytest.mark.parametrize("bits", dimensions)
@pytest.mark.parametrize("metric", hash_metrics)
@pytest.mark.parametrize("connectivity", connectivity_options)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_bitwise_index(
    bits: int,
    metric: MetricKind,
    connectivity: int,
    batch_size: int,
):
    index = Index(ndim=bits, metric=metric, connectivity=connectivity)

    keys = np.arange(batch_size)
    byte_vectors = np.random.randint(2, size=(batch_size, bits))
    bit_vectors = np.packbits(byte_vectors, axis=1)

    index.add(keys, bit_vectors)
    assert np.all(index.get_vectors(keys, ScalarKind.B1) == bit_vectors)

    index.search(bit_vectors, 10)


if __name__ == "__main__":
    pytest.main(args=["python/scripts/test.py", "-s", "-x", "-v"])
