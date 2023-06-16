import os
import pytest
import numpy as np

from usearch.io import load_matrix, save_matrix
from usearch.eval import recall_members, random_vectors

from usearch.index import Index, SparseIndex, MetricKind, ScalarKind, Matches
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


dimensions = [3, 97, 256]
batch_sizes = [1, 33]
index_types = [
    ScalarKind.F32, ScalarKind.F64,
    ScalarKind.F16, ScalarKind.F8,
]
numpy_types = [np.float32, np.float64, np.float16]

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
jit_options = [False]
continuous_metrics = [
    MetricKind.Cos,
    MetricKind.L2sq,
]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


@pytest.mark.parametrize('rows', batch_sizes)
@pytest.mark.parametrize('cols', dimensions)
def test_serializing_fbin_matrix(rows: int, cols: int):

    original = np.random.rand(rows, cols).astype(np.float32)
    save_matrix(original, 'tmp.fbin')
    reconstructed = load_matrix('tmp.fbin')
    assert np.allclose(original, reconstructed)
    os.remove('tmp.fbin')


@pytest.mark.parametrize('rows', batch_sizes)
@pytest.mark.parametrize('cols', dimensions)
def test_serializing_ibin_matrix(rows: int, cols: int):

    original = np.random.randint(0, rows+1, size=(rows, cols)).astype(np.int32)
    save_matrix(original, 'tmp.ibin')
    reconstructed = load_matrix('tmp.ibin')
    assert np.allclose(original, reconstructed)
    os.remove('tmp.ibin')


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('metric', continuous_metrics)
@pytest.mark.parametrize('index_type', index_types)
@pytest.mark.parametrize('numpy_type', numpy_types)
@pytest.mark.parametrize('connectivity', connectivity_options)
@pytest.mark.parametrize('jit', jit_options)
def test_index(
        ndim: int, metric: MetricKind,
        index_type: str, numpy_type: str,
        connectivity: int, jit: bool):

    index = Index(
        metric=metric,
        ndim=ndim,
        dtype=index_type,
        connectivity=connectivity,
        expansion_add=DEFAULT_EXPANSION_ADD,
        expansion_search=DEFAULT_EXPANSION_SEARCH,
        jit=jit,
    )
    assert index.ndim == ndim
    assert index.connectivity == connectivity

    vector = random_vectors(count=1, ndim=ndim, dtype=numpy_type).flatten()

    index.add(42, vector)

    assert 42 in index
    assert 42 in index.labels
    assert 43 not in index
    assert index[42] is not None
    assert index[43] is None
    if numpy_type != np.byte:
        assert np.allclose(index[42], vector, atol=0.1)

    matches, distances, count = index.search(vector, 10)
    assert len(index) == 1
    assert len(matches) == count
    assert len(distances) == count
    assert count == 1
    assert matches[0] == 42
    assert distances[0] == pytest.approx(0, abs=1e-3)

    index.save('tmp.usearch')
    index.clear()
    assert len(index) == 0

    index.load('tmp.usearch')
    assert len(index) == 1


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('metric', continuous_metrics)
@pytest.mark.parametrize('batch_size', batch_sizes)
@pytest.mark.parametrize('index_type', index_types)
@pytest.mark.parametrize('numpy_type', numpy_types)
def test_index_batch(
        ndim: int, metric: MetricKind,
        batch_size: int, index_type: str, numpy_type: str):

    index = Index(ndim=ndim, metric=metric, dtype=index_type)

    labels = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=numpy_type)

    index.add(labels, vectors)
    assert len(index) == batch_size
    assert np.allclose(index.get_vectors(
        labels).astype(numpy_type), vectors, atol=0.1)

    matches: Matches = index.search(vectors, 10)
    assert matches.labels.shape[0] == matches.distances.shape[0]
    assert matches.counts.shape[0] == batch_size
    assert np.all(np.sort(index.labels) == np.sort(labels))

    # With high levels of qunatization we may get collisions even
    # on relatively small batches `assert 0.9696969696969697 == 1`
    # https://github.com/unum-cloud/usearch/actions/runs/5290884538/jobs/9575827350
    if numpy_type != np.byte:
        assert recall_members(index, exact=True) == 1


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('batch_size', batch_sizes)
def test_index_numba(ndim: int, batch_size: int):

    try:
        from numba import cfunc, types, carray
    except ImportError:
        return

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    signature = types.float32(types.CPointer(
        types.float32), types.CPointer(types.float32))

    @cfunc(signature)
    def python_dot(a, b):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        c = 0.0
        for i in range(ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    index = Index(ndim=ndim, metric=python_dot)

    labels = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(labels, vectors)
    matches, distances, count = index.search(vectors, 10)
    assert matches.shape[0] == distances.shape[0]
    assert count.shape[0] == batch_size

    assert recall_members(index, exact=True) == 1


@pytest.mark.parametrize('connectivity', connectivity_options)
def test_sets_index(connectivity: int):

    index = SparseIndex(connectivity=connectivity)
    index.add(10, np.array([10, 12, 15], dtype=np.uint32))
    index.add(11, np.array([11, 12, 15, 16], dtype=np.uint32))
    results = index.search(np.array([12, 15], dtype=np.uint32), 10)
    assert list(results) == [10, 11]


@pytest.mark.parametrize('bits', dimensions)
@pytest.mark.parametrize('metric', hash_metrics)
@pytest.mark.parametrize('connectivity', connectivity_options)
@pytest.mark.parametrize('batch_size', batch_sizes)
def test_bitwise_index(bits: int, metric: MetricKind, connectivity: int, batch_size: int):

    index = Index(ndim=bits, metric=metric, connectivity=connectivity)

    labels = np.arange(batch_size)
    bit_vectors = np.random.randint(2, size=(batch_size, bits))
    bit_vectors = np.packbits(bit_vectors, axis=1)

    index.add(labels, bit_vectors)
    index.search(bit_vectors, 10)
