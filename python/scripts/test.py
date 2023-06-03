import pytest
import numpy as np

from usearch.io import load_matrix
from usearch.index import Index, SetsIndex


dimensions = [13, 99, 100, 256]
sizes = [1, 2, 100]
index_types = ['f32', 'f64', 'f16', 'f8']
numpy_types = [np.float32, np.float64, np.float16, np.byte]
connectivity_options = [3, 13, 50]
jit_options = [False, True]
metrics = ['cos', 'l2sq']


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('metric', metrics)
@pytest.mark.parametrize('index_type', index_types)
@pytest.mark.parametrize('numpy_type', numpy_types)
@pytest.mark.parametrize('connectivity', connectivity_options)
@pytest.mark.parametrize('jit', jit_options)
def test_basics(
        ndim: int, metric: str,
        index_type: str, numpy_type: str,
        connectivity: int, jit: bool):

    index = Index(
        metric=metric,
        ndim=ndim,
        dtype=index_type,
        connectivity=connectivity,
        jit=jit,
    )
    assert index.ndim == ndim
    assert index.connectivity == connectivity

    limit = 0.7 if numpy_type != np.byte else 70
    vector = np.random.uniform(0, limit, (index.ndim)).astype(numpy_type)
    index.add(42, vector)
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
@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('index_type', index_types)
@pytest.mark.parametrize('numpy_type', numpy_types)
def test_l2sq_batch(ndim: int, size: int, index_type: str, numpy_type: str):
    index = Index(ndim=ndim, metric='l2sq', dtype=index_type)

    limit = 0.7 if numpy_type != np.byte else 70
    vectors = np.random.uniform(
        0, limit, (size, index.ndim)).astype(numpy_type)
    labels = np.array(range(size), dtype=np.longlong)

    index.add(labels, vectors)
    matches, distances, count = index.search(vectors, 10)

    assert matches.shape[0] == distances.shape[0]
    assert count.shape[0] == size


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('size', sizes)
def test_user_defined_function(ndim: int, size: int):

    try:
        from numba import cfunc, types, carray
    except ImportError:
        return

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    signature = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
        types.size_t, types.size_t)

    @cfunc(signature)
    def python_dot(a, b, n, m):
        a_array = carray(a, n)
        b_array = carray(b, n)
        c = 0.0
        for i in range(n):
            c += a_array[i] * b_array[i]
        return c

    index = Index(ndim=ndim, metric=python_dot.address)

    vectors = np.random.uniform(
        0, 0.3, (size, index.ndim)).astype(np.float32)
    labels = np.array(range(size), dtype=np.longlong)

    index.add(labels, vectors)
    matches, distances, count = index.search(vectors, 10)

    assert matches.shape[0] == distances.shape[0]
    assert count.shape[0] == size


def test_sets():

    index = SetsIndex()
    index.add(10, np.array([10, 12, 15], dtype=np.uint32))
    index.add(11, np.array([11, 12, 15, 16], dtype=np.uint32))
    results = index.search(np.array([12, 15], dtype=np.uint32), 10)
    assert list(results) == [10, 11]


if __name__ == '__main__':
    pass
