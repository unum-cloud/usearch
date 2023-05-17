import pytest
import numpy as np


from usearch.index import Index, SetsIndex

dimensions = [13, 99, 100, 256]
sizes = [1, 2, 100]
types = ['f32', 'f16', 'f8', 'f64']


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('dtype', types)
def test_cos(ndim: int, dtype: str):
    index = Index(ndim=ndim, metric='cos', dtype=dtype)
    vector = np.random.uniform(0, 0.7, (index.ndim)).astype(np.float32)
    index.add(42, vector)
    matches, distances, count = index.search(vector, 10)

    assert len(matches) == count
    assert len(distances) == count
    assert count == 1
    assert matches[0] == 42
    assert distances[0] == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize('ndim', dimensions)
@pytest.mark.parametrize('size', sizes)
@pytest.mark.parametrize('dtype', types)
def test_cos_batch(ndim: int, size: int, dtype: str):
    index = Index(ndim=ndim, metric='cos', dtype=dtype)

    vectors = np.random.uniform(
        0, 0.3, (size, index.ndim)).astype(np.float32)
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

    index = Index(ndim=ndim, metric_pointer=python_dot.address)

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
    results = index.search(np.array([12, 15], dtype=np.uint32), 20)

    # assert list(results) == [10, 11]


if __name__ == '__main__':
    pass
