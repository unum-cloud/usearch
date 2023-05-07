import numpy as np

import usearch

from numba import cfunc, types, carray

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


count_vectors = 100
count_dimensions = 96
vectors = np.random.uniform(
    0, 0.3, (count_vectors, count_dimensions)).astype(np.float32)
labels = np.array(range(count_vectors), dtype=np.longlong)

index_udf = usearch.Index(dim=96, metric_pointer=python_dot.address)
index_udf.add(labels, vectors, copy=True)
results = index_udf.search(vectors, 10)
print('found', results[0].shape, results[1].shape, results[2].shape)


index = usearch.Index(dim=96)
index.add(labels, vectors, copy=True)
assert index.ndim == count_dimensions
assert index.capacity >= count_vectors
assert len(index) == count_vectors

results = index.search(vectors, 10)
print('found', results[0].shape, results[1].shape, results[2].shape)
