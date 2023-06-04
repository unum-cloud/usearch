import time
import numpy as np
import math

import fire

from usearch.index import Index
from usearch.io import load_matrix


def measure(f) -> float:
    a = time.time_ns()
    result = f()
    b = time.time_ns()
    c = b - a
    secs = c / (10 ** 9)
    print(f'- Took: {secs:.2f} seconds')
    return secs, result


def bench_faiss(index, vectors: np.array, queries: np.array, neighbors: np.array):
    dt, _ = measure(lambda: index.add(vectors))
    print(f'- Performance: {vectors.shape[0]/dt:.2f} insertions/s')

    dt, results = measure(lambda: index.search(queries, neighbors.shape[1]))
    print(f'- Performance: {queries.shape[0]/dt:.2f} queries/s')

    recall_at_one = 0.0
    matches = results[1]
    for i in range(queries.shape[0]):
        recall_at_one += neighbors[i, 0] in matches[i, :]
    recall_at_one /= queries.shape[0]
    print(f'- Recall@1: {recall_at_one * 100:.2f} %')


def bench_usearch(index, vectors: np.array, queries: np.array, neighbors: np.array):
    labels = np.arange(vectors.shape[0], dtype=np.longlong)
    assert len(index) == 0
    dt, _ = measure(lambda: index.add(labels, vectors))
    assert len(index) == vectors.shape[0]
    print(f'- Performance: {vectors.shape[0]/dt:.2f} insertions/s')

    dt, results = measure(lambda: index.search(queries, neighbors.shape[1]))
    print(f'- Performance: {queries.shape[0]/dt:.2f} queries/s')

    recall_at_one = 0.0
    matches = results[0]
    for i in range(queries.shape[0]):
        recall_at_one += neighbors[i, 0] in matches[i, :]
    recall_at_one /= queries.shape[0]
    print(f'- Recall@1: {recall_at_one * 100:.2f} %')


def main(
    vectors: str,
    queries: str,
    neighbors: str,
    connectivity: int = 16,
    expansion_add: int = 128,
    expansion_search: int = 64,
):

    vectors_mat = load_matrix(vectors)
    queries_mat = load_matrix(queries)
    neighbors_mat = load_matrix(neighbors)
    dim = vectors_mat.shape[1]

    print('USearch: HNSW')
    index = Index(
        ndim=dim,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        connectivity=connectivity,
    )
    bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)

    print('USearch: HNSW + Half Precision')
    index = Index(
        ndim=dim,
        dtype='f16',
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        connectivity=connectivity,
    )
    bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)

    print('USearch: HNSW + Quarter Precision')
    index = Index(
        ndim=dim,
        dtype='f8',
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        connectivity=connectivity,
    )
    bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    try:
        from numba import cfunc, types, carray

        signature_f32 = types.float32(
            types.CPointer(types.float32),
            types.CPointer(types.float32),
            types.uint64, types.uint64)

        @cfunc(signature_f32)
        def inner_product_f32(a, b, n, m):
            a_array = carray(a, dim)
            b_array = carray(b, dim)
            c = types.float32(0)
            for i in range(dim):
                c += a_array[i] * b_array[i]
            return types.float32(1 - c)

        signature_f8 = types.float32(
            types.CPointer(types.int8),
            types.CPointer(types.int8),
            types.uint64, types.uint64)

        @cfunc(signature_f8)
        def cos_f8(a, b, n, m):
            a_array = carray(a, dim)
            b_array = carray(b, dim)
            c = types.int32(0)
            a_sq = types.int32(0)
            b_sq = types.int32(0)
            for i in range(dim):
                ai = types.int16(a_array[i])
                bi = types.int16(b_array[i])
                c += ai * bi
                a_sq += ai * ai
                b_sq += bi * bi
            a_norm = math.sqrt(types.float32(a_sq))
            b_norm = math.sqrt(types.float32(b_sq))
            c_norm = types.float32(c) / (a_norm * b_norm)
            return types.float32(1 - c_norm)

        print('USearch: HNSW + Numba JIT')
        index = Index(
            ndim=dim,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
            metric=inner_product_f32.address,
        )
        bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)

        print('USearch: HNSW + Quarter Precision + Numba JIT')
        index = Index(
            ndim=dim,
            dtype='f8',
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
            metric=cos_f8.address,
        )
        bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)
    except ImportError:
        print('Skipping Numba benchmarks')

    # Don't depend on the FAISS installation for benchmarks
    try:
        from faiss import IndexHNSWFlat, IndexIVFPQ

        print('FAISS: HNSW')
        index = IndexHNSWFlat(dim, connectivity)
        index.hnsw.efSearch = expansion_search
        index.hnsw.efConstruction = expansion_add
        bench_faiss(index, vectors_mat, queries_mat, neighbors_mat)

        print('FAISS: HNSW + IVFPQ')
        # Number of inverted lists (number of partitions or cells).
        nlist = 10000
        nsegment = 16  # Number of segments for PQ (number of subquantizers).
        nbit = 8       # Number of bits to encode each segment.
        coarse_quantizer = IndexHNSWFlat(dim, connectivity)
        index = IndexIVFPQ(coarse_quantizer, dim, nlist, nsegment, nbit)
        index.train(vectors_mat)
        index.nprobe = 10
        bench_faiss(index, vectors_mat, queries_mat, neighbors_mat)

    except ImportError:
        print('Skipping FAISS benchmarks')


if __name__ == '__main__':
    fire.Fire(main)
