import datetime
import numpy as np
import fire

import faiss
import usearch

import struct
import numpy as np

def measure(f) -> float:
    a = datetime.datetime.now()
    f()
    b = datetime.datetime.now()
    c = b - a
    print(f'Took: {c.seconds:.2f} seconds')
    return c.seconds


def bench_faiss(index, vectors: np.array, queries: np.array, neighbors: np.array):
    dt = measure(lambda: index.add(vectors))
    print(f'- FAISS: {vectors.shape[0]/dt:.2f} vectors/s')


def bench_usearch(index, vectors: np.array, queries: np.array, neighbors: np.array):
    labels = np.arange(vectors.shape[0], dtype=np.longlong)
    assert len(index) == 0
    dt = measure(lambda: index.add(labels, vectors))
    assert len(index) == vectors.shape[0]
    print(f'- USearch: {vectors.shape[0]/dt:.2f} vectors/s')


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

    # Test FAISS
    index = faiss.IndexHNSWFlat(dim, connectivity)
    index.hnsw.efSearch = expansion_search
    index.hnsw.efConstruction = expansion_add
    bench_faiss(index, vectors_mat, queries_mat, neighbors_mat)

    # Test FAISS with default Product Quantization
    nlist = 10000  # Number of inverted lists (number of partitions or cells).
    nsegment = 16  # Number of segments for PQ (number of subquantizers).
    nbit = 8       # Number of bits to encode each segment.
    coarse_quantizer = faiss.IndexHNSWFlat(dim, connectivity)
    index = faiss.IndexIVFPQ(coarse_quantizer, dim, nlist, nsegment, nbit)
    index.train(vectors_mat)
    index.nprobe = 10
    bench_faiss(index, vectors_mat, queries_mat, neighbors_mat)

    # Test USearch `Index`
    index = usearch.Index(
        ndim=dim,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        connectivity=connectivity,
    )
    bench_usearch(index, vectors_mat, queries_mat, neighbors_mat)


if __name__ == '__main__':
    fire.Fire(main)
