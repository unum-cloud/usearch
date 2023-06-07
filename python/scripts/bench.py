from typing import Union, Optional

import fire

from usearch.index import Index
from usearch.io import load_matrix
from usearch.eval import Benchmark
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


def main(
    vectors: str,
    queries: str,
    neighbors: str,
    connectivity: int = DEFAULT_CONNECTIVITY,
    expansion_add: int = DEFAULT_EXPANSION_ADD,
    expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    k: Optional[int] = None,
):

    vectors_mat = load_matrix(vectors)
    queries_mat = load_matrix(queries)
    neighbors_mat = load_matrix(neighbors)
    dim = vectors_mat.shape[1]
    if k:
        neighbors_mat = neighbors_mat[:, :1]

    for jit in [False, True]:
        for dtype in ['f32', 'f16', 'f8']:
            name = f'USearch: HNSW<{dtype}>'
            if jit:
                name += ' + Numba'
            print(name)

            try:
                index = Index(
                    ndim=dim,
                    jit=jit,
                    dtype=dtype,
                    expansion_add=expansion_add,
                    expansion_search=expansion_search,
                    connectivity=connectivity,
                )
                Benchmark(index, vectors_mat, queries_mat, neighbors_mat).log()
            except (ImportError, ModuleNotFoundError):
                print('... Skipping!')

    # Don't depend on the FAISS installation for benchmarks
    try:
        from index_faiss import IndexFAISS, IndexQuantizedFAISS

        print('FAISS: HNSW')
        index = IndexFAISS(
            ndim=dim,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
        )
        Benchmark(index, vectors_mat, queries_mat, neighbors_mat).log()

        print('FAISS: HNSW + IVFPQ')
        index = IndexQuantizedFAISS(
            train=vectors_mat,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
        )
        Benchmark(index, vectors_mat, queries_mat, neighbors_mat).log()

    except (ImportError, ModuleNotFoundError):
        print('... Skipping FAISS benchmarks')


if __name__ == '__main__':
    fire.Fire(main)
