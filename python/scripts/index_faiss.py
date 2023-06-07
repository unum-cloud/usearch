import numpy as np
from faiss import IndexHNSWFlat, IndexIVFPQ

from usearch.index import Matches
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


class IndexFAISS:

    def __init__(
        self,
        index: IndexHNSWFlat = None,
        ndim: int = 0,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        *args, **kwargs,
    ):

        if index is None:
            index = IndexHNSWFlat(ndim, connectivity)
            index.hnsw.efConstruction = expansion_add
            index.hnsw.efSearch = expansion_search

        self._faiss = index
        self.specs = {
            'Connectivity': connectivity,
            'Expansion@Add': expansion_add,
            'Expansion@Search': expansion_search,
        }

    def add(self, labels, vectors):
        self._faiss.add(vectors)

    def search(self, queries, k: int) -> Matches:
        distances, labels = self._faiss.search(queries, k)
        return Matches(labels, distances, np.array([k] * queries.shape[0]))

    def __len__(self) -> int:
        return self._faiss.ntotal


class IndexQuantizedFAISS(IndexFAISS):

    def __init__(
        self,
        train: np.ndarray,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        *args, **kwargs,
    ):

        ndim = train.shape[1]
        super().__init__(
            ndim=ndim,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            *args, **kwargs,
        )

        nlist = 10000  # Number of inverted lists (number of partitions).
        nsegment = 16  # Number of segments for PQ (number of subquantizers).
        nbit = 8       # Number of bits to encode each segment.

        self._original_faiss = self._faiss
        self._faiss = IndexIVFPQ(
            self._faiss, ndim, nlist, nsegment, nbit)

        self._faiss.train(train)
        self._faiss.nprobe = 10
