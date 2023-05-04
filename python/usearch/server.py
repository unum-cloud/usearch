import ucall.rich_posix as ucall
import usearch

import fire
import numpy as np
from PIL import Image


def serve(dim: int, metric: str = 'ip'):

    server = ucall.Server()
    index = usearch.Index(dim=dim, metric=metric)

    @server
    def add_one(label: int, vector: np.array):
        labels = np.array([label], dtype=np.longlong)
        vectors = vector.reshape(vector.shape[0], 1)
        index.add(labels, vectors, copy=True)

    @server
    def add_many(labels: np.array, vectors: np.array):
        labels = labels.astype(np.longlong)
        index.add(labels, vectors, copy=True)

    @server
    def search_one(vector: np.array, count: int) -> np.ndarray:
        vectors = vector.reshape(vector.shape[0], 1)
        results = index.search(vectors, 3)
        return results[0][:results[2][0]]

    @server
    def size() -> int:
        return len(index)

    @server
    def ndim() -> int:
        return index.ndim

    @server
    def capacity() -> int:
        return index.capacity()

    @server
    def connectivity() -> int:
        return index.connectivity()

    server.run()


if __name__ == '__main__':
    fire(serve)
