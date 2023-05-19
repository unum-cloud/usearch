from typing import Union, Optional

import numpy as np
from ucall.client import Client


def _vector_to_ascii(vector: np.ndarray) -> Optional[str]:
    if vector.dtype != np.int8 and vector.dtype != np.uint8 and vector.dtype != np.byte:
        return None
    if not np.all((vector >= 0) | (vector <= 100)):
        return None

    # Let's map [0, 100] to the range from [23, 123],
    # poking 60 and replacing with the 124.
    vector += 23
    vector[vector == 60] = 124
    ascii = str(vector)
    return ascii


class IndexClient:

    def __init__(self, uri: str = '127.0.0.1', port: int = 8545, use_http: bool = True) -> None:
        self.client = Client(uri=uri, port=port, use_http=use_http)

    def add_one(self, label: int, vector: np.ndarray):
        assert isinstance(label, int)
        assert isinstance(vector, np.ndarray)
        vector = vector.flatten()
        ascii = _vector_to_ascii(vector)
        if ascii:
            self.client.add_ascii(label=label, string=ascii)
        else:
            self.client.add_one(label=label, vectors=vector)

    def add_many(self, labels: np.ndarray, vectors: np.ndarray):
        assert isinstance(labels, int)
        assert isinstance(vectors, np.ndarray)
        assert labels.ndim == 1 and vectors.ndim == 2
        assert labels.shape[0] == vectors.shape[0]
        self.client.add_many(labels=labels, vectors=vectors)

    def add(self, labels: Union[np.ndarray, int], vectors: np.ndarray):
        if isinstance(labels, int) or len(labels) == 1:
            return self.add_one(labels, vectors)
        else:
            return self.add_many(labels, vectors)

    def search_one(self, vector: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matches: list[dict] = []
        vector = vector.flatten()
        ascii = _vector_to_ascii(vector)
        if ascii:
            matches = self.client.search_ascii(string=ascii, count=count)
        else:
            matches = self.client.search_one(vector=vector, count=count)

        print(matches.data)
        matches = matches.json

        labels = np.array((1, count), dtype=np.uint32)
        distances = np.array((1, count), dtype=np.float32)
        counts = np.array((1), dtype=np.uint32)
        for col, result in enumerate(matches):
            labels[0, col] = result['label']
            distances[0, col] = result['distance']
        counts[0] = len(matches)

        return labels, distances, counts

    def search_many(self, vectors: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size: int = vectors.shape[0]
        list_of_matches: list[list[dict]] = self.client.search_many(
            vectors=vectors, count=count)

        labels = np.array((batch_size, count), dtype=np.uint32)
        distances = np.array((batch_size, count), dtype=np.float32)
        counts = np.array((batch_size), dtype=np.uint32)
        for row, matches in enumerate(list_of_matches):
            for col, result in enumerate(matches):
                labels[row, col] = result['label']
                distances[row, col] = result['distance']
            counts[row] = len(results)

        return labels, distances, counts

    def search(self, vectors: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if vectors.ndim == 1 or (vectors.ndim == 2 and vectors.shape[0] == 1):
            return self.search_one(vectors, count)
        else:
            return self.search_many(vectors, count)

    def __len__(self):
        return self.client.size().json()

    @property
    def ndim(self):
        return self.client.ndim().json()

    def capacity(self):
        return self.client.capacity().json()

    def connectivity(self):
        return self.client.connectivity().json()

    def load(self, path: str):
        raise NotImplementedError()

    def view(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()


if __name__ == '__main__':
    index = IndexClient()
    index.add(42, np.array([0.4] * 256, dtype=np.float32))
    results = index.search(np.array([0.4] * 256, dtype=np.float32), 10)
    print(results)
