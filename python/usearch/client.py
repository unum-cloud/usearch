import numpy as np
from ucall.client import Client


class IndexClient:

    def __init__(self, uri: str = '127.0.0.1', port: int = 8545, use_http: bool = True) -> None:
        self.client = Client(uri=uri, port=port, use_http=use_http)

    def add(self, labels: np.ndarray, vectors: np.ndarray):
        if isinstance(labels, int):
            self.client.add_one(label=labels, vectors=vectors)
        else:
            self.client.add_many(labels=labels, vectors=vectors)

    def search(self, vectors: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matches = []
        distances = []
        counts = []
        # return self.client.search_one(vectors=vectors, count=count)
        return matches, distances, counts

    def __len__(self):
        return self.client.size().json

    @property
    def ndim(self):
        return self.client.ndim().json

    def capacity(self):
        return self.client.capacity().json

    def connectivity(self):
        return self.client.connectivity().json

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
