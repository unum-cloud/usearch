import numpy as np

from usearch.index import Index, Matches, MetricKind, BitwiseMetricKind


def vectors(index: Index, count: int) -> np.ndarray:

    if not isinstance(index, Index):
        raise ValueError('Unsupported `index` type')

    dimensions = index.ndim

    if index.metric in BitwiseMetricKind:
        bit_vectors = np.random.randint(2, size=(count, dimensions))
        bit_vectors = np.packbits(bit_vectors, axis=1)
        return bit_vectors

    else:
        x = np.random.rand(count, dimensions).astype(index.numpy_dtype)
        if index.metric == MetricKind.IP:
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        return x


def recall_at_one(index: Index, vectors: np.ndarray) -> float:
    empty_replica: Index = index.fork()

    labels = np.arange(vectors.shape[0])
    empty_replica.add(labels=labels, vectors=vectors)
    matches: Matches = empty_replica.search(empty_replica, 1)

    return np.sum(matches.labels.flatten() == labels) / len(labels)
