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


def recall_members(index: Index, *args, **kwargs) -> float:
    vectors: np.ndarray = np.vstack([
        index[i] for i in range(len(index))
    ])
    labels: np.ndarray = np.arange(vectors.shape[0])
    matches: Matches = index.search(vectors, 1, *args, **kwargs)
    return matches.recall_first(labels)
