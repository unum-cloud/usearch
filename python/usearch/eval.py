from time import time_ns
from typing import Tuple, Any, Callable, Generator, Union, Optional
from dataclasses import dataclass

import numpy as np

from usearch.index import Index, Matches, MetricKind, BitwiseMetricKind, Label


def random_vectors(index: Index, count: int) -> np.ndarray:

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


def measure_seconds(f: Callable) -> Tuple[float, Any]:
    a = time_ns()
    result = f()
    b = time_ns()
    c = b - a
    secs = c / (10 ** 9)
    return secs, result


@dataclass
class BenchmarkResult:

    add_per_second: Optional[float] = None
    search_per_second: Optional[float] = None
    recall_at_one: Optional[float] = None

    def __repr__(self) -> str:
        parts = []
        if self.add_per_second:
            parts.append(f'{self.add_per_second:.2f} add/s')
        if self.search_per_second:
            parts.append(f'{self.search_per_second:.2f} search/s')
        if self.recall_at_one:
            parts.append(f'{self.recall_at_one * 100:.2f}% recall@1')
        return ', '.join(parts)


@dataclass
class Benchmark:

    index: Index
    vectors: Optional[Union[np.ndarray, Generator[np.ndarray, None, None]]]
    queries: Optional[np.ndarray] = None
    neighbors: Optional[np.ndarray] = None

    def log(self):
        for k, v in self.index.specs.items():
            print(f'- {k}: {v}')
        print(self.__call__())

    def __call__(self) -> BenchmarkResult:

        result = BenchmarkResult()

        # Construction
        if isinstance(self.vectors, Generator):

            total_count: int = 0
            total_time: float = 0

            for vectors_batch in self.vectors:

                batch_size: int = vectors_batch.shape[0]
                first_id: int = len(self.index)
                labels = np.arange(first_id, first_id+batch_size, dtype=Label)
                dt, _ = measure_seconds(
                    lambda: self.index.add(labels, vectors_batch))

                assert len(self.index) == first_id + batch_size
                total_count += batch_size
                total_time += dt

            result.add_per_second = total_count / total_time

        elif self.vectors is not None:
            batch_size: int = self.vectors.shape[0]
            first_id: int = len(self.index)
            labels = np.arange(first_id, first_id+batch_size, dtype=Label)
            dt, _ = measure_seconds(
                lambda: self.index.add(labels, self.vectors))

            result.add_per_second = batch_size / dt

        # Search
        if self.queries is not None:

            dt, results = measure_seconds(
                lambda: self.index.search(self.queries, self.neighbors.shape[1]))

            result.search_per_second = self.queries.shape[0]/dt
            result.recall_at_one = results.recall_first(
                self.neighbors[:, 0].flatten())

        else:

            dt, recall_at_one = measure_seconds(
                lambda: recall_members(self.index))
            result.search_per_second = len(self.index)/dt
            result.recall_at_one = recall_at_one

        return result
