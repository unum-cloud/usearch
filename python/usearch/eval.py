from time import time_ns
from typing import Tuple, Any, Callable, Union, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np

from usearch.io import load_matrix
from usearch.index import Index, Matches, MetricKind, MetricKindBitwise


def random_vectors(
    count: int,
    metric: MetricKind = MetricKind.IP,
    dtype: np.generic = np.float32,
    ndim: Optional[int] = None,
    index: Optional[Index] = None,
) -> np.ndarray:
    """Produces a collection of random vectors normalized for the provided `metric`
    and matching wanted `dtype`, which can both be inferred from an existing `index`.
    """

    # Infer default parameters from the `index`, if passed
    if index is not None:
        if not isinstance(index, Index):
            raise ValueError('Unsupported `index` type')

        ndim = index.ndim
        dtype = index.numpy_dtype
        metric = index.metric

    # Produce data
    if metric in MetricKindBitwise:
        bit_vectors = np.random.randint(2, size=(count, ndim))
        bit_vectors = np.packbits(bit_vectors, axis=1)
        return bit_vectors

    else:

        x = np.random.rand(count, ndim)
        if dtype == np.int8:
            x = (x * 100).astype(np.int8)
        else:
            x = x.astype(dtype)
            if metric == MetricKind.IP:
                return x / np.linalg.norm(x, axis=1, keepdims=True)
        return x


def recall_members(index: Index, *args, **kwargs) -> float:
    """Simplest benchmark for a quality of search, which queries every
    existing member of the index, to make sure approximate search finds
    the point itself.

    :param index: Non-empty pre-constructed index
    :type index: Index
    :return: Value from 0 to 1, for the share of found self-references
    :rtype: float
    """
    if len(index) == 0:
        return 0

    matches: Matches = index.search(index.vectors, 1, *args, **kwargs)
    return matches.recall_first(index.labels)


def measure_seconds(f: Callable) -> Tuple[float, Any]:
    """Simple function profiling decorator.

    :param f: Function to be profiled
    :type f: Callable
    :return: Time elapsed in seconds and the result of the execution
    :rtype: Tuple[float, Any]
    """
    a = time_ns()
    result = f()
    b = time_ns()
    c = b - a
    secs = c / (10 ** 9)
    return secs, result


@dataclass
class Dataset:

    labels: np.ndarray
    vectors: np.ndarray
    queries: np.ndarray
    neighbors: np.ndarray

    def crop_neighbors(self, k: int):
        self.neighbors = self.neighbors[:, k]

    @staticmethod
    def build(
        vectors: Optional[str] = None,
        queries: Optional[str] = None,
        neighbors: Optional[str] = None,
        count: Optional[int] = None,
        ndim: Optional[int] = None,
        k: Optional[int] = None,
    ):
        """Either loads an existing dataset from disk, or generates one on the fly.

        :param vectors: _description_, defaults to None
        :type vectors: Optional[str], optional
        :param queries: _description_, defaults to None
        :type queries: Optional[str], optional
        :param neighbors: _description_, defaults to None
        :type neighbors: Optional[str], optional
        :param count: _description_, defaults to None
        :type count: Optional[int], optional
        :param ndim: _description_, defaults to None
        :type ndim: Optional[int], optional
        :param k: _description_, defaults to None
        :type k: Optional[int], optional
        """

        d = Dataset(None, None, None, None)

        if vectors is not None:
            assert ndim is None

            d.vectors = load_matrix(vectors)
            ndim = d.vectors.shape[1]
            count = min(
                d.vectors.shape[0], count) if count is not None else d.vectors.shape[0]
            d.vectors = d.vectors[:count, :]
            d.labels = np.arange(count, dtype=np.longlong)

            if queries is not None:
                d.queries = load_matrix(queries)
            else:
                d.queries = d.vectors

            if neighbors is not None:
                d.neighbors = load_matrix(neighbors)
                if k is not None:
                    d.neighbors = d.neighbors[:, :k]
            else:
                assert k is None, 'Cant ovveride `k`, will retrieve one neighbor'
                d.neighbors = np.reshape(d.labels, (count, 1))

        else:

            assert ndim is not None
            assert count is not None
            assert k is None, 'Cant ovveride `k`, will retrieve one neighbor'

            d.vectors = random_vectors(count=count, ndim=ndim)
            d.queries = d.vectors
            d.labels = np.arange(count, dtype=np.longlong)
            d.neighbors = np.reshape(d.labels, (count, 1))

        return d


@dataclass
class TaskResult:

    add_operations: Optional[int] = None
    add_per_second: Optional[float] = None

    search_operations: Optional[int] = None
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

    @property
    def add_seconds(self) -> float:
        return self.add_operations / self.add_per_second

    @property
    def search_seconds(self) -> float:
        return self.search_operations / self.search_per_second

    def __add__(self, other: TaskResult):
        result = TaskResult()
        if self.add_operations and other.add_operations:
            result.add_operations = self.add_operations + other.add_operations
            result.add_per_second = result.add_operations / \
                (self.add_seconds + other.add_seconds)
        else:
            base = self if self.add_operations else other
            result.add_operations = base.add_operations
            result.add_per_second = base.add_per_second

        if self.search_operations and other.search_operations:
            result.search_operations = self.search_operations + other.search_operations
            result.recall_at_one = (
                self.recall_at_one * self.search_operations + other.recall_at_one *
                other.search_operations) / (self.search_operations + other.search_operations)
            result.search_per_second = result.search_operations / \
                (self.search_seconds + other.search_seconds)
        else:
            base = self if self.search_operations else other
            result.search_operations = base.search_operations
            result.search_per_second = base.search_per_second
            result.recall_at_one = base.recall_at_one

        return result


@dataclass
class AddTask:

    labels: np.ndarray
    vectors: np.ndarray

    def __call__(self, index: Index) -> TaskResult:

        batch_size: int = self.vectors.shape[0]
        old_size: int = len(index)
        dt, _ = measure_seconds(
            lambda: index.add(self.labels, self.vectors))

        assert len(index) == old_size + batch_size
        return TaskResult(
            add_operations=batch_size,
            add_per_second=batch_size/dt,
        )

    @property
    def ndim(self):
        return self.vectors.shape[1]

    @property
    def count(self):
        return self.vectors.shape[0]

    def inplace_shuffle(self):
        """Rorders the `vectors` and `labels`. Often used for robustness benchmarks."""

        new_order = np.arange(self.count)
        np.random.shuffle(new_order)
        self.labels = self.labels[new_order]
        self.vectors = self.vectors[new_order, :]

    def slices(self, batch_size: int) -> List[AddTask]:
        """Splits this dataset into smaller chunks."""

        return [
            AddTask(
                labels=self.labels[start_row:start_row+batch_size],
                vectors=self.vectors[start_row:start_row+batch_size, :],
            ) for start_row in range(0, self.count, batch_size)]

    def clusters(self, number_of_clusters: int) -> List[AddTask]:
        """Splits this dataset into smaller chunks."""

        from sklearn.cluster import KMeans
        clustering = KMeans(
            n_clusters=number_of_clusters,
            random_state=0, n_init='auto',
        ).fit(self.vectors)

        partitioning = defaultdict(list)
        for row, cluster in enumerate(clustering.labels_):
            partitioning[cluster].append(row)

        return [
            AddTask(
                labels=self.labels[rows],
                vectors=self.vectors[rows, :],
            ) for rows in partitioning.values()]


@dataclass
class SearchTask:

    queries: np.ndarray
    neighbors: np.ndarray

    def __call__(self, index: Index) -> TaskResult:

        dt, results = measure_seconds(
            lambda: index.search(self.queries, self.neighbors.shape[1]))

        return TaskResult(
            search_per_second=self.queries.shape[0]/dt,
            recall_at_one=results.recall_first(self.neighbors[:, 0].flatten()),
        )

    def slices(self, batch_size: int) -> List[SearchTask]:
        """Splits this dataset into smaller chunks."""

        return [
            SearchTask(
                queries=self.queries[start_row:start_row+batch_size, :],
                neighbors=self.neighbors[start_row:start_row+batch_size, :],
            ) for start_row in range(0, self.queries.shape[0], batch_size)]


@dataclass
class Evaluation:

    tasks: List[Union[AddTask, SearchTask]]
    count: int
    ndim: int

    @staticmethod
    def for_dataset(
            dataset: Dataset,
            batch_size: int = 0,
            clusters: int = 1) -> Evaluation:

        tasks = []
        add = AddTask(
            vectors=dataset.vectors,
            labels=dataset.labels)
        search = SearchTask(
            queries=dataset.queries,
            neighbors=dataset.neighbors)

        if batch_size:
            tasks.extend(add.slices(batch_size))
            tasks.extend(search.slices(batch_size))
        elif clusters != 1:
            tasks.extend(add.clusters(clusters))
            print(tasks)
            tasks.append(search)
        else:
            tasks.append(add)
            tasks.append(search)

        return Evaluation(
            tasks=tasks,
            count=add.count,
            ndim=add.ndim,
        )

    def __call__(self, index: Index, post_clean: bool = True) -> dict:

        task_result = TaskResult()

        try:
            for task in self.tasks:
                task_result = task_result + task(index)
        except KeyboardInterrupt:
            pass

        if post_clean:
            index.clear()
        return {
            **index.specs,
            **asdict(task_result),
        }
