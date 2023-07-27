from __future__ import annotations

# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
import os
import math
from typing import Optional, Union, NamedTuple, List, Iterable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from usearch.compiled import Index as _CompiledIndex
from usearch.compiled import Indexes as _CompiledIndexes
from usearch.compiled import SparseIndex as _CompiledSetsIndex

from usearch.compiled import IndexMetadata, IndexStats
from usearch.compiled import MetricKind, ScalarKind, MetricSignature
from usearch.compiled import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
    USES_OPENMP,
    USES_SIMSIMD,
    USES_NATIVE_F16,
)

MetricKindBitwise = (
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
)

SparseIndex = _CompiledSetsIndex

Label = np.uint64


def _normalize_dtype(dtype, metric: MetricKind = MetricKind.IP) -> ScalarKind:
    if dtype is None or dtype == "":
        return ScalarKind.B1 if metric in MetricKindBitwise else ScalarKind.F32

    if isinstance(dtype, ScalarKind):
        return dtype

    if isinstance(dtype, str):
        dtype = dtype.lower()

    _normalize = {
        "f64": ScalarKind.F64,
        "f32": ScalarKind.F32,
        "f16": ScalarKind.F16,
        "f8": ScalarKind.F8,
        "b1": ScalarKind.B1,
        "b1x8": ScalarKind.B1,
        np.float64: ScalarKind.F64,
        np.float32: ScalarKind.F32,
        np.float16: ScalarKind.F16,
        np.int8: ScalarKind.F8,
    }
    return _normalize[dtype]


def _to_numpy_compatible_dtype(dtype: ScalarKind) -> ScalarKind:
    _normalize = {
        ScalarKind.F64: ScalarKind.F64,
        ScalarKind.F32: ScalarKind.F32,
        ScalarKind.F16: ScalarKind.F16,
        ScalarKind.F8: ScalarKind.F16,
        ScalarKind.B1: ScalarKind.B1,
    }
    return _normalize[dtype]


def _to_numpy_dtype(dtype: ScalarKind):
    _normalize = {
        ScalarKind.F64: np.float64,
        ScalarKind.F32: np.float32,
        ScalarKind.F16: np.float16,
        ScalarKind.F8: np.float16,
        ScalarKind.B1: np.uint8,
    }
    return _normalize[dtype]


def _normalize_metric(metric):
    if metric is None:
        return MetricKind.IP

    if isinstance(metric, str):
        _normalize = {
            "cos": MetricKind.Cos,
            "ip": MetricKind.IP,
            "l2_sq": MetricKind.L2sq,
            "haversine": MetricKind.Haversine,
            "pearson": MetricKind.Pearson,
            "hamming": MetricKind.Hamming,
            "tanimoto": MetricKind.Tanimoto,
            "sorensen": MetricKind.Sorensen,
        }
        return _normalize[metric.lower()]

    return metric


def _search_in_compiled(
    *,
    compiled: Union[_CompiledIndex, _CompiledIndexes],
    vectors: np.ndarray,
    k: int,
    threads: int,
    exact: bool,
    log: Union[str, bool],
    batch_size: int,
) -> Union[Matches, BatchMatches]:
    #
    assert isinstance(vectors, np.ndarray), "Expects a NumPy array"
    assert vectors.ndim == 1 or vectors.ndim == 2, "Expects a matrix or vector"
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, len(vectors))
    count_vectors = vectors.shape[0]

    def distil_batch(batch_matches: BatchMatches) -> Union[BatchMatches, Matches]:
        return batch_matches[0] if count_vectors == 1 else batch_matches

    if log and batch_size == 0:
        batch_size = int(math.ceil(count_vectors / 100))

    if batch_size:
        tasks = [
            vectors[start_row : start_row + batch_size, :]
            for start_row in range(0, count_vectors, batch_size)
        ]
        tasks_matches = []
        name = log if isinstance(log, str) else "Search"
        pbar = tqdm(
            tasks,
            desc=name,
            total=count_vectors,
            unit="vector",
            disable=log is False,
        )
        for vectors in tasks:
            tuple_ = compiled.search(
                vectors,
                k,
                exact=exact,
                threads=threads,
            )
            tasks_matches.append(BatchMatches(*tuple_))
            pbar.update(vectors.shape[0])

        pbar.close()
        return distil_batch(
            BatchMatches(
                labels=np.vstack([m.labels for m in tasks_matches]),
                distances=np.vstack([m.distances for m in tasks_matches]),
                counts=np.concatenate([m.counts for m in tasks_matches], axis=None),
            )
        )

    else:
        tuple_ = compiled.search(
            vectors,
            k,
            exact=exact,
            threads=threads,
        )
        return distil_batch(BatchMatches(*tuple_))


def _add_to_compiled(
    *,
    compiled,
    labels,
    vectors,
    copy: bool,
    threads: int,
    log: Union[str, bool],
    batch_size: int,
) -> Union[int, np.ndarray]:
    assert isinstance(vectors, np.ndarray), "Expects a NumPy array"
    assert vectors.ndim == 1 or vectors.ndim == 2, "Expects a matrix or vector"
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, len(vectors))

    # Validate or generate the labels
    count_vectors = vectors.shape[0]
    generate_labels = labels is None
    if generate_labels:
        start_id = len(compiled)
        labels = np.arange(start_id, start_id + count_vectors, dtype=Label)
    else:
        if not isinstance(labels, Iterable):
            assert count_vectors == 1, "Each vector must have a label"
            labels = [labels]
        labels = np.array(labels).astype(Label)

    assert len(labels) == count_vectors

    # If logging is requested, and batch size is undefined, set it to grow 1% at a time:
    if log and batch_size == 0:
        batch_size = int(math.ceil(count_vectors / 100))

    # Split into batches and log progress, if needed
    if batch_size:
        labels = [
            labels[start_row : start_row + batch_size]
            for start_row in range(0, count_vectors, batch_size)
        ]
        vectors = [
            vectors[start_row : start_row + batch_size, :]
            for start_row in range(0, count_vectors, batch_size)
        ]
        tasks = zip(labels, vectors)
        name = log if isinstance(log, str) else "Add"
        pbar = tqdm(
            tasks,
            desc=name,
            total=count_vectors,
            unit="vector",
            disable=log is False,
        )
        for labels, vectors in tasks:
            compiled.add(labels, vectors, copy=copy, threads=threads)
            pbar.update(len(labels))

        pbar.close()

    else:
        compiled.add(labels, vectors, copy=copy, threads=threads)

    return labels


@dataclass
class Match:
    label: int
    distance: float


@dataclass
class Matches:
    labels: np.ndarray
    distances: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Match:
        if isinstance(index, int) and index < len(self):
            return Match(
                label=self.labels[index],
                distance=self.distances[index],
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self) -> List[tuple]:
        return [(int(l), float(d)) for l, d in zip(self.labels, self.distances)]

    def __repr__(self) -> str:
        return f"usearch.Matches({len(self)})"


@dataclass
class BatchMatches:
    labels: np.ndarray
    distances: np.ndarray
    counts: np.ndarray

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, index: int) -> Matches:
        if isinstance(index, int) and index < len(self):
            return Matches(
                labels=self.labels[index, : self.counts[index]],
                distances=self.distances[index, : self.counts[index]],
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self) -> List[List[tuple]]:
        lists = [self.__getitem__(row) for row in range(self.__len__())]
        return [item for sublist in lists for item in sublist]

    def recall_first(self, expected: np.ndarray) -> float:
        """Measures recall [0, 1] as of `Matches` that contain the corresponding
        `expected` entry as the first result."""
        return np.sum(self.labels[:, 0] == expected) / len(expected)

    def recall(self, expected: np.ndarray) -> float:
        """Measures recall [0, 1] as of `Matches` that contain the corresponding
        `expected` entry anywhere among results."""
        assert len(expected) == self.batch_size
        recall = 0
        for i in range(self.batch_size):
            recall += expected[i] in self.labels[i]
        return recall / len(expected)

    def __repr__(self) -> str:
        return f"usearch.BatchMatches({np.sum(self.counts)} across {len(self)} queries)"


class CompiledMetric(NamedTuple):
    pointer: int
    kind: MetricKind
    signature: MetricSignature


class Index:
    """Fast JIT-compiled vector-search index for dense equi-dimensional embeddings.

    Vector labels must be integers.
    Vectors must have the same number of dimensions within the index.
    Supports Inner Product, Cosine Distance, Ln measures
    like the Euclidean metric, as well as automatic downcasting
    and quantization.
    """

    def __init__(
        self,
        ndim: int = 0,
        metric: Union[str, MetricKind, CompiledMetric] = MetricKind.IP,
        dtype: Optional[Union[str, ScalarKind]] = None,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
        tune: bool = False,
        path: Optional[os.PathLike] = None,
        view: bool = False,
    ) -> None:
        """Construct the index and compiles the functions, if requested (expensive).

        :param ndim: Number of vector dimensions
        :type ndim: int
            Required for some metrics, optional for others.
            Haversine, for example, only applies to 2-dimensional latitude/longitude
            coordinates. Angular (Cos) and Euclidean (L2sq), obviously, apply to
            vectors with arbitrary number of dimensions.

        :param metric: Distance function, defaults to MetricKind.IP
        :type metric: Union[MetricKind, Callable, str], optional
            Kind of the distance function, or the Numba `cfunc` JIT-compiled object.
            Possible `MetricKind` values: IP, Cos, L2sq, Haversine, Pearson,
            Hamming, Tanimoto, Sorensen.
            Not every kind is JIT-able. For Jaccard distance, use `SparseIndex`.

        :param dtype: Scalar type for internal vector storage, defaults to None
        :type dtype: Optional[Union[str, ScalarKind]], optional
            For continuous metrics can be: f16, f32, f64, or f8.
            For bitwise metrics it's implementation-defined, and can't change.
            Example: you can use the `f16` index with `f32` vectors in Euclidean space,
            which will be automatically downcasted.

        :param connectivity: Connections per node in HNSW, defaults to None
        :type connectivity: Optional[int], optional
            Hyper-parameter for the number of Graph connections
            per layer of HNSW. The original paper calls it "M".
            Optional, but can't be changed after construction.

        :param expansion_add: Traversal depth on insertions, defaults to None
        :type expansion_add: Optional[int], optional
            Hyper-parameter for the search depth when inserting new
            vectors. The original paper calls it "efConstruction".
            Can be changed afterwards, as the `.expansion_add`.

        :param expansion_search: Traversal depth on queries, defaults to None
        :type expansion_search: Optional[int], optional
            Hyper-parameter for the search depth when querying
            nearest neighbors. The original paper calls it "ef".
            Can be changed afterwards, as the `.expansion_search`.

        :param tune: Automatically adjusts hyper-parameters, defaults to False
        :type tune: bool, optional

        :param path: Where to store the index, defaults to None
        :type path: Optional[os.PathLike], optional
        :param view: Are we simply viewing an immutable index, defaults to False
        :type view: bool, optional
        """

        metric = _normalize_metric(metric)
        if isinstance(metric, MetricKind):
            self._metric_kind = metric
            self._metric_jit = None
            self._metric_pointer = 0
            self._metric_signature = MetricSignature.ArrayArraySize
        elif isinstance(metric, CompiledMetric):
            self._metric_jit = metric
            self._metric_kind = metric.kind
            self._metric_pointer = metric.pointer
            self._metric_signature = metric.signature
        else:
            raise ValueError(
                "The `metric` must be a `CompiledMetric` or a `MetricKind`"
            )

        # Validate, that the right scalar type is defined
        dtype = _normalize_dtype(dtype, self._metric_kind)
        self._compiled = _CompiledIndex(
            ndim=ndim,
            dtype=dtype,
            metric=self._metric_kind,
            metric_pointer=self._metric_pointer,
            metric_signature=self._metric_signature,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            tune=tune,
        )

        self.path = path
        if path and os.path.exists(path):
            if view:
                self._compiled.view(path)
            else:
                self._compiled.load(path)

    @staticmethod
    def metadata(path: os.PathLike) -> IndexMetadata:
        return IndexMetadata(path)

    @staticmethod
    def restore(path: os.PathLike, view: bool = False) -> Index:
        if not os.path.exists(path):
            return None
        meta = Index.metadata(path)
        bits_per_scalar = {
            ScalarKind.F8: 8,
            ScalarKind.F16: 16,
            ScalarKind.F32: 32,
            ScalarKind.F64: 64,
            ScalarKind.B1: 1,
        }[meta.scalar_kind]
        ndim = meta.bytes_for_vectors * 8 // meta.size // bits_per_scalar
        return Index(
            ndim=ndim,
            connectivity=meta.connectivity,
            metric=meta.metric,
            path=path,
            view=view,
        )

    def add(
        self,
        labels,
        vectors,
        *,
        copy: bool = True,
        threads: int = 0,
        log: Union[str, bool] = False,
        batch_size: int = 0,
    ) -> Union[int, np.ndarray]:
        """Inserts one or move vectors into the index.

        For maximal performance the `labels` and `vectors`
        should conform to the Python's "buffer protocol" spec.

        To index a single entry:
            labels: int, vectors: np.ndarray.
        To index many entries:
            labels: np.ndarray, vectors: np.ndarray.

        When working with extremely large indexes, you may want to
        pass `copy=False`, if you can guarantee the lifetime of the
        primary vectors store during the process of construction.

        :param labels: Unique identifier(s) for passed vectors, optional
        :type labels: Buffer
        :param vectors: Vector or a row-major matrix
        :type vectors: Buffer
        :param copy: Should the index store a copy of vectors, defaults to True
        :type copy: bool, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        :param log: Whether to print the progress bar, default to False
        :type log: Union[str, bool], optional
        :param batch_size: Number of vectors to process at once, defaults to 0
        :type batch_size: int, optional
        :return: Inserted label or labels
        :type: Union[int, np.ndarray]
        """
        return _add_to_compiled(
            compiled=self._compiled,
            labels=labels,
            vectors=vectors,
            copy=copy,
            threads=threads,
            log=log,
            batch_size=batch_size,
        )

    def search(
        self,
        vectors,
        k: int = 10,
        *,
        threads: int = 0,
        exact: bool = False,
        log: Union[str, bool] = False,
        batch_size: int = 0,
    ) -> Union[Matches, BatchMatches]:
        """
        Performs approximate nearest neighbors search for one or more queries.

        :param vectors: Query vector or vectors.
        :type vectors: Buffer
        :param k: Upper limit on the number of matches to find, defaults to 10
        :type k: int, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        :param exact: Perform exhaustive linear-time exact search, defaults to False
        :type exact: bool, optional
        :param log: Whether to print the progress bar, default to False
        :type log: Union[str, bool], optional
        :param batch_size: Number of vectors to process at once, defaults to 0
        :type batch_size: int, optional
        :return: Approximate matches for one or more queries
        :rtype: Union[Matches, BatchMatches]
        """

        return _search_in_compiled(
            compiled=self._compiled,
            vectors=vectors,
            k=k,
            exact=exact,
            threads=threads,
            log=log,
            batch_size=batch_size,
        )

    def remove(
        self,
        labels: Union[int, Iterable[int]],
        *,
        compact: bool = False,
        threads: int = 0,
    ) -> Union[bool, int]:
        """Removes one or move vectors from the index.

        When working with extremely large indexes, you may want to
        mark some entries deleted, instead of rebuilding a filtered index.
        In other cases, rebuilding - is the recommended approach.

        :param labels: Unique identifier for passed vectors, optional
        :type labels: Buffer
        :param compact: Removes links to removed nodes (expensive), defaults to False
        :type compact: bool, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        :return: Number of removed entries
        :type: Union[bool, int]
        """
        return self._compiled.remove(labels, compact=compact, threads=threads)

    def rename(self, label_from: int, label_to: int) -> bool:
        """Relabel existing entry"""
        return self._compiled.rename(label_from, label_to)

    @property
    def specs(self) -> dict:
        return {
            "Class": "usearch.Index",
            "Connectivity": self.connectivity,
            "Size": self.size,
            "Dimensions": self.ndim,
            "Expansion@Add": self.expansion_add,
            "Expansion@Search": self.expansion_search,
            "OpenMP": USES_OPENMP,
            "SimSIMD": USES_SIMSIMD,
            "NativeF16": USES_NATIVE_F16,
            "JIT": self.jit,
            "DType": self.dtype,
            "Path": self.path,
        }

    def __len__(self) -> int:
        return self._compiled.__len__()

    def __delitem__(self, label: int) -> bool:
        raise self.remove(label)

    def __contains__(self, label: int) -> bool:
        return self._compiled.__contains__(label)

    def __getitem__(self, label: int) -> np.ndarray:
        dtype = self.dtype
        get_dtype = _to_numpy_compatible_dtype(dtype)
        vector = self._compiled.__getitem__(label, get_dtype)
        view_dtype = _to_numpy_dtype(dtype)
        return None if vector is None else vector.view(view_dtype)

    @property
    def jit(self) -> bool:
        return self._metric_jit is not None

    @property
    def size(self) -> int:
        return self._compiled.size

    @property
    def ndim(self) -> int:
        return self._compiled.ndim

    @property
    def metric(self) -> Union[MetricKind, CompiledMetric]:
        return self._metric_jit if self._metric_jit else self._metric_kind

    @property
    def dtype(self) -> ScalarKind:
        return self._compiled.dtype

    @property
    def connectivity(self) -> int:
        return self._compiled.connectivity

    @property
    def capacity(self) -> int:
        return self._compiled.capacity

    @property
    def memory_usage(self) -> int:
        return self._compiled.memory_usage

    @property
    def expansion_add(self) -> int:
        return self._compiled.expansion_add

    @property
    def expansion_search(self) -> int:
        return self._compiled.expansion_search

    @expansion_add.setter
    def expansion_add(self, v: int):
        self._compiled.expansion_add = v

    @expansion_search.setter
    def expansion_search(self, v: int):
        self._compiled.expansion_search = v

    def save(self, path: Optional[os.PathLike] = None):
        path = path if path else self.path
        if path is None:
            raise Exception("Define `path` argument")
        self._compiled.save(path)

    def load(self, path: Optional[os.PathLike] = None):
        path = path if path else self.path
        if path is None:
            raise Exception("Define `path` argument")
        self._compiled.load(path)

    def view(self, path: Optional[os.PathLike] = None):
        path = path if path else self.path
        if path is None:
            raise Exception("Define `path` argument")
        self._compiled.view(path)

    def clear(self):
        self._compiled.clear()

    def close(self):
        self._compiled.close()

    def copy(self) -> Index:
        result = Index(
            ndim=self.ndim,
            metric=self.metric,
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            path=self.path,
        )
        result._compiled = self._compiled.copy()
        return result

    def join(self, other: Index, max_proposals: int = 0, exact: bool = False) -> dict:
        """Performs "Semantic Join" or pairwise matching between `self` & `other` index.
        Is different from `search`, as no collisions are allowed in resulting pairs.
        Uses the concept of "Stable Marriages" from Combinatorics, famous for the 2012
        Nobel Prize in Economics.

        :param other: Another index.
        :type other: Index
        :param max_proposals: Limit on candidates evaluated per vector, defaults to 0
        :type max_proposals: int, optional
        :param exact: Controls if underlying `search` should be exact, defaults to False
        :type exact: bool, optional
        :return: Mapping from labels of `self` to labels of `other`
        :rtype: dict
        """
        return self._compiled.join(
            other=other._compiled,
            max_proposals=max_proposals,
            exact=exact,
        )

    def get_labels(self, offset: int = 0, limit: int = 0) -> np.ndarray:
        if limit == 0:
            limit = 2**63 - 1
        return self._compiled.get_labels(offset, limit)

    @property
    def labels(self) -> np.ndarray:
        """Retrieves the labels of all vectors present in `self`

        :return: Array of labels
        :rtype: np.ndarray
        """
        return self._compiled.labels

    def get_vectors(
        self,
        labels: np.ndarray,
        dtype: ScalarKind = ScalarKind.F32,
    ) -> np.ndarray:
        """Retrieves vectors associated with given `labels`

        :return: Matrix of vectors (row-major)
        :rtype: np.ndarray
        """
        dtype = _normalize_dtype(dtype, self._metric_kind)
        get_dtype = _to_numpy_compatible_dtype(dtype)
        vectors = np.vstack([self._compiled.__getitem__(l, get_dtype) for l in labels])
        view_dtype = _to_numpy_dtype(dtype)
        return vectors.view(view_dtype)

    @property
    def vectors(self) -> np.ndarray:
        return self.get_vectors(self.labels, self.dtype)

    @property
    def max_level(self) -> int:
        return self._compiled.max_level

    @property
    def levels_stats(self) -> IndexStats:
        """Get the accumulated statistics for the entire multi-level graph.

        :return: Statistics for the entire multi-level graph.
        :rtype: IndexStats

        Statistics:
            - ``nodes`` (int): The number of nodes in that level.
            - ``edges`` (int): The number of edges in that level.
            - ``max_edges`` (int): The maximum possible number of edges in that level.
            - ``allocated_bytes`` (int): The amount of allocated memory for that level.
        """
        return self._compiled.levels_stats

    def level_stats(self, level: int) -> IndexStats:
        """Get statistics for one level of the index - one graph.

        :return: Statistics for one level of the index - one graph.
        :rtype: IndexStats

        Statistics:
            - ``nodes`` (int): The number of nodes in that level.
            - ``edges`` (int): The number of edges in that level.
            - ``max_edges`` (int): The maximum possible number of edges in that level.
            - ``allocated_bytes`` (int): The amount of allocated memory for that level.
        """
        return self._compiled.level_stats(level)

    def __repr__(self) -> str:
        f = "usearch.Index({} x {}, {}, expansion: {} & {}, {} vectors in {} levels)"
        return f.format(
            self.dtype,
            self.ndim,
            self.metric,
            self.expansion_add,
            self.expansion_search,
            len(self),
            self.max_level + 1,
        )

    def _repr_pretty_(self) -> str:
        level_stats = [
            f"--- {i}. {self.level_stats(i).nodes} nodes" for i in range(self.max_level)
        ]
        return "\n".join(
            [
                "usearch.Index",
                "- config" f"-- data type: {self.dtype}",
                f"-- dimensions: {self.ndim}",
                f"-- metric: {self.metric}",
                f"-- expansion on addition:{self.expansion_add} candidates",
                f"-- expansion on search: {self.expansion_search} candidates",
                "- state",
                f"-- size: {self.size} vectors",
                f"-- memory usage: {self.memory_usage} bytes",
                f"-- max level: {self.max_level}",
                *level_stats,
            ]
        )


class Indexes:
    def __init__(self, indexes: Iterable[Index]) -> None:
        self._compiled = _CompiledIndexes()
        for index in indexes:
            self._compiled.add(index._compiled)

    def add(self, index: Index):
        self._compiled.add(index._compiled)

    def search(
        self,
        vectors,
        k: int = 10,
        *,
        threads: int = 0,
        exact: bool = False,
    ):
        return _search_in_compiled(
            compiled=self._compiled,
            vectors=vectors,
            k=k,
            exact=exact,
            threads=threads,
            log=False,
            batch_size=None,
        )
