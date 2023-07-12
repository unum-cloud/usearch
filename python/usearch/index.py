from __future__ import annotations

# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
import os
from typing import Optional, Union, NamedTuple, List, Iterable

import numpy as np
from tqdm import tqdm

from usearch.compiled import Index as _CompiledIndex, IndexMetadata
from usearch.compiled import SparseIndex as _CompiledSetsIndex

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
            "perason": MetricKind.Pearson,
            "hamming": MetricKind.Hamming,
            "tanimoto": MetricKind.Tanimoto,
            "sorensen": MetricKind.Sorensen,
        }
        return _normalize[metric.lower()]

    return metric


class Matches(NamedTuple):
    labels: np.ndarray
    distances: np.ndarray
    counts: Union[np.ndarray, int]

    @property
    def is_multiple(self) -> bool:
        return isinstance(self.counts, np.ndarray)

    @property
    def batch_size(self) -> int:
        return len(self.counts) if isinstance(self.counts, np.ndarray) else 1

    @property
    def total_matches(self) -> int:
        return np.sum(self.counts)

    def to_list(self, row: Optional[int] = None) -> Union[List[dict], List[List[dict]]]:
        if not self.is_multiple:
            assert row is None, "Exporting a single sequence is only for batch requests"
            labels = self.labels
            distances = self.distances

        elif row is None:
            return [self.to_list(i) for i in range(self.batch_size)]

        else:
            count = self.counts[row]
            labels = self.labels[row, :count]
            distances = self.distances[row, :count]

        return [
            {"label": int(label), "distance": float(distance)}
            for label, distance in zip(labels, distances)
        ]

    def recall_first(self, expected: np.ndarray) -> float:
        best_matches = self.labels if not self.is_multiple else self.labels[:, 0]
        return np.sum(best_matches == expected) / len(expected)

    def __repr__(self) -> str:
        return (
            f"usearch.Matches({self.total_matches})"
            if self.is_multiple
            else f"usearch.Matches({self.total_matches} across {self.batch_size} queries)"
        )


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

        :param labels: Unique identifier for passed vectors, optional
        :type labels: Buffer
        :param vectors: Collection of vectors.
        :type vectors: Buffer
        :param copy: Should the index store a copy of vectors, defaults to True
        :type copy: bool, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional

        :return: Inserted label or labels
        :type: Union[int, np.ndarray]
        """
        assert isinstance(vectors, np.ndarray), "Expects a NumPy array"
        assert vectors.ndim == 1 or vectors.ndim == 2, "Expects a matrix or vector"
        count_vectors = vectors.shape[0] if vectors.ndim == 2 else 1
        is_multiple = count_vectors > 1
        if not is_multiple:
            vectors = vectors.flatten()

        # Validate of generate teh labels
        generate_labels = labels is None
        if generate_labels:
            start_id = len(self._compiled)
            if is_multiple:
                labels = np.arange(start_id, start_id + count_vectors, dtype=Label)
            else:
                labels = start_id
        else:
            if isinstance(labels, np.ndarray):
                if not is_multiple:
                    labels = int(labels[0])
                else:
                    labels = labels.astype(Label)
        count_labels = len(labels) if isinstance(labels, Iterable) else 1
        assert count_labels == count_vectors

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
                unit="Vector",
                disable=log is False,
            )
            for labels, vectors in tasks:
                self._compiled.add(labels, vectors, copy=copy, threads=threads)
                pbar.update(len(labels))

            pbar.close()

        else:
            self._compiled.add(labels, vectors, copy=copy, threads=threads)

        return labels

    def search(
        self,
        vectors,
        k: int = 10,
        *,
        threads: int = 0,
        exact: bool = False,
        log: Union[str, bool] = False,
        batch_size: int = 0,
    ) -> Matches:
        """Performs approximate nearest neighbors search for one or more queries.

        :param vectors: Query vector or vectors.
        :type vectors: Buffer
        :param k: Upper limit on the number of matches to find, defaults to 10
        :type k: int, optional

        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        :param exact: Perform exhaustive linear-time exact search, defaults to False
        :type exact: bool, optional
        :return: Approximate matches for one or more queries
        :rtype: Matches
        """
        tuple_ = self._compiled.search(
            vectors,
            k,
            exact=exact,
            threads=threads,
        )
        return Matches(*tuple_)

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
        return self._compiled.join(
            other=other._compiled,
            max_proposals=max_proposals,
            exact=exact,
        )

    @property
    def labels(self) -> np.ndarray:
        return self._compiled.labels

    def get_vectors(
        self, labels: np.ndarray, dtype: ScalarKind = ScalarKind.F32
    ) -> np.ndarray:
        dtype = _normalize_dtype(dtype, self._metric_kind)
        get_dtype = _to_numpy_compatible_dtype(dtype)
        vectors = np.vstack([self._compiled.__getitem__(l, get_dtype) for l in labels])
        view_dtype = _to_numpy_dtype(dtype)
        return vectors.view(view_dtype)

    @property
    def vectors(self) -> np.ndarray:
        return self.get_vectors(self.labels, self.dtype)

    def __delitem__(self, label: int):
        raise NotImplementedError()

    def __contains__(self, label: int) -> bool:
        return self._compiled.__contains__(label)

    def __getitem__(self, label: int) -> np.ndarray:
        dtype = self.dtype
        get_dtype = _to_numpy_compatible_dtype(dtype)
        vector = self._compiled.__getitem__(label, get_dtype)
        view_dtype = _to_numpy_dtype(dtype)
        return None if vector is None else vector.view(view_dtype)
