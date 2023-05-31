# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
import os
from math import sqrt
from typing import Optional, Callable, Union, NamedTuple, List

import numpy as np

from usearch.compiled import Index as _CompiledIndex
from usearch.compiled import SetsIndex as _CompiledSetsIndex
from usearch.compiled import HashIndex as _CompiledHashIndex
from usearch.compiled import DEFAULT_CONNECTIVITY, DEFAULT_EXPANSION_ADD, DEFAULT_EXPANSION_SEARCH

SetsIndex = _CompiledSetsIndex
HashIndex = _CompiledHashIndex


class Matches(NamedTuple):
    labels: np.ndarray
    distances: np.ndarray
    counts: np.ndarray


def list_matches(results: Matches, row: int) -> List[dict]:

    count = results[2][row]
    labels = results[0][row, :count]
    distances = results[1][row, :count]
    return [
        {'label': int(label), 'distance': float(distance)}
        for label, distance in zip(labels, distances)
    ]


def jit_metric(ndim: int, metric_name: str, accuracy: str = 'f32') -> Callable:

    try:
        from numba import cfunc, types, carray
    except ImportError:
        raise ModuleNotFoundError(
            'To use JIT install Numba with `pip install numba`.'
            'Alternatively, reinstall usearch with `pip install usearch[jit]`')

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks

    if accuracy == 'f32':
        signature = types.float32(
            types.CPointer(types.float32),
            types.CPointer(types.float32),
            types.size_t, types.size_t)

        if metric_name == 'ip':

            def numba_ip(a, b, _n, _m):
                a_array = carray(a, ndim)
                b_array = carray(b, ndim)
                ab = 0.0
                for i in range(ndim):
                    ab += a_array[i] * b_array[i]
                return ab

            return cfunc(numba_ip, signature)

        if metric_name == 'cos':

            def numba_cos(a, b, _n, _m):
                a_array = carray(a, ndim)
                b_array = carray(b, ndim)
                ab = 0.0
                a_sq = 0.0
                b_sq = 0.0
                for i in range(ndim):
                    ab += a_array[i] * b_array[i]
                    a_sq += a_array[i] * a_array[i]
                    b_sq += b_array[i] * b_array[i]
                return ab / (sqrt(a_sq) * sqrt(b_sq))

            return cfunc(numba_cos, signature)

        if metric_name == 'l2sq':

            def numba_l2sq(a, b, _n, _m):
                a_array = carray(a, ndim)
                b_array = carray(b, ndim)
                ab_delta_sq = 0.0
                for i in range(ndim):
                    ab_delta_sq += (a_array[i] - b_array[i]) * \
                        (a_array[i] - b_array[i])
                return ab_delta_sq

            return cfunc(numba_l2sq, signature)

    return None


class Index:
    """Fast JIT-compiled index for equi-dimensional embeddings.

    Vector labels must be integers.
    Vectors must have the same number of dimensions within the index.
    Supports Inner Product, Cosine Distance, Ln measures
    like the Euclidean metric, as well as automatic downcasting
    and quantization.
    """

    def __init__(
        self,
        ndim: int,
        dtype: str = 'f32',
        metric: Union[str, int] = 'ip',
        jit: bool = False,

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
        :param dtype: Scalar type for vectors, defaults to 'f32'
        :type dtype: str, optional
            Example: you can use the `f16` index with `f32` vectors,
            which will be automatically downcasted.

        :param metric: Distance function, defaults to 'ip'
        :type metric: Union[str, int], optional
            Name of distance function, or the address of the
            Numba `cfunc` JIT-compiled object.

        :param jit: Enable Numba to JIT compile the metric, defaults to False
        :type jit: bool, optional
            This can result in up-to 3x performance difference on very large vectors
            and very recent hardware, as the Python module is compiled with high
            compatibility in mind and avoids very fancy assembly instructions.

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
        if jit:
            assert isinstance(metric, str), 'Name the metric to JIT'
            self._metric_name = metric
            self._metric_jit = jit_metric(
                ndim=ndim,
                metric_name=metric,
                dtype=dtype,
            )
            self._metric_pointer = self._metric_jit.address if \
                self._metric_jit else 0

        elif isinstance(metric, int):
            self._metric_name = ''
            self._metric_jit = None
            self._metric_pointer = metric

        else:
            if metric is None:
                metric = 'ip'
            assert isinstance(metric, str)
            self._metric_name = metric
            self._metric_jit = None
            self._metric_pointer = 0

        self._compiled = _CompiledIndex(
            ndim=ndim,
            metric=self._metric_name,
            metric_pointer=self._metric_pointer,
            dtype=dtype,
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

    def add(
            self, labels, vectors, *,
            copy: bool = True, threads: int = 0):
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

        :param labels: Unique identifier for passed vectors.
        :type labels: Buffer
        :param vectors: Collection of vectors.
        :type vectors: Buffer
        :param copy: Should the index store a copy of vectors, defaults to True
        :type copy: bool, optional
        :param threads: Optimal number of cores to use, defaults to 0
        :type threads: int, optional
        """
        if isinstance(labels, np.ndarray):
            labels = labels.astype(np.longlong)
        self._compiled.add(labels, vectors, copy=copy, threads=threads)

    def search(
            self, vectors, k: int = 10, *,
            threads: int = 0, exact: bool = False) -> Matches:
        tuple_ = self._compiled.search(
            vectors, k,
            exact=exact, threads=threads)
        return Matches(*tuple_)

    def __len__(self) -> int:
        return len(self._compiled)

    @property
    def size(self) -> int:
        return self._compiled.size

    @property
    def ndim(self) -> int:
        return self._compiled.ndim

    @property
    def dtype(self) -> str:
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
    def change_expansion_add(self, v: int):
        self._compiled.expansion_add = v

    @expansion_search.setter
    def change_expansion_search(self, v: int):
        self._compiled.expansion_search = v

    def save(self, path: os.PathLike):
        self._compiled.save(path)

    def load(self, path: os.PathLike):
        self._compiled.load(path)

    def view(self, path: os.PathLike):
        self._compiled.view(path)

    def clear(self):
        self._compiled.clear()

    def remove(self, label: int):
        pass
