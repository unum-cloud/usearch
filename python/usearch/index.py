# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
import os
from typing import Optional, Callable, Union
from math import sqrt

import numpy as np

from usearch.compiled import Index as _CompiledIndex
from usearch.compiled import SetsIndex as _CompiledSetsIndex
from usearch.compiled import HashIndex as _CompiledHashIndex

Triplet = tuple[np.ndarray, np.ndarray, np.ndarray]
SetsIndex = _CompiledSetsIndex
HashIndex = _CompiledHashIndex


def results_to_list(results: Triplet, row: int) -> list[dict]:

    count = results[2][row]
    labels = results[0][row, :count]
    distances = results[1][row, :count]
    return [
        {'label': int(label), 'distance': float(distance)}
        for label, distance in zip(labels, distances)
    ]


def jitted_metric(ndim: int, metric_name: str, accuracy: str = 'f32') -> Callable:

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


class Index(_CompiledIndex):
    """

    """

    def __init__(
        self,
        ndim: int,
        dtype: str = 'f32',
        metric: Union[str, int] = 'ip',
        jit: bool = False,

        capacity: Optional[int] = None,
        connectivity: Optional[int] = None,
        expansion_add: Optional[int] = None,
        expansion_search: Optional[int] = None,
        tune: bool = False,

        path: Optional[os.PathLike] = None,
        view: bool = False,
    ) -> None:

        if jit:
            assert isinstance(metric, str), 'Name the metric to JIT'
            self._metric_name = metric
            self._metric_jitted = jitted_metric(
                ndim=ndim,
                metric_name=metric,
                dtype=dtype,
            )
            self._metric_address = self._metric_jitted.address if \
                self._metric_jitted else 0

        elif isinstance(metric, int):
            self._metric_name = None
            self._metric_jitted = None
            self._metric_address = metric

        super().__init__(
            ndim=ndim,
            metric=self._metric_name,
            metric_address=self._metric_address,
            dtype=dtype,
            capacity=capacity,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            tune=tune,
        )

        if os.path.exists(path):
            if view:
                super().view(path)
            else:
                super().load(path)
