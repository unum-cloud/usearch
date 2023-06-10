# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
from math import sqrt
from typing import Callable

import numpy as np
from numba import cfunc, types, carray

from usearch.compiled import MetricKind


signature_f16args = types.float32(
    types.CPointer(types.float16),
    types.CPointer(types.float16),
    types.uint64, types.uint64)

signature_f32args = types.float32(
    types.CPointer(types.float32),
    types.CPointer(types.float32),
    types.uint64, types.uint64)

signature_f64args = types.float32(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.uint64, types.uint64)


def jit(ndim: int, metric: MetricKind = MetricKind.Cos, dtype: np.generic = np.float32) -> Callable:
    """JIT-compiles a distance metric specifically tuned for the target hardware and number of dimensions.

    Uses Numba `cfunc` functionality, annotating it with Numba `types` instead of `ctypes` to support half-precision.
    https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    """
    normalize_dtype = {
        # Half-precision is still unsupported
        # https://github.com/numba/numba/issues/4402
        # 'f16': np.float16,
        'f32': np.float32,
        'f64': np.float64,
    }
    if dtype in normalize_dtype.keys():
        dtype = normalize_dtype[dtype]
    if dtype not in normalize_dtype.values():
        return None

    scalar_kind_to_accumulator_type = {
        np.float16: types.float32,
        np.float32: types.float32,
        np.float64: types.float64,
    }
    accumulator = scalar_kind_to_accumulator_type[dtype]

    def numba_ip(a, b, _n, _m):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab = accumulator(0)
        for i in range(ndim):
            ab += a_array[i] * b_array[i]
        return types.float32(1 - ab)

    def numba_cos(a, b, _n, _m):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab = accumulator(0)
        a_sq = accumulator(0)
        b_sq = accumulator(0)
        for i in range(ndim):
            ab += a_array[i] * b_array[i]
            a_sq += a_array[i] * a_array[i]
            b_sq += b_array[i] * b_array[i]
        a_norm = sqrt(a_sq)
        b_norm = sqrt(b_sq)
        if a_norm == 0 and b_norm == 0:
            return types.float32(0)
        elif a_norm == 0 or b_norm == 0 or ab == 0:
            return types.float32(1)
        else:
            return types.float32(1 - ab / (a_norm * b_norm))

    def numba_l2sq(a, b, _n, _m):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab_delta_sq = accumulator(0)
        for i in range(ndim):
            ab_delta_sq += (a_array[i] - b_array[i]) * \
                (a_array[i] - b_array[i])
        return types.float32(ab_delta_sq)

    scalar_kind_to_signature = {
        np.float16: signature_f16args,
        np.float32: signature_f32args,
        np.float64: signature_f64args,
    }

    metric_kind_to_function = {
        MetricKind.IP: numba_ip,
        MetricKind.Cos: numba_cos,
        MetricKind.L2sq: numba_l2sq,
    }

    return cfunc(scalar_kind_to_signature[dtype])(metric_kind_to_function[metric])
