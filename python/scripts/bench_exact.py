from time import time
from typing import Literal

from faiss import knn, METRIC_L2, METRIC_INNER_PRODUCT
import fire

from usearch.compiled import hardware_acceleration
from usearch.eval import random_vectors

# Supplemantary imports for CLI arguments normalization
from usearch.index import (
    ScalarKind,
    MetricKind,
    search,
    _normalize_metric,
    _normalize_dtype,
)


def format_duration(duration):
    """Format duration in seconds to milliseconds, nicely formatted."""
    return f"{duration * 1000:,.2f} ms"


def calculate_throughput(duration, count):
    """Calculate and return throughput as calls per second."""
    if duration > 0:
        return f"{count / duration:,.2f} calls/sec"
    return "Inf calls/sec"


def run(
    n: int = 10**5,
    q: int = 10,
    k: int = 100,
    ndim: int = 256,
    dtype: Literal["b1", "i8", "f16", "f32", "f64"] = "f32",
    metric: Literal["cos", "ip", "l2sq"] = "ip",
):

    metric: MetricKind = _normalize_metric(metric)
    dtype: ScalarKind = _normalize_dtype(dtype, ndim=ndim, metric=metric)
    acceleration = hardware_acceleration(
        dtype=dtype,
        ndim=ndim,
        metric_kind=metric,
    )
    print(f"Hardware acceleration in USearch: {acceleration}")

    x = random_vectors(n, ndim=ndim, dtype=dtype)

    start = time()
    _ = search(
        x,
        x[:q],
        k,
        metric=metric,
        exact=True,
    ).keys
    duration = time() - start
    print(f"USearch: {format_duration(duration)} ({calculate_throughput(duration, q)})")

    if metric not in [MetricKind.L2sq, MetricKind.IP]:
        return
    if dtype not in [ScalarKind.I8, ScalarKind.F16, ScalarKind.F32, ScalarKind.F64]:
        return

    start = time()
    faiss_metric = METRIC_L2 if metric == "l2sq" else METRIC_INNER_PRODUCT
    _ = knn(x[:q], x, k, metric=faiss_metric)[1]
    duration = time() - start
    print(f"FAISS:   {format_duration(duration)} ({calculate_throughput(duration, q)})")


if __name__ == "__main__":
    fire.Fire(run)
