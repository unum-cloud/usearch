from time import time

from usearch.index import search, MetricKind, ScalarKind
from usearch.compiled import hardware_acceleration
from faiss import knn, METRIC_L2
import numpy as np
import fire


def run(
    n: int = 10**5,
    q: int = 10,
    k: int = 100,
    ndim: int = 256,
    half: bool = False,
):
    usearch_dtype = ScalarKind.F16 if half else ScalarKind.F32
    acceleration = hardware_acceleration(
        dtype=usearch_dtype,
        ndim=ndim,
        metric_kind=MetricKind.L2sq,
    )
    print("Hardware acceleration in USearch: ", acceleration)

    x = np.random.random((n, ndim))
    x = x.astype(np.float16) if half else x.astype(np.float32)

    start = time()
    _ = search(
        x,
        x[:q],
        k,
        MetricKind.L2sq,
        exact=True,
    ).keys
    print("USearch: ", time() - start)

    start = time()
    _ = knn(x[:q], x, k, metric=METRIC_L2)[1]
    print("FAISS:   ", time() - start)


if __name__ == "__main__":
    fire.Fire(run)
