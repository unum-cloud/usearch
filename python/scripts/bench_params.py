import os
import time

import fire
import plotly.express as px
import numpy as np
import pandas as pd


from usearch.index import Index
from usearch.io import load_matrix


def measure(f) -> float:
    a = time.time_ns()
    result = f()
    b = time.time_ns()
    c = b - a
    secs = c / (10 ** 9)
    print(f'- Took: {secs:.2f} seconds')
    return secs, result


def bench_usearch(index, vectors: np.ndarray) -> float:
    labels = np.arange(vectors.shape[0], dtype=np.longlong)
    assert len(index) == 0
    dt, _ = measure(lambda: index.add(labels, vectors))
    assert len(index) == vectors.shape[0]
    insertions_per_second = vectors.shape[0]/dt
    print(f'- Performance: {insertions_per_second:.2f} insertions/s')
    return insertions_per_second


class Main:

    def dimensions(
        self,
        count: int = 1_000_000,
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):

        dimensions = [
            2, 3, 4, 8, 16, 32, 96, 100,
            256, 384, 512, 768, 1024, 1536]
        results = []

        try:
            for ndim in dimensions:
                vectors_mat = np.random.rand(count, ndim).astype(np.float32)

                print(f'USearch for {ndim}-dimensional f32 vectors')
                index = Index(
                    ndim=ndim,
                    expansion_add=expansion_add,
                    expansion_search=expansion_search,
                    connectivity=connectivity,
                )
                speed = bench_usearch(index, vectors_mat)
                results.append({'ndim': ndim, 'speed': speed})
                del index
        except KeyboardInterrupt:
            pass

        df = pd.DataFrame(results)
        fig = px.line(
            df, x='ndim', y='speed', log_x=True,
            title='HNSW Indexing Speed vs Vector Dimensions')
        fig.write_image(os.path.join(os.path.dirname(
            __file__), 'bench_dimensions.png'))
        fig.show()

    def connectivity(
        self,
        count: int = 1_000_000,
        ndim: int = 256,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):

        connectivities = range(10, 20)
        results = []

        try:
            for connectivity in connectivities:
                vectors_mat = np.random.rand(count, ndim).astype(np.float32)

                print(f'USearch for {connectivity} connectivity')
                index = Index(
                    ndim=ndim,
                    expansion_add=expansion_add,
                    expansion_search=expansion_search,
                    connectivity=connectivity,
                )
                speed = bench_usearch(index, vectors_mat)
                results.append({'connectivity': connectivity, 'speed': speed})
                del index
        except KeyboardInterrupt:
            pass

        df = pd.DataFrame(results)
        fig = px.line(
            df, x='connectivity', y='speed', log_x=True,
            title='HNSW Indexing Speed vs Connectivity')
        fig.write_image(os.path.join(os.path.dirname(
            __file__), 'bench_connectivity.png'))
        fig.show()


if __name__ == '__main__':
    fire.Fire(Main)
