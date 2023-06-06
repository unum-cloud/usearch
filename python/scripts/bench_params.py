import os
import time

import fire
import plotly.express as px
import numpy as np
import pandas as pd

from usearch.index import Index
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,

    USES_OPENMP,
    USES_SIMSIMD,
)


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


def recall_at_one(index: Index, vectors: np.ndarray) -> float:

    labels = np.arange(vectors.shape[0])
    index.add(labels=labels, vectors=vectors)
    matches = index.search(vectors, 1)

    return np.sum(matches.labels.flatten() == labels) / len(labels)


class Main:

    def dimensions(
        self,
        count: int = 1_000_000,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        """Measures indexing speed for different dimensionality vectors.

        :param count: Number of vectors, defaults to 1_000_000
        :type count: int, optional
        """

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
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        """Measures indexing speed and accuracy for different level of
        connectivity in the levels of the hierarchical proximity graph.

        :param count: Number of vectors, defaults to 1_000_000
        :type count: int, optional
        :param ndim: Number of dimensions per vector, defaults to 256
        :type ndim: int, optional
        """

        connectivity_options = range(10, 20)
        results = []

        try:
            for connectivity in connectivity_options:
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

    def robustness(
        self,
        experiments: int = 1,
        count: int = 10000,
        ndim: int = 256,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        """How much does accuracy and speed fluctuate depending on the indexing order.

        :param experiments: How many times to repeat, defaults to 10
        :type experiments: int, optional
        :param count: Number of vectors, defaults to 1_000_000
        :type count: int, optional
        :param ndim: Number of dimensions per vector, defaults to 256
        :type ndim: int, optional
        """

        print(f'Connectivity: {connectivity}')
        print(f'Number of vectors: {count}')
        print(f'Vector dimensions: {ndim}')
        print(f'Expansion on Add: {expansion_add}')
        print(f'Expansion on Search: {expansion_search}')
        print(f'Uses OpenMP: {USES_OPENMP}')
        print(f'Uses SimSIMD: {USES_SIMSIMD}')

        recall_levels = []
        vectors_mat = np.random.rand(count, ndim).astype(np.float32)

        for _ in range(experiments):
            np.random.shuffle(vectors_mat)
            index = Index(
                ndim=ndim,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                connectivity=connectivity,
            )
            bench_usearch(index, vectors_mat)
            recall = recall_at_one(index, vectors_mat) * 100
            print(f'- Recall @ 1 {recall:.2f} %')
            recall_levels.append(recall)
            del index

        min_ = min(recall_levels) * 100
        max_ = max(recall_levels) * 100
        print(f'Recall @ 1 varies between {min_:.3f} and {max_:.3f}')

        # Now let's try clustering and inserting in clusters
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return

        count_clusters = count / expansion_add
        clustering = KMeans(
            n_clusters=count_clusters,
            random_state=0, n_init='auto').fit(vectors_mat)
        NamedTuple()


if __name__ == '__main__':
    Main().robustness()
    fire.Fire(Main)
