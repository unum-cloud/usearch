import os
import time
import collections

import fire
import plotly.express as px
import numpy as np
import pandas as pd


from usearch.index import Index, MetricKind
from usearch.eval import recall_members, Benchmark, BenchmarkResult
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,

    USES_OPENMP,
    USES_SIMSIMD,
)


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
                result: BenchmarkResult = Benchmark(
                    index, vectors_mat).__call__().__dict__()
                result['ndim'] = ndim
                results.append(result)
                del index
        except KeyboardInterrupt:
            pass

        df = pd.DataFrame(results)
        fig = px.line(
            df, x='ndim', y='add_per_second', log_x=True,
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
                result: BenchmarkResult = Benchmark(
                    index, vectors_mat).__call__().__dict__()
                result['connectivity'] = connectivity
                results.append(result)
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
        random_experiments: int = 4,
        number_of_clusters: int = 1_000,
        count: int = 1_000_000,
        ndim: int = 256,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        """How much does accuracy and speed fluctuate depending on the indexing order.

        :param random_experiments: How many times to repeat, defaults to 10
        :type random_experiments: int, optional
        :param count: Number of vectors, defaults to 1_000_000
        :type count: int, optional
        :param ndim: Number of dimensions per vector, defaults to 256
        :type ndim: int, optional
        """

        recall_levels = []
        vectors_mat = np.random.rand(count, ndim).astype(np.float32)

        for _ in range(random_experiments):
            np.random.shuffle(vectors_mat)
            index = Index(
                ndim=ndim,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                connectivity=connectivity,
                jit=True,
            )
            result: BenchmarkResult = Benchmark(
                index, vectors_mat).__call__()
            recall_levels.append(result.recall_at_one)
            del index

        min_ = min(recall_levels) * 100
        max_ = max(recall_levels) * 100
        print(f'Recall @ 1 varies between {min_:.3f} and {max_:.3f}')

        # Now let's try clustering and inserting in clusters
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return

        index = Index(
            ndim=ndim,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            connectivity=connectivity,
            jit=True,
        )

        clustering = KMeans(
            n_clusters=number_of_clusters,
            random_state=0, n_init='auto').fit(vectors_mat)

        partitioning = collections.defaultdict(list)
        for point, cluster in enumerate(clustering.labels_):
            partitioning[cluster].append(point)

        def vector_batches():
            for _, points in partitioning.items():
                labels = np.array(points)
                vectors = vectors_mat[labels, :]
                yield vectors

        result: BenchmarkResult = Benchmark(
            index, vector_batches).__call__()
        print(f'Recall @ 1 for sorted {result.recall_at_one:.3f}')


if __name__ == '__main__':
    Main().robustness()
    fire.Fire(Main)
