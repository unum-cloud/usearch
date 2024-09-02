import os
import argparse

import numpy as np
import faiss
from tqdm import tqdm

import usearch
from usearch.index import Index
from usearch.io import load_matrix


def initialize_centroids(X, k):
    """Randomly choose k data points as initial centroids"""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def assign_clusters_numpy(X, centroids):
    """Assign each data point to the nearest centroid (numpy NumPy implementation)"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def assign_clusters_faiss(X, centroids):
    """Assign each data point to the nearest centroid using FAISS"""
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, labels = index.search(X, 1)
    return labels.flatten()


def assign_clusters_usearch(X, centroids):
    """Assign each data point to the nearest centroid using USearch"""
    dim = centroids.shape[1]
    index = Index(ndim=dim, metric="l2sq")
    index.add(None, centroids)
    results = index.search(X, 1)
    labels = results.keys.flatten()
    return labels


def update_centroids(X, labels, k):
    """Compute new centroids as the mean of all data points assigned to each cluster"""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])


def kmeans(X, k, max_iters=100, tol=1e-4, assign_clusters_func=assign_clusters_numpy):
    centroids = initialize_centroids(X, k)

    for i in tqdm(range(max_iters), desc="KMeans Iterations"):
        labels = assign_clusters_func(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids


def evaluate_clustering(X, labels, centroids):
    """Evaluate clustering quality as average distance to centroids"""
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    return np.mean(distances)


def inverted_kmeans(X, k, max_iters=100, tol=1e-4):
    # This algorithm is a bit different.
    # In a typical KMeans algorithm - we would construct an index of centroids, and search all points in it.
    # That's not very relevant, as there are very few centroids, and many points. Inverted setup makes more sense.

    pass


def custom_faiss_clustering(X, k, max_iters=100):
    verbose = False
    d: int = X.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=max_iters, verbose=verbose)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    return I.flatten(), kmeans.centroids


def custom_usearch_clustering(X, k, max_iters=100):
    # index = Index(ndim=X.shape[1], metric="l2sq")
    # index.add(None, X)
    # clustering = index.cluster(min_count=k, max_count=k)
    # query_ids = clustering.queries.flatten()
    # cluster_ids = clustering.matches.keys.flatten()
    # reordered_cluster_ids = cluster_ids[query_ids]
    # return reordered_cluster_ids, X[query_ids]

    assignments, distances, centroids = usearch.compiled.kmeans(X, k, max_iterations=max_iters)
    return assignments, centroids


def main():
    parser = argparse.ArgumentParser(description="Compare KMeans clustering algorithms")
    parser.add_argument("--vectors", type=str, required=True, help="Path to binary matrix file")
    parser.add_argument("-k", type=int, required=True, help="Number of centroids")
    parser.add_argument("-n", type=int, help="Upper bound on number of points to use")
    parser.add_argument(
        "--method",
        type=str,
        choices=["numpy", "faiss", "usearch", "custom_usearch", "custom_faiss"],
        default="numpy",
        help="Clustering backend",
    )

    args = parser.parse_args()

    X = load_matrix(args.vectors, count_rows=args.n)
    k = args.k
    method = args.method

    if method == "custom_usearch":
        labels, centroids = custom_usearch_clustering(X, k)
    elif method == "custom_faiss":
        labels, centroids = custom_faiss_clustering(X, k)
    else:
        if method == "numpy":
            assign_clusters_func = assign_clusters_numpy
        elif method == "faiss":
            assign_clusters_func = assign_clusters_faiss
        elif method == "usearch":
            assign_clusters_func = assign_clusters_usearch
        labels, centroids = kmeans(X, k, assign_clusters_func=assign_clusters_func)

    quality = evaluate_clustering(X, labels, centroids)
    print(f"Clustering quality (average distance to centroids): {quality:.4f}")

    cluster_sizes = np.unique(labels, return_counts=True)[1]
    cluster_sizes_mean = np.mean(cluster_sizes)
    cluster_sizes_stddev = np.std(cluster_sizes)
    print(f"Cluster sizes: {cluster_sizes_mean:.2f} ± {cluster_sizes_stddev:.2f}")
    print(cluster_sizes)


if __name__ == "__main__":
    main()
