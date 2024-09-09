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


def evaluate_clustering_cosine(X, labels, centroids):
    """Evaluate clustering quality as average cosine distance to centroids"""

    # Normalize both data points and centroids
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    centroids_normalized = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Compute cosine similarity using dot product
    cosine_similarities = np.sum(X_normalized * centroids_normalized[labels], axis=1)

    # Convert cosine similarity to cosine distance
    cosine_distances = 1 - cosine_similarities

    # Return the average cosine distance
    return np.mean(cosine_distances)


def custom_faiss_clustering(X, k, max_iters=100):
    verbose = False
    d: int = X.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=max_iters, verbose=verbose)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    return I.flatten(), kmeans.centroids


def custom_usearch_clustering(X, k, max_iters=100):
    assignments, distances, centroids = usearch.compiled.kmeans(X, k, max_iterations=max_iters, threads=192)
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

    print("centroids: ", centroids.shape, centroids)

    quality = evaluate_clustering(X, labels, centroids)
    quality_cosine = evaluate_clustering_cosine(X, labels, centroids)
    print(f"Clustering quality (average distance to centroids): {quality:.4f}, cosine: {quality_cosine:.4f}")

    # Let's compare it to some random uniform assignment
    random_labels = np.random.randint(0, k, size=X.shape[0])
    random_quality = evaluate_clustering(X, random_labels, centroids)
    random_quality_cosine = evaluate_clustering_cosine(X, random_labels, centroids)
    print(f"- while random assignment quality: {random_quality:.4f}, cosine: {random_quality_cosine:.4f}")

    cluster_sizes = np.unique(labels, return_counts=True)[1]
    cluster_sizes_mean = np.mean(cluster_sizes)
    cluster_sizes_stddev = np.std(cluster_sizes)
    print(f"Cluster sizes: {cluster_sizes_mean:.2f} ± {cluster_sizes_stddev:.2f}")
    print(cluster_sizes)


if __name__ == "__main__":
    main()
