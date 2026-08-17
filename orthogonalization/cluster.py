#!/usr/bin/env python3
"""Leiden clustering + spectral quadrant selection for topic orthogonalization.

Adapted from the discovery engine's question_tensions.py and leiden_clustering.py.

Pipeline:
  1. Compute cosine similarity matrix from L2-normalized embeddings
  2. Threshold to build edge list (topic similarity graph)
  3. Leiden community detection for stable cluster assignments
  4. Compute spectral coordinates for cluster centroids
  5. Select 4 clusters from spectral quadrants (NW, NE, SW, SE) to maximize
     orthogonality between agent personas
"""

import numpy as np
from typing import List, Dict, Tuple

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# Minimum cosine similarity to create an edge between two topics.
# Lower than the discovery engine's V8_QUESTION_SIMILARITY_THRESHOLD (0.5)
# because conversation topics are more diverse than researcher questions.
SIMILARITY_THRESHOLD = 0.35

# Leiden resolution parameter (higher = more, smaller clusters)
LEIDEN_RESOLUTION = 1.0

# Cluster count bounds
MIN_K = 4
MAX_K = 20

# Number of Leiden runs to check partition stability
STABILITY_RUNS = 3


def cluster_topics(
    embeddings: np.ndarray,
    topics: list,
) -> Tuple[List[Dict], np.ndarray]:
    """Cluster topics and select 4 quadrant representatives.

    Full pipeline: similarity matrix -> edge thresholding -> Leiden ->
    spectral coordinates -> quadrant selection.

    Args:
        embeddings: (N, 384) L2-normalized topic embeddings.
        topics: List of N topic strings (same order as embeddings).

    Returns:
        Tuple of:
        - List of 4 cluster dicts, each with quadrant, cluster_id,
          topics, centroid_x, centroid_y
        - sim_matrix: (N, N) cosine similarity matrix

    Raises:
        ImportError: If leidenalg or igraph are not installed.
        ValueError: If no edges found above the similarity threshold.
    """
    if not LEIDEN_AVAILABLE:
        raise ImportError(
            "leidenalg and python-igraph required: pip install leidenalg python-igraph"
        )

    n = len(topics)

    # 1. Cosine similarity (embeddings are L2-normalized, so dot product = cosine)
    sim_matrix = embeddings @ embeddings.T

    # 2. Build edge list from thresholded similarities
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                edges.append((i, j))
                weights.append(float(sim_matrix[i, j]))

    if not edges:
        raise ValueError(
            f"No edges found with similarity threshold {SIMILARITY_THRESHOLD}. "
            f"Got {n} topics. Try lowering SIMILARITY_THRESHOLD or adding more "
            "conversations to the corpus."
        )

    # 3. Leiden clustering
    k, labels = _leiden_cluster(n, edges, weights)

    # 4. Spectral coordinates for cluster centroids
    centroids_xy = _spectral_quadrants(embeddings, labels, k)

    # 5. Select 4 clusters from quadrants
    selected = _select_quadrant_clusters(centroids_xy, labels, topics, k)

    return selected, sim_matrix


def _leiden_cluster(
    n: int,
    edges: list,
    weights: list,
) -> Tuple[int, np.ndarray]:
    """Run Leiden community detection with stability check.

    Runs the Leiden algorithm multiple times and keeps the highest-quality
    partition. Uses RBConfigurationVertexPartition (same as discovery engine).

    Args:
        n: Number of nodes (topics).
        edges: List of (i, j) edge tuples.
        weights: Corresponding edge weights.

    Returns:
        (k, labels) where k is cluster count and labels is per-topic assignments.
    """
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    best_partition = None
    best_quality = -float("inf")

    for seed in range(STABILITY_RUNS):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=LEIDEN_RESOLUTION,
            seed=seed,
        )

        if partition.quality() > best_quality:
            best_quality = partition.quality()
            best_partition = partition

    labels = np.array(best_partition.membership)
    k = len(set(labels))

    return k, labels


def _spectral_quadrants(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute 2D spectral coordinates for cluster centroids.

    Builds a similarity graph between cluster centroids, computes the
    normalized Laplacian, and uses the 2nd and 3rd smallest eigenvectors
    as x and y coordinates.

    Adapted from the discovery engine's spectral_clustering.py.

    Args:
        embeddings: (N, 384) topic embeddings.
        labels: Per-topic cluster assignments.
        k: Number of clusters.

    Returns:
        (k, 2) array of spectral coordinates.
    """
    # Compute cluster centroids (mean of member embeddings)
    centroids = np.zeros((k, embeddings.shape[1]))
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            centroids[i] = embeddings[mask].mean(axis=0)

    # L2 normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids = centroids / norms

    # Adjacency matrix between centroids (cosine similarity)
    W = centroids @ centroids.T
    np.fill_diagonal(W, 0)
    W = np.maximum(W, 0)

    # Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    d = W.sum(axis=1)
    d[d == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(k) - D_inv_sqrt @ W @ D_inv_sqrt

    # Eigendecomposition
    n_components = min(3, k)
    if k <= 3:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    else:
        eigenvalues, eigenvectors = eigsh(
            coo_matrix(L).tocsc(), k=n_components, which="SM"
        )

    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    # Use 2nd and 3rd eigenvectors as (x, y) spectral coordinates
    if k >= 3:
        coords = eigenvectors[:, 1:3]
    else:
        coords = eigenvectors[:, :2]

    return coords


def _select_quadrant_clusters(
    centroids_xy: np.ndarray,
    labels: np.ndarray,
    topics: list,
    k: int,
) -> List[Dict]:
    """Select 4 clusters, one from each spectral quadrant.

    For each quadrant (NW, NE, SW, SE), picks the cluster centroid
    closest to the quadrant's ideal corner to maximize orthogonality.

    Args:
        centroids_xy: (k, 2) spectral coordinates.
        labels: Per-topic cluster assignments.
        topics: All topic strings.
        k: Number of clusters.

    Returns:
        List of 4 cluster dicts.
    """
    quadrant_targets = {
        "NW": np.array([-1.0, 1.0]),
        "NE": np.array([1.0, 1.0]),
        "SW": np.array([-1.0, -1.0]),
        "SE": np.array([1.0, -1.0]),
    }

    selected = []
    used_clusters = set()

    for quadrant, target in quadrant_targets.items():
        best_cluster = None
        best_dist = float("inf")

        for c in range(k):
            if c in used_clusters:
                continue
            dist = float(np.linalg.norm(centroids_xy[c] - target))
            if dist < best_dist:
                best_dist = dist
                best_cluster = c

        if best_cluster is None:
            continue

        used_clusters.add(best_cluster)

        cluster_topics_list = [
            topics[i] for i in range(len(topics)) if labels[i] == best_cluster
        ]

        selected.append({
            "quadrant": quadrant,
            "cluster_id": int(best_cluster),
            "topics": cluster_topics_list,
            "centroid_x": float(centroids_xy[best_cluster, 0]),
            "centroid_y": float(centroids_xy[best_cluster, 1]),
        })

    return selected
