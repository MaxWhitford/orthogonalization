#!/usr/bin/env python3
"""Tests for embedding and clustering with synthetic data."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orthogonalization.embedder import embed_topics
from orthogonalization.cluster import cluster_topics, _leiden_cluster, _spectral_quadrants


def test_embed_topics_shape():
    """Embeddings should have shape (N, 384) and be L2-normalized."""
    topics = ["Machine learning for policy", "Political economy of AI", "Climate adaptation"]
    embeddings = embed_topics(topics)

    assert embeddings.shape == (3, 384)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_topics_similar_topics_closer():
    """Semantically similar topics should have higher cosine similarity."""
    topics = [
        "Bayesian inference in political science",
        "Bayesian statistics for policy evaluation",
        "Deep learning for image classification",
    ]
    embeddings = embed_topics(topics)
    sim_01 = float(embeddings[0] @ embeddings[1])
    sim_02 = float(embeddings[0] @ embeddings[2])
    # Two Bayesian topics should be more similar than Bayesian vs deep learning
    assert sim_01 > sim_02


def test_leiden_cluster_produces_communities():
    """Leiden should find community structure in well-separated data."""
    np.random.seed(42)
    n = 60
    centers = np.random.randn(4, 20) * 3
    embeddings = np.vstack([
        centers[i] + np.random.randn(n // 4, 20) * 0.3
        for i in range(4)
    ])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    sim = embeddings @ embeddings.T
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > 0.3:
                edges.append((i, j))
                weights.append(float(sim[i, j]))

    k, labels = _leiden_cluster(n, edges, weights)
    assert k >= 4
    assert len(labels) == n
    assert len(set(labels)) >= 4


def test_spectral_quadrants_shape():
    """Spectral coordinates should have shape (k, 2)."""
    np.random.seed(42)
    k = 6
    n = 60
    embeddings = np.random.randn(n, 20)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    labels = np.array([i % k for i in range(n)])

    coords = _spectral_quadrants(embeddings, labels, k)
    assert coords.shape == (k, 2)


def test_cluster_topics_returns_4_quadrants():
    """Full pipeline should return 4 quadrant clusters."""
    np.random.seed(42)
    n = 80
    centers = np.random.randn(6, 384) * 2
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings = np.vstack([
        centers[i % 6] + np.random.randn(1, 384) * 0.08
        for i in range(n)
    ])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    topics = [f"Topic {i} about research area {'ABCDEF'[i % 6]}" for i in range(n)]

    clusters, sim_matrix = cluster_topics(embeddings, topics)

    assert len(clusters) == 4
    assert sim_matrix.shape == (n, n)
    quadrants = {c["quadrant"] for c in clusters}
    assert quadrants == {"NW", "NE", "SW", "SE"}
    for c in clusters:
        assert len(c["topics"]) > 0
