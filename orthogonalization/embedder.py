#!/usr/bin/env python3
"""Embed topics using all-MiniLM-L6-v2 for clustering.

Uses the same embedding model as the discovery engine's question_tensions
pipeline for consistency. Produces 384-dimensional embeddings, L2-normalized
for cosine similarity computation via dot product.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Same model as discovery engine (see config.SENTENCE_TRANSFORMER_MODEL)
MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded model singleton
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model.

    The model is loaded once and reused across calls to avoid
    repeated initialization (~1-2s per load).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_topics(topics: list) -> np.ndarray:
    """Embed a list of topic strings into 384-dimensional vectors.

    Args:
        topics: List of topic strings to embed.

    Returns:
        numpy array of shape (N, 384) with L2-normalized embeddings.
        Cosine similarity can be computed as a simple dot product.
    """
    model = get_model()
    embeddings = model.encode(topics, show_progress_bar=len(topics) > 50)

    # L2 normalize so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Guard against zero-norm edge case
    return embeddings / norms
