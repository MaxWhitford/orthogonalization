#!/usr/bin/env python3
"""Tests for topic extractor with mocked Anthropic API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthogonalization.topic_extractor import extract_topics, extract_all_topics


def _mock_anthropic_client(topics_response: list) -> MagicMock:
    """Create a mock Anthropic client that returns the given topics as JSON."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(topics_response))]
    client.messages.create.return_value = response
    return client


def test_extract_topics_returns_list():
    """Topic extraction should return a list of topic strings."""
    expected_topics = [
        "How does Bayesian inference apply to policy evaluation?",
        "What role do informative priors play in causal inference?",
    ]
    client = _mock_anthropic_client(expected_topics)

    conv = {
        "title": "Bayesian policy",
        "messages": [
            {"role": "human", "content": "Let's discuss Bayesian methods in policy"},
            {"role": "assistant", "content": "Bayesian methods allow us to..."},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = extract_topics(conv, client, cache_dir=Path(tmpdir))
        assert result == expected_topics
        assert len(result) == 2


def test_extract_topics_caches_results():
    """Second call with same conversation should hit cache, not API."""
    topics = ["Topic 1", "Topic 2", "Topic 3"]
    client = _mock_anthropic_client(topics)

    conv = {
        "title": "Test conv",
        "messages": [
            {"role": "human", "content": "Discussion about research topics"},
            {"role": "assistant", "content": "Here are my thoughts on the matter..."},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        result1 = extract_topics(conv, client, cache_dir=cache_dir)
        result2 = extract_topics(conv, client, cache_dir=cache_dir)

        # API should only be called once; second call uses cache
        assert client.messages.create.call_count == 1
        assert result1 == result2


def test_extract_all_topics():
    """Batch extraction across multiple conversations."""
    topics = ["Topic A", "Topic B"]
    client = _mock_anthropic_client(topics)

    convs = [
        {"title": f"Conv {i}", "messages": [{"role": "human", "content": f"About topic {i}"}]}
        for i in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = extract_all_topics(convs, client, cache_dir=Path(tmpdir))
        assert len(result) == 3
        assert all(len(t) == 2 for t in result.values())


def test_extract_topics_handles_markdown_wrapped_json():
    """Haiku sometimes wraps JSON in markdown code blocks."""
    topics = ["Topic X", "Topic Y"]
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text='```json\n["Topic X", "Topic Y"]\n```')]
    client.messages.create.return_value = response

    conv = {
        "title": "Markdown test",
        "messages": [{"role": "human", "content": "Research discussion"}],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = extract_topics(conv, client, cache_dir=Path(tmpdir))
        assert result == topics
