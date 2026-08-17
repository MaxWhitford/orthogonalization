#!/usr/bin/env python3
"""Tests for conversation loader."""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthogonalization.loader import load_conversations, conversation_to_text


def test_load_single_file():
    """Load conversations from a single JSON file."""
    conversations = [
        {
            "name": "Bayesian policy analysis",
            "chat_messages": [
                {"sender": "human", "text": "What is Bayesian inference?"},
                {"sender": "assistant", "text": "Bayesian inference is a method of statistical inference..."},
                {"sender": "human", "text": "How does it apply to policy evaluation?"},
                {"sender": "assistant", "text": "In policy analysis, Bayesian methods allow..."},
                {"sender": "human", "text": "Tell me more about informative priors"},
                {"sender": "assistant", "text": "Informative priors encode domain knowledge..."},
                {"sender": "human", "text": "What about conjugate priors specifically?"},
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversations, f)
        f.flush()

        result = load_conversations(f.name)
        assert len(result) == 1
        assert result[0]["title"] == "Bayesian policy analysis"
        assert len(result[0]["messages"]) == 7
        assert result[0]["messages"][0]["role"] == "human"


def test_load_directory():
    """Load conversations from a directory of JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            conv = {
                "name": f"Research conversation {i}",
                "chat_messages": [
                    {"sender": "human", "text": f"Substantive question {j} about topic " * 5}
                    for j in range(8)
                ],
            }
            Path(tmpdir, f"conv_{i}.json").write_text(json.dumps([conv]))

        result = load_conversations(tmpdir)
        assert len(result) == 3


def test_filter_short_conversations():
    """Short or trivial conversations should be filtered out."""
    conversations = [
        {
            "name": "Too short",
            "chat_messages": [
                {"sender": "human", "text": "Hi"},
                {"sender": "assistant", "text": "Hello"},
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversations, f)
        f.flush()

        result = load_conversations(f.name)
        assert len(result) == 0


def test_normalize_role_names():
    """Both sender/text and role/content field names should be handled."""
    conversations = [
        {
            "title": "Alt format",
            "messages": [
                {"role": "user", "content": "Question about research " * 20}
                for _ in range(8)
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversations, f)
        f.flush()

        result = load_conversations(f.name)
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "human"


def test_conversation_to_text():
    """Convert normalized conversation to text."""
    conv = {
        "title": "Test",
        "messages": [
            {"role": "human", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "human", "content": "Question 2"},
        ],
    }

    text = conversation_to_text(conv)
    assert "Question 1" in text
    assert "Answer 1" in text
    assert "Question 2" in text


def test_conversation_to_text_truncation():
    """Long conversations should be truncated."""
    conv = {
        "title": "Long conv",
        "messages": [
            {"role": "human", "content": "x" * 5000},
        ],
    }

    text = conversation_to_text(conv, max_chars=100)
    assert len(text) <= 120
    assert "[truncated]" in text


def test_content_block_fallback():
    """When text is empty, should fall back to content[0].text."""
    conversations = [
        {
            "name": "Content block test",
            "chat_messages": [
                {"sender": "human", "text": "", "content": [{"type": "text", "text": "Real question here about research " * 5}]},
                {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "Real answer here about research " * 5}]},
                {"sender": "human", "text": "", "content": [{"type": "text", "text": "Follow up question " * 5}]},
                {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "Follow up answer " * 5}]},
                {"sender": "human", "text": "", "content": [{"type": "text", "text": "Another question " * 5}]},
                {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "Another answer " * 5}]},
                {"sender": "human", "text": "", "content": [{"type": "text", "text": "Final question " * 5}]},
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversations, f)
        f.flush()

        result = load_conversations(f.name)
        assert len(result) == 1
        assert "Real question here" in result[0]["messages"][0]["content"]
