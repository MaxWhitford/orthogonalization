#!/usr/bin/env python3
"""Extract research topics from conversations using Haiku.

Each conversation is sent to Haiku with a prompt asking for 3-5 distinct
research questions or intellectual topics discussed. Results are cached
by conversation content hash to avoid redundant API calls.

Adapted from the discovery engine's question_tensions.py, which extracts
questions from researcher profiles. Here we extract from conversations.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import Anthropic

from orthogonalization.loader import conversation_to_text

# Haiku for cheap/fast extraction
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Default cache location (relative to the current working directory)
CACHE_DIR = Path("data") / "cache" / "topics"


def extract_topics(
    conversation: dict,
    client: Anthropic,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """Extract 3-5 research topics from a single conversation.

    Uses Haiku to identify the core intellectual topics being explored.
    Focuses on substantive research questions, not meta-discussion about
    AI tools or capabilities.

    Args:
        conversation: Normalized conversation dict with 'title' and 'messages'.
        client: Anthropic API client.
        cache_dir: Directory for caching results. Defaults to CACHE_DIR.

    Returns:
        List of 3-5 topic strings (research questions or descriptions).
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    cache_key = _cache_key(conversation)
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Build the extraction prompt
    conv_text = conversation_to_text(conversation)

    prompt = (
        "Analyze this conversation and extract 3-5 distinct research topics or "
        "intellectual questions being explored. Focus on substantive academic "
        "or research topics, not meta-discussion about AI tools or capabilities.\n\n"
        "Return ONLY a JSON array of strings, each a concise topic description "
        "(1-2 sentences). Example:\n"
        '["How does algorithmic curation affect political polarization on social media?", '
        '"The relationship between Bayesian reasoning and scientific replication crises"]\n\n'
        f"Conversation title: {conversation['title']}\n\n"
        f"{conv_text}"
    )

    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse the JSON array from the response
    text = response.content[0].text.strip()
    # Handle cases where Haiku wraps the JSON in markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    topics = json.loads(text)

    # Cache the result
    cache_file.write_text(json.dumps(topics))

    return topics


def extract_all_topics(
    conversations: List[dict],
    client: Anthropic,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Extract topics from all conversations in batch.

    Processes each conversation through Haiku topic extraction, with
    caching to avoid redundant API calls on re-runs.

    Args:
        conversations: List of normalized conversation dicts.
        client: Anthropic API client.
        cache_dir: Directory for caching results.

    Returns:
        Dict mapping conversation title to list of extracted topics.
    """
    all_topics = {}
    for i, conv in enumerate(conversations):
        title = conv["title"]
        print(f"  [{i + 1}/{len(conversations)}] {title[:60]}...")
        topics = extract_topics(conv, client, cache_dir)
        all_topics[title] = topics
        print(f"    -> {len(topics)} topics")

    total = sum(len(t) for t in all_topics.values())
    print(f"\nExtracted {total} topics from {len(conversations)} conversations")
    return all_topics


def _cache_key(conversation: dict) -> str:
    """Generate a deterministic cache key from conversation content."""
    content = json.dumps(conversation["messages"], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()
