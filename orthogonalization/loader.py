#!/usr/bin/env python3
"""Load and normalize conversation exports from Claude.ai.

Handles the Claude.ai data export format (JSON with chat_messages arrays)
and normalizes it for the topic extraction pipeline.

Expected input: A directory containing JSON files, or a single JSON file
with a list of conversations.

Each conversation needs:
  - title/name (for identification)
  - messages with sender and text fields

Claude.ai export format (per conversation):
    {
        "uuid": "...",
        "name": "Conversation Title",
        "chat_messages": [
            {"sender": "human", "text": "...", "content": [{"type": "text", "text": "..."}]},
            {"sender": "assistant", "text": "...", "content": [{"type": "text", "text": "..."}]}
        ]
    }
"""

import json
from pathlib import Path
from typing import List

# Conversations shorter than this are filtered out as trivial.
# MIN_CHAR_COUNT is a secondary guard against conversations that meet the
# message-count threshold but consist entirely of very short messages (e.g.,
# pure UI noise). 200 chars over 6+ messages is ~33 chars/message average —
# still below any substantive exchange, but above greetings/acks.
MIN_MESSAGES = 6
MIN_CHAR_COUNT = 200


def load_conversations(data_path: str) -> List[dict]:
    """Load conversations from a Claude.ai export.

    Accepts either:
    - A single JSON file containing a list of conversations
    - A directory of JSON files (one conversation per file, or batched)

    Both the Claude.ai format (chat_messages/sender/text) and a generic
    format (messages/role/content) are supported.

    Args:
        data_path: Path to a JSON file or directory of JSON files.

    Returns:
        List of normalized conversation dicts, each with:
        - title: str
        - messages: List[dict] with 'role' and 'content' keys
    """
    path = Path(data_path)

    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        conversations = raw if isinstance(raw, list) else [raw]
    elif path.is_dir():
        conversations = []
        for f in sorted(path.glob("*.json")):
            raw = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                conversations.extend(raw)
            else:
                conversations.append(raw)
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    return [_normalize(c) for c in conversations if _is_substantive(c)]


def _extract_message_text(msg: dict) -> str:
    """Extract text content from a message, checking both text and content fields.

    Claude.ai exports sometimes have empty text but populated content blocks.
    Falls back through: msg["text"] -> msg["content"][0]["text"] -> msg["content"] (str) -> ""
    """
    text = msg.get("text") or ""
    if text:
        return text

    # Fall back to content blocks (Claude.ai format)
    content = msg.get("content")
    if isinstance(content, list) and content:
        first_block = content[0]
        if isinstance(first_block, dict):
            return first_block.get("text", "")
    elif isinstance(content, str):
        return content

    return ""


def _normalize(conversation: dict) -> dict:
    """Normalize a conversation to a standard format.

    Handles both Claude.ai export fields (chat_messages, sender, text)
    and generic fields (messages, role, content).
    """
    raw_messages = (
        conversation.get("chat_messages")
        or conversation.get("messages")
        or []
    )

    messages = []
    for msg in raw_messages:
        role = msg.get("sender") or msg.get("role") or "unknown"
        content = _extract_message_text(msg)
        # Normalize role names to human/assistant
        if role in ("human", "user"):
            role = "human"
        elif role in ("assistant", "ai"):
            role = "assistant"
        messages.append({"role": role, "content": content})

    return {
        "title": conversation.get("name") or conversation.get("title") or "Untitled",
        "messages": messages,
    }


def _is_substantive(conversation: dict) -> bool:
    """Filter out trivial conversations that are too short or shallow.

    A conversation needs at least MIN_MESSAGES messages and MIN_CHAR_COUNT
    total characters to be worth extracting topics from.
    """
    raw_messages = (
        conversation.get("chat_messages")
        or conversation.get("messages")
        or []
    )

    if len(raw_messages) < MIN_MESSAGES:
        return False

    total_chars = sum(len(_extract_message_text(msg)) for msg in raw_messages)

    return total_chars >= MIN_CHAR_COUNT


def conversation_to_text(conversation: dict, max_chars: int = 4000) -> str:
    """Convert a normalized conversation to a truncated text string.

    Includes both human and assistant messages for full context,
    truncated to stay within the topic extraction prompt's budget.

    Args:
        conversation: Normalized conversation dict with 'messages' key.
        max_chars: Maximum character count before truncation.

    Returns:
        Formatted text of the conversation, truncated if needed.
    """
    parts = []
    for msg in conversation["messages"]:
        role = msg["role"].capitalize()
        content = msg["content"]
        parts.append(f"{role}: {content}")

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return text
