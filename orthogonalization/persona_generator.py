#!/usr/bin/env python3
"""Generate agent personas from topic clusters using Opus.

For each of the 4 quadrant clusters, Opus reads the topics and generates
a detailed persona: name, intellectual orientation, methods, domain,
strengths, and blind spots. These become the agents' core memory blocks.

Opus is used because persona generation requires creativity, nuance, and
deep understanding of intellectual traditions.
"""

import json
from typing import List, Dict

from anthropic import Anthropic

# Opus for creative persona generation
OPUS_MODEL = "claude-opus-4-5-20251101"

# Generic fallback identity; callers should pass the real user's name and a
# short description so the personas are grounded in the right person.
DEFAULT_USER_NAME = "the user"
DEFAULT_USER_DESCRIPTION = "a researcher with broad intellectual interests"


def generate_persona(
    cluster: dict,
    all_clusters: list,
    client: Anthropic,
    user_name: str = DEFAULT_USER_NAME,
    user_description: str = DEFAULT_USER_DESCRIPTION,
) -> dict:
    """Generate a persona for one quadrant cluster.

    Opus reads the cluster's topics (and the other clusters for contrast)
    and creates a detailed persona grounded in the intellectual themes
    represented by this cluster.

    Args:
        cluster: Cluster dict with 'quadrant' and 'topics' keys.
        all_clusters: All 4 clusters for differentiation context.
        client: Anthropic API client.
        user_name: The user's first name.
        user_description: Brief description of the user for grounding.

    Returns:
        Dict with keys: name, orientation, methods, domain, strengths,
        blind_spots, quadrant.
    """
    other_summaries = []
    for c in all_clusters:
        if c["quadrant"] != cluster["quadrant"]:
            sample = c["topics"][:3]
            other_summaries.append(f"  {c['quadrant']}: {'; '.join(sample)}")

    prompt = (
        "You are creating a research agent persona. This agent will be one of 4 "
        "orthogonalized agents, each representing a distinct dimension of a "
        "researcher's intellectual interests. The agents will collaborate on "
        "research projects, each bringing their unique perspective.\n\n"
        f"The user is {user_name}, {user_description}.\n\n"
        f"This agent represents the {cluster['quadrant']} quadrant. "
        f"Here are the {len(cluster['topics'])} topics in its cluster:\n"
    )
    for t in cluster["topics"]:
        prompt += f"- {t}\n"

    prompt += (
        "\nThe other 3 agents cover these areas (for contrast, avoid overlap):\n"
        + "\n".join(other_summaries)
        + "\n\n"
        "Generate a persona for this research agent. Return a JSON object with:\n"
        '- "name": A professional first name (real-sounding, not cutesy or generic)\n'
        '- "orientation": 2-3 sentences on this agent\'s intellectual orientation\n'
        '- "methods": What research methods/approaches this agent favors '
        "(qualitative vs quantitative, theoretical vs empirical, etc.)\n"
        '- "domain": What specific topics and questions this agent is tuned to\n'
        '- "strengths": What this agent excels at\n'
        '- "blind_spots": What this agent might miss or undervalue\n\n'
        "Return ONLY the JSON object, no markdown formatting or explanation."
    )

    response = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    persona = json.loads(text)
    persona["quadrant"] = cluster["quadrant"]

    return persona


def generate_all_personas(
    clusters: list,
    client: Anthropic,
    user_name: str = DEFAULT_USER_NAME,
    user_description: str = DEFAULT_USER_DESCRIPTION,
) -> List[Dict]:
    """Generate personas for all 4 quadrant clusters.

    Calls Opus once per cluster. Total cost ~$0.20-0.40 for 4 personas.

    Args:
        clusters: List of 4 cluster dicts from cluster_topics().
        client: Anthropic API client.
        user_name: The user's first name.
        user_description: Brief description of user.

    Returns:
        List of 4 persona dicts.
    """
    personas = []
    for cluster in clusters:
        print(f"  Generating persona for {cluster['quadrant']} "
              f"quadrant ({len(cluster['topics'])} topics)...")
        persona = generate_persona(
            cluster, clusters, client, user_name, user_description
        )
        print(f"    -> {persona['name']}: {persona['orientation'][:80]}...")
        personas.append(persona)

    return personas
