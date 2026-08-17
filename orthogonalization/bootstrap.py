#!/usr/bin/env python3
"""End-to-end orthogonalization pipeline.

Ties together: conversation loading -> topic extraction -> embedding ->
Leiden clustering -> spectral quadrant selection -> Opus persona generation.

Saves output (4 personas + intermediate artifacts) to JSON for use by
the agent creation script.

Usage:
    python -m orthogonalization.bootstrap --data-dir data/conversations/ \
        --user-name Alice --user-description "a sociologist studying online communities"

Output:
    output/topics.json     - extracted topics per conversation
    output/clusters.json   - 4 quadrant clusters with topics
    output/personas.json   - 4 agent personas
"""

import argparse
import json
from pathlib import Path

from anthropic import Anthropic

from orthogonalization.loader import load_conversations
from orthogonalization.topic_extractor import extract_all_topics
from orthogonalization.embedder import embed_topics
from orthogonalization.cluster import cluster_topics
from orthogonalization.persona_generator import generate_all_personas

OUTPUT_DIR = Path("output")

# Generic fallback identity; pass --user-name / --user-description to ground
# the personas in the actual person whose conversations are being analyzed.
DEFAULT_USER_NAME = "the user"
DEFAULT_USER_DESCRIPTION = "a researcher with broad intellectual interests"


def run_pipeline(
    data_dir: str,
    output_dir: str = None,
    user_name: str = DEFAULT_USER_NAME,
    user_description: str = DEFAULT_USER_DESCRIPTION,
) -> list:
    """Run the full orthogonalization pipeline.

    Steps:
      1. Load conversations from Claude.ai export
      2. Extract topics from each conversation via Haiku
      3. Embed all topics with MiniLM (384-d)
      4. Leiden cluster + spectral quadrant selection
      5. Generate 4 personas via Opus

    Args:
        data_dir: Path to conversation export directory or file.
        output_dir: Path to save pipeline output.
        user_name: User's first name.
        user_description: Brief description of user.

    Returns:
        List of 4 persona dicts.
    """
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    client = Anthropic()

    # Step 1
    print("=" * 60)
    print("Step 1: Loading conversations")
    print("=" * 60)
    conversations = load_conversations(data_dir)
    print(f"  Loaded {len(conversations)} substantive conversations\n")

    if len(conversations) < 10:
        print("  WARNING: Fewer than 10 conversations. Topic diversity may be low.")
        print("  Consider adding more conversations for better orthogonalization.\n")

    # Step 2
    print("=" * 60)
    print("Step 2: Extracting topics via Haiku")
    print("=" * 60)
    topics_by_conv = extract_all_topics(conversations, client)

    all_topics = []
    for conv_topics in topics_by_conv.values():
        all_topics.extend(conv_topics)
    print(f"\n  Total topics: {len(all_topics)}\n")

    (out / "topics.json").write_text(json.dumps(topics_by_conv, indent=2))

    # Step 3
    print("=" * 60)
    print("Step 3: Embedding topics with MiniLM")
    print("=" * 60)
    embeddings = embed_topics(all_topics)
    print(f"  Embedded {len(all_topics)} topics -> shape {embeddings.shape}\n")

    # Step 4
    print("=" * 60)
    print("Step 4: Leiden clustering + spectral quadrant selection")
    print("=" * 60)
    clusters, sim_matrix = cluster_topics(embeddings, all_topics)
    for c in clusters:
        print(f"  {c['quadrant']}: {len(c['topics'])} topics (cluster {c['cluster_id']})")
    print()

    (out / "clusters.json").write_text(json.dumps(clusters, indent=2))

    # Step 5
    print("=" * 60)
    print("Step 5: Generating personas via Opus")
    print("=" * 60)
    personas = generate_all_personas(clusters, client, user_name, user_description)
    print()

    (out / "personas.json").write_text(json.dumps(personas, indent=2))

    # Summary
    print("=" * 60)
    print("Orthogonalization complete!")
    print("=" * 60)
    for p in personas:
        print(f"  {p['quadrant']} -> {p['name']}: {p['domain'][:60]}...")
    print(f"\nOutput saved to {out}/")
    print(f"  topics.json    - {len(all_topics)} topics from {len(conversations)} conversations")
    print(f"  clusters.json  - 4 quadrant clusters")
    print(f"  personas.json  - 4 agent personas")

    return personas


def main():
    parser = argparse.ArgumentParser(
        description="Run orthogonalization pipeline on conversation exports"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to conversation export directory or JSON file",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--user-name", default=DEFAULT_USER_NAME, help="User's first name"
    )
    parser.add_argument(
        "--user-description", default=DEFAULT_USER_DESCRIPTION,
        help="Brief user description (grounds the generated personas)",
    )
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.output_dir, args.user_name, args.user_description)


if __name__ == "__main__":
    main()
