# Orthogonalization

Turn your Claude.ai conversation history into four deliberately *orthogonal* AI research
personas — a decomposition of your intellectual life into its most contrasting dimensions.

The pipeline ingests a Claude.ai data export, extracts the research topics you actually
talk about, embeds them, builds a topic-similarity network, finds its community structure,
and then picks the four communities that are maximally spread out in spectral space. An
LLM writes a detailed persona for each one, grounded in your real conversations and
explicitly contrasted against the other three.

The original use case: the four personas seeded the core memory of four autonomous
research agents that collaborated on project ideation, each arguing from a different
corner of the user's own interests.

## Pipeline

```
Claude.ai export (conversations.json)
        │
        ▼
1. loader.py            Parse + normalize; drop trivial conversations
                        (< 6 messages or < 200 chars)
        ▼
2. topic_extractor.py   Claude Haiku extracts 3–5 research topics per
                        conversation (content-hash cached)
        ▼
3. embedder.py          all-MiniLM-L6-v2 sentence embeddings (384-d,
                        L2-normalized so cosine = dot product)
        ▼
4. cluster.py           Cosine similarity graph (threshold 0.35) →
                        Leiden community detection (best of 3 seeded runs) →
                        spectral coordinates of cluster centroids
                        (2nd/3rd eigenvectors of the normalized Laplacian) →
                        select one cluster per spectral quadrant (NW/NE/SW/SE)
        ▼
5. persona_generator.py Claude Opus writes a persona per quadrant:
                        name, orientation, methods, domain, strengths,
                        blind spots — each contrasted against the others
```

The quadrant trick is what makes the personas *orthogonal* rather than just distinct:
the 2nd and 3rd Laplacian eigenvectors place cluster centroids on a 2-D map where
distance reflects dissimilarity, and taking one cluster per quadrant maximizes the
spread of the resulting personas.

## Getting your data

There is no API for this step — export your history manually:

1. Claude.ai → **Settings → Privacy → Export data**
2. You'll receive a download containing `conversations.json`
3. Point `--data-dir` at that file (or a directory of such files)

The loader also accepts a generic `{"messages": [{"role": ..., "content": ...}]}`
format, so transcripts from other sources can be dropped in.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python -m orthogonalization.bootstrap \
    --data-dir path/to/conversations.json \
    --user-name Alice \
    --user-description "a sociologist studying online communities"
```

Try it without your own data using the included synthetic example:

```bash
python -m orthogonalization.bootstrap --data-dir examples/synthetic_conversations.json
```

(Note the example is only 4 conversations — real runs want dozens to hundreds; the
pipeline warns below 10.)

Outputs land in `output/`:

| File | Contents |
|---|---|
| `topics.json` | Extracted topics, keyed by conversation title |
| `clusters.json` | The 4 quadrant clusters with member topics and spectral coordinates |
| `personas.json` | The 4 generated personas |

API cost is small: one Haiku call per conversation (cached by content hash in
`data/cache/topics/`, so re-runs are free) plus four Opus calls (~$0.20–0.40 total).

**Privacy note:** everything under `data/` and `output/` derives from your private
conversation history — titles and topics included. Both directories are gitignored;
keep it that way.

## Tests

```bash
pytest tests/
```

Unit tests run on synthetic data with a mocked Anthropic client; no API key needed.
`test_clustering.py` downloads the MiniLM model (~90 MB) on first run.

## Tuning

Constants at the top of `cluster.py`:

- `SIMILARITY_THRESHOLD` (0.35) — minimum cosine similarity for a graph edge. Raise it
  for tighter, smaller communities; lower it if the graph fragments.
- `LEIDEN_RESOLUTION` (1.0) — higher yields more, smaller clusters.
- `STABILITY_RUNS` (3) — Leiden restarts; the best-quality partition wins.
