# FPPC Opinions Search Lab

Experimental workbench for building and evaluating search engines against ~14,100
FPPC (Fair Political Practices Commission) advisory opinions (1975–present).

See [CLAUDE.md](CLAUDE.md) for repo structure, rules, and detailed documentation.

## Quick start

Run the scorer against an engine:

```bash
python src/scorer.py \
  --search-module src.engines.bm25_baseline \
  --dataset eval/dataset.json \
  --output results/001-bm25-baseline.json
```

Run tests:

```bash
python -m pytest tests/test_scorer.py
```
