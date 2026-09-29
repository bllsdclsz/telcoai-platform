# support-rag: multilingual customer-support assistant

A retrieval-augmented support assistant for a Swiss telecom operator, answering in **German, French, Italian and English** from the help center.

> The operator **"Nordalp Mobile"** and all its help articles, prices and URLs (`*.nordalp.example`) are **fictional**, written for this project. They are not affiliated with any real company.

Status: **retrieval + evaluation gate** (this stage). Next: generation with a local LLM (Ollama, IBM Granite 4.2), guardrails, audit logging and generation evaluation. See [the roadmap](../docs/roadmap.md).

## Pipeline

```
corpus/*.yaml ──▶ load + validate ──▶ chunk (paragraphs, title prefix) ──▶ embed (multilingual-e5-large, ONNX)
 (21 topics × DE/FR/IT/EN)                                                    │
                                                                              ▼
question + UI language ──▶ embed query ──▶ Qdrant search (filter: lang) ──▶ top-k chunks with source URLs
```

- `corpus/`: one YAML per topic with every language (a CMS-export format). Loading fails if any translation is missing, so all languages stay in sync.
- `chunking.py`: packs whole paragraphs up to 120 words and prefixes each chunk with the article title, so a fact is never split in half.
- `embeddings.py`: fastembed (ONNX Runtime) encoders. This is CPU-friendly, has no PyTorch dependency, and handles the model-specific `query:`/`passage:` prefixes that e5 needs.
- `index.py` / `retrieve.py`: Qdrant with stable point IDs (re-ingesting overwrites) and a language filter.
  - Hybrid search (dense + BM25 with reciprocal rank fusion) is available but switched off. The benchmark below shows why.
  - Qdrant runs **embedded** for tests and CI, or as a server via `RAG_QDRANT_URL`.
- `evaluate.py`: the retrieval evaluation harness and CI gate.

## Evaluation

`eval/retrieval_golden.yaml` holds **94 questions** (about 23 per language) phrased the way customers ask, including 10 deliberately confusable ones. Examples: slow mobile data at home versus while roaming; cancelling versus porting a number; the cost of calling the US from Switzerland versus calling from the US.

A question counts as a hit at rank _r_ when the _r_-th retrieved chunk belongs to the expected topic. The harness reports four kinds of numbers:

- hit@1/3/5 and MRR@10 overall and per language, with the UI language known (the production path);
- the same metrics without the language filter, to show cross-lingual robustness;
- latency;
- every miss.

### Embedding model benchmark (2026-09-29, CPU)

| Dense model                           | Search         | hit@1     | hit@3     | MRR       | DE / FR / IT / EN hit@3   | No lang filter hit@3 | ms/query |
| ------------------------------------- | -------------- | --------- | --------- | --------- | ------------------------- | -------------------- | -------- |
| **multilingual-e5-large**             | dense          | **0.872** | **0.979** | **0.927** | 0.96 / 1.00 / 1.00 / 0.96 | **0.979**            | 53       |
| multilingual-e5-large                 | hybrid (+BM25) | 0.830     | 0.957     | 0.901     | 0.96 / 0.96 / 0.96 / 0.96 | 0.968                | 59       |
| jina-embeddings-v3                    | dense          | 0.840     | 0.968     | 0.900     | 0.96 / 1.00 / 0.96 / 0.96 | 0.872                | 1090     |
| paraphrase-multilingual-MiniLM-L12-v2 | hybrid (+BM25) | 0.777     | 0.904     | 0.848     | 0.92 / 0.87 / 0.96 / 0.88 | 0.883                | 11       |
| paraphrase-multilingual-MiniLM-L12-v2 | dense          | 0.702     | 0.883     | 0.802     | 0.83 / 0.87 / 0.91 / 0.92 | 0.809                | 6        |
| potion-multilingual-128M (static)     | dense          | 0.596     | 0.862     | 0.725     | 0.83 / 0.83 / 0.91 / 0.88 | 0.723                | 1        |

**Decision:** multilingual-e5-large, dense-only.

- **Best quality:** best on every metric, and the most robust when the language is unknown.
- **Fast enough:** about 50 ms per query on CPU.
- **Why BM25 is off:** hybrid search _helps_ the weak model but _hurts_ e5. fastembed's BM25 uses English stemming, which adds noise on DE/FR/IT text. A language-specific keyword index would be the next thing to try.

The two remaining misses are genuinely ambiguous: "blocked internet in Thailand" returns the outage article before roaming, and "line blocked although I pay by eBill" returns third-party blocking before late payment. They stay in the golden set rather than being tuned away.

### CI gate

The `support-rag-ci` workflow runs the unit tests (with a hashing encoder, so no model download is needed), then **`rag eval-retrieval --gate`** with the real model. The model is cached between runs. The gate fails the PR when a change (model, chunking, corpus edit) drops below `eval/thresholds.yaml`:

- hit@1 ≥ 0.80, hit@3 ≥ 0.95 and MRR ≥ 0.88 overall;
- hit@3 ≥ 0.90 in every language.

The JSON report is uploaded as a build artifact.

## Usage

```bash
uv run rag ingest                                     # index into data/qdrant (embedded)
uv run rag search "Handy im Zug geklaut" --lang de
uv run rag eval-retrieval --gate                      # the CI gate, locally
uv run rag eval-retrieval --dense <model> --dense <model> [--no-sparse]   # compare models
```

Settings use `RAG_*` environment variables (see `config.py`), e.g. `RAG_QDRANT_URL=http://localhost:6333` or `RAG_SPARSE_MODEL=Qdrant/bm25`.
