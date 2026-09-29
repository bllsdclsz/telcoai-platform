# support-rag: multilingual customer-support assistant

A retrieval-augmented support assistant for a Swiss telecom operator, answering in **German, French, Italian and English** from the help center.

> The operator **"Nordalp Mobile"** and all its help articles, prices and URLs (`*.nordalp.example`) are **fictional**, written for this project. They are not affiliated with any real company.

Status: **retrieval, cited answers from a local LLM, guardrails, and evaluation gates for retrieval and safety**. Next: audit logging/tracing and answer-quality evaluation with an LLM judge. See [the roadmap](../docs/roadmap.md).

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

## Answer generation

```
question + lang ──▶ retrieve top-4 ──▶ best score < 0.78? ──yes──▶ localized fallback (no LLM call)
                                              │ no
                                              ▼
                   numbered sources [1]..[n] + prompts/answer/v1.yaml ──▶ LLM (LiteLLM → Ollama)
                                              │
                    "NO_ANSWER" ──▶ localized fallback      answer ──▶ keep only the sources it cites [n]
```

- **Provider-agnostic** (`llm.py`): one `ChatModel` interface. LiteLLM routes a model string to the provider, so switching is configuration only. The default is **local**: `ollama_chat/granite4.2:8b` (IBM Granite 4.2, needs Ollama ≥ 0.34). `RAG_LLM_MODEL=ollama_chat/qwen2.5` or a hosted model such as `azure/<deployment>` work the same way.
- **Thinking off** (`reasoning_effort: none`, which LiteLLM maps per provider): answering from given sources doesn't need reasoning. With thinking on, Granite spent the whole 400-token budget reasoning and returned _no_ answer. That failure now has its own `generation_failed` reason instead of looking like the model declining. Reasoning that still leaks is stripped, including a dangling `</think>`.
- **Versioned prompts** (`prompts/answer/v1.yaml`, `prompts.py`): prompts live in git and are reviewed like code. Every answer records `answer@v1` and the prompt file's SHA-256, so any response can be traced to the exact prompt text.
- **Grounded, cited answers** (`assistant.py`): sources are numbered once per article, the model must cite `[n]`, and the API returns only the sources the answer actually cites. The prompt tells the model to answer in the customer's language, never invent prices or links, and ignore instructions hidden in the question.
- **Two scope checks:**
  1. **Retrieval-score filter (0.78):** calibrated on the golden set (0/94 in-scope refused) against `eval/out_of_scope.yaml` (24 off-topic questions); it stops 12 of them without an LLM call. The score ranges overlap ("How much does the new iPhone cost at Nordalp?" scores 0.84), so a threshold alone can't separate them.
  2. **Model's `NO_ANSWER`:** the model declines when the sources don't answer the question. Both paths return the same localized message pointing to the app chat and hotline.
- **API** (`api.py`): `POST /ask {question, lang}` returns `{answer, answered, reason, sources[{n,title,url}], prompt, model, latency_ms, request_id}`. `GET /health` is also available.

Checked end to end on the real index and a local model:

| Question                                                    | Result                                                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| DE: stolen phone, what to do and what does a SIM cost?      | Block via app or 0800 700 700, replacement SIM CHF 40, 2 working days [1]           |
| FR: data in Italy with Swiss+?                              | 5 GB per month [1]                                                                  |
| IT: cost of the second reminder?                            | CHF 30 [1]                                                                          |
| DE: how long does porting to Nordalp take?                  | 3 to 5 working days [1]                                                             |
| EN: pizza in Lausanne / "Ignore all previous instructions…" | Fallback from the score filter, no LLM call                                         |
| IT: student offers?                                         | Passes the score filter; the model answers `NO_ANSWER`, so the fallback is returned |

### Choosing the local model

The same 16 cases were run on each model: 12 in-scope questions (3 per language), each with the fact the answer must contain, plus 4 questions to decline. Hardware: RTX 3060 Laptop GPU with 6 GB VRAM.

| Model                        | Facts | Cited | Declines | Answer style                                    | Median latency | GPU fit                 |
| ---------------------------- | ----- | ----- | -------- | ----------------------------------------------- | -------------- | ----------------------- |
| **Granite 4.2 8B** (default) | 12/12 | 12/12 | 4/4      | clean, slightly more complete                   | 5.2 s          | partial (4.3 of 6.2 GB) |
| Granite 4.2 3B               | 12/12 | 12/12 | 4/4      | often echoes instructions or repeats the answer | 0.7 s          | full (2.7 GB)           |
| qwen2.5 7B (2024)            | 12/12 | 12/12 | 4/4      | clean                                           | 1.7 s          | full                    |

Granite 4.2 8B is the default: it's the newest model and its answers are clean. The 3B model's echoes would reach customers, and a fact check alone doesn't catch them. That gap is what the generation evaluation (next stage) adds: an LLM judge for faithfulness and style. The judge will be a different model family (qwen2.5), so no model grades its own answers. qwen2.5 is the faster fallback when latency matters.

## Guardrails

```
question ──▶ PII redaction ──▶ injection rules ──▶ injection classifier ──▶ retrieval + scope ──▶ LLM ──▶ output guards ──▶ answer
               (never sent to        (patterns,            (e5 embedding +                             (prompt-leak canary,
                the model or logs)    DE/FR/IT/EN)          logistic regression)                        citation + number grounding)
```

- **PII redaction** (`guardrails.py`): Swiss phone numbers, e-mails, IBANs, payment cards (Luhn-checked) and AHV/AVS numbers are replaced by `[TYPE]` placeholders before retrieval and the model call. Nordalp's own numbers (hotline, 444) and non-card digit runs are kept. Pattern-based by design: these identifiers have strict formats. Names are not detected (that would need an NER model per language).
- **Prompt injection, layer 1: rules.** Normalized, accent-insensitive patterns in four languages. They are fast and precise, but see below.
- **Prompt injection, layer 2: learned classifier** (`injection_model.py`, `models/injection_classifier.json`). Logistic regression on the same e5 embedding the retriever computes, so it adds no extra model call.
  - **Training data:** [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (Apache-2.0, EN/DE) plus the in-house attack set, with the golden support questions as in-domain benign examples.
  - **Reproducible:** `rag train-injection` rebuilds it, and the 14 KB artifact is versioned in git with its training metadata. Serving only needs a dot product.
- **Output guards:**
  - **Prompt-leak canary:** a per-process token sits in the system prompt (`answer@v2`); an answer that contains it is blocked.
  - **Grounding:** every answer must cite a source, and every number in it must appear in the sources it cites. An invented price becomes `ungrounded` and the fallback is returned.

### What the safety evaluation showed

`rag eval-safety` scores the input guardrails on `eval/safety.yaml`:

| Detector                     | Tuning attacks (36) | **Held-out attacks (16)** | False alarms, held-out benign (38) | False alarms, real questions (94) |
| ---------------------------- | ------------------- | ------------------------- | ---------------------------------- | --------------------------------- |
| Rules only                   | 100%                | **0%**                    | 0%                                 | 0%                                |
| Rules + classifier (shipped) | 100%                | **93.8%**                 | 7.9%                               | 0%                                |

- **The rules overfit completely.** After adding pattern classes they caught 100% of the attacks they were tuned on, yet **none** of 16 attacks written afterwards with new phrasings. Without the held-out set, that 100% would have been reported as the result.
- **The classifier generalizes, even across languages.** It catches French and Italian attacks although its training data has no French or Italian. On deepset's test split it reaches 95% recall at 100% precision.
- **Its false alarms were a domain problem.** Trained on generic benign text only, it blocked 7.6% of real support questions ("Show me how to switch to an eSIM"). Adding the golden questions as in-domain benign examples brought that to 0% on real questions, estimated with 5-fold cross-validation during development. The 3 remaining false alarms come from a benign set written to _look_ like attacks, and one of them ("Write me a Python function…") is off-topic anyway.
- **End to end with Granite 8B,** before the classifier, 10 of the 16 held-out attacks were declined safely by the scope filter or the model's NO_ANSWER. The other 6 got an "answer": nothing harmful, but some described the assistant's rules. With the classifier, **15 of 16 are blocked at input and the last is stopped by the scope filter, so none reach the model.**
- **No regression:** the 16 answer cases still pass 16/16 with all guardrails on.

The CI gate (`eval-gates` job) fails when held-out recall drops below 85%, when false alarms on real questions exceed 2%, when false alarms on the held-out benign set exceed 10%, or when PII recall or precision falls below 95%. Running it with the rules alone fails the gate.

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
uv run rag ask "Handy im Zug geklaut, was nun?" --lang de   # needs Ollama running with the model
uv run rag serve                                      # API on http://127.0.0.1:8100/docs
uv run rag eval-retrieval --gate                      # the CI gate, locally
uv run rag eval-safety --gate                         # safety gate (add --rules-only to compare)
uv run rag train-injection                            # retrain the injection classifier artifact
uv run rag eval-retrieval --dense <model> --dense <model> [--no-sparse]   # compare models
```

Settings use `RAG_*` environment variables (see `config.py`), e.g. `RAG_QDRANT_URL=http://localhost:6333` or `RAG_SPARSE_MODEL=Qdrant/bm25`.
