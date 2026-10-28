# BOTC Assistant

> **Status: in progress.** Interfaces, data formats and results may change at any time.

A rules assistant for [Blood on the Clocktower](https://bloodontheclocktower.com/) (BOTC), a social deduction game
with hundreds of characters and many rule interactions. The goal is to answer players' and Storytellers' rules
questions ("Can the Poisoner poison the Drunk?") with answers grounded in the official wiki and citing their sources.

This is a personal learning project covering the whole chain: building a corpus, retrieval, evaluation, generation
with a small open model, fine-tuning, and serving.

## Overview

- **Corpus** (`src/botc/corpus/`): a rate-limited MediaWiki client downloads every article of the official wiki,
  then the pages are cleaned to Markdown and cut into structural chunks (never across a heading), each with a
  contextual header (page, character type, edition, section).
- **Retrieval** (`src/botc/retrieval/`): sentence-transformers embedder (`BAAI/bge-small-en-v1.5` by default),
  exact cosine search over a flat or sharded in-memory vector store, index saved on disk with a manifest.
- **Serving** (`src/botc/serving/`): a multi-tenant FastAPI service with API-key authentication, per-tenant
  token-bucket rate limiting, dynamic request batching, a per-tenant LRU cache and latency metrics.
  It currently runs on a placeholder hash embedder and a small corpus directory, wiring it to the real index is
  part of the roadmap.

## Roadmap

- [x] Wiki corpus: scraping, cleaning, structural chunking
- [x] Dense retrieval and on-disk index
- [x] Serving layer: auth, rate limiting, batching, cache, metrics
- [Current] Entity-guided retrieval: detect the characters and glossary terms named in a question, boost the matching
      pages, fit the context into a token budget (in progress)
- [ ] Retrieval evaluation set and tuning
- [ ] Answer generation with a small open model (Qwen3)
- [ ] QLoRA fine-tuning on retrieval-augmented examples (RAFT)
- [ ] End-to-end serving of the full pipeline


## Layout

```
src/botc/
  corpus/      scraping, cleaning, chunking, index building
  retrieval/   embedders, vector stores, index, search
  serving/     FastAPI app, auth, rate limiting, batching, cache, metrics
tests/         one folder per package
```

## Disclaimer

This is an unofficial fan project, not affiliated with The Pandemonium Institute. Blood on the Clocktower and the
content of its wiki belong to their respective owners. The wiki text is downloaded locally for retrieval and is not
redistributed in this repository.
