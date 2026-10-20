# Mostro AI

**A trustworthy, prospectus-grounded assistant that helps people in Africa understand IPOs, avoid scams, and learn about investing.**

This repository contains the **AI service** of the Mostro platform: everything that involves language models, embeddings, retrieval, guardrails, and the document ingestion pipeline. The frontend and the other backend microservices live in separate repositories.

> **Status:** in active development (pre-MVP). See [Roadmap](#14-roadmap) for where we are.

---

## Table of contents

1. [Why Mostro exists](#1-why-mostro-exists)
2. [Objectives and non-goals](#2-objectives-and-non-goals)
3. [Target population](#3-target-population)
4. [Product overview](#4-product-overview)
5. [Core design principles](#5-core-design-principles)
6. [Architecture](#6-architecture)
7. [Data model](#7-data-model)
8. [Tech stack](#8-tech-stack)
9. [Repository structure](#9-repository-structure)
10. [Getting started](#10-getting-started)
11. [Useful commands](#11-useful-commands)
12. [Configuration](#12-configuration)
13. [Testing and evaluation](#13-testing-and-evaluation)
14. [Roadmap](#14-roadmap)
15. [Conventions and contributing](#15-conventions-and-contributing)
16. [Security, privacy and compliance](#16-security-privacy-and-compliance)
17. [Open questions](#17-open-questions)
18. [Glossary](#18-glossary)

---

## 1. Why Mostro exists

Large IPOs, such as the recent Dangote refinery IPO, bring many first-time investors into capital markets. Alongside genuine interest come three problems:

- **Scams.** Fake offers, fake "agents", WhatsApp investment groups, and requests to pay into personal accounts.
- **Hard-to-find, hard-to-read information.** The official document is a prospectus of hundreds of pages written in legal and financial language.
- **Skepticism.** Many people distrust anything related to investing, often for good reason.

Mostro answers these with one idea: **let people ask questions directly to the official prospectus, and show them exactly where each answer comes from.** A user can verify any statement by opening the cited page. That verifiability is the product.

## 2. Objectives and non-goals

### Objectives

1. **Accurate answers about a specific IPO**: dates, price, minimum subscription, how to apply, receiving agents and banks, use of proceeds, risks, and so on, always with citations (page, section, short quote).
2. **Financial education**: explain concepts (IPO, allotment, oversubscription, dilution, prospectus) in plain language, adapted to the user's level and language.
3. **Scam protection**: help users check whether an account, agent, or procedure matches what the official prospectus says.
4. **Calendar actions**: turn verified IPO dates (opening, closing, allotment, listing) into calendar events the user confirms.
5. **Honesty over fluency**: when the document doesn't contain the answer, say so.

### Non-goals

- **No personalized investment advice.** We do not tell anyone to buy, sell, or how much to invest.
- **No predictions or guaranteed returns**, ever.
- **No live market data** in v1 (subscription levels, allotment results, listing prices). The prospectus is a snapshot, and we say so.
- **No executing actions** on a user's behalf. The AI service only *proposes*; other services execute after user confirmation.
- **Not a general chatbot.** Off-topic requests are politely redirected.

## 3. Target population

Primary: **first-time and early-stage retail investors across Africa**, including people who are skeptical of investing or have been targeted by scams.

What this implies for the engineering:

| Reality | Design consequence |
|---|---|
| Many users are new to finance | Plain language, adjustable level, glossary for general concepts |
| Anglophone and Francophone markets | English and French from day one; more languages later |
| Mobile-first, variable bandwidth | Streaming responses, short answers by default |
| High scam exposure | Scam-aware behavior, official-source comparison, careful wording |
| Low institutional trust | Every claim cites a page the user can open |
| Different regulators per country | Information and education only; never advice |

## 4. Product overview

The user logs in, picks **one IPO** from the published list, and chats with the assistant. The conversation is **bound to that IPO**: the assistant only knows that IPO's official documents.

What the assistant can do:

- **Answer questions from the prospectus** with page citations.
- **Explain** terms and sections in simple language, in the user's language.
- **Summarize** broad sections ("what are the main risks?").
- **Check legitimacy details**: "Is this bank account one of the official receiving agents?"
- **Propose calendar events** for key dates, which the user confirms in the app.
- **Decline or redirect** advice-seeking, off-topic, and manipulative requests.
- **Abstain** when the document doesn't answer the question.

## 5. Core design principles

1. **The prospectus is the source of truth.** The LLM explains; it does not invent. Facts come from retrieved passages or human-verified structured data.
2. **The IPO is never chosen by the model.** `ipo_id` comes from the session and is applied as a hard database filter. This prevents cross-IPO leakage and prompt-injection context switching.
3. **Abstain rather than guess.** If retrieval confidence is low, the answer is "I couldn't find this in the prospectus", plus where to look.
4. **Deterministic checks on top of the LLM.** Every number and date in an answer must appear in the retrieved sources.
5. **Propose, never execute.** Calendar actions are returned as proposals. Credentials never enter the AI service.
6. **Human in the loop for facts that matter.** Key facts (dates, prices, agents) are extracted by LLM and verified by a person before use.
7. **Everything is auditable.** Each answer stores the query rewrite, retrieved chunks and scores, model, prompt version, and latency.
8. **Quality is measured, not felt.** Evals run on every prompt, model, or chunking change.

## 6. Architecture

### Where the AI service sits

```
Frontend ──► Backend gateway ──► AI service (this repo)
                  │                    │
                  │                    ├─► PostgreSQL + pgvector
                  │                    ├─► Voyage API (embeddings, all environments)
                  │                    └─► DeepSeek (LLM)
                  └─► Calendar execution (user OAuth), users, auth
```

The backend authenticates users and calls the AI service with a shared internal key (`X-Internal-Key`). The AI service never sees user credentials or calendar tokens.

### Chat flow (LangGraph)

The chat is a **fixed graph with conditional edges**, not a free-form tool-calling agent. This is easier to test, debug, and audit.

```
input_guard → understand → route ─┬→ retrieve → abstain gate ─────────┐
                                  ├→ key_facts ────────────────────────┤
                                  ├→ section_summary ──────────────────┼→ generate → output_guard → respond
                                  ├→ calendar_proposal ────────────────┤
                                  └→ refuse / out_of_scope ────────────┘
```

| Step | What happens |
|---|---|
| `input_guard` | Detects prompt injection, advice-seeking, off-topic, and PII (cheap model + rules) |
| `understand` | Detects language, rewrites the question using history, classifies intent |
| `retrieve` | Hybrid search (vector + full-text, fused with RRF), filtered by `ipo_id`, then same-document neighbors expanded; no reranker is enabled yet |
| abstain gate | If the best vector cosine similarity is below a threshold, skip generation and abstain; RRF is used for ranking, not confidence |
| `key_facts` / `section_summary` | Read verified structured facts or precomputed summaries for factual and broad questions |
| `calendar_proposal` | Builds a proposed event from verified key facts, with an explicit IANA time zone |
| `generate` | Answers only from provided passages, cites pages, adapts tone and language |
| `output_guard` | Checks numbers and dates against sources, citations, and blocks advice and guaranteed-return language |
| `respond` | Streams the answer and structured citations; stores the audit trace |

### Ingestion pipeline (offline)

Runs as a separate process, not inside the chat API.

```
Upload PDF → parse (layout-aware) → section-aware chunking → embed → store in Postgres
                     └→ key-facts extraction → human review → publish
                     └→ section summaries
```

Key choices: chunk along headings (about 400 to 800 tokens), prepend a context header such as `Issuer > Risk Factors > Regulatory risks` before embedding, make tables their own chunks, keep page numbers on everything, and treat addenda as new document versions.

### Model roles

| Role | Model (configurable) | Used for |
|---|---|---|
| Fast LLM | `deepseek-v4-flash` | Guardrails, routing, query rewriting |
| Main LLM | `deepseek-v4-pro` | Answer generation, key-fact extraction, summaries |
| Embeddings | `voyage-multimodal-3.5` (1024 dims) | Chunks, glossary, queries |

> DeepSeek renamed its models (the older `deepseek-chat` and `deepseek-reasoner` names were scheduled for deprecation). Check the [official docs](https://api-docs.deepseek.com/) before changing model names. If you change the embedding model, **every vector must be re-embedded** and the `VECTOR(n)` column size updated.

## 7. Data model

The SQLAlchemy models define the current schema; Alembic's initial migration
creates the baseline and subsequent revisions capture changes. The older
`ai_service_schema.sql` is retained as a reference and is not the provisioning
path. Main tables:

| Group | Tables | Notes |
|---|---|---|
| Reference | `ipo` | Synced copy; the backend owns the master record |
| Knowledge | `document`, `chunk`, `key_fact`, `section_summary`, `glossary_term` | Written by ingestion; read-only for the chat API |
| Conversation | `chat_session`, `message`, `message_citation`, `calendar_proposal` | Written at runtime; doubles as the audit log |
| Operations | `ingestion_job` | Tracks document-ingestion work for admin APIs |

Important details:

- `chunk.ipo_id` is denormalized on purpose, so retrieval filters on it directly.
- `document.version` and `status` handle addenda; only `indexed` documents are searchable.
- `key_fact.review_status = 'verified'` is required before a fact is used or turned into a calendar proposal.
- `chat_session.user_id` is an opaque UUID from the backend, with no cross-service foreign key.
- `glossary_term` holds general education content, kept separate from prospectus text.

## 8. Tech stack

| Concern | Choice |
|---|---|
| Language / tooling | Python 3.12, [uv](https://docs.astral.sh/uv/), ruff, mypy, pytest |
| API | FastAPI, Uvicorn, Server-Sent Events (`sse-starlette`) |
| Agent orchestration | LangGraph (+ `langgraph-checkpoint-postgres`) |
| LLM | DeepSeek via `langchain-deepseek` |
| Embeddings | Voyage API (`voyage-multimodal-3.5`, 1024 dims, all environments) |
| Database | PostgreSQL 17 + pgvector (HNSW) + full-text search |
| DB access | SQLAlchemy 2 (async) + psycopg 3, Alembic migrations |
| Document parsing | Docling, PyMuPDF (optional `ingest` dependency group) |
| Resilience / logging | tenacity, structlog |
| Observability | Langfuse (optional) |
| Packaging | Docker (multi-stage: `api` and `ingest` targets) |

## 9. Repository structure

```
ai-service/
├── pyproject.toml, uv.lock
├── .env.example, docker-compose.yml, Dockerfile, .dockerignore
├── ai_service_schema.sql        # legacy/reference SQL; use Alembic to provision
├── alembic.ini
├── migrations/                  # Alembic migrations
├── scripts/                     # smoke_check.py, seed and eval runners
│
├── src/ai/
│   ├── main.py                  # app factory, lifespan (DB pool, graph compile)
│   ├── config.py                # pydantic-settings, the only place reading env
│   │
│   ├── api/                     # HTTP layer only, no business logic
│   │   ├── deps.py              # internal-key auth, DB session, graph
│   │   ├── routes/              # chat (SSE), sessions, ingestion, health
│   │   └── schemas/             # request/response models: the contract with the backend
│   │
│   ├── services/                # use cases
│   │   ├── chat_service.py      # load session, run graph, persist message + citations
│   │   └── calendar_service.py  # build proposals from verified key facts
│   │
│   ├── agent/                   # LangGraph
│   │   ├── state.py             # typed graph state
│   │   ├── graph.py             # nodes and edges
│   │   ├── nodes/               # input_guard, understand, retrieve, key_facts,
│   │   │                        # summary, calendar, generate, output_guard, abstain
│   │   └── prompts/             # versioned prompt templates
│   │
│   ├── retrieval/               # hybrid SQL, RRF, rerank, neighbor expansion, abstention gate
│   ├── guardrails/              # injection, advice, PII, numeric/date check, citation check
│   ├── llm/                     # DeepSeek chat and embedding providers (all provider code)
│   ├── db/                      # session, SQLAlchemy models, repositories
│   ├── ingestion/               # parse, chunk, embed, extract_facts, summarize, pipeline
│   ├── domain/                  # enums, citation and fact types
│   └── core/                    # logging, errors, telemetry
│
├── tests/
│   ├── unit/                    # guardrails, chunking, numeric check, RRF
│   ├── integration/             # real Postgres, mocked LLM
│   └── e2e/                     # full chat flow on a seeded IPO
│
└── evals/
    ├── datasets/                # golden Q&A per IPO (JSONL), trap sets
    └── metrics/                 # faithfulness, citation, abstention
```

### Layering rules

- **Dependency direction:** `api → services → agent → retrieval / guardrails → db / llm`. Nothing imports upward.
- **`llm/` is the only package that knows about model providers.** Swapping a provider touches one folder.
- **`retrieval/` is the only place that writes the filtered search SQL.** It takes `ipo_id` from the caller, never from a prompt or model output.
- **Ingestion is a separate process** with heavier dependencies and different failure modes.
- **Prompts are files, not strings in code.** The prompt version is logged with every message.

## 10. Getting started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- A [DeepSeek API key](https://platform.deepseek.com/) (LLM)
- A [Voyage API key](https://dashboard.voyageai.com/) (embeddings — used in all environments including dev)

### First-time setup

```bash
# 1. Install dependencies (add --extra ingest if you will run ingestion locally)
uv sync

# 2. Configure the environment
cp .env.example .env          # then fill in the keys required for your environment

# 3. Start Postgres with pgvector
docker compose up -d db

# 4. Apply the baseline database migration
uv run --env-file .env alembic upgrade head

# 5. Check that everything is connected
uv run --env-file .env python scripts/smoke_check.py

# 6. Run the API with hot reload
uv run --env-file .env uvicorn ai.main:app --reload
```

The smoke check verifies three things: the database is reachable with pgvector installed, the embedding dimension matches the schema, and DeepSeek answers. **If it fails, fix that before writing any feature code.**

If you are starting the project from scratch (no `pyproject.toml` yet), the dependency setup is:

```bash
uv init --package ai-service && cd ai-service
uv add fastapi "uvicorn[standard]" pydantic-settings sse-starlette structlog tenacity orjson tiktoken
uv add langgraph langgraph-checkpoint-postgres langchain-core langchain-openai langchain-deepseek
uv add "sqlalchemy[asyncio]" "psycopg[binary,pool]" pgvector alembic
uv add --group dev pytest pytest-asyncio httpx respx ruff mypy pre-commit "langgraph-cli[inmem]"
uv add --optional ingest docling pymupdf
```

## 11. Useful commands

### Run and debug

```bash
uv run --env-file .env uvicorn ai.main:app --reload             # API with hot reload
uv run --env-file .env langgraph dev                            # visual graph debugger (needs a langgraph.json)
curl http://localhost:8000/health                               # liveness check
```

The initial chat integration stub is `POST /v1/sessions/{session_id}/messages`.
It requires `X-Internal-Key` and `X-User-Id`, accepts the request shape in
[`api-map.md`](./api-map.md), and streams `token` followed by `done` events by
default. Set `"stream": false` to receive the same `ChatMessage` as JSON. The
stub abstains because it is not connected to prospectus retrieval yet.

### Database

```bash
docker compose up -d db                                          # start Postgres + pgvector
docker compose down                                              # stop (keeps data)
docker compose down -v                                           # stop and delete all data
docker compose exec db psql -U ai -d ai_service                  # open a SQL shell
uv run alembic revision --autogenerate -m "describe change"      # create a migration
uv run alembic upgrade head                                      # apply migrations
uv run alembic downgrade -1                                      # roll back one migration
```

### Ingestion (single-document v1)

```bash
uv sync --extra ingest
uv run --env-file .env python -m ai.ingestion ingest <ipo_id> <prospectus.pdf>  # register and index
uv run --env-file .env python -m ai.ingestion run <document_id>       # re-index an existing document
uv run --env-file .env python -m ai.ingestion inspect <document_id>  # inspect stored chunks
```

The IPO row must already exist in `ai.ipo`. The `ingest` command registers the local PDF as a document, calculates its SHA-256 file hash, and reuses an existing document record for duplicate uploads. The parser preserves page numbers, recognizes heading paths and lists from PDF layout, and emits detected tables as complete Markdown table chunks. Embedding inputs add the section path and nearby same-section text while leaving the source `content` unchanged. Ingestion records progress in `ai.ingestion_job` and replaces a document's chunks only after parsing and embedding succeed.

Always inspect the stored chunks by hand after ingesting a new prospectus: check page numbers, table integrity, and section paths.

### Quality

```bash
uv run ruff check .                             # lint
uv run ruff format --check .                    # format check
uv run mypy src                                  # type check
uv run pytest tests/unit                         # fast tests
uv run pytest tests/integration                  # needs the database running
uv run --env-file .env python scripts/run_eval.py --ipo-id <ipo-uuid>  # retrieval recall@k; embeddings only, no LLM
```

The retrieval dataset in `evals/datasets/dangote_prospectus.jsonl` contains
25 questions with known source pages from the first Dangote refinery
prospectus. Page numbers are 1-based PDF page positions, matching ingestion's
page metadata. Index that prospectus and run the recall evaluation before
advancing to the next phase; if recall is poor, revisit chunking rather than
adding a reranker. The script uses the configured embedding provider and
Postgres retrieval, but does not create a chat client or invoke an LLM.

### Docker images

```bash
docker build -t mostro-ai .                                   # API image (slim)
docker build --target ingest -t mostro-ai-ingest .            # ingestion image (adds docling, pymupdf)
docker run --env-file .env -p 8000:8000 mostro-ai             # run the API container
```

Migrations are **not** run at container startup. Run `alembic upgrade head` as a separate one-off job before deploying, so several replicas never race to migrate.

## 12. Configuration

All configuration comes from environment variables, read in one place (`config.py`). Copy `.env.example` to `.env` for local work. Never commit `.env`. The embedding provider is Voyage (`voyage-multimodal-3.5`, 1536 dims) in **all** environments — dev, test, staging, and production. Runtime providers are available through `ai.llm`; transient failures use bounded retries and successful/failed calls log provider, model, latency, and available token counts without logging input or output text.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection (`postgresql+psycopg://...`) |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | LLM access |
| `LLM_MODEL_FAST`, `LLM_MODEL_MAIN` | Models for guardrails/routing vs generation |
| `VOYAGE_API_KEY` | Voyage embeddings API key (required in all environments) |
| `LLM_TIMEOUT_SECONDS`, `EMBEDDING_TIMEOUT_SECONDS` | Provider request timeouts (default: 30 seconds) |
| `PROVIDER_MAX_ATTEMPTS` | Maximum attempts for transient provider failures (default: 3) |
| `EMBEDDING_DIM` | Must match the `VECTOR(n)` column (currently 1536) |
| `INTERNAL_API_KEY` | Shared secret the backend sends in `X-Internal-Key` |
| `ADMIN_API_KEY` | Separate key reserved for admin endpoints |
| `CORS_ORIGINS` | JSON list of allowed local development browser origins; quote the whole value in `.env` (for example, `CORS_ORIGINS='["http://localhost:3000"]'`) |
| `RETRIEVAL_CANDIDATES` | Maximum candidate count retrieved independently by vector and full-text search |
| `ABSTAIN_SCORE_THRESHOLD` | Below this best vector cosine similarity, the assistant abstains |
| `LANGFUSE_*` | Optional tracing |

## 13. Testing and evaluation

Language-model systems fail silently, so we measure instead of trusting. We use three layers:

1. **Unit tests**: guardrails, chunking, the numeric/date checker, rank fusion. Fast, no network.
2. **Integration tests**: a real Postgres with pgvector and *mocked* LLM calls (`respx`), so tests never spend money.
3. **Evals**: a golden set per IPO (questions with known answers and source pages), plus trap sets.

Metrics we track:

- **Retrieval:** recall@k and MRR against known source pages.
- **Answers:** faithfulness (every claim supported), citation correctness, abstention correctness.
- **Behavior:** refusal quality on trap sets, tool/route correctness.

Trap sets include: advice-seeking ("should I put all my savings in?"), scam messages, prompt injection, questions about a different IPO, French and code-switched queries, and calendar edge cases (time zones, past dates, "next Friday").

**Rule: run the evals after every change to a prompt, model, or chunking setting.**

## 14. Roadmap

Build order follows risk: data quality first, agent later.

- [ ] **Phase 0: Foundation.** Config, DB session, migrations, `/health`, CI, and the frozen API contract with a stub streaming endpoint so the frontend can integrate in parallel.
- [x] **Phase 1: `llm/` layer.** DeepSeek and environment-specific embedding providers behind small interfaces, with retries, timeouts, and logging; fakes for tests.
- [ ] **Phase 2: Ingestion v1.** Parse, chunk, embed, and store one real prospectus. Verify chunks by hand.
- [ ] **Phase 3: Retrieval.** Hybrid search, RRF, neighbor expansion, abstention gate. Start the retrieval evals (recall@k).
- [ ] **Phase 4: Minimal graph.** `understand → retrieve → generate → respond`, persistence, citations, answer evals. *First end-to-end slice.*
- [ ] **Phase 5: Output guardrails.** Numeric/date check, citation check, no-advice filter.
- [ ] **Phase 6: Input guardrails.** Injection, advice-seeking, off-topic, PII, with trap evals.
- [ ] **Phase 7: Key facts and calendar.** Extraction, human review flow, verified-facts-only calendar proposals.
- [ ] **Phase 8: Section summaries** for broad questions.
- [ ] **Phase 9: Scam and legitimacy checks** against official agents, banks, and registrars.
- [ ] **Phase 10: Hardening.** French and mixed-language evals, addenda/versioning, tracing, caching, rate limiting, load tests, production deployment.

**Minimum shippable product: phases 0 to 5.** That gives grounded, cited, abstaining Q&A on one IPO with output checks.

Beyond v1: more languages (Swahili, Hausa, Yoruba, Pidgin, Arabic), more countries and exchanges, WhatsApp/SMS channels, an "announcements" source for live information, multi-IPO comparison.

## 15. Conventions and contributing

- **Branches and commits:** short-lived feature branches, small pull requests, clear commit messages. <!-- TODO: confirm team convention (e.g. Conventional Commits) -->
- **Code style:** ruff for lint and format, mypy for types. CI must pass before merge.
- **Type everything** that crosses a boundary (API schemas, graph state, repository returns).
- **No provider code outside `llm/`.** No raw SQL for retrieval outside `retrieval/`.
- **Prompts live in `agent/prompts/` and are versioned.** Changing a prompt means running the evals.
- **Never accept `ipo_id` from model output.** It comes from the session.
- **Schema changes go through Alembic**, never by editing the database by hand.
- **Write tests with the feature**, not after.

## 16. Security, privacy and compliance

- **Information, not advice.** The product educates and informs. It must never give personalized investment recommendations or promise returns. Disclaimers and wording must be reviewed by a lawyer for each target country, since rules differ by regulator (for example Nigeria's SEC, Kenya's CMA).
- **Service authentication.** The AI service is internal. Every request must carry the internal key; it is not exposed directly to the internet.
- **Isolation.** Retrieval always filters by the session's `ipo_id`. Chat sessions are bound to one IPO for their lifetime.
- **No credentials in this service.** Calendar tokens and user credentials stay in the backend.
- **Prompt injection.** Prospectus text and user messages are treated as untrusted data. The model never decides which IPO to read and never executes actions.
- **Personal data.** User messages may contain personal details. The PII check runs *before* anything is sent to an external LLM provider. Note that the DeepSeek API is hosted outside most users' regions; prospectus text is public, user messages are not. Review data-handling obligations per country.
- **Secrets.** Provided via environment variables or a secret manager. Never committed, never baked into images.
- **Auditability.** Every answer stores its sources, model, and prompt version, so we can explain what the assistant said and why.
- **Wording on scams.** The assistant says things like "this account is not among the receiving agents listed on page X" and always directs users to confirm through official channels. It never declares with certainty that something is or isn't a scam.

## 17. Open questions

These need decisions from the wider team:

1. Which countries and exchanges are in scope for v1? (Drives languages and prospectus sources.)
2. Who curates and verifies key facts, and who uploads prospectuses? (A human reviewer is required.)
3. How should the assistant handle live information (subscription levels, allotment results, listing prices) in v1: "not covered", or a small announcements source?
4. Which channels beyond the app are planned (WhatsApp, SMS)? This affects response format and length.
5. What is the legal position on disclaimers and information licensing in each target country?
6. Where will this service be hosted, and does the Postgres provider support pgvector 0.8+ (for filtered HNSW search)?

## 18. Glossary

| Term | Meaning |
|---|---|
| **IPO** | Initial Public Offering: a company sells shares to the public for the first time |
| **Prospectus** | The official legal document describing the offer, the company, the risks, and how to subscribe |
| **Addendum / supplement** | An official amendment to a prospectus |
| **Allotment** | The process of deciding how many shares each applicant receives |
| **Oversubscription** | More shares requested than offered |
| **Receiving agent** | An authorized institution that collects applications and payments |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant passages, then have the model answer from them |
| **Chunk** | A small piece of the prospectus (text or table) stored with its embedding and page number |
| **Embedding** | A numeric vector representing the meaning of a text, used for similarity search |
| **Hybrid search** | Combining vector similarity with keyword (full-text) search |
| **RRF** | Reciprocal Rank Fusion: a simple way to merge two ranked result lists |
| **Reranker** | A model that re-orders retrieved chunks by relevance to the question |
| **Abstention** | Declining to answer because the sources don't support one |
| **Guardrails** | Checks before and after the model that block unsafe or unsupported behavior |
| **SSE** | Server-Sent Events: how answers are streamed token by token to the client |
| **Key fact** | A structured, human-verified value (for example the offer closing date) with its source page |
