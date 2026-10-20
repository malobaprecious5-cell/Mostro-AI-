-- AI service schema: Postgres 16+ with pgvector >= 0.8
-- Embedding model: OpenAI text-embedding-3-small (1536 dimensions). Re-embed everything if you change it.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS unaccent;   -- French accents in full-text search

CREATE SCHEMA IF NOT EXISTS ai;
SET search_path = ai, public;

-- Full-text config that ignores accents (useful for French queries on English/French text)
CREATE TEXT SEARCH CONFIGURATION ai.fts_unaccent (COPY = pg_catalog.simple);
ALTER TEXT SEARCH CONFIGURATION ai.fts_unaccent
  ALTER MAPPING FOR hword, hword_part, word WITH unaccent, simple;

-- ---------------------------------------------------------------------------
-- Reference data (the backend owns the source of truth; AI service keeps a synced copy)
-- ---------------------------------------------------------------------------
CREATE TABLE ipo (
  id            UUID PRIMARY KEY,                -- same UUID as the backend's IPO id
  issuer_name   TEXT NOT NULL,
  country       CHAR(2) NOT NULL,                -- ISO 3166-1 alpha-2
  exchange      TEXT,
  status        TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft','review','published','archived')),
  published_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Documents (prospectus + addenda, versioned)
-- ---------------------------------------------------------------------------
CREATE TABLE document (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ipo_id        UUID NOT NULL REFERENCES ipo(id) ON DELETE CASCADE,
  version       INT  NOT NULL DEFAULT 1,
  doc_type      TEXT NOT NULL DEFAULT 'prospectus'
                CHECK (doc_type IN ('prospectus','addendum','supplement')),
  storage_uri   TEXT NOT NULL,                   -- object storage path of the original PDF
  file_hash     TEXT NOT NULL,                   -- sha256, detects duplicate uploads
  page_count    INT,
  parser        TEXT,                            -- e.g. 'docling@2.x', for reproducibility
  status        TEXT NOT NULL DEFAULT 'uploaded'
                CHECK (status IN ('uploaded','parsing','indexed','failed','superseded')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (ipo_id, version),
  UNIQUE (ipo_id, file_hash)
);

-- ---------------------------------------------------------------------------
-- Chunks: the retrieval table
-- ---------------------------------------------------------------------------
CREATE TABLE chunk (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ipo_id        UUID NOT NULL REFERENCES ipo(id) ON DELETE CASCADE,   -- denormalized on purpose: hard filter
  document_id   UUID NOT NULL REFERENCES document(id) ON DELETE CASCADE,
  chunk_index   INT  NOT NULL,                   -- order in document, used for neighbor expansion
  chunk_type    TEXT NOT NULL DEFAULT 'text'
                CHECK (chunk_type IN ('text','table','heading','list', 'image')),
  section_path  TEXT,                            -- 'Risk Factors > Regulatory risks'
  page_start    INT NOT NULL,
  page_end      INT NOT NULL,
  content       TEXT NOT NULL,                   -- what the LLM sees and what we quote
  embed_text    TEXT NOT NULL,                   -- context header + content, what was embedded
  token_count   INT,
  embedding     VECTOR(1024) NOT NULL,
  fts           TSVECTOR GENERATED ALWAYS AS
                (to_tsvector('ai.fts_unaccent', coalesce(section_path,'') || ' ' || content)) STORED,
  UNIQUE (document_id, chunk_index)
);

CREATE INDEX chunk_ipo_idx       ON chunk (ipo_id);
CREATE INDEX chunk_doc_order_idx ON chunk (document_id, chunk_index);
CREATE INDEX chunk_fts_idx       ON chunk USING gin (fts);
CREATE INDEX chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- Verified structured facts (power the calendar and the most common questions)
-- ---------------------------------------------------------------------------
CREATE TABLE key_fact (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ipo_id           UUID NOT NULL REFERENCES ipo(id) ON DELETE CASCADE,
  document_id      UUID NOT NULL REFERENCES document(id),
  fact_key         TEXT NOT NULL,                -- 'offer_open_date', 'price_per_share', 'min_subscription', ...
  value            JSONB NOT NULL,               -- {"date":"2026-03-10"} or {"amount":1000,"currency":"NGN"}
  source_page      INT NOT NULL,
  source_chunk_id  UUID REFERENCES chunk(id) ON DELETE SET NULL,
  source_quote     TEXT,
  extracted_by     TEXT,                         -- model name + prompt version
  review_status    TEXT NOT NULL DEFAULT 'pending'
                   CHECK (review_status IN ('pending','verified','rejected')),
  reviewed_by      TEXT,
  reviewed_at      TIMESTAMPTZ,
  UNIQUE (ipo_id, fact_key, document_id)
);
CREATE INDEX key_fact_ipo_idx ON key_fact (ipo_id) WHERE review_status = 'verified';

-- ---------------------------------------------------------------------------
-- Precomputed section summaries (for broad questions)
-- ---------------------------------------------------------------------------
CREATE TABLE section_summary (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ipo_id        UUID NOT NULL REFERENCES ipo(id) ON DELETE CASCADE,
  document_id   UUID NOT NULL REFERENCES document(id),
  section_key   TEXT NOT NULL,                   -- 'risk_factors', 'business', 'financials', 'offer_terms'
  summary       TEXT NOT NULL,
  page_start    INT,
  page_end      INT,
  review_status TEXT NOT NULL DEFAULT 'pending'
                CHECK (review_status IN ('pending','verified','rejected')),
  UNIQUE (ipo_id, section_key, document_id)
);

-- ---------------------------------------------------------------------------
-- Curated glossary (general education, NOT from prospectuses, kept separate)
-- ---------------------------------------------------------------------------
CREATE TABLE glossary_term (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  term          TEXT NOT NULL,
  language      TEXT NOT NULL,                   -- 'en', 'fr', ...
  level         TEXT NOT NULL DEFAULT 'beginner',
  definition    TEXT NOT NULL,
  example       TEXT,
  embedding     VECTOR(1536),
  UNIQUE (term, language)
);
CREATE INDEX glossary_embedding_idx ON glossary_term USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Conversations (user_id is an opaque UUID from the backend: no cross-service FK)
-- ---------------------------------------------------------------------------
CREATE TABLE chat_session (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL,
  ipo_id        UUID NOT NULL REFERENCES ipo(id),   -- immutable for the session's lifetime
  language      TEXT NOT NULL DEFAULT 'en',
  level         TEXT NOT NULL DEFAULT 'beginner',
  summary       TEXT,                               -- rolling summary of older turns
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_active   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_session_user_idx ON chat_session (user_id, last_active DESC);

CREATE TABLE message (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id    UUID NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
  role          TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content       TEXT NOT NULL,
  outcome       TEXT CHECK (outcome IN ('answered','abstained','refused','blocked','error')),
  trace         JSONB,        -- rewritten query, retrieved chunk ids + scores, model, prompt version, latency, tokens
  feedback      SMALLINT,     -- -1 / 0 / 1 from the user
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX message_session_idx ON message (session_id, created_at);

CREATE TABLE message_citation (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id    UUID NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  chunk_id      UUID NOT NULL REFERENCES chunk(id),
  page          INT NOT NULL,
  quote         TEXT                                -- short verbatim excerpt shown in the UI
);
CREATE INDEX message_citation_msg_idx ON message_citation (message_id);

CREATE TABLE calendar_proposal (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id    UUID NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  key_fact_id   UUID NOT NULL REFERENCES key_fact(id),
  title         TEXT NOT NULL,
  starts_at     TIMESTAMPTZ NOT NULL,
  ends_at       TIMESTAMPTZ,
  timezone      TEXT NOT NULL,                      -- IANA, e.g. 'Africa/Lagos'
  reminders     JSONB NOT NULL DEFAULT '[]',        -- [{"minutes_before":1440}]
  status        TEXT NOT NULL DEFAULT 'proposed'
                CHECK (status IN ('proposed','confirmed','rejected','created','failed'))
);

-- ---------------------------------------------------------------------------
-- Least-privilege roles
-- ---------------------------------------------------------------------------
-- CREATE ROLE ai_runtime LOGIN;      -- chat API: read-only on knowledge tables, write on conversation tables
-- CREATE ROLE ai_ingest  LOGIN;      -- ingestion worker: write on document/chunk/key_fact/section_summary
-- GRANT SELECT ON ipo, document, chunk, key_fact, section_summary, glossary_term TO ai_runtime;
-- GRANT SELECT, INSERT, UPDATE ON chat_session, message, message_citation, calendar_proposal TO ai_runtime;

-- ---------------------------------------------------------------------------
-- Hybrid retrieval (vector + full-text, fused with Reciprocal Rank Fusion)
-- $1 = ipo_id (injected by the service, NEVER by the LLM)
-- $2 = query embedding, $3 = query text, $4 = candidates per branch (e.g. 30)
-- ---------------------------------------------------------------------------
-- SET LOCAL hnsw.ef_search = 100;
-- SET LOCAL hnsw.iterative_scan = relaxed_order;   -- pgvector >= 0.8, keeps filtered results complete
--
-- WITH vec AS (
--   SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> $2) AS rnk
--   FROM chunk c
--   JOIN document d ON d.id = c.document_id AND d.status = 'indexed'
--   WHERE c.ipo_id = $1
--   ORDER BY c.embedding <=> $2
--   LIMIT $4
-- ),
-- fts AS (
--   SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.fts, q) DESC) AS rnk
--   FROM chunk c
--   JOIN document d ON d.id = c.document_id AND d.status = 'indexed',
--        websearch_to_tsquery('ai.fts_unaccent', $3) q
--   WHERE c.ipo_id = $1 AND c.fts @@ q
--   ORDER BY ts_rank_cd(c.fts, q) DESC
--   LIMIT $4
-- )
-- SELECT id,
--        coalesce(1.0 / (60 + vec.rnk), 0) + coalesce(1.0 / (60 + fts.rnk), 0) AS score
-- FROM vec FULL OUTER JOIN fts USING (id)
-- ORDER BY score DESC
-- LIMIT 30;     -- then rerank to 5-8 and expand neighbors via (document_id, chunk_index)
