CREATE EXTENSION IF NOT EXISTS vector;

-- ── Main table ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.career_documents (
    -- IDs
    chunk_id                TEXT PRIMARY KEY,
    document_id             TEXT,               -- sha256(filename); groups chunks by source doc
    chunk_index             INT NOT NULL,

    -- Source identity
    filename                TEXT NOT NULL,
    source_type             TEXT,               -- file extension, e.g. '.pdf'

    -- Structure
    title                   TEXT,               -- document title (auto-extracted, low-trust)
    section                 TEXT,               -- top-level heading
    subsection              TEXT,               -- most specific heading (used in embed text)
    page                    INT,

    -- Content
    content                 TEXT NOT NULL,
    word_count              INT,
    embedding               VECTOR(1536) NOT NULL,

    -- Provenance (auto-generated)
    created_date            TIMESTAMPTZ,        -- file mtime at ingest time
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    language                TEXT DEFAULT 'unknown',
    is_ocr                  BOOLEAN DEFAULT FALSE,

    -- Optional / low-trust (auto-extracted where possible; may be NULL)
    author                  TEXT,

    -- Defaults set by pipeline; can be overridden manually post-ingest
    security_classification TEXT DEFAULT 'public',
    version                 TEXT
);

-- ── Vector similarity index ────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS career_documents_embedding_idx
ON public.career_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ── Retrieval RPC ──────────────────────────────────────────────────────────────
DROP FUNCTION IF EXISTS public.match_documents(vector, integer);

CREATE OR REPLACE FUNCTION public.match_documents(
    query_embedding VECTOR(1536),
    match_count     INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id    TEXT,
    filename    TEXT,
    source_type TEXT,
    title       TEXT,
    section     TEXT,
    subsection  TEXT,
    page        INT,
    content     TEXT,
    similarity  FLOAT
)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
    SELECT
        cd.chunk_id,
        cd.filename,
        cd.source_type,
        cd.title,
        COALESCE(cd.section, 'N/A')    AS section,
        COALESCE(cd.subsection, 'N/A') AS subsection,
        cd.page,
        cd.content,
        1 - (cd.embedding <=> query_embedding) AS similarity
    FROM public.career_documents AS cd
    ORDER BY cd.embedding <=> query_embedding
    LIMIT match_count;
$$;


-- ── Migration: apply to an existing installation ───────────────────────────────
-- Run these ALTER statements if the table already exists from a previous schema.
-- Safe to run multiple times (IF NOT EXISTS / IF EXISTS guards where possible).
--
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS document_id             TEXT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS source_type             TEXT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS title                   TEXT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS subsection              TEXT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS word_count              INT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS created_date            TIMESTAMPTZ;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS language                TEXT DEFAULT 'unknown';
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS is_ocr                  BOOLEAN DEFAULT FALSE;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS author                  TEXT;
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS security_classification TEXT DEFAULT 'public';
-- ALTER TABLE public.career_documents ADD COLUMN IF NOT EXISTS version                 TEXT;
