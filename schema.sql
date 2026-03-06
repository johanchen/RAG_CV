CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.career_documents (
    chunk_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    section TEXT,
    page INT,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS career_documents_embedding_idx
ON public.career_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE OR REPLACE FUNCTION public.match_documents(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 3
)
RETURNS TABLE (
    chunk_id TEXT,
    filename TEXT,
    section TEXT,
    page INT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
    SELECT
        cd.chunk_id,
        cd.filename,
        COALESCE(cd.section, 'N/A') AS section,
        cd.page,
        cd.content,
        1 - (cd.embedding <=> query_embedding) AS similarity
    FROM public.career_documents AS cd
    ORDER BY cd.embedding <=> query_embedding
    LIMIT match_count;
$$;
