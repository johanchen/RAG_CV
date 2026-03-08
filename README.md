# RAG Career Portfolio (Streamlit + Supabase + Docling)

Professional streaming RAG assistant for Johan Chen (IT/Cybersecurity Audit), using:
- Streamlit chat UI with token streaming
- Docling multi-format ingestion
- Supabase + pgvector retrieval
- OpenAI embeddings (`text-embedding-3-small`)
- LiteLLM model gateway (OpenAI + OpenRouter)

## Project Files
- `app.py` — Streamlit chat app with RAG retrieval + streaming responses
- `ingest.py` — Document ingestion pipeline with normalisation, deduplication, and vector upsert
- `schema.sql` — Supabase table, pgvector extension, and `match_documents` RPC
- `requirements.txt` — Pinned dependencies
- `.streamlit/secrets.toml.example` — Secrets template

## 1) Install Dependencies
```bash
pip install -r requirements.txt
pip install docling        # not pinned in requirements.txt; install separately
```

## 2) Configure Secrets
This project uses **Streamlit secrets**, not `.env`.

Create `.streamlit/secrets.toml` (copy from `.streamlit/secrets.toml.example`):

```toml
SUPABASE_URL = "https://your-project-ref.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-supabase-service-role-key"
OPENAI_API_KEY = "sk-..."
GROQ_API_KEY = "gsk_..."
LINKEDIN_URL = "https://www.linkedin.com/in/your-linkedin-id"
EMAIL_ADDRESS = "you@example.com"
OPENROUTER_API_KEY = "sk-or-..."  # optional, for OpenRouter models
```

## 3) Initialize Supabase Schema
Run `schema.sql` in the Supabase SQL Editor.

Creates:
- `career_documents` table with full metadata schema
- `ivfflat` vector index
- `match_documents(query_embedding, match_count)` RPC

**Existing installation?** Uncomment the `ALTER TABLE` migration statements at the bottom of `schema.sql` and run those instead.

## 4) Ingest Documents
Put your files in a folder (e.g. `./docs`) and run:
```bash
python ingest.py ./docs
```

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`, `.html`, `.htm`

Ingestion pipeline per file:
1. Docling conversion → text normalisation (unicode, whitespace, OCR artefact removal)
2. Chunking via `HybridChunker` with per-type token limits (PDF 512, DOCX/HTML/MD 384, PPTX/XLSX 256)
3. Repeating header/footer lines stripped (appear in ≥ 40 % of chunks)
4. Near-duplicate chunks removed (`SequenceMatcher` ratio ≥ 0.88)
5. Language detection via `langdetect`; OCR heuristic flagged for PDFs
6. Embeddings batched with `text-embedding-3-small`
7. Upserted by `chunk_id = sha256(filename + chunk_index)`

Re-ingest required after any chunking logic change — delete all rows from `career_documents` first.

## 5) Run the App
```bash
streamlit run app.py
```

Features:
- Streaming assistant responses (`st.write_stream` + LiteLLM)
- Light / dark theme switcher
- Sidebar model switcher: `openai/gpt-5.1` or `openrouter/nvidia/nemotron-3-nano-30b-a3b:free`
- Top-10 retrieval via Supabase RPC, similarity threshold 0.35
- Section › Subsection breadcrumb in retrieved context
- Chat memory capped to last 10 turns
- Retrieval debug panel (toggle in sidebar)

## Notes
- If retrieval returns no results, the app falls back to a LinkedIn/email contact message.
- `.streamlit/secrets.toml`, `.env`, and local doc folders are excluded by `.gitignore`.
