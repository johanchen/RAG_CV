# RAG Career Portfolio (Streamlit + Supabase + Docling)

Professional streaming RAG assistant for **jctx** (IT/Cybersecurity Audit), using:
- Streamlit chat UI with token streaming
- Docling multi-format ingestion
- Supabase + pgvector retrieval
- OpenAI embeddings (`text-embedding-3-small`)
- LiteLLM model gateway (OpenAI + Groq)

## Project Files
- `app.py` - Streamlit chat app with RAG retrieval + streaming responses.
- `ingest.py` - Recursive document ingestion and vector upsert.
- `schema.sql` - Supabase table + pgvector extension + `match_documents` RPC.
- `requirements.txt` - pinned dependencies.
- `.streamlit/secrets.toml.example` - secrets template.

## 1) Install Dependencies
```bash
pip install -r requirements.txt
```

## 2) Configure Secrets
This project uses **Streamlit secrets**, not `.env`.

Create this file:
- `.streamlit/secrets.toml`

You can copy from:
- `.streamlit/secrets.toml.example`

Required keys:
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
Run `schema.sql` in Supabase SQL Editor.

It creates:
- `career_documents` table
- `vector` extension
- `match_documents(query_embedding, match_count)` RPC

## 4) Ingest Documents
Put your files in a folder (example: `./docs`) and run:
```bash
python ingest.py ./docs
```

Supported formats:
- `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`, `.html`, `.htm`

Ingestion behavior:
- Recursive file scan
- Docling conversion + hybrid chunking (`HybridChunker`, ~256 tokens/chunk, `always_emit_headings=True`)
- Skips very short chunks (`len(strip) < 200`)
- Embeds with `text-embedding-3-small`
- Upserts by `chunk_id = sha256(filename + chunk_index)`

## 5) Run the App
```bash
streamlit run app.py
```

Features:
- Streaming assistant responses (`st.write_stream` + LiteLLM)
- Sidebar model switcher:
  - `openai/gpt-5.1`
  - `openrouter/nvidia/nemotron-3-nano-30b-a3b:free`
- Top-15 retrieval via Supabase RPC
- Similarity filter: discard `< 0.35`
- Chat memory capped to last 10 turns

## Notes
- If retrieval has no matching context, the app responds with a LinkedIn contact fallback.
- `.streamlit/secrets.toml`, `.env`, and local doc folders are ignored by `.gitignore`.
