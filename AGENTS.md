# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Streamlit chat application with streaming responses, retrieval, and model switching.
- `ingest.py`: CLI ingestion pipeline for local documents (`python ingest.py ./docs`), chunking, embeddings, and Supabase upsert.
- `schema.sql`: Supabase schema for `career_documents` and `match_documents` RPC.
- `requirements.txt`: pinned runtime dependencies.
- `.streamlit/secrets.toml.example`: template for required secrets.
- `.streamlit/secrets.toml` (local only): actual keys; never commit.
- `rag_cv/`: local virtual environment directory; treat as generated/dev-local content.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install pinned dependencies.
- `streamlit run app.py`: run the RAG chat UI locally.
- `python ingest.py ./docs`: ingest supported documents recursively.
- `python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['app.py','ingest.py']]"`: quick syntax check.

## Coding Style & Naming Conventions
- Language: Python 3.12+.
- Indentation: 4 spaces; UTF-8 source files.
- Naming: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, clear descriptive names.
- Keep functions focused and side effects explicit (API calls, DB writes).
- Prefer small, safe refactors over large rewrites; preserve current app behavior.

## Testing Guidelines
- No formal test suite exists yet. For now, validate changes with:
- syntax check for `app.py` and `ingest.py` (command above),
- a local Streamlit smoke test,
- one ingestion dry run against a small docs folder.
- If adding tests, use `pytest`, place them under `tests/`, and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Git history is not available in this workspace; use Conventional Commits going forward (e.g., `feat: add retrieval score logging`).
- PRs should include:
- what changed and why,
- any schema/config updates,
- local verification steps and results,
- screenshots for UI-impacting changes.

## Security & Configuration Tips
- Store secrets only in `.streamlit/secrets.toml`.
- Required keys: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `LINKEDIN_URL`.
- Never commit secrets, `.env`, or local document corpora.
