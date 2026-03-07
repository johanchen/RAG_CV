import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

import streamlit as st
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from openai import OpenAI
from supabase import Client, create_client


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html", ".htm"}
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
MIN_CHUNK_LENGTH = 200


def load_secret(name: str) -> str:
    value = st.secrets.get(name)
    if not value:
        raise ValueError(f"Missing required secret: {name}")
    return value


def build_clients() -> tuple[Client, OpenAI]:
    supabase = create_client(
        load_secret("SUPABASE_URL"),
        load_secret("SUPABASE_SERVICE_ROLE_KEY"),
    )
    openai_client = OpenAI(api_key=load_secret("OPENAI_API_KEY"))
    return supabase, openai_client


def discover_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def chunk_id_for(filename: str, chunk_index: int) -> str:
    raw = f"{filename}{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _chunk_text(chunk) -> str:
    if hasattr(chunk, "text"):
        return (chunk.text or "").strip()
    if hasattr(chunk, "content"):
        return (chunk.content or "").strip()
    return str(chunk).strip()


def _chunk_section(chunk) -> str:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return "N/A"

    # Docling HierarchicalChunker stores headings as a list (most specific last)
    headings = getattr(meta, "headings", None)
    if headings:
        if isinstance(headings, (list, tuple)) and len(headings) > 0:
            return str(headings[-1])
        return str(headings)

    # Fallback to other common attributes
    for key in ["section", "heading", "title"]:
        value = getattr(meta, key, None)
        if value:
            return str(value)
    if isinstance(meta, dict):
        headings = meta.get("headings")
        if headings and isinstance(headings, list):
            return str(headings[-1])
        for key in ["section", "heading", "title"]:
            if meta.get(key):
                return str(meta[key])
    return "N/A"


def _chunk_page(chunk):
    meta = getattr(chunk, "meta", None)
    if not meta:
        return None

    for key in ["page", "page_no", "page_number"]:
        value = getattr(meta, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    if isinstance(meta, dict):
        for key in ["page", "page_no", "page_number"]:
            value = meta.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
    return None


def _build_embed_text(section: str, content: str) -> str:
    """Prepend section heading to content for richer, context-aware embeddings."""
    if section and section != "N/A":
        return f"[Section: {section}]\n{content}"
    return content


def embed_texts_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in batches, returning embeddings in the same order."""
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        embeddings.extend(batch_embeddings)
    return embeddings


def extract_chunk_records(file_path: Path, chunks) -> List[Dict]:
    """Extract chunk records (without embeddings) from a file's chunks."""
    filename = file_path.name
    records: List[Dict] = []
    for index, chunk in enumerate(chunks):
        content = _chunk_text(chunk)
        if len(content.strip()) < MIN_CHUNK_LENGTH:
            continue
        section = _chunk_section(chunk)
        records.append({
            "chunk_id": chunk_id_for(filename, index),
            "filename": filename,
            "section": section,
            "page": _chunk_page(chunk),
            "chunk_index": index,
            "content": content,
            "_embed_text": _build_embed_text(section, content),
        })
    return records


def ingest_folder(folder_path: Path) -> None:
    supabase, openai_client = build_clients()

    converter = DocumentConverter()
    chunker = HybridChunker(always_emit_headings=True)

    files = discover_files(folder_path)
    if not files:
        print(f"No supported files found in: {folder_path}")
        return

    print(f"Found {len(files)} supported file(s).")

    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            conversion_result = converter.convert(str(file_path))
            document = getattr(conversion_result, "document", conversion_result)
            chunks = chunker.chunk(document)

            records = extract_chunk_records(file_path, chunks)
            if not records:
                print("  - No eligible chunks after filtering.")
                continue

            # Batch embed all chunks for this file in one pass
            embed_texts = [r.pop("_embed_text") for r in records]
            print(f"  - Embedding {len(records)} chunks...")
            embeddings = embed_texts_batch(openai_client, embed_texts)
            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding

            supabase.table("career_documents").upsert(records, on_conflict="chunk_id").execute()
            print(f"  - Upserted {len(records)} chunks.")
        except Exception as exc:
            print(f"  - Failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest career documents into Supabase pgvector.")
    parser.add_argument("folder", type=str, help="Folder path, e.g. python ingest.py ./docs")
    args = parser.parse_args()

    folder_path = Path(args.folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    ingest_folder(folder_path)


if __name__ == "__main__":
    main()
