import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import streamlit as st
from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter
from openai import OpenAI
from supabase import Client, create_client


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html", ".htm"}
EMBEDDING_MODEL = "text-embedding-3-small"


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
    if not meta:
        return "N/A"

    for key in ["section", "heading", "title"]:
        value = getattr(meta, key, None)
        if value:
            return str(value)
    if isinstance(meta, dict):
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


def embed_text(client: OpenAI, text: str) -> List[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def iter_chunk_rows(file_path: Path, chunks: Iterable, openai_client: OpenAI) -> Iterable[Dict]:
    filename = file_path.name
    for index, chunk in enumerate(chunks):
        content = _chunk_text(chunk)
        if len(content.strip()) < 50:
            continue

        yield {
            "chunk_id": chunk_id_for(filename, index),
            "filename": filename,
            "section": _chunk_section(chunk),
            "page": _chunk_page(chunk),
            "chunk_index": index,
            "content": content,
            "embedding": embed_text(openai_client, content),
        }


def ingest_folder(folder_path: Path) -> None:
    supabase, openai_client = build_clients()

    converter = DocumentConverter()
    chunker = HierarchicalChunker(max_tokens=1000, overlap=100)

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

            rows = list(iter_chunk_rows(file_path, chunks, openai_client))
            if not rows:
                print("  - No eligible chunks after filtering.")
                continue

            supabase.table("career_documents").upsert(rows, on_conflict="chunk_id").execute()
            print(f"  - Upserted {len(rows)} chunks.")
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
