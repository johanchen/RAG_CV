"""
ingest.py — Document ingestion pipeline for career_documents Supabase table.

Pipeline per file:
  convert → normalize → chunk (per-type config) → strip headers/footers
  → deduplicate → detect language → embed → upsert
"""

import argparse
import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from openai import OpenAI
from supabase import Client, create_client

try:
    import langdetect
    langdetect.DetectorFactory.seed = 0  # reproducible results
    from langdetect import detect as _langdetect_detect, LangDetectException
    _LANGDETECT = True
except ImportError:
    _LANGDETECT = False


# ── Configuration ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html", ".htm"}
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100

# Characters; shorter chunks are discarded after normalization.
MIN_CHUNK_LENGTH = 150

# Max tokens per chunk by document type.
# - PDF: dense prose → larger chunks preserve more context.
# - PPTX/XLSX: natural slide/cell boundaries → smaller chunks stay coherent.
# - DOCX/MD/HTML/TXT: medium — balanced between structure and context.
CHUNK_MAX_TOKENS: Dict[str, int] = {
    ".pdf":  512,
    ".docx": 384,
    ".pptx": 256,
    ".xlsx": 256,
    ".html": 384,
    ".htm":  384,
    ".md":   384,
    ".txt":  384,
}
DEFAULT_MAX_TOKENS = 384

# Lines appearing in >= this fraction of a document's chunks are treated as
# repeating headers or footers and stripped from all chunks.
HEADER_FOOTER_PREVALENCE = 0.40

# SequenceMatcher ratio at or above which a chunk is considered a near-duplicate
# of a previously accepted chunk and discarded.
NEAR_DUPLICATE_THRESHOLD = 0.88

# Cache chunkers by max_tokens so Docling doesn't reload tokeniser per file.
_chunker_cache: Dict[int, HybridChunker] = {}


# ── Secrets / clients ──────────────────────────────────────────────────────────

def load_secret(name: str) -> str:
    value = st.secrets.get(name)
    if not value:
        raise ValueError(f"Missing required secret: {name}")
    return value


def build_clients() -> Tuple[Client, OpenAI]:
    supabase = create_client(
        load_secret("SUPABASE_URL"),
        load_secret("SUPABASE_SERVICE_ROLE_KEY"),
    )
    openai_client = OpenAI(api_key=load_secret("OPENAI_API_KEY"))
    return supabase, openai_client


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_files(root: Path) -> List[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


# ── ID generation ──────────────────────────────────────────────────────────────

def document_id_for(filename: str) -> str:
    """Stable document-level ID: sha256 of the filename."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()


def chunk_id_for(filename: str, chunk_index: int) -> str:
    """Stable chunk-level ID: sha256 of filename + chunk index."""
    return hashlib.sha256(f"{filename}{chunk_index}".encode("utf-8")).hexdigest()


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    NFC unicode normalise, strip control characters, collapse excessive whitespace.
    Preserves newlines for downstream structure detection.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control chars except \n and \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace Unicode replacement character (common OCR artefact)
    text = text.replace("\ufffd", " ")
    # Collapse runs of spaces/tabs (newlines kept intact)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── OCR detection ──────────────────────────────────────────────────────────────

def is_likely_ocr(file_path: Path, conversion_result) -> bool:
    """
    Heuristic OCR detection for PDFs only.

    Flags a document as OCR-dependent when:
    - The exported text is suspiciously short (< 100 chars) for a PDF, or
    - > 35 % of whitespace-delimited tokens are single characters (garbled text), or
    - > 2 % of characters are Unicode replacement chars (\ufffd).

    Non-PDF formats are always native-digital.
    """
    if file_path.suffix.lower() != ".pdf":
        return False
    try:
        doc = conversion_result.document
        if hasattr(doc, "export_to_markdown"):
            text = doc.export_to_markdown()
        elif hasattr(doc, "export_to_text"):
            text = doc.export_to_text()
        else:
            return False

        if len(text) < 100:
            return True  # near-empty conversion → scanned / image-only PDF

        words = text.split()
        if not words:
            return True
        single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        replacement_ratio = text.count("\ufffd") / len(text)
        return single_char_ratio > 0.35 or replacement_ratio > 0.02
    except Exception:
        return False


# ── Language detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Return ISO 639-1 language code, or 'unknown' if unavailable."""
    if not _LANGDETECT or len(text) < 50:
        return "unknown"
    try:
        return _langdetect_detect(text[:2000])
    except Exception:
        return "unknown"


# ── Document-level metadata ────────────────────────────────────────────────────

def extract_doc_metadata(file_path: Path, conversion_result) -> Dict:
    """
    Extract document-level metadata.

    Auto-generated fields (best-effort, no manual input required):
    - title: from Docling document properties → first Markdown heading → filename stem
    - author: from Docling document properties (None if unavailable)
    - created_date: file modification time in ISO 8601 UTC

    Trust level: low — treat as informational, not authoritative.
    """
    doc = conversion_result.document
    title: Optional[str] = None
    author: Optional[str] = None

    # 1. Docling document description / properties
    try:
        for attr in ("description", "properties", "meta"):
            props = getattr(doc, attr, None)
            if props:
                title = title or getattr(props, "title", None)
                raw_author = getattr(props, "author", None)
                if raw_author and not author:
                    author = ", ".join(str(a) for a in raw_author) if isinstance(raw_author, list) else str(raw_author)
                if title:
                    break
    except Exception:
        pass

    # 2. First Markdown heading in exported text
    if not title:
        try:
            text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = stripped.lstrip("#").strip()[:200]
                    break
        except Exception:
            pass

    # 3. Filename stem as last resort
    if not title:
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

    # File mtime as proxy for document version date
    created_date: Optional[str] = None
    try:
        mtime = file_path.stat().st_mtime
        created_date = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass

    return {"title": title, "author": author or None, "created_date": created_date}


# ── Chunk field extraction helpers ────────────────────────────────────────────

def _chunk_text(chunk) -> str:
    if hasattr(chunk, "text"):
        return (chunk.text or "").strip()
    if hasattr(chunk, "content"):
        return (chunk.content or "").strip()
    return str(chunk).strip()


def _chunk_headings(chunk) -> List[str]:
    """
    Return heading list ordered top-level → most specific.
    HybridChunker stores headings in chunk.meta.headings (most specific last).
    """
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return []
    headings = getattr(meta, "headings", None) if not isinstance(meta, dict) else meta.get("headings")
    if not headings:
        return []
    if isinstance(headings, (list, tuple)):
        return [str(h) for h in headings if h]
    return [str(headings)]


def _chunk_page(chunk) -> Optional[int]:
    meta = getattr(chunk, "meta", None)
    if not meta:
        return None
    for key in ("page", "page_no", "page_number"):
        value = getattr(meta, key, None) if not isinstance(meta, dict) else meta.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


# ── Embed text construction ────────────────────────────────────────────────────

def _build_embed_text(section: str, subsection: str, content: str) -> str:
    """
    Prepend the most specific section heading to content for context-aware embeddings.
    subsection is preferred (more granular); section is used only when there's no subsection.
    The embed text is popped before upsert — Supabase stores only clean content.
    """
    heading = subsection if (subsection and subsection != "N/A") else section
    if heading and heading != "N/A":
        return f"[Section: {heading}]\n{content}"
    return content


# ── Header / footer detection and removal ─────────────────────────────────────

def find_noisy_lines(chunks: List[Dict]) -> Set[str]:
    """
    Identify short lines (< 120 chars) that appear in >= HEADER_FOOTER_PREVALENCE
    of a document's chunks. These are repeating page headers, footers, or watermarks.
    """
    if len(chunks) < 3:
        return set()
    threshold = max(2, int(len(chunks) * HEADER_FOOTER_PREVALENCE))
    counter: Counter = Counter()
    for chunk in chunks:
        seen_in_chunk: Set[str] = set()
        for line in chunk["content"].splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 120 and stripped not in seen_in_chunk:
                counter[stripped] += 1
                seen_in_chunk.add(stripped)
    return {line for line, count in counter.items() if count >= threshold}


def strip_noisy_lines(text: str, noisy: Set[str]) -> str:
    if not noisy:
        return text
    lines = [l for l in text.splitlines() if l.strip() not in noisy]
    return "\n".join(lines).strip()


# ── Near-duplicate detection ───────────────────────────────────────────────────

def is_near_duplicate(text: str, seen: List[str]) -> bool:
    """
    Return True if text is a near-duplicate of any previously accepted chunk.

    Uses stdlib difflib.SequenceMatcher on normalised (lowercased, whitespace-collapsed)
    text. A length shortcircuit skips comparison when strings differ in length by > 50 %,
    which keeps runtime acceptable for documents with many chunks.
    """
    norm = " ".join(text.lower().split())
    norm_len = len(norm)
    for prev in seen:
        prev_norm = " ".join(prev.lower().split())
        # Skip if lengths differ too much — can't reach threshold
        if abs(norm_len - len(prev_norm)) / max(norm_len, len(prev_norm), 1) > 0.5:
            continue
        ratio = SequenceMatcher(None, norm, prev_norm, autojunk=False).ratio()
        if ratio >= NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


# ── Chunk record extraction ────────────────────────────────────────────────────

def extract_chunk_records(
    file_path: Path,
    chunks,
    doc_meta: Dict,
    is_ocr: bool,
) -> List[Dict]:
    """
    Extract raw chunk records from Docling chunks.

    Applies text normalisation and MIN_CHUNK_LENGTH filter.
    Language detection runs over the first three chunks combined (document-level signal).

    Metadata schema per chunk
    ─────────────────────────
    Auto-generated (high-trust):
      document_id, chunk_id, chunk_index, filename, source_type, page,
      word_count, is_ocr, security_classification

    Auto-generated (low-trust / best-effort):
      title, section, subsection, created_date, language, author

    Optional / not auto-populated (set to None):
      version
    """
    filename = file_path.name
    source_type = file_path.suffix.lower()
    doc_id = document_id_for(filename)

    records: List[Dict] = []
    for index, chunk in enumerate(chunks):
        raw_text = _chunk_text(chunk)
        content = normalize_text(raw_text)
        if len(content) < MIN_CHUNK_LENGTH:
            continue

        headings = _chunk_headings(chunk)
        # section = top-level heading (for broad filtering / navigation)
        # subsection = most specific heading (used in embed text for retrieval precision)
        section = headings[0] if headings else "N/A"
        subsection = headings[-1] if len(headings) > 1 else (headings[0] if headings else "N/A")

        records.append({
            # ── IDs ────────────────────────────────────────────────────────
            "document_id": doc_id,
            "chunk_id": chunk_id_for(filename, index),
            "chunk_index": index,
            # ── Source identity ────────────────────────────────────────────
            "filename": filename,        # kept for app.py / match_documents compat
            "source_type": source_type,
            # ── Structure ─────────────────────────────────────────────────
            "title": doc_meta.get("title"),
            "section": section,
            "subsection": subsection,
            "page": _chunk_page(chunk),  # kept as "page" for match_documents compat
            # ── Content ───────────────────────────────────────────────────
            "content": content,
            "word_count": len(content.split()),
            # ── Provenance (auto-generated) ────────────────────────────────
            "created_date": doc_meta.get("created_date"),
            "language": "unknown",       # filled after all records are built
            "is_ocr": is_ocr,
            # ── Optional (low-trust auto-extracted) ───────────────────────
            "author": doc_meta.get("author"),
            # ── Defaults ──────────────────────────────────────────────────
            "security_classification": "public",
            "version": None,
            # ── Ephemeral (popped before upsert) ──────────────────────────
            "_embed_text": _build_embed_text(section, subsection, content),
        })

    # Language detection: sample from first three chunks for a document-level signal.
    if records:
        sample = " ".join(r["content"] for r in records[:3])
        lang = detect_language(sample)
        for r in records:
            r["language"] = lang

    return records


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        ]
        embeddings.extend(batch_embeddings)
    return embeddings


# ── Per-file pipeline ─────────────────────────────────────────────────────────

def get_chunker(max_tokens: int) -> HybridChunker:
    """Return a cached HybridChunker for the given max_tokens value."""
    if max_tokens not in _chunker_cache:
        _chunker_cache[max_tokens] = HybridChunker(
            max_tokens=max_tokens,
            always_emit_headings=True,
        )
    return _chunker_cache[max_tokens]


def process_file(
    file_path: Path,
    converter: DocumentConverter,
    supabase: Client,
    openai_client: OpenAI,
) -> None:
    print(f"\nProcessing: {file_path.name}")
    suffix = file_path.suffix.lower()

    # 1. Convert document
    try:
        result = converter.convert(str(file_path))
        document = getattr(result, "document", result)
    except Exception as exc:
        print(f"  ✗ Conversion failed: {exc}")
        return

    # 2. Detect OCR dependency (PDFs only)
    ocr = is_likely_ocr(file_path, result)
    if ocr:
        print("  ⚠ OCR-dependent PDF detected; text quality may vary.")

    # 3. Extract document-level metadata
    doc_meta = extract_doc_metadata(file_path, result)
    print(f"  Title: \"{doc_meta['title']}\"")

    # 4. Chunk with per-type token limit
    max_tokens = CHUNK_MAX_TOKENS.get(suffix, DEFAULT_MAX_TOKENS)
    chunker = get_chunker(max_tokens)
    try:
        chunks = list(chunker.chunk(document))
    except Exception as exc:
        print(f"  ✗ Chunking failed: {exc}")
        return

    # 5. Normalise and filter short chunks
    records = extract_chunk_records(file_path, chunks, doc_meta, ocr)
    if not records:
        print("  - No eligible chunks after normalisation.")
        return
    print(f"  - {len(records)} chunks after normalisation (lang: {records[0]['language']})")

    # 6. Strip repeating headers / footers
    noisy = find_noisy_lines(records)
    if noisy:
        print(f"  - Stripping {len(noisy)} repeating header/footer line(s).")
        for r in records:
            r["content"] = strip_noisy_lines(r["content"], noisy)
            r["_embed_text"] = strip_noisy_lines(r["_embed_text"], noisy)
            r["word_count"] = len(r["content"].split())
        # Re-apply MIN_CHUNK_LENGTH after stripping
        records = [r for r in records if len(r["content"]) >= MIN_CHUNK_LENGTH]

    # 7. Remove near-duplicate chunks
    seen_texts: List[str] = []
    unique_records: List[Dict] = []
    for r in records:
        if not is_near_duplicate(r["content"], seen_texts):
            seen_texts.append(r["content"])
            unique_records.append(r)
    removed = len(records) - len(unique_records)
    if removed:
        print(f"  - Removed {removed} near-duplicate chunk(s).")
    records = unique_records

    if not records:
        print("  - No chunks remain after deduplication.")
        return

    # 8. Batch embed
    embed_texts = [r.pop("_embed_text") for r in records]
    print(f"  - Embedding {len(records)} chunk(s)...")
    try:
        embeddings = embed_texts_batch(openai_client, embed_texts)
    except Exception as exc:
        print(f"  ✗ Embedding failed: {exc}")
        return
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding

    # 9. Upsert to Supabase
    try:
        supabase.table("career_documents").upsert(records, on_conflict="chunk_id").execute()
        print(f"  ✓ Upserted {len(records)} chunk(s).")
    except Exception as exc:
        print(f"  ✗ Upsert failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_folder(folder_path: Path) -> None:
    supabase, openai_client = build_clients()
    converter = DocumentConverter()

    files = discover_files(folder_path)
    if not files:
        print(f"No supported files found in: {folder_path}")
        return
    print(f"Found {len(files)} supported file(s).")

    for file_path in files:
        process_file(file_path, converter, supabase, openai_client)

    print("\nIngestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest career documents into Supabase pgvector."
    )
    parser.add_argument("folder", type=str, help="Folder containing documents to ingest.")
    args = parser.parse_args()

    folder_path = Path(args.folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    ingest_folder(folder_path)


if __name__ == "__main__":
    main()
