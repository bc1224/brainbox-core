"""
File processing and chunking for different content types.
Handles: PDF, TXT, MD, HTML, JSON, CSV
"""

import os
import re
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Generator

from rich.console import Console

console = Console()


def get_file_hash(filepath: str) -> str:
    """Generate a short hash for deduplication."""
    h = hashlib.md5(filepath.encode() + str(os.path.getmtime(filepath)).encode())
    return h.hexdigest()[:12]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by sentences.
    Tries to respect sentence boundaries for better retrieval.
    """
    # Split into sentences (rough but effective)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        words = len(sentence.split())
        
        if current_length + words > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_text = []
            overlap_len = 0
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_len + s_words > overlap:
                    break
                overlap_text.insert(0, s)
                overlap_len += s_words
            current_chunk = overlap_text
            current_length = overlap_len
        
        current_chunk.append(sentence)
        current_length += words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def extract_pdf(filepath: str) -> str:
    """Extract text from PDF."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        console.print(f"  [red]Error reading PDF {filepath}: {e}[/red]")
        return ""


def extract_text(filepath: str) -> str:
    """Extract text from plain text files."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        console.print(f"  [red]Error reading {filepath}: {e}[/red]")
        return ""


def extract_json(filepath: str) -> str:
    """Extract text from JSON (flatten to readable text)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        console.print(f"  [red]Error reading JSON {filepath}: {e}[/red]")
        return ""


def extract_csv(filepath: str) -> str:
    """Extract text from CSV."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return ""
        # Convert to readable text with headers
        headers = rows[0]
        text_parts = []
        for row in rows[1:]:
            entry = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
            if entry:
                text_parts.append(entry)
        return "\n".join(text_parts)
    except Exception as e:
        console.print(f"  [red]Error reading CSV {filepath}: {e}[/red]")
        return ""


# Map extensions to extractors
EXTRACTORS = {
    ".pdf": extract_pdf,
    ".txt": extract_text,
    ".md": extract_text,
    ".markdown": extract_text,
    ".html": extract_text,
    ".htm": extract_text,
    ".json": extract_json,
    ".csv": extract_csv,
    ".tsv": extract_text,
    ".log": extract_text,
    ".py": extract_text,
    ".js": extract_text,
    ".ts": extract_text,
    ".tsx": extract_text,
    ".jsx": extract_text,
    ".yaml": extract_text,
    ".yml": extract_text,
    ".toml": extract_text,
    ".env": extract_text,
}

SUPPORTED_EXTENSIONS = set(EXTRACTORS.keys())


def process_file(filepath: str, chunk_size: int, chunk_overlap: int) -> Generator[dict, None, None]:
    """
    Process a single file into chunks with metadata.
    Yields dicts with: id, text, metadata
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    
    if ext not in EXTRACTORS:
        return
    
    extractor = EXTRACTORS[ext]
    text = extractor(filepath)
    
    if not text or len(text.strip()) < 20:
        return
    
    file_hash = get_file_hash(filepath)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_hash}_{i}"
        yield {
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                "source": str(path.name),
                "source_path": str(path),
                "file_type": ext.lstrip("."),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now().isoformat(),
            }
        }


def discover_files(directory: str) -> list[str]:
    """Recursively find all supported files in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return sorted(files)
