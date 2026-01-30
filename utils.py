import os
import io
import pdfplumber
from docx import Document


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def read_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    if ext in (".txt", ".md"):
        return read_txt(path)
    # Fallback: try reading as text
    try:
        return read_txt(path)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 70):
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap
    return chunks
