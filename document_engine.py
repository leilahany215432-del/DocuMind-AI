from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4}\b",
]


def extract_pages(file_name: str, data: bytes) -> list[tuple[str, int | None]]:
    ext = os.path.splitext(file_name.lower())[1]
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(text, None)]
    if ext in {".txt", ".md"}:
        return [(data.decode("utf-8", errors="ignore"), None)]
    raise ValueError("نوع الملف غير مدعوم. استخدم PDF أو DOCX أو TXT أو MD.")


def split_chunks(pages: Iterable[tuple[str, int | None]], file_name: str, max_chars: int = 900) -> list[Chunk]:
    chunks: list[Chunk] = []
    for text, page in pages:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            continue
        sentences = re.split(r"(?<=[.!؟?؛])\s+|\n+", clean)
        current = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(Chunk(current, file_name, page))
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(Chunk(current, file_name, page))
    return chunks


def build_index(chunks: list[Chunk]) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return vectorizer, matrix


def search_chunks(query: str, chunks: list[Chunk], vectorizer: TfidfVectorizer, matrix: object, top_k: int = 5) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in indices if scores[i] > 0]


def local_summary(chunks: list[Chunk], vectorizer: TfidfVectorizer, matrix: object, count: int = 5) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []
    centroid = matrix.mean(axis=0)
    scores = cosine_similarity(matrix, centroid).ravel()
    indices = scores.argsort()[::-1][:count]
    return [(chunks[i], float(scores[i])) for i in indices]


def extract_entities(text: str) -> dict[str, list[str]]:
    dates: list[str] = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
    names = []
    # Conservative heuristic: capitalized Latin names and Arabic phrases following common labels.
    names.extend(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b", text))
    names.extend(re.findall(r"(?:الاسم|اسم العميل|المسؤول|السيد|السيدة)\s*[:：-]\s*([^،,؛;\n]{2,50})", text))
    def unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(v.strip() for v in values if v.strip()))[:30]
    return {"التواريخ": unique(dates), "البريد الإلكتروني": unique(emails), "أرقام الهاتف": unique(phones), "أسماء محتملة": unique(names)}


def answer_with_local_model(question: str, results: list[tuple[Chunk, float]]) -> str:
    if not results:
        return "لم أجد مقطعًا ذا صلة كافية داخل المستند. جرّبي صياغة السؤال بكلمات موجودة في الملف."
    best = results[:3]
    return "\n\n".join(f"• {chunk.text}" for chunk, _ in best)


def answer_with_optional_llm(question: str, results: list[tuple[Chunk, float]]) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key or not results:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_API_BASE") or None)
        context = "\n\n".join(f"[المصدر: {c.source}, الصفحة: {c.page or 'غير محددة'}]\n{c.text}" for c, _ in results[:6])
        prompt = f"أجب بالعربية اعتمادًا على السياق فقط. إذا لم توجد الإجابة قل ذلك بوضوح. السؤال: {question}\n\nالسياق:\n{context}"
        response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return response.choices[0].message.content
    except Exception:
        return None
