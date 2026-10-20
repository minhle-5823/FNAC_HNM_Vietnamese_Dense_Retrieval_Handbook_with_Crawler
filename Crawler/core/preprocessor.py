"""Retrieval-oriented preprocessing for Vietnamese web documents.

REVIEW MAP
----------
SECTION 1 : Unicode / whitespace cleanup
SECTION 2 : boilerplate-line deduplication
SECTION 3 : Vietnamese-language heuristic
SECTION 4 : topic relevance / quality validation
SECTION 5 : retrieval passage chunking
SECTION 6 : full preprocessing pipeline

Design principle for dense retrieval:
- preserve Vietnamese diacritics, punctuation, numbers, product/course names;
- DO NOT stem, remove stopwords, or aggressively lowercase the corpus;
- keep title + section context because semantic retrieval benefits from natural text.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

VI_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
VI_COMMON = {
    "và","của","cho","là","có","được","không","trong","với","khi","tại","theo",
    "khách","hàng","hỗ","trợ","dịch","vụ","đăng","ký","hướng","dẫn",
    "tuyển","sinh","học","phí","sinh","viên","đào","tạo","trường","ngành",
}

BAD_PAGE_PATTERNS = [
    r"\b404\b", r"not found", r"access denied", r"verify you are human",
    r"captcha", r"just a moment", r"đăng nhập để tiếp tục",
]


@dataclass
class ProcessedDocument:
    clean_title: str
    clean_text: str
    language_score: float
    relevance_score: float
    quality_score: float
    valid: bool
    rejection_reason: str
    processed_hash: str
    passages: list[dict]


# ============================================================
# SECTION 1 - NORMALIZE TEXT
# ============================================================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# SECTION 2 - REMOVE REPEATED / BOILERPLATE-LIKE LINES
# ============================================================
def dedupe_lines(text: str) -> str:
    out = []
    seen = set()
    for raw in re.split(r"\n+", text):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        key = line.casefold()
        # Repeated menus / footer lines are common on crawled pages.
        if key in seen and len(line) < 180:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


# ============================================================
# SECTION 3 - VIETNAMESE-LANGUAGE HEURISTIC
# ============================================================
def vietnamese_score(text: str) -> float:
    words = re.findall(r"\b[\wÀ-ỹ]+\b", (text or "").lower(), flags=re.UNICODE)
    if not words:
        return 0.0
    common_hits = sum(1 for w in words if w in VI_COMMON)
    char_hits = sum(1 for ch in (text or "").lower() if ch in VI_CHARS)
    # Conservative, dependency-free heuristic. 1.0 means strongly Vietnamese.
    word_component = min(1.0, common_hits / max(6.0, len(words) * 0.04))
    char_component = min(1.0, char_hits / max(8.0, len(text) * 0.015))
    return round(0.55 * word_component + 0.45 * char_component, 4)


# ============================================================
# SECTION 4 - TOPIC RELEVANCE + DATA QUALITY
# ============================================================
def _keyword_score(title: str, text: str, keywords: Iterable[str]) -> tuple[float, int]:
    title_cf = (title or "").casefold()
    text_cf = (text or "").casefold()
    kws = [k.casefold().strip() for k in keywords if str(k).strip()]
    if not kws:
        return 1.0, 0
    score = 0.0
    hits = 0
    for kw in kws:
        in_title = kw in title_cf
        in_text = kw in text_cf
        if in_title or in_text:
            hits += 1
            score += 2.0 if in_title else 1.0
    return min(1.0, score / max(3.0, min(8.0, len(kws)))), hits


def validate_document(title: str, text: str, profile: dict) -> tuple[bool, str, float, float, float]:
    cfg = profile.get("preprocess", {})
    min_words = int(cfg.get("min_words", 60))
    min_chars = int(cfg.get("min_chars", 300))
    min_vi = float(cfg.get("min_vietnamese_score", 0.35))
    min_rel = float(cfg.get("min_relevance_score", 0.20))
    min_kw_hits = int(cfg.get("min_keyword_hits", 1))
    keywords = cfg.get("relevance_keywords", [])
    required_any = [
        str(x).casefold().strip()
        for x in cfg.get("required_any_keywords", [])
        if str(x).strip()
    ]
    reject_any = [
        str(x).casefold().strip()
        for x in cfg.get("reject_any_keywords", [])
        if str(x).strip()
    ]

    merged = f"{title}\n{text}".strip()
    words = re.findall(r"\b[\wÀ-ỹ]+\b", merged, flags=re.UNICODE)
    vi = vietnamese_score(merged)
    rel, hits = _keyword_score(title, text, keywords)

    low = merged.casefold()

    if required_any and not any(k in low for k in required_any):
        return False, "missing_required_domain_signal", vi, rel, 0.0

    if reject_any and any(k in low for k in reject_any):
        return False, "excluded_topic_signal", vi, rel, 0.0

    for pattern in cfg.get("reject_text_regex", BAD_PAGE_PATTERNS):
        if re.search(pattern, low, flags=re.I):
            return False, f"reject_pattern:{pattern}", vi, rel, 0.0

    if len(text) < min_chars:
        return False, "too_short_chars", vi, rel, 0.0
    if len(words) < min_words:
        return False, "too_short_words", vi, rel, 0.0
    if vi < min_vi:
        return False, "not_vietnamese_enough", vi, rel, 0.0
    if hits < min_kw_hits or rel < min_rel:
        return False, "topic_not_relevant_enough", vi, rel, 0.0

    length_score = min(1.0, len(words) / 250.0)
    quality = round(0.40 * vi + 0.40 * rel + 0.20 * length_score, 4)
    return True, "", vi, rel, quality


# ============================================================
# SECTION 5 - CHUNK FOR DENSE RETRIEVAL
# ============================================================
def chunk_text(title: str, text: str, profile: dict) -> list[dict]:
    cfg = profile.get("preprocess", {})
    target = int(cfg.get("chunk_words", 220))
    overlap = int(cfg.get("chunk_overlap_words", 40))
    min_chunk = int(cfg.get("min_chunk_words", 45))

    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, current = [], []

    def flush():
        nonlocal current
        words = " ".join(current).split()
        if len(words) >= min_chunk:
            idx = len(chunks)
            chunk = " ".join(words)
            chunks.append({
                "passage_index": idx,
                "text": chunk,
                "word_count": len(words),
            })
            current = words[-overlap:] if overlap > 0 else []
        else:
            current = words

    for p in paragraphs:
        p_words = p.split()
        if current and len(current) + len(p_words) > target:
            flush()
        current.extend(p_words)
        while len(current) >= target + max(20, overlap):
            flush()

    if current:
        words = current
        if len(words) >= min_chunk or not chunks:
            chunks.append({
                "passage_index": len(chunks),
                "text": " ".join(words),
                "word_count": len(words),
            })

    # Put title context into every passage without destroying natural language.
    clean_title = normalize_text(title)
    for c in chunks:
        if clean_title and clean_title.casefold() not in c["text"][: max(200, len(clean_title) + 20)].casefold():
            c["text"] = f"{clean_title}\n{c['text']}"
            c["word_count"] = len(c["text"].split())
    return chunks


# ============================================================
# SECTION 6 - FULL PIPELINE
# ============================================================
def preprocess_document(title: str, extracted_text: str, profile: dict) -> ProcessedDocument:
    clean_title = normalize_text(title)
    clean_text = dedupe_lines(normalize_text(extracted_text))
    valid, reason, vi, rel, quality = validate_document(clean_title, clean_text, profile)
    passages = chunk_text(clean_title, clean_text, profile) if valid else []
    fp = hashlib.sha256(clean_text.encode("utf-8", errors="ignore")).hexdigest() if clean_text else ""
    return ProcessedDocument(
        clean_title=clean_title,
        clean_text=clean_text,
        language_score=vi,
        relevance_score=rel,
        quality_score=quality,
        valid=valid,
        rejection_reason=reason,
        processed_hash=fp,
        passages=passages,
    )
