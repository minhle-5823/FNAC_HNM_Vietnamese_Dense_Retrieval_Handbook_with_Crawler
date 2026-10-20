"""Auditable site-aware content extraction on *already crawled* text.

Conservative by design: an uncertain extraction is quarantined, rather than
silently turning navigation/recommendations into training documents.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ExtractedUnit:
    text: str
    title: str
    rule: str
    quality: str = 'auto_high'


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', html.unescape(str(text or '')))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\u200b-\u200d\ufeff\x00-\x08\x0b-\x1f\x7f]', '', text)
    # Do NOT erase URLs indiscriminately: they may be meaningful admission links.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_fpt_footer(text: str) -> str:
    return re.sub(r'\s*Chia\s+sẻ\s*:\s*Các\s+bài\s+viết\s+liên\s+quan\s+Footer\s*$', '', text, flags=re.I).strip()


def _momo(text: str, title: str, url: str) -> tuple[list[ExtractedUnit], str]:
    if urlsplit(url).path.rstrip('/') == '/hoi-dap':
        return [], 'momo_listing_home'
    # Topic pages have a list of other questions but no standalone answer.
    marker = re.search(r'GỌI\s+NGAY\s+(?:Quay\s+lại\s+)?', text, flags=re.I)
    if not marker:
        return [], 'momo_missing_article_anchor'
    tail = text[marker.end():].strip()
    # The answer must immediately follow the page's title, otherwise it is a listing.
    if not tail.casefold().startswith(title.casefold()):
        return [], 'momo_category_or_title_mismatch'
    # Category listings do not have the article back-link. Some real Q&A
    # pages omit it; allow those only for question-like titles.
    if not re.search(r'GỌI\s+NGAY\s+Quay\s+lại', text[:marker.end()], flags=re.I) and '?' not in title:
        return [], 'momo_category_listing_without_answer'
    body = tail[len(title):].strip()
    # Rating and related links are the common ending, not response text.
    end = re.search(r'\s+Đánh\s+giá\s*:\s*\d+(?:\s*/\s*\d+)?\b', body, flags=re.I)
    if not end:
        return [], 'momo_missing_end_anchor'
    body = body[:end.start()].strip()
    body = re.sub(r'^Quay\s+lại\s+', '', body, flags=re.I).strip()
    if len(body) < 55 or (body.count('?') > 4 and 'Quay lại' not in text[marker.start():marker.end()]):
        return [], 'momo_answer_too_short_or_empty'
    # Handle titles explicitly instead of relying on a generic menu expression.
    return [ExtractedUnit(title+'\n'+body, title, 'momo_anchor_qa')], ''


_UEL_MENU_END = 'Tuyển sinh khác'
_UEL_QUESTION_START = re.compile(
    r'(?:Trường\s+Đại\s+học|Học\s+phí|Chương\s+trình|Làm\s+sao|Muốn\s+xem|Có\s+chứng\s+chỉ|Học\s+sinh|Thí\s+sinh|'
    r'Điểm\s+chuẩn|Tại\s+sao|Những\s+đối\s+tượng|Khi\s+nào|Nhà\s+trường|Trường\s+có|Các\s+ngành|'
    r'Sinh\s+viên|Địa\s+chỉ|Nếu\s+|Lệ\s+phí|Hồ\s+sơ|Lịch\s+trình)',
    re.I,
)


def _uel(text: str, title: str, url: str) -> tuple[list[ExtractedUnit], str]:
    # Repeated site-wide menu is a distinctive, exact delimiter. Never remove
    # the text of an admission requirement based on generic words such as "Học phí".
    i = text.find(_UEL_MENU_END)
    if i < 0:
        return [], 'uel_navigation_anchor_not_found'
    body = text[i+len(_UEL_MENU_END):].strip()
    # The site uses "Tags: Sao chép" to separate FAQ cards.
    cards = re.split(r'\s+Tags\s*:\s*(?:Sao\s+chép\s*)+', body, flags=re.I)
    results: list[ExtractedUnit] = []
    for card_index, card in enumerate(cards):
        # An open last card can be a crawler-truncated response.
        if card_index == len(cards)-1 and not re.search(r'Tags\s*:\s*(?:Sao\s+chép\s*)+$', body, flags=re.I):
            continue
        card = re.sub(r'^(?:Sao\s+chép\s*)+', '', card.strip(), flags=re.I).strip()
        if not card:
            continue
        # Breadcrumbs only in the first card. Find first plausible question
        # whose question mark is followed by an answer.
        questions=[j for j,ch in enumerate(card[:260]) if ch=='?']
        if not questions:
            continue
        options=list(_UEL_QUESTION_START.finditer(card[:questions[0]+1]))
        if not options:
            continue
        # A breadcrumb may mention 'Học phí' before the actual FAQ title.
        # Breadcrumbs are separated by ' - '; choose the earliest plausible
        # start AFTER the final breadcrumb, not an entity inside the question.
        boundary = card.rfind(' - ', 0, questions[0])
        later = [m for m in options if m.start() > boundary+3] if boundary>=0 else []
        m = later[0] if later else options[0]
        # On filtered FAQ pages, an unseparated 'Điểm chuẩn' category label
        # precedes a real question. Do not include the category in title.
        if boundary < 0 and card.lower().startswith('điểm chuẩn ') and len(options)>1:
            m = options[1]
        card = card[m.start():].strip()
        # Keep multiple related questions in the question title (e.g. 2-part UEL FAQ).
        question_marks = [j for j,ch in enumerate(card[:270]) if ch=='?']
        if not question_marks:
            continue
        qend = question_marks[0]
        if len(question_marks)>1 and question_marks[1]-qend<105:
            qend = question_marks[1]
        q = card[:qend+1].strip()
        answer = card[qend+1:].strip()
        answer = re.sub(r'\s+(?:Tags\s*:\s*Sao\s+chép\s*)+$','',answer, flags=re.I).strip()
        if len(q)<15 or len(answer)<38:
            continue
        results.append(ExtractedUnit(q+'\n'+answer, q, 'uel_faq_card'))
    if not results:
        return [], 'uel_no_safe_qa_cards'
    return results, ''


_MOBI_NAV = ('Thoát KHÁM PHÁ ', 'Câu hỏi thường gặp')


def _mobifone(text: str, title: str, url: str) -> tuple[list[ExtractedUnit], str]:
    # This source is the MobiFone FAQ. The adjacent 5G marketing/category pages
    # are NOT question-answer documents, and their CTA tables are poorly structured.
    if 'cau-hoi-thuong-gap' not in urlsplit(url).path:
        return [], 'mobi_non_faq_marketing_or_category_page'
    m=re.search(r'\b1\.\s+Mạng\s+5G\s+là\s+gì\?',text,flags=re.I)
    if not m:
        return [], 'mobi_faq_questions_missing'
    body=text[m.start():]
    starts=list(re.finditer(r'(?<!\S)(\d{1,2})\.\s+(?=(?:[A-ZÀ-Ỹ]|\dG))',body))
    results=[]
    for i, marker in enumerate(starts):
        chunk=body[marker.end():starts[i+1].start() if i+1<len(starts) else len(body)].strip()
        qm=chunk.find('?')
        if qm<8 or qm>140:
            continue
        q=chunk[:qm+1].strip(); answer=chunk[qm+1:].strip()
        if len(answer)<35:continue
        results.append(ExtractedUnit(q+'\n'+answer,q,'mobifone_numbered_faq'))
    if len(results)<8:
        return [], 'mobi_incomplete_faq_extraction'
    return results, ''


# Parse only the *two known 2026 admission-table URLs*. The source already
# flattened HTML table cells to plain text. Do not infer missing rows or quotas.
_HCMUS_2026_TABLE_PATHS = {
    '/chi-tieu/',
    '/2026-phuong-huong-tuyen-sinh-trinh-do-dai-hoc-du-kien/',
}
# HCMUS official 2026 program codes are seven characters long and may
# contain a letter in the sixth position (e.g. 74401a1, 75202a1).
# Optional suffix such as _DKD denotes another program variant.
# Restrict the shape to known table-code forms; do not treat arbitrary numbers
# elsewhere in a policy paragraph as a row header.
_HCMUS_ROW = re.compile(
    r'(?<!\w)(\d{1,2})\s+((?:\d{7}|\d{5}[A-Za-z]\d)(?:_[A-Za-z0-9]+)?)\s+'
)
_HCMUS_NAME_QUOTA = re.compile(r'(.{4,160}?)\s+(\d{1,4})\s+(?=Phương\s+thức)', re.I)


def extract_hcmus_admission_table(text: str, title: str, url: str) -> tuple[list[ExtractedUnit], dict] | None:
    """Turn each reliably detected HCMUS 2026 table row into ONE document.

    Keep source wording of method descriptions, conditions and combinations.
    Store a traceable split; DO NOT synthesize table entries missing in the raw
    crawler text. Source raw passages include STT 8/32 with alpha-numeric codes.
    """
    if urlsplit(url).path not in _HCMUS_2026_TABLE_PATHS:
        return None
    starts = list(_HCMUS_ROW.finditer(text))
    numbers = [int(m.group(1)) for m in starts]
    if len(starts) != 39 or numbers != list(range(1, 40)):
        return None
    # On the long /2026-phuong-huong... article, 'Lưu ý:' follows the table.
    # Its legal instructions belong to the article, NOT to degree program #39.
    table_end = len(text)
    if urlsplit(url).path != '/chi-tieu/':
        match = re.search(r'\s+Lưu\s+ý\s*:\s*', text[starts[-1].end():], flags=re.I)
        if not match:
            return None  # ambiguous table boundary: quarantine rather than guess
        table_end = starts[-1].end() + match.start()
    rows = []
    for i,match in enumerate(starts):
        finish = starts[i+1].start() if i+1 < len(starts) else table_end
        row_text = text[match.end():finish].strip()
        # Defense in depth: no second program's row header may be hidden inside
        # a single document. Fail closed if parser delimiters become ambiguous.
        if _HCMUS_ROW.search(row_text):
            return None
        parsed = _HCMUS_NAME_QUOTA.match(row_text)
        if not parsed or len(row_text) < 80:
            return None
        program_name, quota = parsed.group(1).strip(), parsed.group(2)
        method_and_subjects = row_text[parsed.end():].strip()
        if not method_and_subjects.lower().startswith('phương thức'):
            return None
        row_title = f'{program_name} (mã ngành {match.group(2)}, chỉ tiêu {quota}, 2026)'
        # Add field labels, keep original policy clauses and subject combinations.
        content = (f'Chỉ tiêu tuyển sinh 2026 | Mã ngành: {match.group(2)} | '
                   f'Tên ngành: {program_name} | Chỉ tiêu: {quota} | '
                   f'{method_and_subjects}')
        rows.append(ExtractedUnit(content, row_title, 'hcmus_structured_admission_row'))
    units = []
    prefix = text[:starts[0].start()].strip()
    prefix = re.sub(r'\s*STT\s+Mã\s+ngành\s+xét\s+tuyển.*$', '', prefix, flags=re.I)
    if len(prefix) > 130 and urlsplit(url).path != '/chi-tieu/':
        units.append(ExtractedUnit(prefix, title, 'hcmus_admission_article_preface'))
    units.extend(rows)
    suffix = text[table_end:].strip()
    if len(suffix) > 100:
        units.append(ExtractedUnit(suffix, title, 'hcmus_admission_article_after_table'))
    meta = {'table_rows': len(rows), 'stt_present': numbers,
            'stt_missing': sorted(set(range(1, 40)) - set(numbers)),
            'url': url, 'status': 'parsed_complete_39_rows'}
    return units, meta


# Navigation-only HCMUS directory, NOT an article. Its URLs are preserved
# in the upstream crawler database; exclude just this retrieval document.
_HCMUS_NAVIGATION_PATHS = {'/phuong-thuc-tuyen-sinh/'}
_HCMUS_CTA = re.compile(r'\b(?:Xem\s+thông\s+báo\s+chi\s+tiết|Xem\s+kết\s+quả\s+xét\s+tuyển)(?:\s+tại\s+đây)?\s*(?:→)?', re.I)
# Case-sensitive on purpose. 'các văn bản liên quan' lower-case within a
# genuine policy sentence must NEVER be interpreted as a footer.
_HCMUS_ATTACHMENT_HEADING = re.compile(
    r'(?<!\w)(?:(?:\d{1,2}|[IVX]{1,5})\.\s+)?CÁC\s+VĂN\s+BẢN\s+LIÊN\s+QUAN\s*:?(?=\s|$)'
)
_HCMUS_ARTICLE_RESTART = re.compile(r'(?<!\w)1\.\s+ĐIỀU\s+KIỆN\s+ĐƯỢC\s+THAM\s+GIA\s+DỰ\s+TUYỂN', re.I)
_HCMUS_INSTRUCTION_AFTER_ATTACHMENT = re.compile(r'Đối\s+với\s+phụ\s+lục\s+\d', re.I)


def clean_hcmus_article(text: str, url: str) -> str:
    """Conservatively strip navigation and attachment lists without policy loss.

    Only canonical *uppercase* attachment section markers are removed. A
    genuine lower-case mention of related legislation remains untouched.
    An attachment section that cannot be safely bounded is preserved and
    flagged for human verification rather than deleted speculatively.
    """
    # Pure navigation CTA; do not delete method descriptions around the CTA.
    text = _HCMUS_CTA.sub(' ', text)
    text = re.sub(r'\s+Xem\s+các\s+phương\s+thức\s+(?:xét\s+tuyển\s+)?khác\s*$', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    # Repeat to handle an article text pasted twice by the source page.
    for _ in range(4):
        m = _HCMUS_ATTACHMENT_HEADING.search(text)
        if not m:
            break
        head, after = text[:m.start()].rstrip(), text[m.end():].strip()
        # In the 2024 article the trailing attachment link is followed by an
        # accidental restart of the full policy. Keep policy, not the link.
        restart = _HCMUS_ARTICLE_RESTART.search(after)
        if restart and restart.start() < 120:
            text = normalize_text(head + ' ' + after[restart.start():])
            continue
        # One announcement has important instructions after the attachment list.
        inst = _HCMUS_INSTRUCTION_AFTER_ATTACHMENT.search(after)
        if inst and inst.start() < 400:
            text = normalize_text(head + ' ' + after[inst.start():])
            continue
        # Wholly trailing link/attachment block of bounded length.
        if len(after) <= 420:
            text = head
            continue
        # Ambiguous heading in middle of substantive content. Preserve it.
        break
    return text.strip()


def _hcmus(text: str, title: str, url: str) -> tuple[list[ExtractedUnit], str]:
    path = urlsplit(url).path
    if path.rstrip('/') == '':
        return [], 'hcmus_landing_navigation'
    if path.rstrip('/') + '/' in _HCMUS_NAVIGATION_PATHS:
        return [], 'hcmus_admission_method_directory_navigation'
    text = clean_hcmus_article(text,url)
    if len(text) < 100:
        return [], 'hcmus_short_article'
    parsed = extract_hcmus_admission_table(text, title, url)
    if parsed is not None:
        return parsed[0], ''
    if path in _HCMUS_2026_TABLE_PATHS:
        return [], 'hcmus_table_structure_ambiguous_review_raw'
    return [ExtractedUnit(text, title, 'hcmus_article_trim')], ''


def extract_page(raw_text: str, source_site: str, title: str, url: str) -> tuple[list[ExtractedUnit], str]:
    raw = normalize_text(raw_text)
    title = normalize_text(title)
    if source_site == 'momo_help':
        return _momo(raw,title,url)
    if source_site == 'uel_admissions_faq':
        return _uel(raw,title,url)
    if source_site == 'mobifone_5g_faq':
        return _mobifone(raw,title,url)
    if source_site == 'hcmus_admissions':
        return _hcmus(raw,title,url)
    if source_site == 'fpt_telecom_support':
        body = remove_fpt_footer(raw)
        # Audited exact transcription correction from uploaded DB (no fuzzy spellcheck).
        body = re.sub(r'\bthiết\s+bioj\b', 'thiết bị', body, flags=re.I)
        return ([ExtractedUnit(body,title,'fpt_footer_trim')], '') if len(body)>=75 else ([], 'fpt_short_article')
    return [], 'unrecognized_source_site'


# High-specificity leftovers; all matches should be zero on clean dataset.
RESIDUAL_PATTERNS = {
    'momo_navigation': re.compile(r'Câu hỏi theo chủ đề|Đánh giá\s*:\s*\d+\s*/|Bài viết này có hữu ích với bạn|Bài viết liên quan\s*:',re.I),
    'fpt_footer': re.compile(r'Chia\s+sẻ\s*:\s*Các\s+bài\s+viết\s+liên\s+quan\s+Footer',re.I),
    'mobi_navigation': re.compile(r'Thoát\s+KHÁM\s+PHÁ\s+Về\s+5G',re.I),
    'uel_chrome': re.compile(r'You\s+are\s+here\s*:\s*Home\s+FAQ|Tags\s*:\s*Sao\s+chép',re.I),
    'hcmus_navigation': re.compile(r'Xem\s+các\s+phương\s+thức\s+xét\s+tuyển\s+khác|Xem\s+thông\s+báo\s+chi\s+tiết|Xem\s+kết\s+quả\s+xét\s+tuyển',re.I),
    'hcmus_attachment_header': re.compile(r'(?<!\w)(?:\d{1,2}\.\s+)?CÁC\s+VĂN\s+BẢN\s+LIÊN\s+QUAN(?=\s|$)'),
}
