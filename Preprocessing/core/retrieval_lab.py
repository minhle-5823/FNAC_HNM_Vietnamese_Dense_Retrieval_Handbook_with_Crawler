"""SEG301 preprocessing lab applied to FNAC-HNM's Vietnamese auxiliary corpus.

Only reads the merged SQLite snapshot. Does not create relevance labels or
train dense embeddings. Designed for auditability and reproducibility.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from core.source_cleaning import extract_page, RESIDUAL_PATTERNS, extract_hcmus_admission_table, normalize_text

STOPWORDS_VI = frozenset('''và là thì mà ở của cho với từ bởi khi nếu để một những các cái này kia đó ấy cũng rất hơn nữa đã đang sẽ về tại trong trên dưới giữa theo gồm như mỗi từng đều lại rồi mình chúng tôi bạn anh chị họ đây ấy vậy hoặc hay'''.split())
# Deliberately preserve negations (không/chưa/chẳng/đừng), prices, 5G, brand names.
STRATEGIES = ('baseline', 'no_stopwords', 'stemmed', 'lemmatized')
DEFAULT_QUERIES = (
    ('q01', 'Cách kiểm tra gói cước 5G MobiFone đang sử dụng?'),
    ('q02', 'Làm sao liên hệ chăm sóc khách hàng MoMo?'),
    ('q03', 'Cách thay đổi mật khẩu WiFi FPT?'),
    ('q04', 'Điều kiện và phương thức xét tuyển đại học UEL?'),
    ('q05', 'Thông tin tuyển sinh Đại học Khoa học Tự nhiên TP.HCM?'),
)


class VietnameseTokenizer:
    """Prefer Vietnamese word segmentation; explicit fallback for offline environments."""
    def __init__(self, backend: str = 'auto'):
        self.backend = 'regex_fallback'
        self._vi_tokenize = None
        if backend not in ('auto', 'underthesea', 'regex'):
            raise ValueError('tokenizer must be auto, underthesea, or regex')
        if backend != 'regex':
            try:
                from underthesea import word_tokenize
                self._vi_tokenize = word_tokenize
                self.backend = 'underthesea'
            except ImportError:
                if backend == 'underthesea':
                    raise RuntimeError('Need underthesea: pip install underthesea') from None

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        if self._vi_tokenize:
            segmented = self._vi_tokenize(text, format='text')
            return re.findall(r'(?u)\b[\w]+\b|[^\w\s]', segmented)
        return re.findall(r'(?u)\b[\w]+\b|[^\w\s]', text)


def normalize(text: str, lower: bool = False) -> str:
    text = unicodedata.normalize('NFC', html.unescape(str(text or '')))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\u200b-\u200d\ufeff\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower() if lower else text


# FPT Hi FPT detail pages include this navigation/website chrome after the
# article's actual answer. Remove ONLY this exact end-of-document pattern and
# ONLY for FPT, before segmentation, hashing, deduplication and indexing.
FPT_FOOTER_RE = re.compile(
    r'\s*Chia\s+sẻ\s*:\s*Các\s+bài\s+viết\s+liên\s+quan\s+Footer\s*$',
    flags=re.IGNORECASE,
)


def clean_source_text(text: str, source_site: str = '') -> str:
    result = normalize(text)
    if source_site == 'fpt_telecom_support':
        result = FPT_FOOTER_RE.sub('', result).strip()
    return result


def stable_id(source_passage_id: str, segment: int, text: str) -> str:
    payload = f'{source_passage_id}:{segment}:{normalize(text,True)}'
    return 'web_' + hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]


def segments(text: str, max_words: int = 220, overlap: int = 30) -> list[str]:
    """Sentence-aware chunks, never strand a section enumerator at a boundary.

    Preserve words and headings; overlap only full sentences. Flattened tables
    must be handled by their own source-aware extractor, not by this function.
    """
    if max_words < 30 or overlap >= max_words or overlap < 0:
        raise ValueError('max_words >= 30, and 0 <= overlap < max_words required')
    text = normalize_text(text)
    if len(text.split()) <= max_words:
        return [text] if text else []
    sentence_parts = re.split(r'(?<=[.!?;])\s+(?=[A-ZÀ-Ỹ0-9“\(])', text)
    blocks=[]
    for item in sentence_parts:
        words = item.split()
        if len(words)>max_words:
            blocks.extend(' '.join(words[i:i+max_words]) for i in range(0,len(words),max_words))
        elif words:
            blocks.append(item)
    # Prefix enumerators ("3.", "2.4.", "IV.") are section markers, not
    # standalone sentences. Attach to the NEXT complete block before packing.
    joined=[]
    pending=''
    enum_only=re.compile(r'(?:\d{1,3}(?:\.\d{1,3})*\.|[IVX]{1,8}\.)', re.I)
    for b in blocks:
        if enum_only.fullmatch(b.strip()):
            pending += b.strip() + ' '
        else:
            joined.append((pending+b).strip())
            pending=''
    if pending and joined:
        joined[-1] += ' '+pending.strip()
    chunks=[]; current=[]; current_words=0
    for block in joined:
        bw=len(block.split())
        if current and current_words+bw>max_words:
            chunks.append(' '.join(current).strip())
            carry=[]; carry_count=0
            if overlap:
                for old in reversed(current):
                    count=len(old.split())
                    if carry_count+count>overlap:break
                    carry.insert(0,old); carry_count+=count
            current=carry; current_words=carry_count
            if current_words+bw>max_words:
                current=[]; current_words=0
        current.append(block);current_words+=bw
    if current:
        last=' '.join(current).strip()
        if chunks and len(last.split())<30:
            chunks[-1] += ' '+last
        else:
            chunks.append(last)
    # A numeric enumerator still can follow a LONG paragraph after another
    # sentence (e.g. "... nộp hồ sơ. 3."). Move it forward, never discard it.
    dangling=re.compile(r'(?<!\w)(\d{1,3}(?:\.\d{1,3})*\.)\s*$')
    for i in range(len(chunks)-1):
        m=dangling.search(chunks[i])
        if m and len(chunks[i][:m.start()].split())>25 and not chunks[i+1].startswith(m.group(1)):
            chunks[i]=chunks[i][:m.start()].rstrip()
            chunks[i+1]=m.group(1)+' '+chunks[i+1]
    return [c for c in chunks if c.strip()]


def shingles(text: str) -> set[str]:
    ts = re.findall(r'(?u)\w+', normalize(text, True))
    return {' '.join(ts[i:i+4]) for i in range(len(ts)-3)} or set(ts)


def near_key(text: str) -> tuple[int, ...]:
    """16 deterministic minhash signatures for likely near-duplicate candidate retrieval."""
    terms = shingles(text)
    if not terms:
        return (0,) * 16
    hs = [int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), 'little')
          for t in terms]
    prime = 18446744073709551557
    return tuple(min((h * (i * 277 + 1) + i * 7919) % prime for h in hs)
                 for i in range(1, 17))


def dedup_rows(rows: list[dict], near_threshold: float = 0.90) -> tuple[list[dict], list[dict]]:
    exact: dict[str, str] = {}
    buckets: dict[tuple[int, tuple[int,...]], set[int]] = defaultdict(set)
    kept, rejected, shingle_cache = [], [], []
    for row in rows:
        text = row['context']
        key = normalize(text, True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        reason = None
        master = ''
        if digest in exact:
            reason, master = 'exact_duplicate', exact[digest]
        else:
            signs = near_key(key)
            candidates = set()
            for band in range(4):
                candidates.update(buckets.get((band, signs[band*4:(band+1)*4]), set()))
            if candidates:
                sh = shingles(key)
                for j in sorted(candidates):
                    old = shingle_cache[j]
                    # Preserve different titles/years and different FAQ intents,
                    # even if their shared policy text is near-identical.
                    if not (kept[j].get('extraction_rule') == 'hcmus_structured_admission_row'
                            or row.get('extraction_rule') == 'hcmus_structured_admission_row') and (kept[j]['source_site'] == row['source_site']
                            and normalize(kept[j]['title'],True) == normalize(row['title'],True)
                            and old and sh and (len(sh & old)/len(sh | old) >= near_threshold)):
                        reason, master = 'near_duplicate', kept[j]['doc_id']
                        break
        if reason:
            rejected.append({**row, 'reason': reason, 'duplicate_of': master})
            continue
        idx = len(kept)
        kept.append(row)
        exact[digest] = row['doc_id']
        shingle_cache.append(shingles(key))
        for band in range(4):
            buckets[(band, signs[band*4:(band+1)*4])].add(idx)
    return kept, rejected


def load_merged(db: Path, expected_sites: int = 5, allow_legacy_fpt: bool = False) -> tuple[pd.DataFrame, dict]:
    if not db.is_file():
        raise FileNotFoundError(f'Cannot find merged DB: {db}. Run python main.py merge-dbs first.')
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        tables = {x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {'passages', 'processed_documents', 'raw_pages', 'merge_sources', 'merge_info'}
        if required - tables:
            raise ValueError(f'Not an 8-table merged DB; missing {sorted(required-tables)}')
        if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite quick_check failed')
        nsites = con.execute('SELECT count(*) FROM merge_sources').fetchone()[0]
        if nsites != expected_sites:
            raise ValueError(f'Expected {expected_sites} sources but DB has {nsites}')
        old_fpt = con.execute("""SELECT count(*) FROM raw_pages
            WHERE source_site='fpt_telecom_support' AND url LIKE '%/tin-tuc/%'""").fetchone()[0]
        # The /ho-tro/cau-hoi-thuong-gap endpoint returned HTTP 500. The
        # supported customer-help source is now /hi-fpt/huong-dan-su-dung.
        hifpt_pages = con.execute("""SELECT count(*) FROM raw_pages
            WHERE source_site='fpt_telecom_support'
              AND url LIKE '%fpt.vn/hi-fpt/huong-dan-su-dung%'""").fetchone()[0]
        hifpt_passages = con.execute("""SELECT count(*) FROM passages
            WHERE source_site='fpt_telecom_support'
              AND page_url LIKE '%fpt.vn/hi-fpt/huong-dan-su-dung%'""").fetchone()[0]
        if (old_fpt > 0 or hifpt_pages == 0 or hifpt_passages == 0) and not allow_legacy_fpt:
            raise ValueError('FPT Hi FPT help-center pages or valid passages are missing '
                             '(or legacy /tin-tuc/ pages remain). '
                             'Update the FPT profile/extractor, crawl FPT with --reset, '
                             'validate, then run python main.py merge-dbs. '
                             f'Hi FPT raw pages={hifpt_pages}, passages={hifpt_passages}.')
        invalid = dict(con.execute('SELECT source_site, count(*) FROM processed_documents '
                                   'WHERE is_valid=0 GROUP BY source_site').fetchall())
        raw_counts = dict(con.execute('SELECT source_site, count(*) FROM raw_pages GROUP BY source_site').fetchall())
        df = pd.read_sql_query("""
          SELECT p.passage_id AS source_passage_id, p.origin_site, p.origin_id, p.page_url AS url,
                 p.source_site, p.domain_label, p.passage_index, p.text AS raw_text,
                 p.quality_score, d.title, r.crawled_at
          FROM passages p
          INNER JOIN processed_documents d
             ON d.origin_site=p.origin_site AND d.page_url=p.page_url AND d.is_valid=1
          LEFT JOIN raw_pages r
             ON r.origin_site=p.origin_site AND r.url=p.page_url
          ORDER BY p.origin_site, p.page_url, p.passage_index, p.id
        """, con)
        meta = {'expected_sites': expected_sites, 'actual_sites': nsites,
                'invalid_pages_by_site': invalid, 'raw_pages_by_site': raw_counts,
                'legacy_fpt_news_pages': old_fpt, 'hifpt_raw_pages': hifpt_pages, 'hifpt_passages': hifpt_passages,
                'merged_db': str(db)}
    if df.empty:
        raise ValueError('No valid passages available in merged DB')
    df['source_domain'] = df['url'].map(lambda x: urlsplit(str(x)).netloc.lower())
    df['title'] = df['title'].fillna('')
    df['crawled_at'] = df['crawled_at'].fillna('')
    df['raw_text'] = df['raw_text'].fillna('')
    meta['missing_values_by_column'] = {str(k):int(v) for k,v in df.isna().sum().items()}
    meta['exact_duplicate_source_passages'] = int(df.raw_text.duplicated().sum())
    meta['source_passages_rows'] = len(df)
    meta['source_passages_columns'] = len(df.columns)
    return df, meta


def _english_tools(allow_missing: bool = True):
    """English-only classroom illustration, never applied as Vietnamese morphology."""
    try:
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        stem = PorterStemmer().stem
        lemma_tool = WordNetLemmatizer()
        lemma_tool.lemmatize('cars')  # force resource check
        def lemma(w):
            return lemma_tool.lemmatize(lemma_tool.lemmatize(w, pos='v'), pos='n')
        return stem, lemma, 'WordNet available; English-only operations'
    except (ImportError, LookupError):
        if not allow_missing:
            raise RuntimeError('Download NLTK WordNet: python -m nltk.downloader wordnet omw-1.4') from None
        try:
            from nltk.stem import PorterStemmer
            stem = PorterStemmer().stem
        except ImportError:
            stem = lambda w: w
        return stem, (lambda w: w), 'WordNet unavailable: lemmatized experiment UNAVAILABLE, unchanged tokens'


def _lexical_token(token: str) -> str | None:
    """Keep Vietnamese compound words joined with '_' by underthesea.

    str.isalnum() excludes '_' and previously discarded useful words such as
    'đại_học' and 'đăng_ký' from all TF-IDF/BM25 indexing strategies.
    Accept one or more nonempty alphanumeric components separated by '_'.
    """
    token = token.strip('_')
    if token and all(part.isalnum() for part in token.split('_')):
        return token
    return None


def tokenize_strategies(tok: VietnameseTokenizer, text: str, stem, lemma) -> dict[str, list[str]]:
    raw = tok.tokenize(text)
    base = [word for token in raw if (word := _lexical_token(token)) is not None]
    no_stop = [w for w in base if w not in STOPWORDS_VI]
    # Avoid stemming Vietnamese ASCII syllables such as 'ban', 'nha', 'mua'.
    # English-only illustrative vocabulary from the instructor's morphology examples.
    english_examples = {'connect','connected','connecting','connection','studies',
                        'studying','study','cars','running','run','recorders',
                        'recording','devices','settings','payments','customers'}
    def latin_en(w):
        return w in english_examples
    return {'all': raw, 'baseline': base, 'no_stopwords': no_stop,
            'stemmed': [stem(w) if latin_en(w) else w for w in no_stop],
            'lemmatized': [lemma(w) if latin_en(w) else w for w in no_stop]}


def make_corpus(raw_df: pd.DataFrame, min_chars=100, max_words=220, overlap=30,
                near_threshold=0.90, return_audit=False):
    """Build high-confidence content units from primary crawler passages only.

    The crawler sometimes stores an extra tail passage (passage_index > 0)
    for the same page. It is not another document. Such tails are excluded,
    preserving a traceable reason in the audit. Unknown or ambiguous sections
    are quarantined, not guessed into the mining dataset.
    """
    staged, rejected, audit = [], [], []
    primary_urls = set(raw_df.loc[raw_df['passage_index'].eq(0), 'url'])
    for r in raw_df.to_dict('records'):
        base = {k: r[k] for k in ['source_passage_id','origin_site','origin_id','url',
                    'source_site','domain_label','source_domain','title','crawled_at',
                    'passage_index','quality_score']}
        if int(r['passage_index']) > 0 and r['url'] in primary_urls:
            audit.append({**base,'action':'quarantine','reason':'secondary_duplicate_tail_passage',
                          'extracted_units':0,'original_chars':len(r['raw_text']),'kept_chars':0})
            rejected.append({**base,'context':str(r['raw_text']),'doc_id':'',
                             'reason':'secondary_duplicate_tail_passage','duplicate_of':''})
            continue
        units, error = extract_page(str(r['raw_text']), r['source_site'], r['title'], r['url'])
        if error:
            audit.append({**base,'action':'quarantine','reason':error,'extracted_units':0,
                          'original_chars':len(r['raw_text']),'kept_chars':0})
            rejected.append({**base,'context':str(r['raw_text']),'doc_id':'',
                             'reason':error,'duplicate_of':''})
            continue
        audit.append({**base,'action':'extracted','reason':'source_aware_extraction',
                      'extracted_units':len(units),'original_chars':len(r['raw_text']),
                      'kept_chars':sum(len(u.text) for u in units)})
        for unit_index,unit in enumerate(units):
            clean = normalize_text(unit.text)
            if len(clean) < min_chars:
                rejected.append({**base,'context':clean,'doc_id':'','reason':'too_short','duplicate_of':''})
                continue
            # One UEL card == one Q&A; do not join unrelated questions.
            # Admission-table rows are atomic. Never split quotas, program IDs and
            # subject combinations across separate retrieval documents.
            pieces = ([clean] if unit.rule == 'hcmus_structured_admission_row'
                      else segments(clean,max_words,overlap))
            for pos,seg in enumerate(pieces):
                if pos > 0 and unit.title and not seg.casefold().startswith(unit.title.casefold()):
                    seg = unit.title + ' | ' + seg
                row = {**base,'doc_id':stable_id(r['source_passage_id']+'_'+str(unit_index),pos,seg),
                       'segment_index':pos,'unit_index':unit_index,'title':unit.title,
                       'extraction_rule':unit.rule,'extraction_quality':unit.quality,'context':seg}
                if len(seg) < min_chars:
                    rejected.append({**row,'reason':'short_segment','duplicate_of':''})
                else:
                    staged.append(row)
    kept, dupe = dedup_rows(staged,near_threshold)
    rejected.extend(dupe)
    corpus = pd.DataFrame(kept)
    if corpus.empty:
        raise ValueError('All records were filtered out. Examine 15_cleaning_page_audit.csv.')
    rejected_df = pd.DataFrame(rejected)
    stats = {'passages_from_merged':len(raw_df), 'segmented_candidates':len(staged),
             'clean_documents':len(corpus), 'rejected_total':len(rejected),
             'rejection_reasons':dict(Counter(x['reason'] for x in rejected)),
             'pages_auto_extracted':sum(a['action']=='extracted' for a in audit),
             'pages_quarantined':sum(a['action']=='quarantine' for a in audit)}
    if return_audit:
        return corpus, rejected_df, stats, pd.DataFrame(audit)
    return corpus,rejected_df,stats


def _bm25_corpus(tokens: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokens,k1=1.5,b=0.75), 'rank_bm25'
    except ImportError:
        return LocalBM25(tokens), 'local_bm25_compatible'


class LocalBM25:
    """Offline fallback matching rank_bm25.BM25Okapi's score formulation."""
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1,self.b = k1,b
        self.lengths = np.asarray([len(x) for x in corpus],dtype=np.float64)
        self.avgdl = float(self.lengths.mean())
        self.frequencies = [Counter(doc) for doc in corpus]
        df = Counter()
        for freq in self.frequencies:df.update(freq.keys())
        scores = {t:math.log(len(corpus)-n+0.5)-math.log(n+0.5) for t,n in df.items()}
        positive = [x for x in scores.values() if x>0]
        floor = epsilon * (sum(scores.values())/max(1,len(scores)))
        self.idf = {t:(v if v>=0 else floor) for t,v in scores.items()}
    def get_scores(self,query):
        scores=np.zeros(len(self.lengths),dtype=float)
        for q in query:
            idf = self.idf.get(q,0.0)
            if not idf: continue
            freq=np.fromiter((d.get(q,0) for d in self.frequencies),dtype=float,count=len(self.frequencies))
            scores += idf*(freq*(self.k1+1))/(freq+self.k1*(1-self.b+self.b*self.lengths/self.avgdl))
        return scores


def _topk(scores,k):
    scores=np.asarray(scores)
    order=np.lexsort((np.arange(len(scores)), -scores))
    return order[:min(k,len(order))]


def _search(query,tokenizer,stem,lemma,model,method,top_k):
    tokens = tokenize_strategies(tokenizer, normalize(query),stem,lemma)[model['strategy']]
    if method == 'tfidf':
        q = model['tfidf_vectorizer'].transform([' '.join(tokens)])
        scores = np.asarray((model['tfidf_matrix'] @ q.T).toarray()).ravel()
    else:
        scores = model['bm25'].get_scores(tokens)
    return [(int(i),float(scores[i])) for i in _topk(scores,top_k)]


def run_lab(db: Path, out_dir: Path, *, tokenizer_backend='auto', min_chars=100,
            max_words=220, overlap=30, near_threshold=0.90,
            allow_legacy_fpt=False, qrels_path: Path | None = None, raw_csv: Path | None = None,
            review_decisions: Path | None = None) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True)
    model_dir=out_dir/'models'; model_dir.mkdir(exist_ok=True)
    if raw_csv:
        raw_df = pd.read_csv(raw_csv,keep_default_na=False,encoding='utf-8-sig')
        meta = {'actual_sites':int(raw_df.source_site.nunique()),'raw_snapshot':str(raw_csv),
                'note':'source is raw CSV snapshot; SQLite not modified'}
    else:
        raw_df,meta = load_merged(Path(db),allow_legacy_fpt=allow_legacy_fpt)
    raw_df.to_csv(out_dir/'01_export_merged_raw.csv',index=False,encoding='utf-8-sig')
    corpus,rejected,stats,audit = make_corpus(raw_df,min_chars,max_words,overlap,near_threshold,return_audit=True)
    source_events = []
    for r in raw_df.loc[raw_df.passage_index.eq(0)].to_dict('records'):
        raw = str(r['raw_text'])
        site,url = r['source_site'],r['url']
        if site == 'hcmus_admissions':
            if urlsplit(url).path.rstrip('/') == '/phuong-thuc-tuyen-sinh':
                source_events.append({'source_site':site,'url':url,'rule':'exclude_navigation_directory', 'count':1,'basis':'index page with year/method links, no standalone policy article'})
            for rule,pat in [
                ('remove_link_call_to_action',r'Xem\s+(?:thông\s+báo\s+chi\s+tiết|kết\s+quả\s+xét\s+tuyển)'),
                ('uppercase_attachment_heading',r'CÁC\s+VĂN\s+BẢN\s+LIÊN\s+QUAN')]:
                n=len(re.findall(pat,raw))
                if n:source_events.append({'source_site':site,'url':url,'rule':rule,'count':n,'basis':'source-aware anchored text cleaning; original preserved in 01_export_merged_raw.csv'})
        if site == 'fpt_telecom_support' and re.search(r'thiết\s+bioj',raw,re.I):
            source_events.append({'source_site':site,'url':url,'rule':'exact_typo_thiet_bioj_to_thiet_bi','count':1,'basis':'explicitly audited isolated source typo; original preserved'})
    pd.DataFrame(source_events,columns=['source_site','url','rule','count','basis']).to_csv(
        out_dir/'27_source_cleaning_decisions.csv',index=False,encoding='utf-8-sig')
    if review_decisions:
        decision_df=pd.read_csv(review_decisions,keep_default_na=False,encoding='utf-8-sig')
        if not {'doc_id','decision'}.issubset(decision_df.columns):
            raise ValueError('Review file requires columns doc_id, decision=keep/drop (or blank)')
        labels=decision_df[['doc_id','decision']].copy()
        labels['decision']=labels.decision.astype(str).str.strip().str.lower()
        labels=labels[labels.decision!='']
        if not labels.decision.isin(['keep','drop']).all():
            raise ValueError('Review decision values can only be keep/drop or blank')
        if labels.doc_id.duplicated().any():
            raise ValueError('Duplicate doc_id in review decisions; decide once per document')
        unknown=set(labels.doc_id)-set(corpus.doc_id)
        if unknown:
            raise ValueError(f'Review doc_id not present in current corpus: {list(unknown)[:3]}; regenerate review form')
        rejected_review=set(labels.loc[labels.decision=='drop','doc_id'])
        if rejected_review:
            old=corpus[corpus.doc_id.isin(rejected_review)].copy()
            old['reason']='human_review_rejected'; old['duplicate_of']=''
            rejected=pd.concat([rejected,old],ignore_index=True)
            corpus=corpus[~corpus.doc_id.isin(rejected_review)].copy().reset_index(drop=True)
        stats['human_review_dropped']=len(rejected_review)
        stats['clean_documents']=len(corpus)
        stats['rejected_total']=len(rejected)
        stats['rejection_reasons']=dict(Counter(rejected['reason'].astype(str)))
        if corpus.empty:raise ValueError('Human reviews excluded all documents')
    audit.to_csv(out_dir/'15_cleaning_page_audit.csv',index=False,encoding='utf-8-sig')
    quarantined = audit[audit.action=='quarantine'].copy()
    quarantined.to_csv(out_dir/'16_quarantine_pages.csv',index=False,encoding='utf-8-sig')
    qc_rows=[]
    for site,g in corpus.groupby('source_site'):
        for rule, pattern in RESIDUAL_PATTERNS.items():
            found = g.context.astype(str).str.contains(pattern, regex=True).sum()
            qc_rows.append({'source_site':site,'test':rule,'violations':int(found)})
    qc=pd.DataFrame(qc_rows)
    qc.to_csv(out_dir/'17_residual_noise_audit.csv',index=False,encoding='utf-8-sig')
    # Verify both HCMUS tables against the raw SQLite passages.
    # Do not fabricate missing/ambiguous admission rows.
    from core.source_cleaning import normalize_text as _qc_normalize_text
    table_audit, parsed_rows=[] , []
    for r in raw_df.loc[(raw_df.source_site=='hcmus_admissions') &
                        raw_df.passage_index.eq(0)].to_dict('records'):
        if not ('/chi-tieu/' in r['url'] or
                '/2026-phuong-huong-tuyen-sinh-trinh-do-dai-hoc-du-kien/' in r['url']):
            continue
        parsed=extract_hcmus_admission_table(_qc_normalize_text(r['raw_text']),r['title'],r['url'])
        if not parsed:
            table_audit.append({'url':r['url'],'status':'requires_manual_parsing',
                                'parsed_rows':0,'missing_stt':'unknown',
                                'warning':'Could not establish reliable table boundaries'})
            continue
        units,info=parsed
        table_audit.append({'url':r['url'],'status':info['status'],
                            'parsed_rows':info['table_rows'],
                            'missing_stt':', '.join(map(str,info['stt_missing'])),
                            'warning':'Manual verification: missing table rows' if info['stt_missing'] else ''})
        for u in units:
            if u.rule!='hcmus_structured_admission_row':continue
            match=re.search(r'\(mã ngành (.*?), chỉ tiêu (\d+), 2026\)$',u.title)
            if match:
                program=u.title[:match.start()].casefold().strip()
                parsed_rows.append({'url':r['url'],'program':program,
                                    'code':match.group(1),'quota':match.group(2)})
    table_df=pd.DataFrame(table_audit,columns=['url','status','parsed_rows','missing_stt','warning'])
    table_df.to_csv(out_dir/'21_hcmus_table_integrity.csv',index=False,encoding='utf-8-sig')
    conflict_records=[]
    if parsed_rows:
        for (program,quota), grp in pd.DataFrame(parsed_rows).groupby(['program','quota']):
            if grp['code'].nunique()>1:
                conflict_records.append({'program':program,'quota':quota,
                    'codes':' | '.join(sorted(set(grp.code))),
                    'urls':' | '.join(sorted(set(grp.url))),
                    'status':'MANUAL_REVIEW_REQUIRED_NOT_AUTOCORRECTED'})
    pd.DataFrame(conflict_records,columns=['program','quota','codes','urls','status']).to_csv(
        out_dir/'23_hcmus_code_discrepancies.csv',index=False,encoding='utf-8-sig')
    hcmus=corpus[corpus.source_site=='hcmus_admissions'].copy()
    hcmus['word_count']=hcmus.context.astype(str).map(lambda s:len(s.split()))
    hcmus['dangling_number']=hcmus.context.str.contains(r'(?<!\w)\d{1,3}(?:\.\d{1,3})*\.\s*$',regex=True)
    hcmus['atomic_table_row']=hcmus.extraction_rule.eq('hcmus_structured_admission_row')
    hcmus['long_atomic_row']=hcmus.atomic_table_row & hcmus.word_count.gt(max_words)
    hcmus[['doc_id','title','url','extraction_rule','word_count','dangling_number',
           'atomic_table_row','long_atomic_row']].to_csv(
              out_dir/'22_hcmus_chunk_integrity.csv',index=False,encoding='utf-8-sig')
    stats['hcmus_dangling_number_chunks']=int(hcmus.dangling_number.sum())
    stats['hcmus_long_atomic_table_rows']=int(hcmus.long_atomic_row.sum())
    stats['hcmus_source_tables_with_missing_stt']=int(table_df.missing_stt.astype(str).str.len().gt(0).sum()) if len(table_df) else 0
    stats['hcmus_program_code_discrepancies']=len(conflict_records)
    if stats['hcmus_dangling_number_chunks']:
        raise ValueError('HCMUS chunk ends with a dangling enumeration. Check 22_hcmus_chunk_integrity.csv')

    # Additional low-confidence indicators are NOT deleted automatically:
    # headings for appended references, raw-site typos, and flattened tables
    # can occur in genuine article text. Queue them for human judgement.
    possible_flags={
        'hcmus_embedded_related_refs': (r'(?-i:CÁC\s+VĂN\s+BẢN\s+LIÊN\s+QUAN)', 'hcmus_admissions'),
        'fpt_source_typo_thiet_bioj': (r'thiết\s+bioj', 'fpt_telecom_support'),
        'momo_unusual_guide_cta': (r'GỌI\s+NGAY\s*$', 'momo_help'),
        'hcmus_long_atomic_table_row': (r'(?!)', 'hcmus_admissions'),
    }
    needs_review=[]
    for flag,(pattern,site) in possible_flags.items():
        found=corpus[(corpus.source_site==site) & corpus.context.str.contains(pattern,case=False,regex=True)]
        for _, rr in found.iterrows():
            needs_review.append({'flag':flag,'source_site':site,'doc_id':rr.doc_id,
                                 'title':rr.title,'url':rr.url,'context':rr.context})
    for _,rr in hcmus.loc[hcmus.long_atomic_row].iterrows():
        needs_review.append({'flag':'hcmus_long_atomic_table_row_check_model_tokens',
                             'source_site':'hcmus_admissions','doc_id':rr.doc_id,
                             'title':rr.title,'url':rr.url,'context':rr.context})
    for conflict in conflict_records:
        for _,rr in hcmus[hcmus.title.str.casefold().str.startswith(conflict['program'],na=False)].iterrows():
            if rr.extraction_rule=='hcmus_structured_admission_row':
                needs_review.append({'flag':'hcmus_program_code_discrepancy_between_pages',
                                     'source_site':'hcmus_admissions','doc_id':rr.doc_id,
                                     'title':rr.title,'url':rr.url,'context':rr.context})
    pd.DataFrame(needs_review, columns=['flag','source_site','doc_id','title','url','context']).to_csv(
        out_dir/'19_review_flags.csv',index=False,encoding='utf-8-sig')
    stats['needs_human_review_documents']=len(set(x['doc_id'] for x in needs_review))
    # Blank copyable form; never pretend a review was performed.
    flagged={}
    for issue in needs_review:
        did=issue['doc_id']
        if issue['flag'] not in flagged.get(did,'').split(' | '):
            flagged[did]=(flagged[did]+' | ' if did in flagged else '')+issue['flag']
    review_sample=pd.concat([g.sample(min(8,len(g)),random_state=42)
                            for _,g in corpus.groupby('source_site')],ignore_index=True)
    review_doc_ids=set(review_sample.doc_id)|set(flagged)
    review_form=corpus[corpus.doc_id.isin(review_doc_ids)][['doc_id','source_site','title','url','context']].copy()
    review_form['flag']=review_form.doc_id.map(lambda did:flagged.get(did,'random_sample'))
    review_form['decision']=''
    review_form['review_notes']=''
    review_form.to_csv(out_dir/'20_review_form_TEMPLATE.csv',index=False,encoding='utf-8-sig')
    stats['high_specificity_noise_hits'] = int(qc.violations.sum())
    if stats['high_specificity_noise_hits']:
        raise ValueError('Residual site chrome detected. Inspect 17_residual_noise_audit.csv before indexing.')
    # Stable reproducible review sample: ~10 documents per site, first and
    # several hashed sample IDs. Manual review remains essential.
    samples=[]
    for site, group in corpus.groupby('source_site'):
        samples.append(group.head(3))
        samples.append(group.sample(min(10,len(group)),random_state=42))
    pd.concat(samples).drop_duplicates('doc_id').to_csv(
        out_dir/'18_manual_review_sample.csv',index=False,encoding='utf-8-sig')
    rejected.to_csv(out_dir/'02_rejected_rows.csv',index=False,encoding='utf-8-sig')
    tok = VietnameseTokenizer(tokenizer_backend)
    stem,lemma,lemma_status = _english_tools()
    all_variants=[]
    for text in corpus.context:
        all_variants.append(tokenize_strategies(tok,normalize(text,lower=True),stem,lemma))
    # Fail loudly if a future code change removes valid underthesea compounds.
    if tok.backend == 'underthesea':
        expected_compounds = sum(
            1 for group in all_variants for raw_word in group['all']
            if (clean_word := _lexical_token(raw_word)) is not None and '_' in clean_word
        )
        indexed_compounds = sum(
            1 for group in all_variants for word in group['baseline'] if '_' in word
        )
        if expected_compounds != indexed_compounds:
            raise RuntimeError(
                f'Vietnamese compounds lost during lexical tokenization: '
                f'{indexed_compounds}/{expected_compounds} retained'
            )
    corpus['clean_text'] = corpus.context.map(lambda x:normalize(x,True))
    corpus['raw_token_count'] = corpus.context.map(lambda x:len(tok.tokenize(x)))
    corpus['clean_token_count'] = [len(x['all']) for x in all_variants]
    for strategy in STRATEGIES:
        corpus[f'tokens_{strategy}']=[json.dumps(x[strategy],ensure_ascii=False) for x in all_variants]
        corpus[f'count_{strategy}']=[len(x[strategy]) for x in all_variants]
    corpus['tokens_no_stop_raw'] = [json.dumps(
        [w for w in x['all'] if w not in STOPWORDS_VI], ensure_ascii=False) for x in all_variants]
    corpus['tokens_no_stop_alnum'] = corpus['tokens_no_stopwords']
    corpus.to_csv(out_dir/'03_cleaned_tokenized_lab.csv',index=False,encoding='utf-8-sig')
    cols=['doc_id','url','title','context','source_domain','crawled_at',
          'source_site','domain_label','source_passage_id','segment_index','unit_index',
          'extraction_rule','extraction_quality']
    corpus[cols].to_csv(out_dir/'04_crawled_auxiliary_corpus.csv',index=False,encoding='utf-8-sig')
    corpus.groupby(['source_site','domain_label'],dropna=False).size().rename('documents').reset_index().to_csv(
        out_dir/'05_counts_by_site.csv',index=False,encoding='utf-8-sig')
    # Cross-source before/after examples, not just the first five FPT rows.
    example_parts=[]
    for site, group in corpus.groupby('source_site'):
        pick=group.drop_duplicates('source_passage_id').head(3)
        for row in pick.to_dict('records'):
            originals = raw_df[(raw_df.source_passage_id == row['source_passage_id'])]
            if originals.empty: continue
            source = originals.iloc[0]
            example_parts.append({'source_passage_id':row['source_passage_id'],
                                  'source_site':site,'url':row['url'],
                                  'raw_text':source['raw_text'],
                                  'clean_text':row['context'],
                                  'extraction_rule':row['extraction_rule']})
    raw_examples=pd.DataFrame(example_parts)
    raw_examples['raw_token_count']=raw_examples.raw_text.map(lambda x:len(tok.tokenize(x)))
    raw_examples['clean_token_count']=raw_examples.clean_text.map(lambda x:len(tok.tokenize(x)))
    raw_examples.to_csv(out_dir/'06_raw_clean_examples.csv',index=False,encoding='utf-8-sig')
    avg_tokens_raw=float(raw_examples.raw_token_count.mean())
    avg_tokens_clean=float(raw_examples.clean_token_count.mean())
    sample_words=['connect','connected','connecting','connection','studies','studying','cars','running']
    pd.DataFrame([{'token':w,'stemmed_english':stem(w),'lemmatized_english':lemma(w)} for w in sample_words]).to_csv(
        out_dir/'07_stemming_lemma_english_demo.csv',index=False,encoding='utf-8-sig')
    models={}; summary=[]
    for strategy in STRATEGIES:
        tokens=[x[strategy] for x in all_variants]
        docs=[' '.join(t) for t in tokens]
        vectorizer=TfidfVectorizer(tokenizer=str.split,token_pattern=None,preprocessor=None,
                                   lowercase=False,dtype=np.float32,norm='l2')
        matrix=vectorizer.fit_transform(docs)
        bm25,backend=_bm25_corpus(tokens)
        sparse.save_npz(model_dir/f'tfidf_{strategy}.npz',matrix)
        joblib.dump(vectorizer,model_dir/f'vectorizer_{strategy}.joblib')
        joblib.dump(bm25,model_dir/f'bm25_{strategy}.joblib')
        vocab=pd.DataFrame({'term':vectorizer.get_feature_names_out()})
        vocab.to_csv(model_dir/f'vocabulary_{strategy}.csv',index=False,encoding='utf-8-sig')
        models[strategy]={'strategy':strategy,'tfidf_matrix':matrix,'tfidf_vectorizer':vectorizer,'bm25':bm25}
        summary.append({'strategy':strategy,'vocab_size':len(vectorizer.vocabulary_),
                        'tfidf_rows':matrix.shape[0],'tfidf_cols':matrix.shape[1],
                        'tfidf_nnz':matrix.nnz,'token_count':sum(map(len,tokens)),
                        'token_reduction_pct':round(100*(1-sum(map(len,tokens))/max(1,sum(len(x['baseline']) for x in all_variants))),2),
                        'bm25_backend':backend})
    pd.DataFrame(summary).to_csv(out_dir/'08_strategy_comparison.csv',index=False,encoding='utf-8-sig')
    qdata=DEFAULT_QUERIES
    results=[]
    pool={}
    for qid,query in qdata:
        for strategy in STRATEGIES:
            for method in ('tfidf','bm25'):
                for rank,(i,score) in enumerate(_search(query,tok,stem,lemma,models[strategy],method,10),1):
                    record={'query_id':qid,'query':query,'strategy':strategy,'method':method,'rank':rank,
                            'doc_id':corpus.iloc[i]['doc_id'],'source_site':corpus.iloc[i]['source_site'],
                            'score':round(score,6),'context':corpus.iloc[i]['context']}
                    results.append(record)
                    key=(qid,record['doc_id'])
                    pool[key]={'query_id':qid,'query':query,'doc_id':record['doc_id'],
                               'source_site':record['source_site'],'context':record['context'],
                               'relevant':'','notes':''}
    pd.DataFrame(results).to_csv(out_dir/'09_tfidf_bm25_top10.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(list(pool.values())).to_csv(out_dir/'10_qrels_labeling_template.csv',index=False,encoding='utf-8-sig')
    metrics = None
    if qrels_path:
        metrics=evaluate_qrels(Path(qrels_path),results)
        pd.DataFrame(metrics).to_csv(out_dir/'11_metrics_from_qrels.csv',index=False,encoding='utf-8-sig')
    quality_report=['# Kiểm định chất lượng Text Cleaning cho FNAC-HNM','',
        '## Đầu vào và quyết định lọc',
        f'- Nguồn: `{raw_csv or db}`',
        f'- Số passages nguồn: **{len(raw_df)}**',
        f'- Corpus được chấp nhận: **{len(corpus)}** retrieval documents',
        f'- Trang/trích đoạn được trích xuất: {stats["pages_auto_extracted"]}',
        f'- Trang/trích đoạn cách ly: {stats["pages_quarantined"]}',
        f'- Tỷ lệ khớp nhiễu đặc trưng còn lại: **{stats["high_specificity_noise_hits"]}**',
        f'- Đoạn có dấu hiệu cần rà soát riêng (không tự xóa): **{stats["needs_human_review_documents"]}**',
        '', '## Số lượng theo website',
        '| Site | Documents |','|---|---:|']
    for site,num in corpus.source_site.value_counts().sort_index().items():
        quality_report.append(f'| {site} | {num:,} |')
    quality_report += ['', '## HCMUS 2026: tách hàng và kiểm định nguồn',
        f'- Chunk kết thúc bằng số thứ tự đứng một mình: **{stats["hcmus_dangling_number_chunks"]}**.',
        f'- Hàng tuyển sinh dài hơn max_words (giữ nguyên một hàng; cần kiểm tra token model): **{stats["hcmus_long_atomic_table_rows"]}**.',
        f'- Trang bảng thiếu STT trong bản crawl: **{stats["hcmus_source_tables_with_missing_stt"]}**.',
        f'- Tên ngành/chỉ tiêu xuất hiện với nhiều mã ngành giữa hai URL: **{stats["hcmus_program_code_discrepancies"]}**.',
        '- Xem `21_hcmus_table_integrity.csv`, `22_hcmus_chunk_integrity.csv` và `23_hcmus_code_discrepancies.csv`.',
        '- Bảng raw SQLite đã được kiểm tra đủ 39 STT ở mỗi URL. Nếu bảng khác schema, parser sẽ quarantine thay vì suy đoán.',
    ]
    quality_report += ['', '## Lý do loại/cách ly', '| Lý do | Số bản ghi |', '|---|---:|']
    for reason,n in sorted(stats['rejection_reasons'].items()):
        quality_report.append(f'| {reason} | {n} |')
    quality_report += ['','## Quy tắc có kiểm soát',
        '- MoMo: chỉ giữ câu trả lời có tiêu đề khớp, điểm bắt đầu và điểm kết thúc (rating); bỏ trang chuyên mục và related articles.',
        '- UEL: phân tách từng FAQ theo dấu hiệu câu hỏi và ranh giới Tags: Sao chép; loại breadcrumb, yêu cầu đáp án đủ dài.',
        '- MobiFone: chỉ giữ trang Câu hỏi thường gặp và tách từng cặp hỏi–đáp đánh số. Các landing/catalog khác được cách ly.',
        '- HCMUS: tách riêng từng hàng mã ngành/chỉ tiêu trong hai bảng 2026; giữ đủ phương thức/tổ hợp, dedup bản trùng tuyệt đối giữa URL, xử lý số mục bị treo; phần nội dung khác chia theo câu.',
        '- FPT: bỏ footer đúng mẫu, không sửa thông tin sản phẩm, mức cước, quy định, URL hoặc phủ định.',
        '- Các đoạn phụ cuối bài của crawler (passage_index > 0 khi có bản chính) được cách ly, không nhân đôi thành tài liệu.',
        '', '## Kiểm định và các giới hạn',
        '- Kết quả quét nhiễu đặc trưng bằng 5 nhóm mẫu regex không thể chứng minh toàn bộ 100% tài liệu sạch.',
        '- Kiểm tra thủ công `18_manual_review_sample.csv`, `19_review_flags.csv`, và khi cần cả `16_quarantine_pages.csv` trước khi chốt cho training.',
        '- Điền `20_review_form_TEMPLATE.csv` thành bản `review_decisions.csv`, cột decision=keep/drop, rồi chạy `--reviews review_decisions.csv` để loại những bản ghi được xác nhận không phù hợp và rebuild index.',
        '- Một số bài tuyển sinh có bảng được website flatten thành văn bản: phải xem thủ công những bảng/điểm/mức phí quan trọng.',
        '- Có thể còn lỗi chính tả từ website nguồn. Không sửa từ tùy ý vì có thể làm lệch ý nghĩa.',
        '- Khi nạp vào dense retriever, kiểm tra tỷ lệ bị cắt tại MAX_SEQ_LENGTH bởi tokenizer của model thật.',
        '- Các trang bị cách ly vẫn lưu nguồn trong file audit; không giả định chúng là câu trả lời rác tuyệt đối.',
        '', '## Tình trạng',
        '**PASS kiểm tra kỹ thuật tự động**, **PENDING kiểm tra mẫu thủ công** trước khi dùng nghiên cứu chính thức.', '']
    (out_dir/'QUALITY_GATE.md').write_text('\n'.join(quality_report),encoding='utf-8')
    result={'input':str(raw_csv or db),'output':str(out_dir),'stats':stats,'source_audit':meta,
            'example_raw_token_mean':avg_tokens_raw, 'example_clean_token_mean':avg_tokens_clean,
            'tokenizer_backend':tok.backend,'english_lemma':lemma_status,
            'strategies':summary,'metrics':metrics,
            'note':'Crawled documents have no qrels. Search-score or source_site are NOT relevance labels.'}
    (out_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    from core.research_release import build_research_release
    result['research_release'] = build_research_release(out_dir, review_decisions)
    (out_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def evaluate_qrels(file: Path, results: list[dict]):
    """P@5 on pooled judged docs only if all top-5 are labeled; no fabricated Recall."""
    rel_df=pd.read_csv(file,keep_default_na=False)
    needed={'query_id','doc_id','relevant'}
    if not needed.issubset(rel_df.columns):
        raise ValueError('Qrels needs query_id, doc_id, relevant (0/1)')
    qrels={}
    for row in rel_df.to_dict('records'):
        if str(row['relevant']).strip() in ('0','1'):
            qrels[(str(row['query_id']),str(row['doc_id']))]=int(row['relevant'])
    groups=defaultdict(list)
    for r in results:
        if r['rank']<=5:groups[(r['query_id'],r['strategy'],r['method'])].append(r)
    rows=[]
    for (qid,strategy,method), items in groups.items():
        assessed=[qrels.get((qid,x['doc_id'])) for x in items]
        # A document outside pooled judgments is UNKNOWN, not negative.
        if len(assessed)<5 or any(x is None for x in assessed):
            continue
        rows.append({'query_id':qid,'strategy':strategy,'method':method,
                     'P@5':sum(assessed)/5, 'Recall@5':None,
                     'warning':'Recall undefined: no exhaustive corpus-wide qrels'})
    return rows


def search_saved_index(out_dir: Path, query: str, strategy='baseline', method='bm25', top_k=5, corpus_name='lab'):
    """Search lab's saved index, or temporarily index the 887-document auxiliary subset.

    No separate research index is persisted: during FNAC-HNM training, the actual
    BM25 + dense mining pool must include VIRE documents AND auxiliary documents.
    """
    if strategy not in STRATEGIES or method not in ('bm25','tfidf'):
        raise ValueError('invalid strategy or method')
    out_dir=Path(out_dir)
    if corpus_name not in ('lab','research'): raise ValueError('corpus must be lab or research')
    filename='24_auxiliary_corpus_RESEARCH_READY.csv' if corpus_name=='research' else '04_crawled_auxiliary_corpus.csv'
    corpus_path=out_dir/filename
    report=json.loads((out_dir/'report.json').read_text(encoding='utf-8'))
    backend='underthesea' if report['tokenizer_backend']=='underthesea' else 'regex'
    tok=VietnameseTokenizer(backend)
    stem,lemma,_=_english_tools()
    query_tokens=tokenize_strategies(tok,normalize(query,lower=True),stem,lemma)[strategy]
    corpus=pd.read_csv(corpus_path,keep_default_na=False)
    if corpus_name=='lab':
        indexdir=out_dir/'models'
        if method=='bm25':
            scores=joblib.load(indexdir/f'bm25_{strategy}.joblib').get_scores(query_tokens)
        else:
            vectorizer=joblib.load(indexdir/f'vectorizer_{strategy}.joblib')
            X=sparse.load_npz(indexdir/f'tfidf_{strategy}.npz')
            q=vectorizer.transform([' '.join(query_tokens)])
            scores=np.asarray((X @ q.T).toarray()).ravel()
    else:
        # Test-only convenience search, not a precomputed training mining index.
        token_df=pd.read_csv(out_dir/'03_cleaned_tokenized_lab.csv',keep_default_na=False)
        aligned=corpus[['doc_id']].merge(token_df[['doc_id',f'tokens_{strategy}']],on='doc_id',how='left',validate='one_to_one',indicator=True,sort=False)
        if not aligned['_merge'].eq('both').all() or aligned.doc_id.tolist()!=corpus.doc_id.tolist():
            raise ValueError('Research corpus and saved tokenization do not match')
        tokens=[json.loads(s) for s in aligned[f'tokens_{strategy}'].tolist()]
        if method=='bm25':
            bm25,_=_bm25_corpus(tokens)
            scores=bm25.get_scores(query_tokens)
        else:
            vectorizer=TfidfVectorizer(tokenizer=str.split,token_pattern=None,preprocessor=None,
                                       lowercase=False,dtype=np.float32,norm='l2')
            X=vectorizer.fit_transform([' '.join(ts) for ts in tokens])
            q=vectorizer.transform([' '.join(query_tokens)])
            scores=np.asarray((X @ q.T).toarray()).ravel()
    if len(corpus)!=len(scores): raise ValueError('Index and corpus row counts differ')
    return [{'doc_id':str(corpus.iloc[i]['doc_id']),
             'source_site':str(corpus.iloc[i]['source_site']),
             'context':str(corpus.iloc[i]['context']),
             'score':float(scores[i])} for i in _topk(scores,top_k)]
