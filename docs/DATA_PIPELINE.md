# Data pipeline for VIRE hard-negative mining

## Raw
`raw_pages` giữ nguyên HTML + text BeautifulSoup đã extract để audit.

## Processed document
`preprocessor.py`:
1. Unicode NFC
2. remove zero-width/control
3. whitespace normalization
4. duplicate boilerplate lines
5. Vietnamese-language score
6. domain relevance score
7. quality gate
8. retrieval chunking

## Retrieval passage
Mặc định 180-220 words, overlap 35-40 words tùy profile.
Mỗi chunk gắn `source_site`, `domain_label`, `page_url`, `quality_score`.

## Downstream hard-negative mining
Không nằm trong crawler:
- index `passages`;
- BM25 top-k;
- baseline dense retriever top-k;
- remove known positives;
- false-negative filtering;
- difficulty scoring;
- curriculum semi-hard -> hard.

Crawler KHÔNG tự gán relevance label/ground truth cho web documents.

## Full SQLite merger

- `python main.py crawl-all --group vire_web_corpus --reset` automatically invokes
  the SQLite merger after all five crawlers complete without errors.
- `python main.py merge-dbs` can be run independently when the five site DBs exist.
- Full merged DB: `data/vire_web_corpus_merged.db` (all eight crawler tables,
  including RAW HTML, processed docs, passage chunks, URL states, links,
  robots checks, events, run history).
- Each record receives `origin_site`, `origin_id` while preserving all original
  fields. New global `id` avoids collisions across per-site databases.
- Merge is atomic; missing DBs or changed profile seeds cause an error.
- The older `build-corpus` produces a separate passages-only export unchanged.
