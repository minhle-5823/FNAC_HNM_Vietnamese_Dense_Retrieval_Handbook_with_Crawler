"""Standalone SEG301 preprocessing add-on; does not modify crawler main.py or SQLite.

Usage:
    python preprocess_merged.py --tokenizer underthesea
    python preprocess_merged.py search --query "Chăm sóc khách hàng MoMo"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        ap = argparse.ArgumentParser(description='Query saved TF-IDF/BM25 indices')
        ap.add_argument('search')
        ap.add_argument('--out', default=str(ROOT/'data'/'preprocessing'))
        ap.add_argument('--query', required=True)
        ap.add_argument('--method', choices=['bm25','tfidf'], default='bm25')
        ap.add_argument('--strategy', choices=['baseline','no_stopwords','stemmed','lemmatized'], default='baseline')
        ap.add_argument('--top-k', type=int, default=5)
        ap.add_argument('--corpus', choices=['lab','research'], default='lab')
        a=ap.parse_args()
        from core.retrieval_lab import search_saved_index
        rows=search_saved_index(Path(a.out),a.query,a.strategy,a.method,a.top_k, corpus_name=a.corpus)
        for i,r in enumerate(rows,1):
            print(f"{i}. [{r['score']:.4f}] {r['source_site']} | {r['doc_id']}")
            print('   '+r['context'][:230])
        return
    ap = argparse.ArgumentParser(description='Merged 5-site SQLite -> FNAC-HNM preprocessing corpus and SEG301 lab')
    ap.add_argument('--reviews',help='Optional manually filled review_decisions.csv with doc_id, decision keep/drop')
    ap.add_argument('--raw-csv', help='Use existing 01_export_merged_raw.csv snapshot when merged DB is unavailable')
    ap.add_argument('--db', default=str(ROOT/'data'/'vire_web_corpus_merged.db'))
    ap.add_argument('--out', default=str(ROOT/'data'/'preprocessing'))
    ap.add_argument('--tokenizer', choices=['auto','underthesea','regex'], default='auto')
    ap.add_argument('--min-chars', type=int, default=100)
    ap.add_argument('--max-words', type=int, default=220)
    ap.add_argument('--overlap', type=int, default=30)
    ap.add_argument('--near-threshold', type=float, default=.90)
    ap.add_argument('--qrels', help='Optional hand-reviewed CSV (0/1 per query/document)')
    a=ap.parse_args()
    from core.retrieval_lab import run_lab
    result=run_lab(Path(a.db),Path(a.out),tokenizer_backend=a.tokenizer,
                   min_chars=a.min_chars,max_words=a.max_words,overlap=a.overlap,
                   near_threshold=a.near_threshold,qrels_path=Path(a.qrels) if a.qrels else None,
                   raw_csv=Path(a.raw_csv) if a.raw_csv else None,
                   review_decisions=Path(a.reviews) if a.reviews else None)
    print(json.dumps({'output':result['output'],'stats':result['stats'],
                      'tokenizer':result['tokenizer_backend']},ensure_ascii=False,indent=2))

if __name__=='__main__': cli()
