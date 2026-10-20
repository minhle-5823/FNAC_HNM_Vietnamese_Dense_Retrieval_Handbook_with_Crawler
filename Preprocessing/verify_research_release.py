"""Reproducibility checks for the SEG301 lab and unlabeled FNAC-HNM auxiliary CSV.

Lab index (889 documents) lives in models/; no standalone 887-document
mining index is needed. Build final BM25 + dense mining with VIRE TRAIN in training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz


def sha(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            hash_.update(b)
    return hash_.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'data/preprocessing')
    args=p.parse_args()
    out=args.out
    report=json.loads((out/'report.json').read_text(encoding='utf8'))
    manifest=json.loads((out/'26_research_release_manifest.json').read_text(encoding='utf8'))
    full=pd.read_csv(out/'04_crawled_auxiliary_corpus.csv',keep_default_na=False)
    aux=pd.read_csv(out/'24_auxiliary_corpus_RESEARCH_READY.csv',keep_default_na=False)
    pending=pd.read_csv(out/'25_pending_manual_review.csv',keep_default_na=False)
    token_df=pd.read_csv(out/'03_cleaned_tokenized_lab.csv',keep_default_na=False)
    problems=[]
    if report.get('tokenizer_backend')!='underthesea': problems.append('Tokenizer report is not underthesea')
    if len(full)!=len(aux)+len(pending): problems.append('Full/research/pending row counts conflict')
    if full.doc_id.duplicated().any() or aux.doc_id.duplicated().any() or full.context.eq('').any() or aux.context.eq('').any():
        problems.append('Missing content or duplicate IDs')
    if set(aux.doc_id)&set(pending.doc_id): problems.append('Pending and research IDs overlap')
    if not set(aux.doc_id).issubset(set(full.doc_id)): problems.append('Research IDs not in full corpus')
    if list(token_df.doc_id)!=list(full.doc_id): problems.append('Token rows do not match full corpus order')
    if sha(out/'04_crawled_auxiliary_corpus.csv')!=manifest['full_corpus_sha256']:
        problems.append('Full-corpus hash differs from manifest')
    if sha(out/'24_auxiliary_corpus_RESEARCH_READY.csv')!=manifest['research_ready_sha256']:
        problems.append('Research-corpus hash differs from manifest')
    compound=sum('_' in w for s in token_df.tokens_baseline for w in json.loads(s))
    if compound==0: problems.append('Vietnamese compound tokens are absent')
    for strategy in ('baseline','no_stopwords','stemmed','lemmatized'):
        X=load_npz(out/'models'/f'tfidf_{strategy}.npz')
        if X.shape[0]!=len(full):problems.append('Lab TF-IDF index mismatch: '+strategy)
    result={'status':'PASS' if not problems else 'FAIL','lab_documents':len(full),
            'auxiliary_documents':len(aux),'pending_review':len(pending),
            'compound_tokens_kept':compound,'standalone_research_index':False,
            'issues':problems}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    (out/'RELEASE_TOKENIZATION_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return 0 if not problems else 2

if __name__=='__main__':raise SystemExit(main())
