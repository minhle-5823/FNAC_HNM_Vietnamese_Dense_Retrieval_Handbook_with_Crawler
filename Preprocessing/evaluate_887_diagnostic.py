"""Supplementary *partial-label* diagnostic on all crawled 887 documents.

WARNING: This is NOT Precision@K or Recall@K. Only known-positive documents
from the 50-doc manually inspected subset (AI-assisted preliminary labels) are
used. Other documents among the 887 are UNKNOWN, not judged negatives.
No CSConDa / EduCoQA data is used.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from core.retrieval_lab import VietnameseTokenizer, STRATEGIES, _english_tools, _bm25_corpus, normalize, tokenize_strategies, _topk

BASE=Path('data/preprocessing')
DEFAULT_CORPUS=BASE/'24_auxiliary_corpus_RESEARCH_READY.csv'
DEFAULT_EVAL=BASE/'crawl_eval'


def main(corpus_path:Path=DEFAULT_CORPUS, eval_path:Path=DEFAULT_EVAL, tokenizer_backend:str='underthesea'):
    docs=pd.read_csv(corpus_path,encoding='utf-8-sig',keep_default_na=False)
    queries=pd.read_csv(eval_path/'eval_queries.csv',encoding='utf-8-sig',keep_default_na=False)
    judgments=pd.read_csv(eval_path/'qrels_REVIEWED.csv',encoding='utf-8-sig',keep_default_na=False)
    assert docs.doc_id.is_unique and docs.context.str.strip().ne('').all()
    assert judgments.relevant.astype(str).isin(['0','1']).all()
    known={str(q):set(x.doc_id.astype(str)) for q,x in judgments[judgments.relevant.astype(int)==1].groupby('query_id')}
    docids=set(docs.doc_id.astype(str))
    assert len(queries)>0
    for q in queries.query_id:
        if not known.get(str(q)) or not known[str(q)].issubset(docids):
            raise ValueError(f'Known relevant docs missing for query {q}')
    tok=VietnameseTokenizer(tokenizer_backend)
    stem,lemma,_=_english_tools()
    doc_toks=[tokenize_strategies(tok,normalize(str(t),True),stem,lemma) for t in docs.context]
    query_toks=[tokenize_strategies(tok,normalize(str(q),True),stem,lemma) for q in queries['query']]
    details=[]; ranks=[]
    for strategy in STRATEGIES:
        token_docs=[v[strategy] for v in doc_toks]
        vec=TfidfVectorizer(tokenizer=str.split,token_pattern=None,preprocessor=None,lowercase=False,norm='l2',dtype=np.float32)
        X=vec.fit_transform([' '.join(v) for v in token_docs])
        bm25,_=_bm25_corpus(token_docs)
        for qi,qr in queries.reset_index(drop=True).iterrows():
            qid=str(qr.query_id); qtok=query_toks[qi][strategy]
            qvec=vec.transform([' '.join(qtok)])
            for method, scores in [('tfidf',np.asarray((X@qvec.T).toarray()).ravel()),('bm25',np.asarray(bm25.get_scores(qtok)))]:
                if qvec.nnz==0:
                    details.append({'strategy':strategy,'method':method,'query_id':qid,'known_positive_count':len(known[qid]),'first_known_positive_rank':np.nan,
                     'AnyKnownPositive_at_1':np.nan,'AnyKnownPositive_at_5':np.nan,'AnyKnownPositive_at_10':np.nan,'KnownPositiveCoverage_at_5':np.nan,'OOV':True})
                    continue
                # Full ranking is 887 rows. This is NOT a true binary-label ranking because other documents are unjudged.
                order=np.argsort(-scores,kind='stable')
                first=next((i+1 for i,di in enumerate(order) if str(docs.iloc[di].doc_id) in known[qid]),None)
                for j,di in enumerate(order[:10],1):
                    d=docs.iloc[di]
                    ranks.append({'query_id':qid,'query':qr['query'],'strategy':strategy,'method':method,'rank':j,'doc_id':d.doc_id,
                        'known_positive':int(str(d.doc_id) in known[qid]),'otherwise':'UNJUDGED; not negative' if str(d.doc_id) not in known[qid] else 'KNOWN POSITIVE',
                        'score':float(scores[di]),'title':d.title})
                cov5=sum(str(docs.iloc[di].doc_id) in known[qid] for di in order[:5])/len(known[qid])
                details.append({'strategy':strategy,'method':method,'query_id':qid,'known_positive_count':len(known[qid]),'first_known_positive_rank':first,
                  'AnyKnownPositive_at_1':int(first<=1),'AnyKnownPositive_at_5':int(first<=5),'AnyKnownPositive_at_10':int(first<=10),
                  'KnownPositiveCoverage_at_5':cov5,'OOV':False})
    detail=pd.DataFrame(details)
    summ=detail.groupby(['strategy','method'],as_index=False).agg(
        AnyKnownPositive_at_1=('AnyKnownPositive_at_1','mean'),
        AnyKnownPositive_at_5=('AnyKnownPositive_at_5','mean'),
        AnyKnownPositive_at_10=('AnyKnownPositive_at_10','mean'),
        KnownPositiveCoverage_at_5=('KnownPositiveCoverage_at_5','mean'),
        mean_first_known_positive_rank=('first_known_positive_rank','mean'),
        queries_scored=('AnyKnownPositive_at_5','count'),oov_queries=('OOV','sum'))
    detail.to_csv(eval_path/'full_887_diagnostic_per_query.csv',index=False,encoding='utf-8-sig')
    summ.to_csv(eval_path/'full_887_diagnostic_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ranks).to_csv(eval_path/'full_887_diagnostic_top10.csv',index=False,encoding='utf-8-sig')
    with (eval_path/'REPORT_FULL_887_DIAGNOSTIC.md').open('w',encoding='utf-8') as f:
        f.write('# Kiểm tra độ khó truy xuất trên 887 tài liệu crawl\n\n')
        f.write('**KHÔNG PHẢI Precision@K/Recall@K**: chỉ biết một phần tài liệu positive từ tập 50 documents. Những tài liệu còn lại **chưa đánh giá**, không được coi là negative.\n\n')
        f.write('Các nhãn known-positive ban đầu do AI hỗ trợ, chưa được con người kiểm chứng độc lập. Kết quả phụ thuộc tokenizer.\n\n')
        f.write(summ.round(4).to_markdown(index=False)+'\n\n')
        f.write('**AnyKnownPositive@5** = tỷ lệ câu hỏi có ít nhất một document được gán positive trong Top-5 trên 887 documents.\n\n')
        f.write('**KnownPositiveCoverage@5** = phần các document known-positive đã tìm thấy ở Top-5; **không phải Recall thật** vì các positive khác trong 887 chưa được gán nhãn.\n\n')
    print('Diagnostic completed: 887-doc candidate set, 40 queries, partial labels. Results in',eval_path)
    return summ

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    p.add_argument('--eval-dir',type=Path,default=DEFAULT_EVAL)
    p.add_argument('--tokenizer',choices=['underthesea','regex','auto'],default='underthesea')
    args=p.parse_args()
    main(args.corpus,args.eval_dir,args.tokenizer)
