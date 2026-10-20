"""SEG301 retrieval evaluation solely on the five websites crawled by the group.

PREPARE creates a FIXED, small, stratified document corpus; draft queries; and
an EMPTY (unjudged) exhaustive query/document relevance worksheet. Researchers
must label EVERY document for EVERY query before EVALUATE will compute true
Precision@5 and Recall@5 on this EVALUATION SUBSET.

No ViRE/CSConDa/EduCoQA data; evaluation may use explicitly declared AI-assisted relevance drafts rather than passing them off as human gold labels.
The 887-document research corpus is not overwritten or assigned labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from core.retrieval_lab import VietnameseTokenizer, STRATEGIES, _english_tools, _bm25_corpus, normalize, tokenize_strategies, _topk

ENCODING = 'utf-8-sig'
CORPUS_DEFAULT = Path('data/preprocessing/24_auxiliary_corpus_RESEARCH_READY.csv')
OUT_DEFAULT = Path('data/preprocessing/crawl_eval')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f'Not found: {path}')
    return pd.read_csv(path, keep_default_na=False, encoding=ENCODING, dtype=str)


def prepare(corpus_path: Path = CORPUS_DEFAULT, out: Path = OUT_DEFAULT,
            docs_per_site: int = 10, queries_per_site: int = 2, seed: int = 42) -> dict:
    """Creates an evaluable, stratified benchmark subset; NEVER invents labels."""
    source = _read(corpus_path)
    required = {'doc_id','context','title','source_site','url'}
    if not required.issubset(source.columns):
        raise ValueError(f'Corpus missing fields: {sorted(required-set(source.columns))}')
    if source.doc_id.duplicated().any() or source.context.str.strip().eq('').any():
        raise ValueError('Invalid corpus: duplicate doc_id or empty context')
    if docs_per_site < 5 or not 1 <= queries_per_site <= docs_per_site:
        raise ValueError('docs_per_site >= 5; queries_per_site within 1..docs_per_site')
    out.mkdir(parents=True,exist_ok=True)
    # Avoid silently destroying the independently reviewed evaluation worksheet.
    if (out/'qrels_REVIEWED.csv').is_file():
        raise FileExistsError(
            f'{out}/qrels_REVIEWED.csv already exists. '
            'PREPARE would regenerate questions and documents and invalidate these labels. '
            'Back up the entire crawl_eval folder before intentionally rebuilding it.'
        )
    sampled=[]
    for site, part in source.groupby('source_site', sort=True):
        if len(part)<docs_per_site:
            raise ValueError(f'{site}: only {len(part)} docs; choose --docs-per-site <= {len(part)}')
        # Deterministic per-site sample; no synthetic documents.
        sampled.append(part.sample(n=docs_per_site,random_state=seed).sort_values('doc_id'))
    docs=pd.concat(sampled,ignore_index=True).sort_values(['source_site','doc_id']).reset_index(drop=True)
    queries=[]
    for site,part in docs.groupby('source_site',sort=True):
        # Distinct titles where possible; if a page was chunked into repeated-title
        # pieces, choose one question from that title and request human rewrite.
        chosen=[]
        titles=set()
        for r in part.itertuples(index=False):
            name=normalize(str(r.title))
            if name.lower() not in titles:
                titles.add(name.lower()); chosen.append(r)
            if len(chosen)==queries_per_site: break
        if len(chosen)<queries_per_site:
            chosen.extend([r for r in part.itertuples(index=False) if r.doc_id not in {v.doc_id for v in chosen}][:queries_per_site-len(chosen)])
        for idx,r in enumerate(chosen,1):
            query=str(r.title).strip()
            if site=='hcmus_admissions':
                import re
                yr=re.search(r'20\d{2}', query)
                if 'PHƯƠNG THỨC XÉT TUYỂN' in query.upper():
                    query=f'HCMUS có những phương thức xét tuyển nào năm {yr.group(0) if yr else "gần đây"}?'
                elif 'chỉ tiêu' in query.lower():
                    query=f'Chỉ tiêu tuyển sinh của HCMUS cho {query} là bao nhiêu?'
            queries.append({'query_id':f'{site}_q{idx:02d}', 'query':query,
                            'source_site':site, 'source_doc_id_for_drafting':str(r.doc_id),
                            'review_note':'Rewrite query into a natural question if title is too broad. The source document is NOT automatically labeled relevant.'})
    qdf=pd.DataFrame(queries)
    # Each of the K queries is judged against every document in the fixed Kdoc benchmark.
    labels=pd.DataFrame([
        {'query_id':str(q.query_id),'query':str(q.query),
         'doc_id':str(d.doc_id),'document_site':str(d.source_site),
         'document_title':str(d.title),'document_context':str(d.context),
         'relevant':'','annotator':'','note':''}
        for q in qdf.itertuples(index=False) for d in docs.itertuples(index=False)
    ])
    docs.to_csv(out/'eval_documents.csv',index=False,encoding=ENCODING)
    qdf.to_csv(out/'eval_queries.csv',index=False,encoding=ENCODING)
    labels.to_csv(out/'qrels_TO_BE_LABELED.csv',index=False,encoding=ENCODING)
    metadata={'scope':'5-site crawled corpus ONLY','type':'stratified fixed evaluation subset',
              'original_corpus_path':str(corpus_path),'original_corpus_sha256':_sha(corpus_path),
              'evaluation_doc_count':len(docs),'evaluation_query_count':len(qdf),
              'judgments_required':len(labels),'doc_count_by_site':docs.source_site.value_counts().sort_index().to_dict(),
              'seed':seed,'labels_status':'ALL BLANK: not evaluated; no relevance scores are valid yet',
              'validity':'When ALL pairs are judged 0/1, P@5 and Recall@5 apply ONLY to this fixed evaluation subset, NOT to the full original corpus.'}
    (out/'EVALUATION_PROTOCOL.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def evaluate(out: Path = OUT_DEFAULT, labels_path: Path|None=None,
             tokenizer_backend: str='underthesea') -> dict:
    out=Path(out)
    docs=_read(out/'eval_documents.csv')
    queries=_read(out/'eval_queries.csv')
    labels_path=Path(labels_path) if labels_path else out/'qrels_REVIEWED.csv'
    labels=_read(labels_path)
    required={'query_id','query','doc_id','relevant'}
    if not required.issubset(labels.columns):
        raise ValueError('Reviewed qrels must include query_id, doc_id and relevant')
    # A valid Recall denominator requires EXHAUSTIVE judgments for the chosen corpus.
    expected={(q,d) for q in queries.query_id for d in docs.doc_id}
    observed=list(zip(labels.query_id,labels.doc_id))
    if len(observed)!=len(set(observed)):
        raise ValueError('Duplicate query_id/doc_id pair in reviewed labels')
    if set(observed)!=expected:
        raise ValueError(f'Incomplete/foreign qrels: missing {len(expected-set(observed))}; extra {len(set(observed)-expected)}')
    stripped=labels.relevant.astype(str).str.strip()
    if not stripped.isin(['0','1']).all():
        remaining=int((~stripped.isin(['0','1'])).sum())
        raise ValueError(f'{remaining} unjudged pairs. Label ALL pairs with 0 or 1; do not fake a Recall denominator.')
    # The reviewed sheet owns the question text. This permits human rewriting
    # the automatically drafted query, but all copied rows must agree.
    revised=labels.groupby('query_id')['query'].agg(lambda x:set(x.astype(str).str.strip()))
    if any(len(x)!=1 or not next(iter(x)) for x in revised):
        raise ValueError('One query_id has inconsistent or blank text. Update the query in ALL its worksheet rows.')
    qtext={qid:next(iter(values)) for qid,values in revised.items()}
    queries=queries.copy()
    queries['query']=queries.query_id.map(qtext)
    labels['rel']=stripped.astype(int)
    per_query=labels.groupby('query_id')['rel'].sum().to_dict()
    empty=[q for q in queries.query_id if per_query.get(q,0)==0]
    if empty:
        raise ValueError(f'No positive document in evaluation set for {len(empty)} queries: {empty[:3]}. Edit query or expand docs; do not force fake labels.')
    lab={(str(r.query_id),str(r.doc_id)):int(r.rel) for r in labels.itertuples(index=False)}
    tok=VietnameseTokenizer(tokenizer_backend)
    stem,lemma,_=_english_tools()
    doc_tok=[tokenize_strategies(tok,normalize(x,True),stem,lemma) for x in docs.context]
    q_tok=[tokenize_strategies(tok,normalize(x,True),stem,lemma) for x in queries['query']]
    details=[];rankings=[]
    for strategy in STRATEGIES:
        td=[x[strategy] for x in doc_tok]
        vec=TfidfVectorizer(tokenizer=str.split, token_pattern=None, preprocessor=None,
                            lowercase=False,norm='l2',dtype=np.float32)
        X=vec.fit_transform([' '.join(z) for z in td])
        bm25,_=_bm25_corpus(td)
        for qi,q in queries.reset_index(drop=True).iterrows():
            qt=q_tok[qi][strategy]
            qvec=vec.transform([' '.join(qt)])
            for method,scores in (('tfidf',np.asarray((X@qvec.T).toarray()).ravel()),
                                  ('bm25',np.asarray(bm25.get_scores(qt)))):
                # No positive signal -> no meaningful ranking/metric.
                if qvec.nnz==0:
                    details.append({'query_id':q.query_id,'strategy':strategy,'method':method,
                                    'Precision@5':np.nan,'Recall@5':np.nan,'Hit@1':np.nan,
                                    'MRR@5':np.nan,'nDCG@5':np.nan,
                                    'relevant_total':per_query[q.query_id],
                                    'eval_status':'OOV QUERY: no shared vocabulary'})
                    continue
                indices=_topk(scores,5)
                hits=0
                relevant_positions=[]
                for rank,di in enumerate(indices,1):
                    did=str(docs.iloc[di].doc_id);rel=lab[(str(q.query_id),did)];hits+=rel
                    if rel:
                        relevant_positions.append(rank)
                    rankings.append({'query_id':q.query_id,'query':q['query'],'strategy':strategy,
                                     'method':method,'rank':rank,'doc_id':did,'relevant':rel,
                                     'score':float(scores[di]),'title':str(docs.iloc[di].title)})
                dcg=sum(1.0/np.log2(pos+1) for pos in relevant_positions)
                ideal_dcg=sum(1.0/np.log2(pos+1) for pos in range(1,min(5,per_query[q.query_id])+1))
                details.append({'query_id':q.query_id,'strategy':strategy,'method':method,
                                'Precision@5':hits/5.0,'Recall@5':hits/per_query[q.query_id],
                                'Hit@1':int(1 in relevant_positions),
                                'MRR@5':(1.0/min(relevant_positions) if relevant_positions else 0.0),
                                'nDCG@5':(dcg/ideal_dcg if ideal_dcg else float('nan')),
                                'relevant_total':per_query[q.query_id], 'eval_status':'EXHAUSTIVELY JUDGED SUBSET'})
    detail_df=pd.DataFrame(details)
    summary=detail_df.groupby(['strategy','method'],as_index=False).agg(
        P_at_5=('Precision@5','mean'),Recall_at_5=('Recall@5','mean'),
        Hit_at_1=('Hit@1','mean'),MRR_at_5=('MRR@5','mean'),nDCG_at_5=('nDCG@5','mean'),
        queries_scored=('Precision@5','count'),queries_total=('query_id','count'))
    detail_df.to_csv(out/'per_query_metrics.csv',index=False,encoding=ENCODING)
    summary.to_csv(out/'metric_summary.csv',index=False,encoding=ENCODING)
    pd.DataFrame(rankings).to_csv(out/'top5_rankings.csv',index=False,encoding=ENCODING)
    protocol=json.loads((out/'EVALUATION_PROTOCOL.json').read_text(encoding='utf-8'))
    has_ai_labels=labels.get('annotator', pd.Series('',index=labels.index)).astype(str).str.contains('AI',case=False,regex=False).any()
    label_origin=('AI-assisted; NOT independently human-verified' if has_ai_labels else 'Annotator labels from supplied qrels')
    protocol.update({'label_file_sha256':_sha(labels_path),'status':'AI_ASSISTED_PRELIMINARY' if has_ai_labels else 'LABELED SUBSET EVALUATED',
       'annotation_provenance':label_origin,
       'scoring':'Exhaustive 0/1 judgments for all query-doc pairs inside the fixed subset; each query has >=1 relevant document.',
       'scope_warning':'Scores VALID for fixed crawl evaluation subset ONLY; NOT for all 887 or 889 documents.'})
    (out/'EVALUATION_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    table=summary.round(4).to_markdown(index=False)
    lines=['# SEG301 | P@5 / Recall@5 chỉ trên dữ liệu crawl',
           '',f'- Corpus: **{len(docs)} tài liệu** được lấy từ đúng 5 website.',
           f'- Queries: **{len(queries)} câu hỏi**, mỗi câu được chấm với **toàn bộ {len(docs)} tài liệu**.',
           f'- Judgments: **{len(labels)} nhãn 0/1**. Nguồn nhãn: **{label_origin}**.',
           '- TF-IDF/BM25 dùng cùng tập corpus, queries và nhãn; đánh giá 4 chiến lược preprocessing.',
           '- **Giới hạn:** chỉ phản ánh tập mẫu, không được gọi là Recall@5 trên toàn bộ 887 tài liệu.',
           '- **Lưu ý kiểm định:** câu hỏi được xây dựng từ nội dung chính corpus đánh giá, nên dù diễn đạt lại và có đối chứng dễ nhầm, kết quả có thể lạc quan. Nhãn AI là kết quả sơ bộ, nên rà lại các ca được đánh dấu nếu công bố chính thức.',
           '', '## Kết quả', '',table,'',
           '## Công thức', '- Precision@5 = số tài liệu relevant ở Top-5 / 5.',
           '- Recall@5 = số relevant ở Top-5 / tổng relevant đã gán trong corpus đánh giá cố định.',
           '- Hit@1 = 1 nếu top-1 relevant, ngược lại 0; trung bình trên các queries.',
           '- MRR@5 = 1/vị trí relevant đầu tiên trong top-5 (không có thì 0), trung bình queries.',
           '- nDCG@5 = DCG chuẩn hóa theo thứ hạng lý tưởng, dùng relevance nhị phân và log2.',
           '', '## Bằng chứng', '`eval_documents.csv`, `eval_queries.csv`, `qrels_REVIEWED.csv`, `per_query_metrics.csv`, `top5_rankings.csv`, `EVALUATION_PROTOCOL.json`.']
    (out/'REPORT_CRAWL_EVALUATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'subset_docs':len(docs),'queries':len(queries),
            'judgments_total':len(labels),
            'ai_assisted_judgments':len(labels) if has_ai_labels else 0,
            'human_verified_judgments':0 if has_ai_labels else len(labels),
            'scored_combinations':len(summary),'report':str(out/'REPORT_CRAWL_EVALUATION.md')}


def main():
    parser=argparse.ArgumentParser(description='Crawl-only relevance evaluation, with human exhaustive labels for a fixed sample')
    subs=parser.add_subparsers(dest='action',required=True)
    p1=subs.add_parser('prepare',help='Create unjudged documents, queries and labeling worksheet')
    p1.add_argument('--corpus',type=Path,default=CORPUS_DEFAULT)
    p1.add_argument('--out',type=Path,default=OUT_DEFAULT)
    p1.add_argument('--docs-per-site',type=int,default=10)
    p1.add_argument('--queries-per-site',type=int,default=2)
    p1.add_argument('--seed',type=int,default=42)
    p2=subs.add_parser('evaluate',help='Only run after human review of ALL query-doc pairs')
    p2.add_argument('--out',type=Path,default=OUT_DEFAULT)
    p2.add_argument('--labels',type=Path,default=None)
    p2.add_argument('--tokenizer',choices=['underthesea','regex','auto'],default='underthesea')
    args=parser.parse_args()
    result=(prepare(args.corpus,args.out,args.docs_per_site,args.queries_per_site,args.seed)
            if args.action=='prepare' else evaluate(args.out,args.labels,args.tokenizer))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
