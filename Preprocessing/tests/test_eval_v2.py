"""Regression checks for accuracy of evaluation metric definitions."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from evaluate_crawled_corpus import evaluate,prepare

class EvaluationMetricsV2Test(unittest.TestCase):
    def test_hit_mrr_ndcg_and_review_guard(self):
        sites=['fpt_telecom_support','hcmus_admissions','mobifone_5g_faq','momo_help','uel_admissions_faq']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'corpus.csv';out=root/'crawl_eval'
            rows=[]
            for s in sites:
                for i in range(5):
                    rows.append({'source_site':s,'doc_id':f'{s}_{i}',
                        'title':f'Văn bản {s} {i}', 'url':f'https://example.org/{s}/{i}', 'context':f'{s} giải thích hướng dẫn dịch vụ thông tin {i}'} )
            pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
            prepare(path,out,docs_per_site=5,queries_per_site=1)
            labs=pd.read_csv(out/'qrels_TO_BE_LABELED.csv',keep_default_na=False)
            qs=pd.read_csv(out/'eval_queries.csv')
            gold=dict(zip(qs.query_id,qs.source_doc_id_for_drafting))
            labs['relevant']=[int(did==gold[qid]) for qid,did in zip(labs.query_id,labs.doc_id)]
            labs['annotator']='AI-assisted unit-test-only synthetic'
            labs.to_csv(out/'qrels_REVIEWED.csv',index=False,encoding='utf-8-sig')
            with self.assertRaises(FileExistsError):prepare(path,out,5,1)
            evaluate(out,tokenizer_backend='regex')
            sums=pd.read_csv(out/'metric_summary.csv');det=pd.read_csv(out/'per_query_metrics.csv')
            for k in ['Hit_at_1','MRR_at_5','nDCG_at_5']:
                self.assertTrue(sums[k].between(0,1).all())
            for k in ['Hit@1','MRR@5','nDCG@5']:
                self.assertTrue(det[k].between(0,1).all())
            self.assertTrue(sums.P_at_5.between(0,1).all())

if __name__=='__main__':unittest.main()
