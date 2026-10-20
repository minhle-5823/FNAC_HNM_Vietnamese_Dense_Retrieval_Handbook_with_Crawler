"""Synthetic samples for testing logic ONLY. They are NEVER research results."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from evaluate_crawled_corpus import prepare, evaluate

SITES=['fpt_telecom_support','hcmus_admissions','mobifone_5g_faq','momo_help','uel_admissions_faq']

class CrawlEvaluationTests(unittest.TestCase):
    def _make(self,root):
        rows=[]
        for site in SITES:
            for i in range(5):
                rows.append({'doc_id':f'{site}_{i}','source_site':site,'url':f'https://example.com/{site}/{i}',
                             'title':f'Hướng dẫn {site} dịch vụ số {i}?',
                             'context':f'Hướng dẫn {site} dịch vụ số {i} và quy định chi tiết',})
        path=root/'crawl.csv';pd.DataFrame(rows).to_csv(path,index=False)
        return path

    def test_template_is_unjudged_and_crawl_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self._make(root);out=root/'eval'
            res=prepare(src,out,docs_per_site=5,queries_per_site=1)
            self.assertEqual(res['evaluation_doc_count'],25)
            self.assertEqual(res['evaluation_query_count'],5)
            self.assertEqual(res['judgments_required'],125)
            labs=pd.read_csv(out/'qrels_TO_BE_LABELED.csv',keep_default_na=False)
            self.assertTrue(labs.relevant.eq('').all())
            self.assertFalse((out/'metric_summary.csv').exists())
            self.assertEqual(set(pd.read_csv(out/'eval_documents.csv').source_site),set(SITES))
            with self.assertRaises(FileNotFoundError):evaluate(out,tokenizer_backend='regex')

    def test_refuses_missing_or_unjudged_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self._make(root);out=root/'eval';prepare(src,out,5,1)
            labs=pd.read_csv(out/'qrels_TO_BE_LABELED.csv',keep_default_na=False)
            labs.loc[0,'relevant']='1';labs.to_csv(out/'qrels_REVIEWED.csv',index=False)
            with self.assertRaisesRegex(ValueError,'unjudged'):evaluate(out,tokenizer_backend='regex')
            labs=labs.iloc[:-1];labs['relevant']='0';labs.to_csv(out/'qrels_REVIEWED.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Incomplete'):evaluate(out,tokenizer_backend='regex')

    def test_fully_judged_subset_produces_real_metric_shape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self._make(root);out=root/'eval';prepare(src,out,5,1)
            labs=pd.read_csv(out/'qrels_TO_BE_LABELED.csv',keep_default_na=False)
            queries=pd.read_csv(out/'eval_queries.csv',keep_default_na=False)
            source_doc=dict(zip(queries.query_id,queries.source_doc_id_for_drafting))
            labs['relevant']=[int(doc==source_doc[q]) for q,doc in zip(labs.query_id,labs.doc_id)]
            labs.to_csv(out/'qrels_REVIEWED.csv',index=False)
            result=evaluate(out,tokenizer_backend='regex')
            self.assertEqual(result['scored_combinations'],8)
            self.assertTrue((out/'metric_summary.csv').is_file())
            summary=pd.read_csv(out/'metric_summary.csv')
            self.assertTrue(summary.P_at_5.between(0,1).all())
            self.assertTrue(summary.Recall_at_5.between(0,1).all())
            self.assertTrue((out/'top5_rankings.csv').is_file())
            self.assertIn('ONLY', (out/'EVALUATION_PROTOCOL.json').read_text())

if __name__=='__main__':unittest.main()
