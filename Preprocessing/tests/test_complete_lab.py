"""Regression checks for the optional label-aware SEG301 completion workflow."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from complete_lab import prepare_vire_pairs, _p5_from_labels, run_vire_evaluation


class CompletionTests(unittest.TestCase):
    def test_vire_qrels_groups_multiple_positives(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'vire.csv'
            pd.DataFrame([
                {'qid':'q1','question':'Hỏi học phí?','context':'Học phí trường A.'},
                {'qid':'q1','question':'Hỏi học phí?','context':'Học phí trường B.'},
                {'qid':'q2','question':'Điều kiện xét tuyển?','context':'Xét tuyển theo điểm.'},
            ]).to_csv(f,index=False)
            docs, queries, qrels=prepare_vire_pairs(f)
            self.assertEqual(len(docs),3)
            self.assertEqual(len(queries),2)
            self.assertEqual(len(qrels['q1']),2)

    def test_qrels_has_no_auto_negatives_from_website(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'vire.csv'
            pd.DataFrame([
                {'question':'Gói FPT nào?','context':'Chọn gói Internet.'},
                {'question':'Gói FPT nào?','context':'Chọn gói mạng FPT.'}
            ]).to_csv(f,index=False)
            _, _, qrels=prepare_vire_pairs(f)
            self.assertEqual(len(next(iter(qrels.values()))),2)

    def test_manual_p5_needs_all_top5_labels(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'labels.csv'
            pd.DataFrame([{'query_id':'q1','doc_id':f'd{i}','relevant':str(int(i < 2)) if i<4 else ''}
                          for i in range(5)]).to_csv(f,index=False)
            ranked=pd.DataFrame([{'query_id':'q1','doc_id':f'd{i}','strategy':'baseline',
                                  'method':'bm25','rank':i+1} for i in range(5)])
            metrics=_p5_from_labels(f,ranked)
            self.assertTrue(pd.isna(metrics.iloc[0]['P@5']))
            pd.DataFrame([{'query_id':'q1','doc_id':f'd{i}','relevant':int(i<2)}
                          for i in range(5)]).to_csv(f,index=False)
            metrics=_p5_from_labels(f,ranked)
            self.assertAlmostEqual(metrics.iloc[0]['P@5'],.4)
            self.assertTrue(pd.isna(metrics.iloc[0]['Recall@5']))

    def test_conflicting_labels_not_silently_used(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'labels.csv'
            pd.DataFrame([{'query_id':'q1','doc_id':'d1','relevant':'1'},
                          {'query_id':'q1','doc_id':'d1','relevant':'0'}]).to_csv(f,index=False)
            ranked=pd.DataFrame([{'query_id':'q1','doc_id':'d1','strategy':'baseline','method':'tfidf','rank':1}])
            with self.assertRaises(ValueError): _p5_from_labels(f,ranked)

    def test_vire_metric_csv_generated(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            rows=[]
            for i,topic in enumerate(['học phí','đăng ký 5G','xét tuyển','chuyển khoản']):
                rows.append({'question':f'Cách {topic}?',
                             'context':f'Hướng dẫn {topic} chi tiết số {i} tại hệ thống.'})
            f=path/'pairs.csv'
            pd.DataFrame(rows).to_csv(f,index=False)
            metrics,cfg=run_vire_evaluation(f,path,tokenizer_backend='regex',max_queries=3)
            self.assertEqual(len(metrics),8)
            self.assertTrue((path/'14_vire_labeled_metric_summary.csv').exists())
            self.assertTrue((path/'13_vire_labeled_per_query_metrics.csv').exists())
            self.assertEqual(cfg['evaluated_queries'],3)

if __name__=='__main__':
    unittest.main()
