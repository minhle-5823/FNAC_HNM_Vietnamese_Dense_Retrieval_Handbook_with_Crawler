"""SEG301 score tests: only human-labeled crawled documents; NO ViRE benchmarks."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from complete_lab import _p5_from_labels


class CompletionTests(unittest.TestCase):
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

if __name__=='__main__': unittest.main()
