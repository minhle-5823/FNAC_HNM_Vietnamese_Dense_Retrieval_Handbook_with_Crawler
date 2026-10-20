"""Core regression tests for preprocessing. Run python -m unittest discover -s tests."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.retrieval_lab import (
    normalize, segments, dedup_rows, stable_id, VietnameseTokenizer,
    tokenize_strategies, evaluate_qrels, load_merged, _english_tools,
    clean_source_text, make_corpus
)


class TestPreprocessing(unittest.TestCase):
    def test_cleaning_keeps_vietnamese_and_numbers(self):
        self.assertEqual(normalize(' Chăm sóc  5G!   \nhttps://test.vn/ '),'Chăm sóc 5G!')
        self.assertEqual(normalize('<strong>KHÔNG</strong> được',True),'không được')

    def test_fpt_footer_removed_only_at_end(self):
        raw = ('Làm sao đăng ký Internet FPT? Bước 1: Mở Hi FPT. '
               'Bước 2: Chọn gói cước. Chia sẻ: Các bài viết liên quan Footer')
        expected = ('Làm sao đăng ký Internet FPT? Bước 1: Mở Hi FPT. '
                    'Bước 2: Chọn gói cước.')
        self.assertEqual(clean_source_text(raw, 'fpt_telecom_support'), expected)
        self.assertEqual(clean_source_text(raw, 'momo_help'), raw)
        self.assertEqual(clean_source_text(raw.replace(' Footer', ' Footer phần tiếp theo'),
                                           'fpt_telecom_support'),
                         raw.replace(' Footer', ' Footer phần tiếp theo'))
        self.assertEqual(clean_source_text('Đăng bài viết liên quan về WiFi',
                                           'fpt_telecom_support'),
                         'Đăng bài viết liên quan về WiFi')

    def test_fpt_footer_cleaned_before_chunking_and_dedup(self):
        import pandas as pd
        raw = pd.DataFrame([{
            'source_passage_id':'p1','origin_site':'fpt_telecom_support',
            'origin_id':1, 'url':'https://fpt.vn/hi-fpt/huong-dan-su-dung',
            'source_site':'fpt_telecom_support', 'domain_label':'customer_support',
            'source_domain':'fpt.vn','title':'Hướng dẫn', 'crawled_at':'2026-10-19',
            'passage_index':0,'quality_score':1,
            'raw_text':('Quý khách có thể đăng ký dịch vụ Internet FPT '
                        'trực tiếp trên ứng dụng Hi FPT theo các bước sau. '
                        'Bước một chọn Internet, bước hai chọn gói cước, '
                        'bước ba hoàn tất thanh toán. '
                        'Chia sẻ: Các bài viết liên quan Footer')
        }])
        corpus,rejected,stats=make_corpus(raw,min_chars=100)
        self.assertEqual(len(corpus),1)
        self.assertEqual(stats['clean_documents'],1)
        self.assertNotIn('Chia sẻ:',corpus.iloc[0]['context'])
        self.assertNotIn('Footer',corpus.iloc[0]['context'])
        self.assertEqual(len(rejected),0)

    def test_segmentation_never_infinite(self):
        self.assertEqual(segments('abc def',max_words=30,overlap=5),['abc def'])
        text=' '.join('a'+str(i) for i in range(235))
        docs=segments(text,100,20)
        self.assertGreaterEqual(len(docs),2)
        self.assertIn('a234',docs[-1])
        self.assertGreater(len(docs[0].split()),20)

    def test_dedup_preserves_master(self):
        src='Tư vấn học phí và đăng ký xét tuyển đại học tại trường này. '*9
        a={'doc_id':'id1','context':src}
        b={'doc_id':'id2','context':src}
        clean,bad=dedup_rows([a,b])
        self.assertEqual([x['doc_id'] for x in clean],['id1'])
        self.assertEqual(bad[0]['reason'],'exact_duplicate')
        self.assertEqual(bad[0]['duplicate_of'],'id1')

    def test_stable_ids(self):
        self.assertEqual(stable_id('aa',0,'Tư vấn 5G'),stable_id('aa',0,'Tư vấn 5G'))
        self.assertNotEqual(stable_id('aa',0,'Tư vấn 5G'),stable_id('aa',1,'Tư vấn 5G'))

    def test_negation_preserved(self):
        stem,lemma,_=_english_tools()
        res=tokenize_strategies(VietnameseTokenizer('regex'),'không được hủy tài khoản',stem,lemma)
        self.assertIn('không',res['no_stopwords'])
        self.assertIn('được',res['no_stopwords'])
        self.assertIn('hủy',res['no_stopwords'])

    def test_partial_qrels_no_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'qrels.csv'
            with f.open('w',newline='',encoding='utf-8') as fh:
                w=csv.writer(fh);w.writerow(['query_id','doc_id','relevant'])
                for i in range(5):w.writerow(['q1',f'd{i}',int(i<2)])
            results=[{'query_id':'q1','doc_id':f'd{i}','strategy':'baseline',
                      'method':'bm25','rank':i+1} for i in range(5)]
            rows=evaluate_qrels(f,results)
            self.assertAlmostEqual(rows[0]['P@5'],0.4)
            self.assertIsNone(rows[0]['Recall@5'])

    def test_missing_db_raises_no_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'missing.db'
            with self.assertRaises(FileNotFoundError):load_merged(p)
            self.assertFalse(p.exists())

if __name__ == '__main__':unittest.main()
