"""Regression checks for source-cleaning v5 and research-only search artifacts."""
import json
import unittest
from pathlib import Path
import pandas as pd
from scipy import sparse
from core.source_cleaning import extract_page, clean_hcmus_article
from core.retrieval_lab import search_saved_index
from core.research_release import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'preprocessing'


class V5CleaningAndIndexTests(unittest.TestCase):
    def test_hcmus_directory_is_not_a_retrieval_document(self):
        items,error=extract_page('Phương thức tuyển sinh các năm '+('Xem thông báo chi tiết → '*20),
                                 'hcmus_admissions','Phương thức tuyển sinh',
                                 'https://tuyensinh.hcmus.edu.vn/phuong-thuc-tuyen-sinh/')
        self.assertFalse(items)
        self.assertIn('navigation',error)

    def test_capital_attachment_heading_removed_but_lowercase_legal_phrase_preserved(self):
        body=('Quy chế tuyển sinh năm 2026 và các văn bản liên quan. Thí sinh làm hồ sơ đầy đủ. '
              'CÁC VĂN BẢN LIÊN QUAN Thông báo về phương thức 2 năm 2026')
        cleaned=clean_hcmus_article(body,'https://tuyensinh.hcmus.edu.vn/2026-thong-bao/')
        self.assertIn('và các văn bản liên quan.',cleaned)
        self.assertNotIn('CÁC VĂN BẢN LIÊN QUAN',cleaned)
        self.assertIn('Thí sinh làm hồ sơ đầy đủ.',cleaned)

    def test_preserve_instructions_following_attachment_list(self):
        body=('Nội dung xét tuyển năm 2025. '+'Quy định tuyển sinh và hồ sơ hợp lệ. '*4+
              'CÁC VĂN BẢN LIÊN QUAN Phụ lục 3.1 Mẫu đăng ký Phụ lục 3.4 '+
              'Đối với phụ lục 3.4 thí sinh phải in và ký xác nhận.')
        clean=clean_hcmus_article(body,'https://tuyensinh.hcmus.edu.vn/2025-thong-bao/')
        self.assertIn('Đối với phụ lục 3.4 thí sinh phải in',clean)
        self.assertNotIn('CÁC VĂN BẢN LIÊN QUAN',clean)

    def test_usable_corpus_does_not_include_old_boilerplate(self):
        full=pd.read_csv(OUT/'04_crawled_auxiliary_corpus.csv',keep_default_na=False)
        research=pd.read_csv(OUT/'24_auxiliary_corpus_RESEARCH_READY.csv',keep_default_na=False)
        pending=pd.read_csv(OUT/'25_pending_manual_review.csv',keep_default_na=False)
        self.assertEqual(len(full),len(research)+len(pending))
        self.assertEqual(set(full.source_site),set(research.source_site))
        self.assertFalse(full.url.str.contains(r'/phuong-thuc-tuyen-sinh/$',regex=True).any())
        self.assertFalse(full.context.str.contains(r'CÁC VĂN BẢN LIÊN QUAN',regex=True).any())
        self.assertFalse(full.context.str.contains(r'Xem thông báo chi tiết|Xem kết quả xét tuyển|thiết bioj',case=False,regex=True).any())
        self.assertFalse(full.doc_id.duplicated().any())

    def test_auxiliary_csv_is_subset_and_can_be_searched_without_persisted_index(self):
        corpus=pd.read_csv(OUT/'24_auxiliary_corpus_RESEARCH_READY.csv',keep_default_na=False)
        full=pd.read_csv(OUT/'04_crawled_auxiliary_corpus.csv',keep_default_na=False)
        manifest=json.loads((OUT/'26_research_release_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['research_ready_sha256'],sha256(OUT/'24_auxiliary_corpus_RESEARCH_READY.csv'))
        self.assertTrue(set(corpus.doc_id).issubset(set(full.doc_id)))
        self.assertFalse((OUT/'models_research').exists())
        from importlib.util import find_spec
        if find_spec('underthesea') is None:
            self.skipTest('Search tokenizer requires underthesea; installed on user machine')
        for method in ('bm25','tfidf'):
            results=search_saved_index(OUT,'hồ sơ tuyển sinh đại học',method=method,top_k=3,corpus_name='research')
            self.assertEqual(len(results),3)
            self.assertTrue(all(r['doc_id'] in set(corpus.doc_id) for r in results))

if __name__=='__main__': unittest.main()
