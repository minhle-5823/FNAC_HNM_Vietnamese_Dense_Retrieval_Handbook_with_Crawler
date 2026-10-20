"""Regression tests on the raw CSV exported from the uploaded 5-source SQLite DB."""
import unittest
from pathlib import Path
import pandas as pd
from core.source_cleaning import normalize_text, extract_hcmus_admission_table
from core.research_release import build_research_release

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'preprocessing'

class RealSnapshotRegression(unittest.TestCase):
    def test_2026_tables_parsed_all_39_from_actual_sqlite_export(self):
        df=pd.read_csv(OUT/'01_export_merged_raw.csv',encoding='utf-8-sig',keep_default_na=False)
        tables=df[(df.source_site=='hcmus_admissions') & (df.passage_index.eq(0))
            & (df.url.str.contains('/chi-tieu/|/2026-phuong-huong-tuyen-sinh-trinh-do-dai-hoc-du-kien/',regex=True))]
        self.assertEqual(len(tables),2)
        for r in tables.to_dict('records'):
            parsed=extract_hcmus_admission_table(normalize_text(r['raw_text']),r['title'],r['url'])
            self.assertIsNotNone(parsed)
            units, meta=parsed
            self.assertEqual(meta['stt_present'],list(range(1,40)))
            rows=[u for u in units if u.rule=='hcmus_structured_admission_row']
            self.assertEqual(len(rows),39)
            self.assertEqual(sum('74401a1' in u.title for u in rows),1)
            self.assertEqual(sum('75202a1' in u.title for u in rows),1)
            for row in rows:
                self.assertEqual(row.text.count('Mã ngành:'),1)
                self.assertEqual(row.text.count('Chỉ tiêu:'),1)

    def test_conservative_release_does_not_discard_pending_reviews(self):
        manifest=build_research_release(OUT)
        full=pd.read_csv(OUT/'04_crawled_auxiliary_corpus.csv',encoding='utf-8-sig')
        ready=pd.read_csv(OUT/'24_auxiliary_corpus_RESEARCH_READY.csv',encoding='utf-8-sig')
        pending=pd.read_csv(OUT/'25_pending_manual_review.csv',encoding='utf-8-sig')
        self.assertEqual(len(full),len(ready)+len(pending))
        self.assertFalse(set(ready.doc_id)&set(pending.doc_id))
        self.assertEqual(manifest['pending_review_documents'],len(pending))

    def test_no_8_32_program_row_conflation(self):
        df=pd.read_csv(OUT/'04_crawled_auxiliary_corpus.csv',encoding='utf-8-sig')
        rows=df.loc[df.extraction_rule=='hcmus_structured_admission_row']
        for code in ('74401a1','75202a1'):
            found=rows[rows.context.str.contains('Mã ngành: '+code+' \\|',regex=True)]
            self.assertGreaterEqual(len(found),1)
        # Previously the preceding row inadvertently contained row-header
        # material for #8/#32. Such content must no longer appear in them.
        parents=rows[rows.title.str.contains(r'mã ngành (?:7440107|7520207),',regex=True)]
        self.assertFalse(parents.context.str.contains(r'\b(?:74401a1|75202a1)\b',regex=True).any())

if __name__=='__main__':unittest.main()
