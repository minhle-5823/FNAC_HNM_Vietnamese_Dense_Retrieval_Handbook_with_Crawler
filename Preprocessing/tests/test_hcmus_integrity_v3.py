"""Regression tests for source-faithful HCMUS 2026 preprocessing."""
import re
import unittest
from core.retrieval_lab import segments, dedup_rows
from core.source_cleaning import extract_page, extract_hcmus_admission_table


def fake_table():
    header='Thông tin năm 2026. STT Mã ngành xét tuyển Tên ngành Chỉ tiêu 2026 Phương thức xét tuyển '
    rows=[]
    for n in range(1,40):
        code = {8:"74401a1",32:"75202a1"}.get(n, str(7420000+n))
        rows.append(f'{n} {code} Ngành học {n} 90 Phương thức 1a: Tuyển thẳng. '
                    f'Phương thức 2: Xét tuyển theo điểm thi THPT 2026 hoặc ĐGNL và học bạ các lớp 10, 11, 12. '
                    f'A00: Toán - Vật lý - Hóa học. B00: Toán - Hóa học - Sinh học.')
    return header + ' '.join(rows)


class Hcmus2026Tests(unittest.TestCase):
    def test_table_with_alphanumeric_codes_parses_all_39_rows(self):
        url='https://tuyensinh.hcmus.edu.vn/chi-tieu/'
        result=extract_hcmus_admission_table(fake_table(),'Chỉ tiêu 2026',url)
        self.assertIsNotNone(result)
        rows,meta=result
        self.assertEqual(meta['table_rows'],39)
        self.assertEqual(meta['stt_missing'],[])
        self.assertEqual(len([u for u in rows if u.rule=='hcmus_structured_admission_row']),39)
        self.assertIn('7420001',rows[0].text)
        self.assertIn('A00: Toán',rows[0].text)
        self.assertTrue(any('74401a1' in u.title for u in rows))
        self.assertTrue(any('75202a1' in u.title for u in rows))
        for u in rows:
            self.assertNotRegex(u.text,r'\b\d{1,2}\s+(?:74401a1|75202a1)\b')

    def test_missing_real_row_quarantines_table(self):
        raw=fake_table().replace('8 74401a1 Ngành học 8', 'ĐÃ MẤT ĐẦU HÀNG Ngành học 8')
        self.assertIsNone(extract_hcmus_admission_table(raw, 'Chỉ tiêu 2026',
            'https://tuyensinh.hcmus.edu.vn/chi-tieu/'))

    def test_tail_article_is_separate_from_last_program(self):
        url='https://tuyensinh.hcmus.edu.vn/2026-phuong-huong-tuyen-sinh-trinh-do-dai-hoc-du-kien/'
        text=fake_table()+' Lưu ý: Đây là thông tin bổ sung về thủ tục nhập học và trường hợp hỗ trợ người học. '+('Thí sinh vẫn cần làm theo thông báo chính thức. '*6)
        units,reason=extract_page(text,'hcmus_admissions','Phương hướng 2026',url)
        self.assertEqual(reason,'')
        rows=[u for u in units if u.rule=='hcmus_structured_admission_row']
        self.assertEqual(len(rows),39)
        self.assertNotIn('Lưu ý:',rows[-1].text)
        suffix=[u for u in units if u.rule=='hcmus_admission_article_after_table']
        self.assertEqual(len(suffix),1)
        self.assertIn('Lưu ý:',suffix[0].text)

    def test_ambiguous_table_is_quarantined(self):
        # Missing anchors means raw snapshot is not safe to parse as a table.
        units,reason=extract_page('Chỉ tiêu tuyển sinh 2026 '+'Thông tin tuyển sinh dự kiến. '*20,
              'hcmus_admissions','Chỉ tiêu', 'https://tuyensinh.hcmus.edu.vn/chi-tieu/')
        self.assertFalse(units)
        self.assertIn('ambiguous',reason)

    def test_dangling_numeric_header_moves_forward(self):
        text=('Một thông báo tuyển sinh trình bày những điều kiện chính và thời hạn nộp hồ sơ. '*11 +
              '3. Hồ sơ bao gồm các giấy tờ theo hướng dẫn của trường. ' +
              'Các giấy tờ phải được nộp đúng thời hạn công bố. '*15)
        chunks=segments(text,max_words=70,overlap=10)
        self.assertGreater(len(chunks),1)
        self.assertFalse(any(re.search(r'(?<!\w)\d+\.\s*$',c) for c in chunks[:-1]))
        self.assertIn('3.', ' '.join(chunks))

    def test_dedup_exact_only_preserves_differing_table_rule(self):
        full='Chỉ tiêu năm 2026: Ngành học A, 100. Phương thức xét tuyển theo Bộ GDĐT.'
        rows=[{'context':full,'doc_id':'a','title':'Ngành học A','source_site':'hcmus_admissions','extraction_rule':'hcmus_structured_admission_row'},
              {'context':full,'doc_id':'b','title':'Ngành học A','source_site':'hcmus_admissions','extraction_rule':'hcmus_structured_admission_row'},
              {'context':full.replace('100','101'),'doc_id':'c','title':'Ngành học A','source_site':'hcmus_admissions','extraction_rule':'hcmus_structured_admission_row'}]
        kept,deleted=dedup_rows(rows,0.6)
        self.assertEqual(len(kept),2)
        self.assertEqual(len(deleted),1)
        self.assertEqual(deleted[0]['reason'],'exact_duplicate')

if __name__=='__main__':unittest.main()
