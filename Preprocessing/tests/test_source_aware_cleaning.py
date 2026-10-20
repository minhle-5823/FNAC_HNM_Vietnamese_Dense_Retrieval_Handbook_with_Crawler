import unittest
from core.source_cleaning import extract_page, RESIDUAL_PATTERNS
from core.retrieval_lab import segments, dedup_rows


class SourceAwareCleaningTests(unittest.TestCase):
    def test_fpt_retains_answer_removes_only_footer(self):
        title='Làm sao để đổi mật khẩu WiFi?'
        content=title+' Người dùng không nên chia sẻ mật khẩu; hãy đổi trên Hi FPT. Chia sẻ: Các bài viết liên quan Footer'
        units,err=extract_page(content,'fpt_telecom_support',title,'https://fpt.vn/hi-fpt/huong-dan-su-dung/abc')
        self.assertFalse(err)
        self.assertIn('không nên',units[0].text)
        self.assertNotIn('Footer',units[0].text)

    def test_momo_extracts_qa_not_menu_and_related(self):
        title='Làm thế nào để nạp tiền?'
        text='Hỏi đáp Hỏi đáp Câu hỏi theo chủ đề Nạp tiền GỌI NGAY Quay lại '+title + ' Bạn mở MoMo, chọn Nạp tiền và xác nhận số tiền. Các bước hướng dẫn có trong ứng dụng. Đánh giá : 5 / 0 Bài viết này có hữu ích với bạn? Bài viết liên quan: - Lỗi nạp tiền'
        units,err=extract_page(text,'momo_help',title,'https://www.momo.vn/hoi-dap/nap-tien')
        self.assertFalse(err)
        self.assertEqual(len(units),1)
        self.assertNotIn('Câu hỏi theo chủ đề',units[0].text)
        self.assertNotIn('Bài viết liên quan',units[0].text)
        self.assertIn('xác nhận số tiền',units[0].text)

    def test_momo_category_is_quarantined(self):
        units,err=extract_page('GỌI NGAY Bảo mật Công nghệ gì? Bảo vệ gì? Đánh giá : 5 / 0','momo_help','Bảo mật','https://www.momo.vn/hoi-dap/bao-mat')
        self.assertFalse(units)
        self.assertIn('category',err)

    def test_uel_cards_split_and_breadcrumb_removed(self):
        text='FAQ You are here: Home FAQ Hỏi đáp chung Tuyển sinh khác Học phí - Học bổng - Hỏi đáp khác Học phí của Trường như thế nào? Mỗi năm mức học phí theo chương trình và theo ngành sẽ khác nhau. Tags: Sao chép Sao chép Khi nào nộp hồ sơ? Hồ sơ được nộp trong thời gian có thông báo tuyển sinh chính thức. Tags: Sao chép'
        units,err=extract_page(text,'uel_admissions_faq','FAQ','https://tuyensinh.uel.edu.vn/faq/')
        self.assertFalse(err)
        self.assertEqual(len(units),2)
        self.assertEqual(units[0].title,'Học phí của Trường như thế nào?')
        self.assertNotIn('You are here',units[0].text)

    def test_mobifone_only_numbered_faq(self):
        qa=' '.join(f'{i}. Câu hỏi số {i} về 5G là gì? Đây là giải thích rất cụ thể về công nghệ mạng thế hệ 5 và các bước sử dụng dịch vụ của MobiFone.' for i in range(1,12))
        text='Thoát KHÁM PHÁ Về 5G Câu hỏi thường gặp '+qa
        # Site extractor expects the actual FAQ's first question title.
        qa='1. Mạng 5G là gì? 5G là công nghệ mạng di động thế hệ thứ năm, tốc độ cao hơn. '+ ' '.join(f'{i}. Làm thế nào để sử dụng mạng 5G thứ {i}? Đây là hướng dẫn dành cho khách hàng MobiFone sử dụng dịch vụ 5G tại nhà.' for i in range(2,12))
        units,err=extract_page('Thoát KHÁM PHÁ Về 5G Câu hỏi thường gặp '+qa,'mobifone_5g_faq','CÂU HỎI THƯỜNG GẶP','https://5g.mobifone.vn/mobifone5g/danh-muc-chinh/cau-hoi-thuong-gap')
        self.assertFalse(err)
        self.assertEqual(len(units),11)
        self.assertNotIn('Thoát',units[0].text)
        reject,reason=extract_page('GÓI CƯỚC 5G Đăng ký Đăng ký Đăng ký','mobifone_5g_faq','GÓI CƯỚC 5G','https://5g.mobifone.vn/mobifone5g/danh-muc-chinh/goi-cuoc-5g')
        self.assertFalse(reject)
        self.assertIn('non_faq',reason)

    def test_hcmus_preserves_policy_and_removes_navigation(self):
        title='THÔNG BÁO XÉT TUYỂN NĂM 2026'
        x=title+' Điều kiện xét tuyển: thí sinh không được thiếu giấy tờ, thời gian đăng ký năm 2026 theo yêu cầu. Xem các phương thức xét tuyển khác'
        u,e=extract_page(x,'hcmus_admissions',title,'https://tuyensinh.hcmus.edu.vn/2026-thong-bao/')
        self.assertFalse(e)
        self.assertIn('không được',u[0].text)
        self.assertNotIn('Xem các phương thức',u[0].text)
        homepage,reason=extract_page('Giới thiệu về trường','hcmus_admissions','Trang chủ','https://tuyensinh.hcmus.edu.vn/')
        self.assertFalse(homepage)
        self.assertIn('landing',reason)

    def test_nearduplicate_preserves_different_titles(self):
        a=' '.join(['Thông báo học phí dự kiến năm học 2026 và điều kiện xét tuyển']*15)
        rows=[{'context':a,'doc_id':'a','title':'Thông báo năm 2026','source_site':'hcmus_admissions'},
              {'context':a.replace('2026','2027'), 'doc_id':'b','title':'Thông báo năm 2027','source_site':'hcmus_admissions'}]
        kept,rejected=dedup_rows(rows,.85)
        self.assertEqual(len(kept),2)

    def test_sentence_chunks_preserve_long_text(self):
        text=' '.join(f'Câu {i} của thông báo tuyển sinh với điều kiện rõ ràng.' for i in range(75))
        out=segments(text,max_words=80,overlap=12)
        self.assertGreater(len(out),2)
        self.assertTrue(all(s for s in out))
        self.assertIn('Câu 74',out[-1])


if __name__=='__main__':unittest.main()
