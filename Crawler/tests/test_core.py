import tempfile
import unittest
from pathlib import Path

from core.database import CrawlDB
from core.duplicate import DuplicateDetector
from core.extractor import extract_links, parse_page
from core.frontier import URLFrontier
from core.preprocessor import preprocess_document
from core.profile_loader import load_group, load_site
from core.url_tools import is_allowed_url, normalize_url

ROOT = Path(__file__).resolve().parents[1]


class FrontierTests(unittest.TestCase):
    def test_bfs_fifo(self):
        f = URLFrontier()
        self.assertTrue(f.add("A", 0))
        self.assertTrue(f.add("B", 1))
        self.assertTrue(f.add("C", 1))
        self.assertEqual(f.pop().url, "A")
        self.assertEqual(f.pop().url, "B")
        self.assertEqual(f.pop().url, "C")

    def test_queue_and_visited_duplicate(self):
        f = URLFrontier()
        self.assertTrue(f.add("A", 0))
        self.assertFalse(f.add("A", 1))
        item = f.pop()
        f.mark_visited(item.url)
        self.assertFalse(f.add("A", 2))


class ActiveProfileTests(unittest.TestCase):
    def test_group_has_five_active_sites(self):
        self.assertEqual(
            load_group(ROOT, "vire_web_corpus"),
            [
                "mobifone_5g_faq",
                "momo_help",
                "fpt_telecom_support",
                "uel_admissions_faq",
                "hcmus_admissions",
            ],
        )

    def test_momo_scope_and_self_link(self):
        profile = load_site(ROOT, "momo_help")
        html = b"""
        <html><body>
          <a href='/hoi-dap/vi-sao-giao-dich-that-bai'>FAQ</a>
          <a href='/blog/promo'>Blog</a>
          <a href='/hoi-dap'>Self</a>
        </body></html>
        """
        parsed = parse_page(html, "https://www.momo.vn/hoi-dap", profile)
        links, rejected = extract_links(
            parsed["soup"], "https://www.momo.vn/hoi-dap", profile
        )
        self.assertEqual(
            links,
            ["https://www.momo.vn/hoi-dap/vi-sao-giao-dich-that-bai"],
        )
        self.assertGreaterEqual(rejected.get("self_link", 0), 1)

    def test_hcmus_scope(self):
        profile = load_site(ROOT, "hcmus_admissions")
        html = b"""
        <html><body>
          <a href='/hoc-phi/'>Hoc phi</a>
          <a href='/cau-hoi-thuong-gap/'>FAQ</a>
          <a href='/gioi-thieu-chung/'>About</a>
        </body></html>
        """
        parsed = parse_page(html, "https://tuyensinh.hcmus.edu.vn/", profile)
        links, _ = extract_links(
            parsed["soup"], "https://tuyensinh.hcmus.edu.vn/", profile
        )
        self.assertIn("https://tuyensinh.hcmus.edu.vn/hoc-phi/", links)
        self.assertIn(
            "https://tuyensinh.hcmus.edu.vn/cau-hoi-thuong-gap/", links
        )
        self.assertNotIn(
            "https://tuyensinh.hcmus.edu.vn/gioi-thieu-chung/", links
        )

    def test_fpt_hifpt_scope(self):
        profile = load_site(ROOT, "fpt_telecom_support")
        root = "https://fpt.vn/hi-fpt/huong-dan-su-dung"
        self.assertEqual(profile["seed_urls"], [root])
        self.assertFalse(is_allowed_url("https://fpt.vn/tin-tuc/tin-tuc-123.html", profile)[0])
        self.assertFalse(is_allowed_url("https://fpt.vn/ho-tro/cau-hoi-thuong-gap", profile)[0])
        self.assertTrue(is_allowed_url(root + "/quan-ly-internet", profile)[0])
        self.assertTrue(is_allowed_url(root + "/quan-ly-dich-vu-internet/huong-dan-chia-se-mat-khau-wi-fi", profile)[0])

    def test_hifpt_listing_category_and_detail(self):
        profile = load_site(ROOT, "fpt_telecom_support")
        root = "https://fpt.vn/hi-fpt/huong-dan-su-dung"
        category = root + "/quan-ly-internet"
        detail = root + "/quan-ly-dich-vu-internet/huong-dan-chia-se-mat-khau-wi-fi"
        listing_html = f"""<html><body><header><a href='/tin-tuc/'>News</a></header>
            <main><h1>Chúng tôi có thể giúp gì cho bạn?</h1>
            <a href='{category}'>Quản lý internet</a>
            <a href='/hi-fpt//huong-dan-su-dung/quan-ly-dich-vu-internet/huong-dan-chia-se-mat-khau-wi-fi'>Chia sẻ Wi-Fi</a>
            <a href='/tin-tuc/news'>Tin công nghệ</a></main>
            </body></html>"""
        parsed = parse_page(listing_html, root, profile)
        self.assertEqual(parsed["page_type"], "help_listing")
        self.assertEqual(parsed["content"], "")
        links, _ = extract_links(parsed["soup"], root, profile)
        self.assertEqual(set(links), {category, detail})
        cat_parsed = parse_page(f"<html><main><h1>Quản lý internet</h1><a href='{detail}'>Bài viết</a></main></html>", category, profile)
        self.assertEqual(cat_parsed["page_type"], "help_category")
        self.assertEqual(cat_parsed["content"], "")
        cat_links, _ = extract_links(cat_parsed["soup"], category, profile)
        self.assertEqual(cat_links, [detail])

        detail_html = """<html><body><header><p>Tin tức Thể thao</p></header>
          <main><p>Trang chủ Hi FPT</p><h1>Hướng dẫn chia sẻ mật khẩu Wi-Fi</h1>
          <div><p>Quý Khách có thể chia sẻ mật khẩu Wi-Fi đơn giản theo hướng dẫn sau:</p>
            <ul><li>Bước 1: Nhấn vào icon 3 chấm bên cạnh tên Wi-Fi tại trang chủ.</li>
            <li>Bước 2: Nhấn chọn Chia sẻ Wi-Fi (QR Code Wi-Fi).</li>
            <li>Bước 3: Người thân và bạn bè quét mã QR để kết nối Wi-Fi.</li></ul></div>
          <section><h2>Nội dung này có hữu ích không?</h2><p>Có / Không</p></section>
          </main><footer>Giấy chứng nhận ĐKDN và tin tức công ty</footer></body></html>"""
        result = parse_page(detail_html, detail, profile)
        self.assertEqual(result["page_type"], "help_detail")
        self.assertEqual(result["title"], "Hướng dẫn chia sẻ mật khẩu Wi-Fi")
        self.assertIn("Bước 1", result["content"])
        self.assertIn("Bước 3", result["content"])
        self.assertNotIn("Nội dung này có hữu ích", result["content"])
        self.assertNotIn("Giấy chứng nhận", result["content"])
        self.assertNotIn("Tin tức Thể thao", result["content"])
        self.assertTrue(preprocess_document(result["title"], result["content"], profile).valid)


class PreprocessTests(unittest.TestCase):
    def test_customer_support_document_accept(self):
        profile = load_site(ROOT, "momo_help")
        text = (
            "Nếu giao dịch thanh toán không thành công, khách hàng kiểm tra số dư "
            "tài khoản và ngân hàng liên kết. Nếu vẫn lỗi, liên hệ bộ phận hỗ trợ. "
        ) * 18
        p = preprocess_document("Vì sao giao dịch thanh toán không thành công?", text, profile)
        self.assertTrue(p.valid)
        self.assertGreaterEqual(len(p.passages), 1)

    def test_irrelevant_document_reject(self):
        profile = load_site(ROOT, "fpt_telecom_support")
        text = ("Đội bóng thi đấu và giành chiến thắng trong trận chung kết. " * 30)
        p = preprocess_document("Kết quả bóng đá hôm nay", text, profile)
        self.assertFalse(p.valid)


class URLAndDuplicateTests(unittest.TestCase):
    def test_normalize_tracking_query(self):
        profile = load_site(ROOT, "momo_help")
        u = normalize_url(
            "/hoi-dap/demo?utm_source=x#abc",
            "https://www.momo.vn/hoi-dap",
            profile,
        )
        self.assertEqual(u, "https://www.momo.vn/hoi-dap/demo")
        self.assertTrue(is_allowed_url(u, profile)[0])

    def test_sha_duplicate(self):
        d = DuplicateDetector()
        self.assertFalse(d.check_and_add("same text")[0])
        self.assertTrue(d.check_and_add("same text")[0])


class DatabaseResumeTests(unittest.TestCase):
    def test_persistent_frontier_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.db"
            db = CrawlDB(path)
            db.discover_url("https://example.com/", 0)
            db.discover_url(
                "https://example.com/a",
                1,
                "https://example.com/",
            )
            db.mark_url_done("https://example.com/", "visited")
            db.conn.commit()

            self.assertEqual(
                db.queued_urls(),
                [("https://example.com/a", 1)],
            )
            self.assertIn(
                "https://example.com/",
                db.visited_urls(),
            )
            db.close()


if __name__ == "__main__":
    unittest.main()
