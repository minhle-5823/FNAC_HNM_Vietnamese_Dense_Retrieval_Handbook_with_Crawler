import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.crawler import UnifiedCrawler
from core.validator import validate_database


LONG_ROOT = (
    "Đây là trang hỗ trợ khách hàng về tài khoản, dịch vụ, đăng ký và hướng dẫn. "
    "Khách hàng có thể xem thông tin hỗ trợ và cách xử lý các vấn đề thường gặp. "
) * 8
LONG_LIST = (
    "Danh sách hướng dẫn dịch vụ dành cho khách hàng, bao gồm đăng ký, tài khoản "
    "và hỗ trợ xử lý lỗi. "
) * 10
LONG_DOC = (
    "Hướng dẫn khách hàng xử lý lỗi kết nối dịch vụ. Kiểm tra tài khoản, "
    "thông tin đăng ký và liên hệ hỗ trợ nếu vấn đề vẫn còn. "
) * 12


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        pages = {
            "/robots.txt": (
                "text/plain",
                "User-agent: *\nAllow: /\n",
            ),
            "/": (
                "text/html",
                f"<html><head><title>Hỗ trợ</title></head><body>"
                f"<main>{LONG_ROOT}<a href='/list'>List</a>"
                f"<a href='/'>Self</a></main></body></html>",
            ),
            "/list": (
                "text/html",
                f"<html><head><title>Danh sách hỗ trợ</title></head><body>"
                f"<main>{LONG_LIST}<a href='/doc/demo'>Demo</a></main></body></html>",
            ),
            "/doc/demo": (
                "text/html",
                f"<html><head><title>Hướng dẫn xử lý lỗi</title></head><body>"
                f"<main>{LONG_DOC}</main></body></html>",
            ),
        }

        if self.path not in pages:
            self.send_response(404)
            self.end_headers()
            return

        ctype, body = pages[self.path]
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LocalIntegrationTest(unittest.TestCase):
    def test_end_to_end_bfs_raw_processed_passages(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
        )
        thread.start()

        host = f"127.0.0.1:{server.server_address[1]}"
        base = f"http://{host}"

        profile = {
            "id": "local_test",
            "name": "Local Test",
            "topic": "Test",
            "domain_label": "customer_support",
            "seed_urls": [base + "/"],
            "allowed_domains": [host],
            "user_agent": "VIRETestCrawler/1.0",
            "crawl": {
                "max_depth": 2,
                "max_pages": 10,
                "max_requests": 20,
                "request_timeout": 5,
            },
            "rate_limit": {
                "initial_delay": 0.0,
                "min_delay": 0.0,
                "max_delay": 1.0,
                "max_retries": 2,
                "retry_statuses": [429, 503],
                "success_window": 2,
                "recovery_factor": 0.8,
            },
            "http": {"follow_redirects": False, "headers": {}},
            "policy": {
                "robots": "respect",
                "robots_unavailable": "block",
                "stop_host_statuses": [401, 403],
                "challenge_patterns": [],
            },
            "url": {
                "allow_path_regex": ["^/"],
                "deny_path_regex": [],
                "ignored_extensions": [],
                "strip_query_prefixes": ["utm_"],
                "strip_query_keys": [],
                "sort_query": True,
                "trailing_slash": "preserve",
            },
            "page_types": [
                {"name": "root", "url_regex": "^/$"},
                {"name": "list", "url_regex": "^/list$"},
                {"name": "doc", "url_regex": "^/doc/"},
            ],
            "extract": {
                "parser": "html.parser",
                "title": {"selectors": ["title", "h1"]},
                "content": {
                    "container_selectors": ["main", "body"],
                    "remove_selectors": ["script", "style"],
                },
                "metadata": [],
            },
            "links": {
                "default_follow": True,
                "default_selector": "a[href]",
                "rules": [],
            },
            "preprocess": {
                "min_words": 10,
                "min_chars": 80,
                "min_vietnamese_score": 0.05,
                "min_relevance_score": 0.05,
                "min_keyword_hits": 1,
                "relevance_keywords": [
                    "khách hàng",
                    "hỗ trợ",
                    "tài khoản",
                    "dịch vụ",
                ],
                "required_any_keywords": ["hỗ trợ", "khách hàng"],
                "chunk_words": 80,
                "chunk_overlap_words": 10,
                "min_chunk_words": 10,
            },
        }

        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "data").mkdir()
                (root / "reports").mkdir()

                crawler = UnifiedCrawler(root, profile, reset=True)
                try:
                    summary = crawler.run()
                finally:
                    crawler.close()

                self.assertEqual(summary["raw_saved"], 3)
                self.assertEqual(summary["processed_valid"], 3)
                self.assertGreaterEqual(summary["passages_saved"], 3)
                self.assertEqual(
                    summary["stop_reason"],
                    "URL Frontier is empty",
                )

                result = validate_database(
                    root / "data" / "local_test.db"
                )
                self.assertEqual(result["result"], "PASS")
                self.assertEqual(
                    result["by_depth"],
                    {"0": 1, "1": 1, "2": 1},
                )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
