"""No external HTTP requests: verifies a full five-database SQLite merge."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.database import CrawlDB
from core.database_merge import TABLES, merge_group_databases


class MergeDatabaseTests(unittest.TestCase):
    def _setup(self, root: Path):
        sites = ["s1", "s2", "s3", "s4", "s5"]
        (root / "data").mkdir()
        (root / "groups").mkdir()
        (root / "groups" / "testing.yaml").write_text(
            "sites:\n" + "".join(f"  - {s}\n" for s in sites), encoding="utf-8"
        )
        for i, site in enumerate(sites):
            seed = f"https://site{i}.example/faq"
            profile_dir = root / "site_profiles" / site
            profile_dir.mkdir(parents=True)
            (profile_dir / "profile.yaml").write_text(
                f"id: {site}\nname: {site}\nseed_urls: [{seed}]\n"
                f"allowed_domains: [site{i}.example]\n", encoding="utf-8"
            )
            db = CrawlDB(root / "data" / f"{site}.db")
            db.discover_url(seed, 0)
            db.mark_url_done(seed)
            db.save_raw_page({
                "url": seed, "domain": f"site{i}.example", "source_site": site,
                "domain_label": "customer_support", "page_type": "faq_detail",
                "title": "FAQ", "raw_html": f"<p>{site}</p>", "extracted_text": "Câu hỏi",
                "depth": 0, "status_code": 200, "metadata": {},
            })
            valid = SimpleNamespace(
                clean_title="FAQ", clean_text="Một câu hỏi có đáp án", language_score=.8,
                relevance_score=.9, quality_score=.9, valid=True,
                rejection_reason="", processed_hash=site,
                passages=[{"passage_index": 0, "text": "Một câu hỏi có đáp án", "word_count": 6}],
            )
            db.save_processed(seed, site, "customer_support", valid)
            db.save_link(seed, seed + "/more", 1)
            db.event(seed, 0, "test_event")
            db.robots_check(seed, True, "allowed")
            run = db.start_run(site, 2, 10, 30)
            db.finish_run(run, {"raw_saved": 1, "processed_valid": 1, "passages_saved": 1})
            db.close()
        return sites

    def test_merges_all_eight_tables_without_id_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sites = self._setup(root)
            for _ in range(2):  # replacement is idempotent
                result = merge_group_databases(root, "testing")
                self.assertEqual(result["sites"], sites)
                self.assertTrue(all(result["totals"][table] == 5 for table in TABLES))
                with sqlite3.connect(result["database"]) as conn:
                    for table in TABLES:
                        self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 5)
                        self.assertEqual(conn.execute(f"SELECT COUNT(DISTINCT id) FROM {table}").fetchone()[0], 5)
                        self.assertEqual(conn.execute(f"SELECT COUNT(DISTINCT origin_site) FROM {table}").fetchone()[0], 5)
                        self.assertEqual(conn.execute(f"SELECT COUNT(DISTINCT origin_id) FROM {table}").fetchone()[0], 1)
                    self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                    self.assertEqual(conn.execute("SELECT site_count FROM merge_info").fetchone()[0], 5)
                    row = conn.execute("SELECT row_counts_json FROM merge_sources LIMIT 1").fetchone()
                    self.assertEqual(json.loads(row[0])["passages"], 1)
            # None of the original site databases was overwritten.
            with sqlite3.connect(root / "data" / "s1.db") as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0], 1)

    def test_rejects_stale_profile_and_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._setup(root)
            result = merge_group_databases(root, "testing")
            prior = Path(result["database"]).read_bytes()
            profile = root / "site_profiles" / "s3" / "profile.yaml"
            profile.write_text(profile.read_text().replace("/faq", "/new-faq"))
            with self.assertRaisesRegex(ValueError, "seeds from a different profile"):
                merge_group_databases(root, "testing")
            self.assertEqual(Path(result["database"]).read_bytes(), prior)

    def test_missing_db_does_not_create_partial_merged_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._setup(root)
            (root / "data" / "s4.db").unlink()
            with self.assertRaises(FileNotFoundError):
                merge_group_databases(root, "testing")
            self.assertFalse((root / "data" / "testing_merged.db").exists())


if __name__ == "__main__":
    unittest.main()
