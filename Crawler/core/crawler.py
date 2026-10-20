"""Unified profile-driven BFS crawler.

REVIEW MAP
----------
SECTION 1 : initialize engine
SECTION 2 : resume/fresh seed
SECTION 3 : robots.txt preflight
SECTION 4 : BFS crawl loop
SECTION 5 : fetch + HTTP/block handling
SECTION 6 : parse HTML + extract data
SECTION 7 : extract/filter links + depth+1
SECTION 8 : SQLite/checkpoint
SECTION 9 : summary + validation

Important: anti-bot challenges are reported and stopped, never bypassed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from .block_detector import BlockDetector
from .database import CrawlDB, utc_now
from .duplicate import DuplicateDetector
from .preprocessor import preprocess_document
from .extractor import extract_links, parse_page
from .fetcher import Fetcher
from .frontier import URLFrontier
from .rate_limiter import AdaptiveRateLimiter
from .report import write_report
from .robots_policy import RobotsPolicy
from .url_tools import is_allowed_url, normalize_url
from .validator import validate_database


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower()


class UnifiedCrawler:
    # ========================================================
    # SECTION 1 - INITIALIZE ENGINE
    # ========================================================
    def __init__(
        self,
        repo_root: Path,
        profile: dict,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
        max_requests: int | None = None,
        reset: bool = False,
    ):
        self.repo_root = repo_root
        self.profile = deepcopy(profile)
        if max_pages is not None:
            self.profile["crawl"]["max_pages"] = int(max_pages)
        if max_depth is not None:
            self.profile["crawl"]["max_depth"] = int(max_depth)
        if max_requests is not None:
            self.profile["crawl"]["max_requests"] = int(max_requests)

        self.site_id = self.profile["id"]
        self.db_path = repo_root / "data" / f"{self.site_id}.db"
        self.report_dir = repo_root / "reports" / self.site_id
        if reset:
            CrawlDB.reset_file(self.db_path)

        self.db = CrawlDB(self.db_path)
        self.frontier = URLFrontier()
        self.duplicates = DuplicateDetector()
        self.blocked_hosts: set[str] = set()
        self.stats = Counter()
        self.status_counts = Counter()
        self.depth_counts = Counter()
        self.type_counts = Counter()
        self.filter_reasons = Counter()

        self.session = requests.Session()
        headers = {
            "User-Agent": self.profile.get(
                "user_agent", "SEG301UnifiedCrawler/1.0 (educational assignment)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        headers.update(self.profile.get("http", {}).get("headers", {}))
        self.session.headers.update(headers)

        self.limiter = AdaptiveRateLimiter(self.profile)
        self.detector = BlockDetector(self.profile)
        self.fetcher = Fetcher(self.profile, self.session, self.limiter)
        self.robots = RobotsPolicy(
            self.profile,
            self.session,
            self.limiter,
            event_callback=self._event,
        )

    def close(self):
        self.db.close()
        self.session.close()

    def _event(self, url, depth, kind, status_code=None, message=""):
        self.db.event(url, depth, kind, status_code, message)

    # ========================================================
    # SECTION 2 - RESUME OR FRESH SEED
    # ========================================================
    def _restore_or_seed(self) -> bool:
        queued = self.db.queued_urls()
        visited = self.db.visited_urls()
        hashes = self.db.existing_hashes()

        if queued or visited:
            # A changed seed (e.g., old FPT technology news -> new FAQ) must
            # never silently resume the unrelated crawl state.
            expected_seeds = {
                normalize_url(seed, seed, self.profile)
                for seed in self.profile["seed_urls"]
            }
            actual_seeds = {
                row[0] for row in self.db.conn.execute(
                    "SELECT url FROM url_state WHERE depth=0 AND discovered_from IS NULL"
                )
            }
            if actual_seeds != expected_seeds:
                raise RuntimeError(
                    f"Seed URLs changed for {self.site_id}. Re-run this site "
                    f"with --reset to crawl the new profile: "
                    f"python main.py crawl --site {self.site_id} --reset"
                )
            self.frontier.restore_visited(visited)
            self.duplicates.restore(hashes)
            for url, depth in queued:
                self.frontier.add(url, depth)
            self.stats["raw_saved"] = self.db.conn.execute(
                "SELECT COUNT(*) FROM raw_pages"
            ).fetchone()[0]
            self.stats["links_saved"] = self.db.conn.execute(
                "SELECT COUNT(*) FROM links"
            ).fetchone()[0]
            for row in self.db.conn.execute(
                "SELECT depth,page_type,status_code,COUNT(*) n FROM raw_pages GROUP BY depth,page_type,status_code"
            ):
                self.depth_counts[row["depth"]] += row["n"]
                self.type_counts[row["page_type"]] += row["n"]
                self.status_counts[row["status_code"]] += row["n"]
            print("[RESUME] Đã khôi phục state từ SQLite")
            print(f"         Raw pages    : {self.stats['raw_saved']:,}")
            print(f"         Pending URLs  : {len(self.frontier):,}")
            return True

        added = 0
        for seed in self.profile["seed_urls"]:
            normalized = normalize_url(seed, seed, self.profile)
            if not normalized:
                self._event(seed, 0, "invalid_seed", None, "normalize_failed")
                continue
            allowed, reason = is_allowed_url(normalized, self.profile)
            if not allowed:
                self._event(normalized, 0, "invalid_seed", None, reason)
                continue
            if self.frontier.add(normalized, 0):
                self.db.discover_url(normalized, 0, None)
                added += 1
        self.db.conn.commit()
        if added == 0:
            raise RuntimeError("Không có seed URL hợp lệ sau khi normalize/filter")
        return False

    # ========================================================
    # SECTION 3 - ROBOTS.TXT PREFLIGHT
    # ========================================================
    def preflight(self) -> list[dict]:
        rows = []
        for seed in self.profile["seed_urls"]:
            normalized = normalize_url(seed, seed, self.profile)
            if not normalized:
                rows.append({"url": seed, "allowed": False, "reason": "invalid_seed"})
                continue
            decision = self.robots.can_fetch(normalized)
            self.db.robots_check(
                normalized, decision.allowed, decision.reason, decision.status_code
            )
            rows.append({
                "url": normalized,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "crawl_delay": decision.crawl_delay,
                "robots_url": decision.robots_url,
                "status_code": decision.status_code,
            })
        return rows

    def print_configuration(self):
        c = self.profile["crawl"]
        r = self.profile["rate_limit"]
        print("=" * 78)
        print("SEG301 UNIFIED PROFILE-DRIVEN WEB CRAWLER")
        print("=" * 78)
        print(f"Site            : {self.profile['name']} ({self.site_id})")
        print(f"Topic           : {self.profile.get('topic', '')}")
        print(f"Seed URLs       : {len(self.profile['seed_urls'])}")
        print(f"Allowed domains : {', '.join(self.profile['allowed_domains'])}")
        print(f"Traversal       : BFS / deque URL Frontier")
        print(f"MAX_DEPTH       : {c['max_depth']}")
        print(f"MAX_PAGES       : {c['max_pages']:,}")
        print(f"MAX_REQUESTS    : {c['max_requests']:,}")
        print(f"Initial delay   : {float(r['initial_delay']):.1f}s (adaptive)")
        print(f"Output DB       : {self.db_path}")
        print("=" * 78)

    # ========================================================
    # SECTION 4 - BFS CRAWL LOOP
    # ========================================================
    def run(self) -> dict:
        self.print_configuration()
        self._restore_or_seed()

        max_pages = int(self.profile["crawl"]["max_pages"])
        max_depth = int(self.profile["crawl"]["max_depth"])
        max_requests = int(self.profile["crawl"]["max_requests"])
        run_id = self.db.start_run(self.site_id, max_depth, max_pages, max_requests)
        stop_reason = "URL Frontier is empty"

        try:
            while not self.frontier.empty():
                if self.stats["raw_saved"] >= max_pages:
                    stop_reason = "MAX_PAGES reached"
                    break
                if self.stats["requests_sent"] >= max_requests:
                    stop_reason = "MAX_REQUESTS reached"
                    break

                item = self.frontier.pop()
                if item is None:
                    break
                url, depth = item.url, item.depth

                if depth > max_depth:
                    self.stats["depth_skipped"] += 1
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "depth_skipped")
                    continue

                host = _host(url)
                if host in self.blocked_hosts:
                    self.stats["blocked_host_skipped"] += 1
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "blocked_host")
                    continue

                decision = self.robots.can_fetch(url)
                self.db.robots_check(url, decision.allowed, decision.reason, decision.status_code)
                if not decision.allowed:
                    self.stats["robots_skipped"] += 1
                    self._event(url, depth, "robots_skip", decision.status_code, decision.reason)
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "robots_skipped")
                    continue

                print(f"\n[CRAWL] depth={depth} | {url}")

                # ====================================================
                # SECTION 5 - FETCH + HTTP / BLOCK HANDLING
                # ====================================================
                response, elapsed, attempts, error = self.fetcher.fetch(
                    url, robots_delay=decision.crawl_delay
                )
                self.stats["requests_sent"] += attempts

                if error is not None or response is None:
                    self.stats["failed_requests"] += 1
                    self._event(url, depth, "request_error", None, str(error))
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "request_error")
                    continue

                status = int(response.status_code)
                self.status_counts[status] += 1
                if status in {429, 503}:
                    self.stats["rate_limit_responses"] += 1

                # Safe redirect handling: validate Location before requesting it.
                if status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    target = normalize_url(location, url, self.profile) if location else None
                    if target:
                        allowed, reason = is_allowed_url(target, self.profile)
                    else:
                        allowed, reason = False, "invalid_redirect"
                    if allowed:
                        if self.db.save_link(url, target, depth):
                            self.stats["links_saved"] += 1
                        if self.frontier.add(target, depth):
                            self.db.discover_url(target, depth, url)
                        self._event(url, depth, "redirect", status, target)
                    else:
                        self._event(url, depth, "redirect_rejected", status, reason)
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "redirect")
                    self.db.conn.commit()
                    continue

                classification = self.detector.classify_status(status)
                if status != 200:
                    self.stats["failed_requests"] += 1
                    self._event(url, depth, "http_error", status, classification)
                    if classification == "blocked":
                        self.blocked_hosts.add(host)
                        self.stats["hosts_blocked"] += 1
                        self._event(url, depth, "host_blocked", status, "explicit HTTP block")
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "http_error")
                    continue

                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    self.stats["non_html_skipped"] += 1
                    self._event(url, depth, "non_html", status, content_type)
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "non_html")
                    continue

                # ====================================================
                # SECTION 6 - PARSE HTML + EXTRACT DATA
                # ====================================================
                final_url = normalize_url(response.url or url, url, self.profile) or url
                parsed = parse_page(response.content, final_url, self.profile)
                title = parsed["title"]
                content = parsed["content"]
                page_type = parsed["page_type"]
                metadata = parsed["metadata"]

                challenge = self.detector.detect_challenge(title, content)
                if challenge:
                    self.stats["hosts_blocked"] += 1
                    self.blocked_hosts.add(host)
                    self._event(url, depth, "challenge", status, challenge)
                    self.frontier.mark_visited(url)
                    self.db.mark_url_done(url, "challenge")
                    print(f"[BLOCK] Challenge detected: {challenge}. Host dừng, không bypass.")
                    continue

                # Extract links before duplicate-content decision. Even a duplicate
                # page can still contribute hyperlink graph edges.
                links, rejected = extract_links(parsed["soup"], final_url, self.profile)
                self.filter_reasons.update(rejected)

                # ====================================================
                # SECTION 6B - SAVE RAW -> PREPROCESS -> QUALITY GATE
                # ====================================================
                # Raw HTML is always preserved first for reproducibility/audit.
                raw_html = response.text
                import hashlib
                raw_hash = hashlib.sha256(raw_html.encode("utf-8", errors="ignore")).hexdigest()
                self.db.save_raw_page({
                    "url": final_url,
                    "domain": _host(final_url),
                    "source_site": self.site_id,
                    "domain_label": self.profile.get("domain_label", "unknown"),
                    "page_type": page_type,
                    "title": title,
                    "raw_html": raw_html,
                    "extracted_text": content,
                    "depth": depth,
                    "status_code": status,
                    "response_time": elapsed,
                    "crawled_at": utc_now(),
                    "raw_hash": raw_hash,
                    "metadata": metadata,
                })
                self.stats["raw_saved"] += 1
                self.depth_counts[depth] += 1
                self.type_counts[page_type] += 1

                processed = preprocess_document(title, content, self.profile)
                duplicate, fp = self.duplicates.check_and_add(processed.clean_text)

                if processed.valid and duplicate:
                    processed.valid = False
                    processed.rejection_reason = "duplicate_processed_content"
                    self.stats["duplicate_content"] += 1

                passage_count = self.db.save_processed(
                    final_url,
                    self.site_id,
                    self.profile.get("domain_label", "unknown"),
                    processed,
                )

                if processed.valid:
                    self.stats["processed_valid"] += 1
                    self.stats["passages_saved"] += passage_count
                    print(
                        f"[DATA] ACCEPT | vi={processed.language_score:.2f} "
                        f"rel={processed.relevance_score:.2f} q={processed.quality_score:.2f} "
                        f"| passages={passage_count}"
                    )
                else:
                    self.stats["processed_rejected"] += 1
                    self._event(url, depth, "data_rejected", status, processed.rejection_reason)
                    print(f"[DATA] REJECT | {processed.rejection_reason}")

                # ====================================================
                # SECTION 7 - LINKS -> NORMALIZE/FILTER -> DEPTH + 1
                # ====================================================
                next_depth = depth + 1
                for target in links:
                    if self.db.save_link(final_url, target, next_depth):
                        self.stats["links_saved"] += 1

                    if next_depth > max_depth:
                        self.stats["depth_limit_links"] += 1
                        continue

                    if self.frontier.add(target, next_depth):
                        if self.db.discover_url(target, next_depth, final_url):
                            self.stats["unique_urls_discovered"] += 1
                    else:
                        self.stats["duplicate_urls_skipped"] += 1

                # ====================================================
                # SECTION 8 - CHECKPOINT / RESUME STATE
                # ====================================================
                self.frontier.mark_visited(url)
                self.db.mark_url_done(url, "visited")
                self.db.conn.commit()

                print(
                    f"[OK] {status} | {page_type} | title={title[:70]!r} | "
                    f"links={len(links)} | frontier={len(self.frontier):,} | "
                    f"delay={self.limiter.current_delay(url):.1f}s"
                )

        except KeyboardInterrupt:
            stop_reason = "Interrupted by user; safe to resume"
            print("\n[STOP] Người dùng ngắt. SQLite đã giữ checkpoint để resume.")
        finally:
            # ========================================================
            # SECTION 9 - SUMMARY + VALIDATION
            # ========================================================
            self.db.conn.commit()
            summary = self._build_summary(stop_reason)
            self.db.finish_run(run_id, summary)
            validation = validate_database(self.db_path)
            write_report(self.report_dir, self.profile, summary, validation)
            self._print_summary(summary, validation)

        return summary

    def _build_summary(self, stop_reason: str) -> dict:
        return {
            "site_id": self.site_id,
            "raw_saved": int(self.stats["raw_saved"]),
            "processed_valid": int(self.stats["processed_valid"]),
            "processed_rejected": int(self.stats["processed_rejected"]),
            "passages_saved": int(self.stats["passages_saved"]),
            "requests_sent": int(self.stats["requests_sent"]),
            "links_saved": int(self.stats["links_saved"]),
            "unique_urls_discovered": int(self.stats["unique_urls_discovered"]),
            "duplicate_urls_skipped": int(self.stats["duplicate_urls_skipped"]),
            "duplicate_content": int(self.stats["duplicate_content"]),
            "failed_requests": int(self.stats["failed_requests"]),
            "robots_skipped": int(self.stats["robots_skipped"]),
            "rate_limit_responses": int(self.stats["rate_limit_responses"]),
            "hosts_blocked": int(self.stats["hosts_blocked"]),
            "frontier_remaining": len(self.frontier),
            "stop_reason": stop_reason,
            "http_statuses": dict(self.status_counts),
            "by_depth": dict(self.depth_counts),
            "by_page_type": dict(self.type_counts),
            "filter_reasons": dict(self.filter_reasons),
        }

    def _print_summary(self, summary: dict, validation: dict):
        print("\n" + "=" * 78)
        print("CRAWLING SUMMARY")
        print("=" * 78)
        print(f"Site                  : {self.profile['name']}")
        print(f"Raw pages saved       : {summary['raw_saved']:,}")
        print(f"Valid documents       : {summary.get('processed_valid',0):,}")
        print(f"Rejected documents    : {summary.get('processed_rejected',0):,}")
        print(f"Passages saved        : {summary.get('passages_saved',0):,}")
        print(f"Requests sent         : {summary['requests_sent']:,}")
        print(f"Links saved           : {summary['links_saved']:,}")
        print(f"Failed requests       : {summary['failed_requests']:,}")
        print(f"Robots skipped        : {summary['robots_skipped']:,}")
        print(f"Rate-limit responses  : {summary['rate_limit_responses']:,}")
        print(f"Hosts blocked         : {summary['hosts_blocked']:,}")
        print(f"Frontier remaining    : {summary['frontier_remaining']:,}")
        print(f"Stop reason           : {summary['stop_reason']}")
        for d, n in sorted(summary["by_depth"].items()):
            print(f"Depth {d:<2}               : {n:,}")
        for status, n in sorted(summary["http_statuses"].items()):
            print(f"HTTP {status:<3}              : {n:,}")
        print(f"Validation            : {validation.get('result')}")
        print(f"DB                    : {self.db_path}")
        print(f"Report                : {self.report_dir / 'RESULTS.md'}")
        print("=" * 78)
