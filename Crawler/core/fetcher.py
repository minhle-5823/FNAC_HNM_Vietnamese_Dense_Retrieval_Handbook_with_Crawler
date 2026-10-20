"""HTTP fetcher with adaptive rate limiting and safe retries."""
from __future__ import annotations

import time

import requests

from .rate_limiter import AdaptiveRateLimiter


class Fetcher:
    def __init__(self, profile: dict, session: requests.Session, limiter: AdaptiveRateLimiter):
        self.profile = profile
        self.session = session
        self.limiter = limiter
        cfg = profile.get("rate_limit", {})
        self.max_retries = int(cfg.get("max_retries", 4))
        self.retry_statuses = {int(x) for x in cfg.get("retry_statuses", [429, 503])}
        self.timeout = float(profile.get("crawl", {}).get("request_timeout", 20))
        self.follow_redirects = bool(profile.get("http", {}).get("follow_redirects", False))

    def fetch(self, url: str, robots_delay: float = 0.0):
        last_response = None
        last_elapsed = None
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            self.limiter.before_request(url, robots_delay)
            started = time.perf_counter()
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=self.follow_redirects,
                )
                elapsed = time.perf_counter() - started
                self.limiter.mark_request(url)
                last_response, last_elapsed = response, elapsed

                if response.status_code in self.retry_statuses and attempt < self.max_retries:
                    before, after, wait = self.limiter.on_throttle(url, response.headers, attempt)
                    print(
                        f"[RATE] HTTP {response.status_code} | {before:.1f}s -> {after:.1f}s | "
                        f"retry sau {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                if 200 <= response.status_code < 300:
                    before, after = self.limiter.on_success(url)
                    if after != before:
                        print(f"[RATE] ổn định | delay {before:.1f}s -> {after:.1f}s")
                return response, elapsed, attempt, None

            except requests.RequestException as exc:
                self.limiter.mark_request(url)
                last_error = exc
                if attempt < self.max_retries:
                    wait = min(
                        float(self.profile["rate_limit"].get("max_delay", 120)),
                        max(2.0, float(self.profile["rate_limit"].get("initial_delay", 2)) * (2 ** (attempt - 1))),
                    )
                    print(f"[RETRY] {type(exc).__name__} | chờ {wait:.1f}s")
                    time.sleep(wait)
                    continue

        if last_response is not None:
            return last_response, last_elapsed, self.max_retries, None
        return None, None, self.max_retries, last_error
