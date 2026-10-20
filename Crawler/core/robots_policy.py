"""Live robots.txt policy checker.

REVIEW MAP
----------
SECTION 1 : origin / safe robots redirect checks
SECTION 2 : load robots.txt with retry/backoff
SECTION 3 : 200 parse, 404/410 = no robots restrictions
SECTION 4 : can_fetch decision

Safety behavior:
- Respect robots.txt.
- 404/410 means no robots file, not an automatic block.
- Follow only a small number of SAFE redirects for robots.txt.
- Redirect target must stay inside profile allowed_domains.
- 429/503 uses existing adaptive backoff.
- Other errors remain fail-closed unless profile explicitly says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from .rate_limiter import AdaptiveRateLimiter


@dataclass
class RobotsDecision:
    allowed: bool
    crawl_delay: float
    reason: str
    robots_url: str
    status_code: int | None


class RobotsPolicy:
    def __init__(
        self,
        profile: dict,
        session: requests.Session,
        limiter: AdaptiveRateLimiter,
        event_callback=None,
    ):
        self.profile = profile
        self.session = session
        self.limiter = limiter
        self.event_callback = event_callback
        self.cache: dict[str, tuple[RobotFileParser | None, float, str, int | None, str]] = {}
        self.user_agent = profile.get("user_agent", "VIRERetrievalCrawler/1.0")

    # ========================================================
    # SECTION 1 - ORIGIN + SAFE ROBOTS REDIRECTS
    # ========================================================
    def _origin(self, url: str) -> str:
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    def _redirect_allowed(self, url: str) -> bool:
        p = urlsplit(url)
        allowed = {str(x).lower() for x in self.profile.get("allowed_domains", [])}
        return (
            p.scheme in {"http", "https"}
            and bool(p.hostname)
            and ((p.hostname or "").lower() in allowed or p.netloc.lower() in allowed)
        )

    def _get_robots_response(self, start_url: str):
        current = start_url
        max_redirects = 3
        max_retries = int(self.profile.get("rate_limit", {}).get("max_retries", 4))

        for redirect_no in range(max_redirects + 1):
            response = None
            for attempt in range(1, max_retries + 1):
                self.limiter.before_request(current)
                response = self.session.get(
                    current,
                    timeout=float(self.profile["crawl"].get("request_timeout", 20)),
                    allow_redirects=False,
                )
                self.limiter.mark_request(current)

                if response.status_code in {429, 503} and attempt < max_retries:
                    _, _, wait = self.limiter.on_throttle(
                        current, response.headers, attempt
                    )
                    import time
                    time.sleep(wait)
                    continue
                break

            if response is None:
                return None, current, "robots_no_response"

            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                target = urljoin(current, location) if location else ""
                if not target or not self._redirect_allowed(target):
                    return response, current, "robots_redirect_rejected"
                if redirect_no >= max_redirects:
                    return response, current, "robots_too_many_redirects"
                current = target
                continue

            return response, current, "robots_response"

        return None, current, "robots_no_response"

    # ========================================================
    # SECTION 2/3 - LOAD + PARSE ROBOTS
    # ========================================================
    def _load(self, url: str):
        origin = self._origin(url)
        if origin in self.cache:
            return self.cache[origin]

        robots_url = origin + "/robots.txt"
        unavailable_policy = self.profile.get("policy", {}).get(
            "robots_unavailable", "block"
        )
        parser: RobotFileParser | None = None
        crawl_delay = 0.0
        reason = ""
        status = None
        effective_robots_url = robots_url

        try:
            response, effective_robots_url, redirect_reason = self._get_robots_response(
                robots_url
            )
            status = response.status_code if response is not None else None

            if (
                status == 200
                and response is not None
                and "<html" not in response.text[:1000].lower()
            ):
                parser = RobotFileParser()
                parser.set_url(effective_robots_url)
                parser.parse(response.text.splitlines())
                crawl_delay = (
                    parser.crawl_delay(self.user_agent)
                    or parser.crawl_delay("*")
                    or 0.0
                )
                reason = (
                    "robots_loaded_after_redirect"
                    if effective_robots_url != robots_url
                    else "robots_loaded"
                )
                self.limiter.on_success(effective_robots_url)

            elif status in {404, 410}:
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
                reason = "robots_not_found"

            else:
                if redirect_reason != "robots_response":
                    reason = redirect_reason
                else:
                    reason = f"robots_unavailable_http_{status}"

                if unavailable_policy == "allow":
                    parser = RobotFileParser()
                    parser.parse(["User-agent: *", "Allow: /"])

        except requests.RequestException as exc:
            reason = f"robots_request_error:{type(exc).__name__}"
            if unavailable_policy == "allow":
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])

        self.cache[origin] = (
            parser,
            float(crawl_delay),
            reason,
            status,
            effective_robots_url,
        )

        if self.event_callback:
            self.event_callback(
                url=effective_robots_url,
                depth=None,
                kind="robots",
                status_code=status,
                message=reason,
            )
        return self.cache[origin]

    # ========================================================
    # SECTION 4 - FINAL ROBOTS DECISION
    # ========================================================
    def can_fetch(self, url: str) -> RobotsDecision:
        default_robots_url = self._origin(url) + "/robots.txt"

        if self.profile.get("policy", {}).get("robots", "respect") != "respect":
            return RobotsDecision(
                True, 0.0, "robots_check_disabled_by_profile",
                default_robots_url, None
            )

        parser, crawl_delay, reason, status, effective_robots_url = self._load(url)

        if parser is None:
            return RobotsDecision(
                False, crawl_delay, reason, effective_robots_url, status
            )

        allowed = parser.can_fetch(self.user_agent, url)
        final_reason = reason if not allowed else f"allow:{reason}"
        if not allowed:
            final_reason = f"disallow:{reason}"

        return RobotsDecision(
            allowed=bool(allowed),
            crawl_delay=crawl_delay,
            reason=final_reason,
            robots_url=effective_robots_url,
            status_code=status,
        )
