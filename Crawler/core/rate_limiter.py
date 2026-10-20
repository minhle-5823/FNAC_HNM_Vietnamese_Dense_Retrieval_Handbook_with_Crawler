"""Adaptive per-host crawl delay.

Normal success traffic never goes below profile.min_delay.
429/503 increases delay. Stable successes gradually recover toward min_delay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit


@dataclass
class HostRateState:
    delay: float
    last_request_at: float = 0.0
    success_streak: int = 0


class AdaptiveRateLimiter:
    def __init__(self, profile: dict):
        cfg = profile.get("rate_limit", {})
        self.initial_delay = float(cfg.get("initial_delay", 2.0))
        self.min_delay = float(cfg.get("min_delay", self.initial_delay))
        self.max_delay = float(cfg.get("max_delay", 120.0))
        self.success_window = int(cfg.get("success_window", 8))
        self.recovery_factor = float(cfg.get("recovery_factor", 0.8))
        self.states: dict[str, HostRateState] = {}

    def _state(self, host: str) -> HostRateState:
        if host not in self.states:
            self.states[host] = HostRateState(delay=max(self.initial_delay, self.min_delay))
        return self.states[host]

    def before_request(self, url: str, robots_delay: float = 0.0) -> float:
        host = urlsplit(url).netloc.lower()
        state = self._state(host)
        effective_delay = max(state.delay, float(robots_delay or 0.0))
        elapsed = time.monotonic() - state.last_request_at
        wait = max(0.0, effective_delay - elapsed)
        if wait > 0:
            time.sleep(wait)
        return wait

    def mark_request(self, url: str) -> None:
        host = urlsplit(url).netloc.lower()
        self._state(host).last_request_at = time.monotonic()

    def on_success(self, url: str) -> tuple[float, float]:
        host = urlsplit(url).netloc.lower()
        state = self._state(host)
        before = state.delay
        state.success_streak += 1
        if state.success_streak >= self.success_window and state.delay > self.min_delay:
            state.delay = max(self.min_delay, state.delay * self.recovery_factor)
            state.success_streak = 0
        return before, state.delay

    def on_throttle(self, url: str, headers: dict, attempt: int) -> tuple[float, float, float]:
        host = urlsplit(url).netloc.lower()
        state = self._state(host)
        before = state.delay
        retry_after = self._parse_retry_after(headers.get("Retry-After"))
        exponential = max(self.min_delay, self.initial_delay * (2 ** max(0, attempt - 1)))
        wait = retry_after if retry_after is not None else exponential
        state.delay = min(self.max_delay, max(state.delay * 2, wait, self.min_delay))
        state.success_streak = 0
        return before, state.delay, min(self.max_delay, max(wait, state.delay))

    def current_delay(self, url: str) -> float:
        return self._state(urlsplit(url).netloc.lower()).delay

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
