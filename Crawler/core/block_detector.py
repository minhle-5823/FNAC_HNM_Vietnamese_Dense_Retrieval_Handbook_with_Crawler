"""Detect access blocks/challenges without bypassing them."""
from __future__ import annotations


class BlockDetector:
    def __init__(self, profile: dict):
        policy = profile.get("policy", {})
        self.stop_host_statuses = {int(x) for x in policy.get("stop_host_statuses", [401, 403])}
        self.challenge_patterns = [str(x).lower() for x in policy.get("challenge_patterns", [])]

    def classify_status(self, status_code: int) -> str:
        if status_code in self.stop_host_statuses:
            return "blocked"
        if status_code == 429:
            return "rate_limited"
        if status_code == 503:
            return "temporary_unavailable"
        return "ok"

    def detect_challenge(self, title: str, text: str) -> str | None:
        haystack = f"{title}\n{text[:10000]}".lower()
        for marker in self.challenge_patterns:
            if marker and marker in haystack:
                return marker
        return None
