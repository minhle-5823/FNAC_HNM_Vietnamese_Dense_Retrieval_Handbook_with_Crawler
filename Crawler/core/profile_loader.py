"""Load and validate YAML site profiles.

REVIEW MAP
----------
SECTION 1: defaults shared by all sites
SECTION 2: load one profile
SECTION 3: discover available profiles/groups
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# SECTION 1 - DEFAULT PROFILE VALUES
# ============================================================
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "crawl": {
        "max_depth": 2,
        "max_pages": 300,
        "max_requests": 1000,
        "request_timeout": 20,
    },
    "rate_limit": {
        "initial_delay": 2.0,
        "min_delay": 2.0,
        "max_delay": 120.0,
        "max_retries": 4,
        "retry_statuses": [429, 503],
        "success_window": 8,
        "recovery_factor": 0.8,
    },
    "http": {
        "follow_redirects": False,
        "headers": {},
    },
    "policy": {
        "robots": "respect",
        "robots_unavailable": "block",
        "stop_host_statuses": [401, 403],
        "challenge_patterns": [
            "just a moment",
            "access denied",
            "verify you are human",
            "captcha",
            "cf-chl-",
        ],
    },
    "url": {
        "allow_path_regex": [],
        "deny_path_regex": [],
        "ignored_extensions": [
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
            ".css", ".js", ".pdf", ".zip", ".rar", ".7z", ".gz",
            ".mp3", ".mp4", ".avi", ".mov", ".webm", ".ogg", ".wav",
            ".woff", ".woff2", ".ttf",
        ],
        "strip_query_prefixes": ["utm_"],
        "strip_query_keys": ["fbclid", "gclid", "mc_cid", "mc_eid"],
        "sort_query": True,
        "trailing_slash": "preserve",
    },
    "page_types": [],
    "extract": {
        "parser": "html.parser",
        "title": {"selectors": ["title"]},
        "content": {
            "container_selectors": ["main", "article", "body"],
            "remove_selectors": ["script", "style", "noscript", "template"],
        },
        "metadata": [],
    },
    "links": {
        "default_follow": True,
        "default_selector": "a[href]",
        "rules": [],
    },
    "domain_label": "unknown",
    "preprocess": {
        "min_words": 60,
        "min_chars": 300,
        "min_vietnamese_score": 0.35,
        "min_relevance_score": 0.20,
        "min_keyword_hits": 1,
        "relevance_keywords": [],
        "chunk_words": 220,
        "chunk_overlap_words": 40,
        "min_chunk_words": 45,
        "reject_text_regex": [
            "\\b404\\b", "not found", "access denied", "verify you are human",
            "captcha", "just a moment"
        ],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# ============================================================
# SECTION 2 - LOAD + VALIDATE ONE SITE PROFILE
# ============================================================
def load_profile(profile_path: Path) -> dict[str, Any]:
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    profile = _deep_merge(DEFAULTS, raw)
    profile["_profile_path"] = str(profile_path)

    required = ["id", "name", "seed_urls", "allowed_domains"]
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(
            f"Profile {profile_path} thiếu field bắt buộc: {', '.join(missing)}"
        )

    if not isinstance(profile["seed_urls"], list):
        raise ValueError("seed_urls phải là list")
    if not isinstance(profile["allowed_domains"], list):
        raise ValueError("allowed_domains phải là list")

    return profile


# ============================================================
# SECTION 3 - PROFILE / GROUP DISCOVERY
# ============================================================
def profile_path(repo_root: Path, site_id: str) -> Path:
    path = repo_root / "site_profiles" / site_id / "profile.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy site profile: {site_id}")
    return path


def load_site(repo_root: Path, site_id: str) -> dict[str, Any]:
    return load_profile(profile_path(repo_root, site_id))


def list_sites(repo_root: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    root = repo_root / "site_profiles"
    for path in sorted(root.glob("*/profile.yaml")):
        try:
            p = load_profile(path)
            result.append({
                "id": p["id"],
                "name": p["name"],
                "topic": p.get("topic", ""),
                "enabled": str(bool(p.get("enabled", True))),
            })
        except Exception as exc:  # show broken profile instead of hiding it
            result.append({
                "id": path.parent.name,
                "name": f"BROKEN PROFILE: {exc}",
                "topic": "",
                "enabled": "False",
            })
    return result


def load_group(repo_root: Path, group_id: str) -> list[str]:
    path = repo_root / "groups" / f"{group_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy group: {group_id}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    sites = data.get("sites") or []
    if not isinstance(sites, list):
        raise ValueError("groups/<name>.yaml phải chứa sites: [..]")
    return [str(x) for x in sites]
