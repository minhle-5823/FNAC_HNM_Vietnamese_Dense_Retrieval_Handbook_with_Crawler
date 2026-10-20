"""URL normalization + filtering shared by every website profile."""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlencode, unquote, urljoin, urlsplit, urlunsplit


NON_WEB_SCHEMES = ("mailto:", "javascript:", "tel:", "data:")


def normalize_url(href: str, base_url: str, profile: dict) -> str | None:
    """Convert relative href -> canonical HTTP(S) URL used by the frontier."""
    if not href:
        return None
    href = str(href).strip()
    if not href or href.lower().startswith(NON_WEB_SCHEMES) or href.startswith("#"):
        return None

    try:
        parsed = urlsplit(urljoin(base_url, href))
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None

        scheme = parsed.scheme.lower()
        host = parsed.hostname.lower()
        port = parsed.port
        if port and (scheme, port) not in {("http", 80), ("https", 443)}:
            host = f"{host}:{port}"

        path = parsed.path or "/"
        slash_policy = profile.get("url", {}).get("trailing_slash", "preserve")
        if slash_policy == "remove" and path != "/":
            path = path.rstrip("/") or "/"

        url_cfg = profile.get("url", {})
        strip_prefixes = tuple(x.lower() for x in url_cfg.get("strip_query_prefixes", []))
        strip_keys = {x.lower() for x in url_cfg.get("strip_query_keys", [])}
        query_items = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            kl = key.lower()
            if kl in strip_keys or any(kl.startswith(p) for p in strip_prefixes):
                continue
            query_items.append((key, value))
        if url_cfg.get("sort_query", True):
            query_items.sort()
        query = urlencode(query_items)

        return urlunsplit((scheme, host, path, query, ""))
    except (TypeError, ValueError):
        return None


def is_allowed_url(url: str, profile: dict) -> tuple[bool, str]:
    """Global site-level filter before a URL can enter the frontier."""
    try:
        p = urlsplit(url)
    except ValueError:
        return False, "malformed_url"

    if p.scheme not in {"http", "https"}:
        return False, "non_http"

    allowed_domains = {d.lower() for d in profile.get("allowed_domains", [])}
    if p.netloc.lower() not in allowed_domains and (p.hostname or "").lower() not in allowed_domains:
        return False, "outside_domain"

    decoded_path = unquote(p.path or "/")
    suffix = PurePosixPath(decoded_path.lower()).suffix
    ignored = {x.lower() for x in profile.get("url", {}).get("ignored_extensions", [])}
    if suffix and suffix in ignored:
        return False, "ignored_extension"

    cfg = profile.get("url", {})
    allow_patterns = cfg.get("allow_path_regex", [])
    deny_patterns = cfg.get("deny_path_regex", [])

    if allow_patterns and not any(re.search(pattern, decoded_path) for pattern in allow_patterns):
        return False, "path_not_allowed"
    if any(re.search(pattern, decoded_path) for pattern in deny_patterns):
        return False, "path_denied"

    return True, "accepted"
