"""Generic HTML parser/extractor driven by YAML site profiles.

REVIEW MAP
----------
SECTION 1: parse HTML + classify page
SECTION 2: extract title/content
SECTION 3: metadata extractors (CSS / JSON-LD / label-value table)
SECTION 4: hyperlink extraction + focused rules
"""
from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlsplit, urlunsplit

from bs4 import BeautifulSoup, NavigableString
import soupsieve

from .url_tools import is_allowed_url, normalize_url


# ============================================================
# SECTION 1 - PARSE HTML + PAGE TYPE
# ============================================================
def make_soup(html: str | bytes, profile: dict) -> BeautifulSoup:
    parser = profile.get("extract", {}).get("parser", "html.parser")
    return BeautifulSoup(html, parser)


def classify_page(url: str, profile: dict) -> str:
    path = unquote(urlsplit(url).path or "/")
    for rule in profile.get("page_types", []):
        if re.search(rule.get("url_regex", r"$^"), path):
            return str(rule.get("name", "page"))
    return "page"


# ============================================================
# SECTION 2 - TITLE + VISIBLE CONTENT
# ============================================================
def _first_node(soup: BeautifulSoup, selectors: list[str]):
    for selector in selectors:
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None
        if node is not None:
            return node
    return None


def extract_title(soup: BeautifulSoup, profile: dict) -> str:
    selectors = profile.get("extract", {}).get("title", {}).get("selectors", ["title"])
    node = _first_node(soup, selectors)
    return node.get_text(" ", strip=True) if node else ""


def extract_content(soup: BeautifulSoup, profile: dict) -> str:
    cfg = profile.get("extract", {}).get("content", {})
    root = _first_node(soup, cfg.get("container_selectors", ["main", "article", "body"]))
    root = root or soup.body or soup

    # Copy selected content before deleting nodes so metadata/link extraction can
    # still use the original soup.
    working = BeautifulSoup(str(root), profile.get("extract", {}).get("parser", "html.parser"))
    for selector in cfg.get("remove_selectors", ["script", "style", "noscript", "template"]):
        try:
            for node in working.select(selector):
                node.decompose()
        except Exception:
            continue

    return re.sub(r"\s+", " ", working.get_text(" ", strip=True)).strip()


# ============================================================
# SECTION 3 - METADATA EXTRACTORS
# ============================================================
def _css_text(soup: BeautifulSoup, rule: dict):
    selectors = rule.get("selectors") or ([rule["selector"]] if rule.get("selector") else [])
    node = _first_node(soup, selectors)
    return node.get_text(" ", strip=True) if node else None


def _css_attr(soup: BeautifulSoup, rule: dict):
    selectors = rule.get("selectors") or ([rule["selector"]] if rule.get("selector") else [])
    node = _first_node(soup, selectors)
    if not node:
        return None
    value = node.get(rule.get("attribute", "content"))
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return value


def _class_choice(soup: BeautifulSoup, rule: dict):
    node = _first_node(soup, rule.get("selectors") or [rule.get("selector", "")])
    if not node:
        return None
    classes = set(node.get("class", []))
    for choice in rule.get("choices", []):
        if choice in classes:
            return choice
    return None


def _iter_json_ld_objects(soup: BeautifulSoup):
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        stack = data if isinstance(data, list) else [data]
        for item in stack:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                for child in graph:
                    if isinstance(child, dict):
                        yield child
            yield item


def _json_path(value, path: str):
    """Tiny JSON path resolver supporting `director[].name`."""
    current = [value]
    for token in path.split("."):
        expand = token.endswith("[]")
        key = token[:-2] if expand else token
        next_values = []
        for item in current:
            if not isinstance(item, dict) or key not in item:
                continue
            child = item[key]
            if expand:
                if isinstance(child, list):
                    next_values.extend(child)
                elif child is not None:
                    next_values.append(child)
            else:
                next_values.append(child)
        current = next_values

    flat = []
    for item in current:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    flat = [x for x in flat if x not in (None, "")]
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    return ", ".join(str(x) for x in flat)


def _json_ld(soup: BeautifulSoup, rule: dict):
    types = {str(x) for x in rule.get("object_types", [])}
    for item in _iter_json_ld_objects(soup):
        obj_type = item.get("@type")
        actual_types = set(obj_type if isinstance(obj_type, list) else [obj_type])
        if types and not (types & actual_types):
            continue
        value = _json_path(item, rule.get("path", ""))
        if value not in (None, ""):
            return value
    return None


def _label_value_table(soup: BeautifulSoup, rule: dict) -> dict[str, str]:
    container = _first_node(soup, rule.get("container_selectors", []))
    if not container:
        return {}

    fields = rule.get("fields", {})
    result: dict[str, str] = {}
    for row in container.select(rule.get("row_selector", "tr")):
        label_node = row.select_one(rule.get("label_selector", "th"))
        value_node = row.select_one(rule.get("value_selector", "td"))
        if not label_node or not value_node:
            continue
        label = label_node.get_text(" ", strip=True)
        target_key = fields.get(label)
        if target_key:
            result[str(target_key)] = value_node.get_text(" ", strip=True)
    return result


def extract_metadata(soup: BeautifulSoup, page_type: str, profile: dict) -> dict[str, object]:
    result: dict[str, object] = {}
    for rule in profile.get("extract", {}).get("metadata", []):
        only_types = set(rule.get("when_page_types", []))
        if only_types and page_type not in only_types:
            continue

        kind = rule.get("type")
        if kind == "label_value_table":
            result.update(_label_value_table(soup, rule))
            continue

        name = rule.get("name")
        if not name:
            continue
        value = None
        if kind == "css_text":
            value = _css_text(soup, rule)
        elif kind == "css_attr":
            value = _css_attr(soup, rule)
        elif kind == "class_choice":
            value = _class_choice(soup, rule)
        elif kind == "json_ld":
            value = _json_ld(soup, rule)

        if value not in (None, "", []):
            result[str(name)] = value
    return result


# ============================================================
# SECTION 4 - HYPERLINK EXTRACTION + FOCUSED RULES
# ============================================================
def _find_matching_ancestor(node, selector: str):
    parent = node.parent
    while parent is not None and getattr(parent, "name", None):
        try:
            if soupsieve.match(selector, parent):
                return parent
        except Exception:
            return None
        parent = parent.parent
    return None


def _rule_matches_source(rule: dict, current_url: str) -> bool:
    path = unquote(urlsplit(current_url).path or "/")
    pattern = rule.get("source_path_regex")
    return not pattern or bool(re.search(pattern, path))


def extract_links(soup: BeautifulSoup, current_url: str, profile: dict):
    link_cfg = profile.get("links", {})
    rules = [r for r in link_cfg.get("rules", []) if _rule_matches_source(r, current_url)]

    if not rules:
        if not link_cfg.get("default_follow", True):
            return [], {"no_link_rule": 1}
        rules = [{"selector": link_cfg.get("default_selector", "a[href]")}]

    accepted: list[str] = []
    seen: set[str] = set()
    rejected: dict[str, int] = {}

    def reject(reason: str):
        rejected[reason] = rejected.get(reason, 0) + 1

    for rule in rules:
        selector = rule.get("selector", "a[href]")
        try:
            anchors = soup.select(selector)
        except Exception:
            reject("bad_selector")
            continue

        for anchor in anchors:
            href = anchor.get("href")
            if not href:
                reject("missing_href")
                continue

            anchor_text = anchor.get_text(" ", strip=True)
            allow_anchor = rule.get("anchor_text_allow_regex")
            deny_anchor = rule.get("anchor_text_deny_regex")
            if allow_anchor and not re.search(allow_anchor, anchor_text):
                reject("anchor_text_not_allowed")
                continue
            if deny_anchor and re.search(deny_anchor, anchor_text):
                reject("anchor_text_denied")
                continue

            ancestor_selector = rule.get("ancestor_selector")
            if ancestor_selector:
                ancestor = _find_matching_ancestor(anchor, ancestor_selector)
                if ancestor is None:
                    reject("ancestor_missing")
                    continue
                ancestor_regex = rule.get("ancestor_text_regex")
                if ancestor_regex and not re.search(
                    ancestor_regex, ancestor.get_text(" ", strip=True)
                ):
                    reject("ancestor_text_not_matched")
                    continue

            target = normalize_url(href, current_url, profile)
            if target and profile.get("extract", {}).get("strategy") == "fpt_hifpt":
                # FPT emits some links as /hi-fpt//huong-dan-su-dung/...
                # Canonicalize only this site's URLs, without changing other crawlers.
                parsed_target = urlsplit(target)
                normalized_path = re.sub(r"/{2,}", "/", parsed_target.path)
                target = urlunsplit((parsed_target.scheme, parsed_target.netloc,
                                     normalized_path, parsed_target.query, ""))
            if not target:
                reject("normalize_rejected")
                continue

            # Current page often links to itself through logo/menu/category tabs.
            # Do not re-enqueue it at depth+1.
            current_normalized = normalize_url(current_url, current_url, profile)
            if current_normalized and target == current_normalized:
                reject("self_link")
                continue

            ok, reason = is_allowed_url(target, profile)
            if not ok:
                reject(reason)
                continue

            target_path = unquote(urlsplit(target).path or "/")
            allow_target = rule.get("target_path_regex")
            deny_target = rule.get("deny_target_path_regex")
            if allow_target and not re.search(allow_target, target_path):
                reject("target_rule_not_allowed")
                continue
            if deny_target and re.search(deny_target, target_path):
                reject("target_rule_denied")
                continue

            if target in seen:
                reject("duplicate_link")
                continue
            seen.add(target)
            accepted.append(target)

    return accepted, rejected


def _extract_fpt_faq(soup: BeautifulSoup, page_type: str) -> tuple[str, str]:
    """Extract just the FAQ answer, excluding the long FPT menu/category list.

    On FPT's FAQ pages the question heading precedes the answer and the
    'Nội dung này có hữu ích không?' feedback heading follows it. Using these
    structural boundaries is more reliable than treating the entire page as an
    article, since each detail page also embeds the whole FAQ navigation list.
    """
    if page_type == "faq_listing":
        return "Câu hỏi thường gặp FPT Telecom", ""  # seed is for discovering links

    feedback = soup.find(string=lambda s: bool(s and "nội dung này có hữu ích không" in s.casefold()))
    if feedback is None:
        # Pages whose FAQ data is still loading should be retained in RAW only.
        return "", ""
    heading = feedback.parent.find_previous(["h2", "h3", "h4", "h5", "h6"])
    if heading is None:
        return "", ""
    question = heading.get_text(" ", strip=True)
    if not question or question.casefold().startswith(("chọn chủ đề", "câu hỏi thường gặp")):
        return "", ""

    answer_parts = []
    for node in heading.next_elements:
        if node is feedback:
            break
        if not isinstance(node, NavigableString):
            continue
        if any(getattr(p, 'name', '') in {'script', 'style', 'noscript', 'template'}
               for p in node.parents):
            continue
        part = ' '.join(str(node).split())
        if part and part.casefold() not in {'sao chép liên kết', 'copy link'}:
            answer_parts.append(part)
    return question, ' '.join(answer_parts).strip()



def _extract_fpt_hifpt(soup: BeautifulSoup, page_type: str) -> tuple[str, str]:
    """Extract the H1 question and its answer from a Hi FPT help article.

    Listing/category pages exist only to discover links. On a detail page the
    answer follows its H1; the site's main navigation and footer must never
    become retrieval text.  No guessed text is generated for missing content.
    """
    if page_type != "help_detail":
        return "Hướng dẫn sử dụng Hi FPT", ""

    heading = _first_node(soup, ["main h1", "article h1", "h1"])
    if heading is None:
        return "", ""
    title = heading.get_text(" ", strip=True)
    if not title or "chúng tôi có thể giúp gì" in title.casefold():
        return "", ""

    # Relevant answer ends at feedback widget, recommended-articles block,
    # or the corporate footer. Avoid collecting links from the surrounding UI.
    stop_texts = (
        "nội dung này có hữu ích", "câu hỏi liên quan", "bài viết liên quan",
        "bài viết khác", "xem thêm bài viết", "về fpt telecom",
        "công ty cổ phần viễn thông fpt", "giấy chứng nhận đkdn",
        "gửi đánh giá", "đánh giá nội dung", "các chủ đề khác",
    )
    forbidden_ancestors = {"script", "style", "noscript", "template", "nav",
                           "header", "footer", "aside", "form", "button", "svg"}
    out: list[str] = []
    for node in heading.next_elements:
        # next_elements begins inside the H1; skip its own text.
        if not isinstance(node, NavigableString):
            continue
        if heading in node.parents:
            continue
        ancestors = list(node.parents)
        if any(getattr(a, "name", "") in forbidden_ancestors for a in ancestors):
            continue
        value = " ".join(str(node).split())
        if not value:
            continue
        low = value.casefold()
        if any(marker in low for marker in stop_texts):
            break
        if low in {"sao chép liên kết", "copy link", "chia sẻ", "trang chủ"}:
            continue
        # Standalone related-article links tend to follow the answer; allow
        # links inside actual paragraphs/lists, but not a sidebar link farm.
        if any(getattr(a, "name", "") == "a" for a in ancestors):
            if not any(getattr(a, "name", "") in {"p", "li"} for a in ancestors):
                continue
        out.append(value)
        # Guard against unbounded markup with no footer/feedback marker.
        if len(" ".join(out)) > 14000:
            break
    answer = re.sub(r"\s+", " ", " ".join(out)).strip()
    return title, answer

def parse_page(html: str | bytes, url: str, profile: dict) -> dict:
    soup = make_soup(html, profile)
    page_type = classify_page(url, profile)
    strategy = profile.get("extract", {}).get("strategy")
    if strategy == "fpt_faq":
        title, content = _extract_fpt_faq(soup, page_type)
    elif strategy == "fpt_hifpt":
        title, content = _extract_fpt_hifpt(soup, page_type)
    else:
        title, content = extract_title(soup, profile), extract_content(soup, profile)
    return {
        "soup": soup,
        "page_type": page_type,
        "title": title,
        "content": content,
        "metadata": extract_metadata(soup, page_type, profile),
    }
