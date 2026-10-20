# Code Review Guide

File này dành cho lúc thầy mở code và hỏi trực tiếp từng đoạn.

## `main.py`

- CLI `sites`, `inspect`, `robots`, `crawl`, `crawl-all`, `validate`.
- `cmd_crawl()` tạo `UnifiedCrawler` và gọi `.run()`.

## `core/crawler.py`

Search trực tiếp các comment:

```text
SECTION 1 - INITIALIZE ENGINE
SECTION 2 - RESUME OR FRESH SEED
SECTION 3 - ROBOTS.TXT PREFLIGHT
SECTION 4 - BFS CRAWL LOOP
SECTION 5 - FETCH + HTTP / BLOCK HANDLING
SECTION 6 - PARSE HTML + EXTRACT DATA
SECTION 7 - LINKS -> NORMALIZE/FILTER -> DEPTH + 1
SECTION 8 - CHECKPOINT / RESUME STATE
SECTION 9 - SUMMARY + VALIDATION
```

Đây là file điều phối toàn pipeline.

## `core/frontier.py`

BFS:

```python
self.queue.append(...)
self.queue.popleft()
```

URL mới vào cuối, URL cũ nhất ra đầu → FIFO → BFS.

## `core/url_tools.py`

- relative → absolute URL;
- bỏ fragment/tracking query;
- domain filter;
- path allow/deny;
- extension filter.

## `core/robots_policy.py`

`can_fetch()` quyết định URL có được phép crawl theo robots.txt.

Không lấy được robots và profile đặt `block` → không suy đoán là được phép.

## `core/rate_limiter.py`

- delay theo host;
- đọc `Retry-After`;
- 429/503 tăng delay;
- stable success giảm dần về `min_delay`.

## `core/fetcher.py`

Chỗ gửi HTTP thật:

```python
self.session.get(...)
```

Có retry cho timeout/429/503.

## `core/block_detector.py`

Nhận diện HTTP block và challenge text.

Không bypass.

## `core/extractor.py`

- `make_soup()` → BeautifulSoup;
- `extract_title()`;
- `extract_content()`;
- `extract_metadata()`;
- `extract_links()`.

Website-specific selector không hardcode ở file này, nó đến từ YAML.

## `core/duplicate.py`

```python
hashlib.sha256(...)
```

Hash content để phát hiện exact duplicate.

## `core/database.py`

- `url_state` = persistent frontier / resume;
- `pages` = documents;
- `links` = hyperlink graph;
- `metadata` = field riêng của site;
- `events` = error/block;
- `crawl_runs` = statistics.

## `site_profiles/*/profile.yaml`

Đây là nơi thầy hỏi:

> “Selector lấy ở đâu?”

Trả lời:

> Inspect DOM của website, xác định tag/class/id chứa dữ liệu, rồi khai báo selector/rule trong profile. Engine chung đọc profile để parse.
