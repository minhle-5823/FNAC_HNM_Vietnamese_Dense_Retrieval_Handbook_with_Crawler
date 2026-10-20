# Cách thêm website mới

Mục tiêu của repo: **không copy crawler.py**.

Tạo:

```text
site_profiles/new_site/
├── profile.yaml
└── NOTES.md
```

## 1. Bắt buộc

```yaml
id: new_site
name: New Site
seed_urls:
  - https://example.com/movies/
allowed_domains:
  - example.com
```

## 2. Crawl limits

```yaml
crawl:
  max_depth: 2
  max_pages: 300
  max_requests: 1000
  request_timeout: 20
```

## 3. Policy + rate limit

```yaml
policy:
  robots: respect
  robots_unavailable: block
  stop_host_statuses: [401, 403]

rate_limit:
  initial_delay: 2
  min_delay: 2
  max_delay: 120
  max_retries: 4
  retry_statuses: [429, 503]
```

## 4. URL rules

```yaml
url:
  allow_path_regex:
    - '^/movie/'
    - '^/movies/'
  deny_path_regex:
    - '^/login'
    - '^/user/'
```

## 5. Selector title/content

Dùng browser DevTools → Inspect HTML thật → xác định tag/class/id.

```yaml
extract:
  title:
    selectors:
      - 'h1.movie-title'
      - 'h1'
      - 'title'
  content:
    container_selectors:
      - 'main'
      - 'article'
      - 'body'
```

Selector đầu không thấy thì engine thử selector kế tiếp.

## 6. Metadata bằng CSS

```yaml
metadata:
  - name: price
    type: css_text
    selector: '.price'
```

Hoặc attribute:

```yaml
  - name: poster
    type: css_attr
    selector: 'img.poster'
    attribute: src
```

## 7. JSON-LD

```yaml
  - name: director
    type: json_ld
    object_types: [Movie]
    path: director[].name
```

Engine hỗ trợ path như:

```text
name
aggregateRating.ratingValue
director[].name
```

## 8. Bảng label/value

Hợp với Wikipedia infobox:

```yaml
  - type: label_value_table
    container_selectors: ['table.infobox']
    row_selector: tr
    label_selector: th
    value_selector: td
    fields:
      'Directed by': directed_by
      'Running time': running_time
```

## 9. Hyperlink rules

Đơn giản:

```yaml
links:
  default_follow: true
  default_selector: 'a[href]'
```

Focused hơn:

```yaml
links:
  default_follow: false
  rules:
    - source_path_regex: '^/movies/$'
      selector: 'a[href]'
      target_path_regex: '^/movie/'
```

Có thể yêu cầu link phải nằm trong một ancestor có text nhất định:

```yaml
    - selector: 'li a[href]'
      ancestor_selector: li
      ancestor_text_regex: '\b20\d{2}\b'
```

## 10. Kiểm tra

```powershell
python main.py inspect --site new_site
python main.py robots --site new_site
python main.py crawl --site new_site --max-pages 20 --reset
```
