# Pipeline chung

```text
Site Profile YAML
        ↓
Load Config
        ↓
Seed URL (depth 0)
        ↓
robots.txt check
        ↓
URL Frontier
        ↓
BFS
        ↓
Adaptive Rate Limiter
        ↓
HTTP Request
        ↓
Response
   ┌────┼───────────────┐
   │    │               │
  200  429/503        401/403/challenge
   │    │               │
   │    ↓               ↓
   │  backoff        report + stop host
   ↓
HTML
   ↓
BeautifulSoup
   ├───────────────┐
   ↓               ↓
Extract Data    Extract Links
   ↓               ↓
SHA-256        Normalize / Filter
   ↓               ↓
SQLite          depth + 1
                   ↓
                Frontier
                   ↺
```

## Output từng bước

| Bước | Output |
|---|---|
| Profile | config đã parse từ YAML |
| Seed | `FrontierItem(url, depth=0)` |
| robots | ALLOW/BLOCK + Crawl-delay |
| Frontier | URL + depth tiếp theo |
| Requests | HTTP Response |
| HTML | `response.content` |
| BeautifulSoup | DOM / `soup` |
| Extract Data | title + content + metadata |
| Extract Links | danh sách hyperlink |
| Normalize/Filter | URL hợp lệ |
| SHA-256 | content hash |
| SQLite | pages + links + metadata + state |
| Validation | PASS / CHECK REQUIRED + statistics |
