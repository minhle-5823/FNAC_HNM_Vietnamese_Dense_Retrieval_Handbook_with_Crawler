# Sources reviewed when building this unified repo

Repo chung được tổng hợp từ 5 ZIP do nhóm cung cấp:

1. `Wikipedia.zip`
2. `Metacritic_Crawler.zip`
3. `letterboxd_crawler_v4.zip`
4. `rottentomatoes_assignment_crawler_MAX300.zip`
5. `books_crawler_project.zip`

Các phần được giữ lại/chuẩn hóa:

- seed URL và allowed domain;
- BFS + URL Frontier;
- depth/page limit;
- Requests + BeautifulSoup;
- URL normalize/filter;
- robots.txt handling;
- crawl delay;
- SQLite pages/links;
- selector/metadata riêng của Wikipedia, Letterboxd và Books;
- policy/block behavior an toàn từ crawler Metacritic;
- cấu hình MAX300 của Rotten Tomatoes.

Các phần bị loại khỏi repo chung:

- `.venv` nằm trong ZIP Metacritic;
- `__pycache__`;
- code BFS/parser/database bị copy lặp ở từng crawler;
- database/output cũ;
- những kết luận live-site chưa được crawler thành viên xác minh.

Website live có thể thay đổi sau thời điểm các source trên được viết. Vì vậy policy/robots được kiểm tra lại lúc chạy và selector nằm trong YAML để cập nhật mà không sửa core engine.


## Replacement source

- Letterboxd was removed from the active movies group after live HTTP 403 tests.
- Added official Library of Congress `Selections from the National Film Registry` profile.
