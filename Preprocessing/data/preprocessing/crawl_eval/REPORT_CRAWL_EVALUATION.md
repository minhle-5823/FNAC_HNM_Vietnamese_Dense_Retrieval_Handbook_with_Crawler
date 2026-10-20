# SEG301 | P@5 / Recall@5 chỉ trên dữ liệu crawl

- Corpus: **50 tài liệu** được lấy từ đúng 5 website.
- Queries: **40 câu hỏi**, mỗi câu được chấm với **toàn bộ 50 tài liệu**.
- Judgments: **2000 nhãn 0/1**. Nguồn nhãn: **AI-assisted; NOT independently human-verified**.
- TF-IDF/BM25 dùng cùng tập corpus, queries và nhãn; đánh giá 4 chiến lược preprocessing.
- **Giới hạn:** chỉ phản ánh tập mẫu, không được gọi là Recall@5 trên toàn bộ 887 tài liệu.
- **Lưu ý kiểm định:** câu hỏi được xây dựng từ nội dung chính corpus đánh giá, nên dù diễn đạt lại và có đối chứng dễ nhầm, kết quả có thể lạc quan. Nhãn AI là kết quả sơ bộ, nên rà lại các ca được đánh dấu nếu công bố chính thức.

## Kết quả

| strategy     | method   |   P_at_5 |   Recall_at_5 |   Hit_at_1 |   MRR_at_5 |   nDCG_at_5 |   queries_scored |   queries_total |
|:-------------|:---------|---------:|--------------:|-----------:|-----------:|------------:|-----------------:|----------------:|
| baseline     | bm25     |    0.23  |        0.9917 |      0.975 |     0.9875 |      0.9849 |               40 |              40 |
| baseline     | tfidf    |    0.225 |        0.9833 |      0.925 |     0.95   |      0.9539 |               40 |              40 |
| lemmatized   | bm25     |    0.23  |        0.9917 |      0.95  |     0.975  |      0.9757 |               40 |              40 |
| lemmatized   | tfidf    |    0.23  |        0.9917 |      0.875 |     0.9271 |      0.9389 |               40 |              40 |
| no_stopwords | bm25     |    0.23  |        0.9917 |      0.95  |     0.975  |      0.9757 |               40 |              40 |
| no_stopwords | tfidf    |    0.23  |        0.9917 |      0.875 |     0.9271 |      0.9389 |               40 |              40 |
| stemmed      | bm25     |    0.23  |        0.9917 |      0.95  |     0.975  |      0.9757 |               40 |              40 |
| stemmed      | tfidf    |    0.23  |        0.9917 |      0.875 |     0.9271 |      0.9389 |               40 |              40 |

## Công thức
- Precision@5 = số tài liệu relevant ở Top-5 / 5.
- Recall@5 = số relevant ở Top-5 / tổng relevant đã gán trong corpus đánh giá cố định.
- Hit@1 = 1 nếu top-1 relevant, ngược lại 0; trung bình trên các queries.
- MRR@5 = 1/vị trí relevant đầu tiên trong top-5 (không có thì 0), trung bình queries.
- nDCG@5 = DCG chuẩn hóa theo thứ hạng lý tưởng, dùng relevance nhị phân và log2.

## Bằng chứng
`eval_documents.csv`, `eval_queries.csv`, `qrels_REVIEWED.csv`, `per_query_metrics.csv`, `top5_rankings.csv`, `EVALUATION_PROTOCOL.json`.
