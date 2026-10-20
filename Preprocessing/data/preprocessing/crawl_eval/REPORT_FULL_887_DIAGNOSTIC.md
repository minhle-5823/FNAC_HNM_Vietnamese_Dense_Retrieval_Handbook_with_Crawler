# Kiểm tra độ khó truy xuất trên 887 tài liệu crawl

**KHÔNG PHẢI Precision@K/Recall@K**: chỉ biết một phần tài liệu positive từ tập 50 documents. Những tài liệu còn lại **chưa đánh giá**, không được coi là negative.

Các nhãn known-positive ban đầu do AI hỗ trợ, chưa được con người kiểm chứng độc lập. Kết quả phụ thuộc tokenizer.

| strategy     | method   |   AnyKnownPositive_at_1 |   AnyKnownPositive_at_5 |   AnyKnownPositive_at_10 |   KnownPositiveCoverage_at_5 |   mean_first_known_positive_rank |   queries_scored |   oov_queries |
|:-------------|:---------|------------------------:|------------------------:|-------------------------:|-----------------------------:|---------------------------------:|-----------------:|--------------:|
| baseline     | bm25     |                   0.65  |                   0.95  |                    0.975 |                       0.9292 |                            1.9   |               40 |             0 |
| baseline     | tfidf    |                   0.65  |                   0.925 |                    0.95  |                       0.8792 |                            2.375 |               40 |             0 |
| lemmatized   | bm25     |                   0.6   |                   0.975 |                    0.975 |                       0.9542 |                            1.9   |               40 |             0 |
| lemmatized   | tfidf    |                   0.625 |                   0.9   |                    0.95  |                       0.8542 |                            2.875 |               40 |             0 |
| no_stopwords | bm25     |                   0.6   |                   0.975 |                    0.975 |                       0.9542 |                            1.9   |               40 |             0 |
| no_stopwords | tfidf    |                   0.625 |                   0.9   |                    0.95  |                       0.8542 |                            2.875 |               40 |             0 |
| stemmed      | bm25     |                   0.6   |                   0.975 |                    0.975 |                       0.9542 |                            1.9   |               40 |             0 |
| stemmed      | tfidf    |                   0.625 |                   0.9   |                    0.95  |                       0.8542 |                            2.875 |               40 |             0 |

**AnyKnownPositive@5** = tỷ lệ câu hỏi có ít nhất một document được gán positive trong Top-5 trên 887 documents.

**KnownPositiveCoverage@5** = phần các document known-positive đã tìm thấy ở Top-5; **không phải Recall thật** vì các positive khác trong 887 chưa được gán nhãn.

