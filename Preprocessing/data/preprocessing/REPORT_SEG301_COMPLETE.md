# SEG301 | Báo cáo thực hành hoàn chỉnh: Vietnamese Preprocessing & Retrieval

> Corpus crawl gốc là **auxiliary, unlabeled**; một **tập mẫu cố định** có thể được gán nhãn riêng để đánh giá retrieval. Không suy rộng metric của tập mẫu ra toàn corpus.

## Bước 1: Load, audit dữ liệu SQLite → CSV
- Corpus đầu ra: **889 đoạn**; nguồn SQLite trích **1,022 passages**.
- Số cột ban đầu: **12**; bản ghi `raw_text` trùng chính xác: **6**.
- Missing values trong CSV gốc theo cột: `{}`.

| source_site | documents |
| --- | --- |
| fpt_telecom_support | 88 |
| hcmus_admissions | 395 |
| mobifone_5g_faq | 12 |
| momo_help | 384 |
| uel_admissions_faq | 10 |

Loại trùng text phù hợp với kho tìm kiếm phụ trợ để giảm bản sao, nhưng **phải giữ metadata nguồn** và không được tự gán nhãn relevance từ domain.

## Bước 2: Cleaning / normalization
Áp dụng chuẩn Unicode NFC, HTML unescape, lọc markup/điều hướng theo website và khoảng trắng thừa (giữ URL có ý nghĩa trong `context`); giữ dấu tiếng Việt, số 5G, gói cước, và phủ định. Bản `context` vẫn giữ dấu câu để dùng dense retrieval.
Kiểm tra ví dụ chi tiết trong `06_raw_clean_examples.csv` và `SEG301_Vietnamese_Preprocessing_Lab.ipynb`.
- Kết quả cleaning: **889** đoạn; 543 passages được cách ly (bao gồm tail phụ).
- Dò mẫu nhiễu đặc trưng: **0**; có **2** đoạn cần kiểm tra thủ công trong `19_review_flags.csv`.
- Không tự suy ra tài liệu đã sạch tuyệt đối từ 0 lỗi kiểm tra bằng regex; xem `QUALITY_GATE.md`.
Việc xóa tất cả chữ số, dấu gạch nối và dấu câu có thể làm mất thông tin như **5G, Wi-Fi, mức phí, điều kiện không được hủy**.

## Bước 3: Tokenization / token count
- Tổng token baseline: **93,829**; tokenizer **thực sự dùng khi sinh dữ liệu**: `underthesea`.
- Tokenizer được yêu cầu khi tạo báo cáo: `underthesea`; nếu khác trên, phải chạy lại `preprocess_merged.py --tokenizer underthesea` rồi sinh lại báo cáo.
- Dữ liệu token của mỗi văn bản nằm trong `03_cleaned_tokenized_lab.csv`, gồm raw, clean, baseline và các biến thể.
- Với tiếng Việt dùng tách từ (underthesea) thay cho word_tokenize tiếng Anh; đọc 3 dòng đầu để xác nhận token ghép có dấu gạch dưới.

## Bước 4: Stop-word experiments A/B
- Sau stop-word removal: **78,046** token, giảm **16.82%** so với baseline.
- Thí nghiệm A giữ dấu câu nhưng bỏ stopword; B vừa bỏ stopword vừa chỉ giữ token chữ/số. So sánh thêm trong notebook.
- Giữ phủ định **không, chưa, chẳng, đừng**: xóa chúng có thể đảo ngược nghĩa câu truy vấn.
- Việc giảm token/vocabulary không đảm bảo P@5/Recall@5 tốt hơn.

## Bước 5: Stemming vs Lemmatization
- **Porter stemmer / WordNet lemmatizer được thiết kế cho tiếng Anh**, không phải hệ thống stemming/lemmatization tiếng Việt.
- File `07_stemming_lemma_english_demo.csv` minh họa sự khác nhau giữa hậu tố heuristic và lemma dạng từ điển; WordNet còn phụ thuộc POS.
- Ở corpus Việt, các cột `stemmed`, `lemmatized` **chỉ thực thi English-only trên từ whitelist demo**, không có nghĩa là toàn bộ văn bản được stem/lemma thực sự.
- Nếu chúng giống stopword variant về top-5, đó là **kết quả hợp lý do hạn chế ngôn ngữ**, không phải lỗi.

## Bước 6: TF-IDF / BM25, cùng query và chính sách tokenization
- TF-IDF xây dựng sparse matrix (`models/tfidf_*.npz`), cosine similarity nhờ l2 normalized vectors.
- BM25 dùng token frequency, inverse document frequency và độ dài văn bản (k1=1.5, b=0.75).
- Cùng 5 câu hỏi, cùng tokenization cho queries và documents; xếp hạng thực tế trong `09_tfidf_bm25_top10.csv`.
- **Không** so điểm BM25 thô với cosine vì khác thang đo.

## Bước 7: 4 strategies x 2 rankers
| strategy | vocab_size | token_reduction_pct | tfidf_rows | tfidf_cols |
| --- | --- | --- | --- | --- |
| baseline | 3193 | 0.0 | 889 | 3193 |
| no_stopwords | 3143 | 16.82 | 889 | 3143 |
| stemmed | 3143 | 16.82 | 889 | 3143 |
| lemmatized | 3143 | 16.82 | 889 | 3143 |

- Vocabulary: baseline **3,193**, bỏ stopwords **3,143**.
- Bảng `11_top5_rankings_all_strategies.csv` gồm tất cả top-5 query/doc ID; bảng `12_top5_comparison.csv` cho biết phần giao nhau với baseline và khác thứ tự.
- Để đo ảnh hưởng preprocessing, **giữ ranker cố định**. Để so TF-IDF/BM25, **giữ chiến lược preprocessing cố định**.

## Bước 8: Precision@5, Recall@5 và deliverables
- Corpus crawl gốc **không có nhãn relevance đầy đủ**. Nhóm đã xây dựng **một tập đánh giá cố định**, tách biệt khỏi corpus không nhãn dùng cho dự án.
- Tập đánh giá gồm **50 documents**, **40 queries** và **2000 cặp query–document** gán nhãn 0/1 trong chính tập này.
- **Nguồn nhãn:** AI-assisted; NOT independently human-verified. Nhãn AI hỗ trợ là kết quả sơ bộ, **chưa phải ground truth do con người kiểm chứng độc lập**.
- Precision@5, Recall@5 và nDCG@5 chỉ phản ánh tập tài liệu đã gán nhãn đầy đủ này, **không phải 887/889 documents**.
- Các query có nguồn gốc/paraphrase từ văn bản trong tập đánh giá nên phép đo có nguy cơ dễ hơn truy vấn mới độc lập.

## Bước 9: Discussion questions
1. **Stopword có luôn tốt?** Không. Có thể bỏ mất từ mang ý định, ngữ nghĩa và phủ định; chỉ đánh giá mới biết.
2. **Stemming / lemmatization?** Stemming cắt từ theo luật, có thể sinh gốc không có trong từ điển; lemma dựa từ điển và POS, có dạng chuẩn.
3. **Vì sao Top-5 khác nhau?** Vocabulary, tần suất, token hóa và weighting/normalization thay đổi score từng document.
4. **BM25 điều chỉnh độ dài thế nào?** Chuẩn hóa term-frequency theo độ dài document so với độ dài trung bình, điều khiển bằng `b`.
5. **Vì sao không so trực tiếp BM25 và cosine?** BM25 là điểm xếp hạng với IDF/TF riêng; cosine trên TF-IDF l2 normalized, thang khác nhau.
6. **Query OOV sẽ sao?** Không có token query trong vocabulary thì vector TF-IDF bằng 0, thứ hạng sẽ hòa điểm/không mang ý nghĩa; cần kiểm OOV.
7. **Bigrams hữu ích?** Giữ thứ tự cặp token như `chăm_sóc khách_hàng`, `phí đăng_ký`, giảm nhầm nghĩa so với unigram; tăng sparse vocabulary.
8. **Cần nhãn gì?** Relevance query–document được con người/qrels đánh giá; P@5 yêu cầu Top-5 đủ nhãn; Recall@5 yêu cầu danh sách relevant đầy đủ trong corpus đánh giá đã định.

## Kết luận và hạn chế
- Dùng `24_auxiliary_corpus_RESEARCH_READY.csv` làm **unlabeled auxiliary mining pool** của OURS-WEB; file `04_...csv` là toàn bộ dữ liệu cho lab và audit.
- Tập crawler **chưa có relevance labels** và không được biến thành labeled training triplets trong bài SEG301.
- `models/` hiện chỉ là các chỉ mục TF-IDF/BM25 của bài lab, không phải dense model đã fine-tune.
- Hai mã chương trình HCMUS 7440107 và 7440107_DKD được cách ly, không tự sửa vì có thể là hai hệ chương trình khác nhau.
- Xem kiểm định `QUALITY_GATE.md`, `15_cleaning_page_audit.csv`, `27_source_cleaning_decisions.csv`, `18_manual_review_sample.csv`; khi báo cáo ghi rõ kiểm định tự động không thay cho thẩm định toàn corpus.
- Stem/lemma tiếng Anh chỉ là demo; muốn triển khai morphology tiếng Việt thực sự cần tool/định nghĩa riêng, không tráo nhãn xử lý.

## Phụ lục A: P@5/Recall@5 trên tập crawl có nhãn
Đánh giá trên tập mẫu từ năm website crawl với nhãn AI hỗ trợ, chưa có kiểm chứng độc lập; KHÔNG phải toàn bộ corpus.
| strategy | method | P_at_5 | Recall_at_5 | Hit_at_1 | MRR_at_5 | nDCG_at_5 | queries_scored | queries_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | bm25 | 0.23 | 0.9917 | 0.975 | 0.9875 | 0.9849 | 40 | 40 |
| baseline | tfidf | 0.225 | 0.9833 | 0.925 | 0.95 | 0.9539 | 40 | 40 |
| lemmatized | bm25 | 0.23 | 0.9917 | 0.95 | 0.975 | 0.9757 | 40 | 40 |
| lemmatized | tfidf | 0.23 | 0.9917 | 0.875 | 0.9271 | 0.9389 | 40 | 40 |
| no_stopwords | bm25 | 0.23 | 0.9917 | 0.95 | 0.975 | 0.9757 | 40 | 40 |
| no_stopwords | tfidf | 0.23 | 0.9917 | 0.875 | 0.9271 | 0.9389 | 40 | 40 |
| stemmed | bm25 | 0.23 | 0.9917 | 0.95 | 0.975 | 0.9757 | 40 | 40 |
| stemmed | tfidf | 0.23 | 0.9917 | 0.875 | 0.9271 | 0.9389 | 40 | 40 |
Chi tiết quy mô và xác nhận nhãn trong `crawl_eval/EVALUATION_PROTOCOL.json`.


## Phụ lục B: Known-positive retrieval trên toàn corpus crawl
Mở rộng vùng **xếp hạng** sang toàn bộ 887 documents nhưng nhãn vẫn chỉ biết một phần.
**Không gọi đây là Recall@5 trên 887 documents**: tất cả kết quả chưa gán nhãn bên ngoài tập 50 có trạng thái UNKNOWN, không phải negative.
| strategy | method | AnyKnownPositive_at_1 | AnyKnownPositive_at_5 | AnyKnownPositive_at_10 | KnownPositiveCoverage_at_5 | mean_first_known_positive_rank | queries_scored | oov_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | bm25 | 0.65 | 0.95 | 0.975 | 0.9292 | 1.9 | 40 | 0 |
| baseline | tfidf | 0.65 | 0.925 | 0.95 | 0.8792 | 2.375 | 40 | 0 |
| lemmatized | bm25 | 0.6 | 0.975 | 0.975 | 0.9542 | 1.9 | 40 | 0 |
| lemmatized | tfidf | 0.625 | 0.9 | 0.95 | 0.8542 | 2.875 | 40 | 0 |
| no_stopwords | bm25 | 0.6 | 0.975 | 0.975 | 0.9542 | 1.9 | 40 | 0 |
| no_stopwords | tfidf | 0.625 | 0.9 | 0.95 | 0.8542 | 2.875 | 40 | 0 |
| stemmed | bm25 | 0.6 | 0.975 | 0.975 | 0.9542 | 1.9 | 40 | 0 |
| stemmed | tfidf | 0.625 | 0.9 | 0.95 | 0.8542 | 2.875 | 40 | 0 |
Xem `crawl_eval/REPORT_FULL_887_DIAGNOSTIC.md` và `full_887_diagnostic_per_query.csv` để kiểm tra các trường hợp rơi khỏi Top-5.

