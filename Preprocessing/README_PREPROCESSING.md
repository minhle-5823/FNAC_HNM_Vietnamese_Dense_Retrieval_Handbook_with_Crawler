# SEG301 | Tiền xử lý và truy hồi văn bản tiếng Việt từ 5 website crawl

**Repo sử dụng dữ liệu crawl của nhóm, KHÔNG sử dụng CSConDa/EduCoQA để tiền xử lý hoặc đánh giá môn SEG301.**

## Tình trạng kiểm định và tài sản dữ liệu

- `data/preprocessing/01_export_merged_raw.csv`: **1.022 passages** từ SQLite hợp nhất (nguồn thực địa do `Crawler` quản lý).
- `data/preprocessing/04_crawled_auxiliary_corpus.csv`: **889 documents** sau cleaning cho bài thực hành.
- `data/preprocessing/24_auxiliary_corpus_RESEARCH_READY.csv`: **887 documents** dự kiến dùng làm *unlabeled auxiliary corpus* sau này, **2 HCMUS documents** còn nghi vấn để riêng trong `25_pending_manual_review.csv`.
- `data/preprocessing/models/`: chỉ mục TF-IDF/BM25 trên **889 documents**, KHÔNG phải mô hình dense đã fine-tune.
- `data/preprocessing/crawl_eval/`: **50 documents x 40 queries = 2.000 cặp nhãn 0/1**, trong đó 47 cặp relevant. Nhãn **AI-assisted preliminary**, chưa được thẩm định độc lập bởi con người. P@5/Recall@5 **chỉ đo trên tập 50 documents đã gán nhãn đầy đủ**.
- `data/preprocessing/crawl_eval/full_887_diagnostic_summary.csv`: tìm lại các known positives giữa **887 documents**; đây là *diagnostic*, KHÔNG PHẢI Precision/Recall toàn bộ 887 documents.

## Cấu trúc thư mục dự kiến

```text
SEG+TMG_Final_Project/
├─ Crawler/
│  └─ data/vire_web_corpus_merged.db
└─ Preprocessing/
   ├─ preprocess_merged.py
   ├─ complete_lab.py
   ├─ evaluate_crawled_corpus.py
   ├─ evaluate_887_diagnostic.py
   ├─ english_morphology_experiment.py
   ├─ core/
   ├─ tests/
   ├─ SEG301_Vietnamese_Preprocessing_Lab.ipynb
   └─ data/preprocessing/
```

## Tái lập thí nghiệm (Windows / PowerShell trong thư mục Preprocessing)

```powershell
python -m pip install -r requirements-preprocessing.txt
python -m unittest discover -s tests -v
python verify_research_release.py --out data/preprocessing
```

**Không cần tái crawl hoặc re-run cleaning nếu các CSV đã tồn tại và checksum khớp.** Nếu có DB ở thư mục ngang hàng, có thể tái tạo kết quả bằng:

```powershell
python preprocess_merged.py --db "..\Crawler\data\vire_web_corpus_merged.db" --tokenizer underthesea
```

Sau đó, để chạy lại phép đánh giá (sẽ tính **từ nhãn 2.000 cặp có sẵn**, không chạy lệnh `prepare` vì sẽ làm thay đổi tập câu hỏi và yêu cầu gán nhãn mới):

```powershell
python evaluate_crawled_corpus.py evaluate --tokenizer underthesea
python evaluate_887_diagnostic.py --tokenizer underthesea
python complete_lab.py --out data/preprocessing --tokenizer underthesea
```

Có thể chạy tắt bằng `RUN_EVAL_V2_WINDOWS.bat`. Nếu muốn chạy lại thí nghiệm morphology tiếng Anh **đầy đủ POS-aware**, dùng:

```powershell
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.4
python english_morphology_experiment.py
python complete_lab.py --out data/preprocessing --tokenizer underthesea
```

Khi chạy thành công, xuất `29_english_morphology_POS_examples.csv`, `29_english_morphology_stats.json` và `29_REPORT_ENGLISH_MORPHOLOGY.md`. Những file này chưa có trong bản snapshot nếu bộ dữ liệu NLTK chưa được tải tại máy chạy. **Không coi kết quả POS-aware đã hoàn thành nếu chưa có output thật.**

## Thí nghiệm SEG301 theo đúng phạm vi và giới hạn

1. **Load/explore**: SQLite -> CSV + thống kê thiếu/trùng/theo nguồn.
2. **Cleaning/normalization**: Unicode NFC, HTML entity/markup, rule theo 5 nguồn, chunk theo câu, dedup SHA-256 và MinHash/Jaccard; kiểm tra residual noise.
3. **Tokenization**: `underthesea`, giữ token ghép tiếng Việt có `_`.
4. **Stopwords**: danh sách tiếng Việt tự khai báo, giữ từ phủ định quan trọng; đo % giảm token.
5. **Stemming/Lemmatization**: NLTK Porter/WordNet là thực nghiệm riêng cho **tiếng Anh**, không được nói đã stem/lemma toàn bộ corpus Việt. Phần corpus Việt sau stopwords gần như không đổi ở hai variants này.
6. **Retrieval**: TF-IDF + cosine và BM25 (Okapi, k1=1.5, b=0.75).
7. **Compare**: bốn preprocessing strategies × hai rankers; giữ điều kiện so sánh nhất quán.
8. **Report/evaluation**: subset 50 có nhãn AI sơ bộ đo P@5/R@5/Hit@1/MRR@5/nDCG@5; diagnostic 887 có nhãn một phần, chỉ đánh giá known-positive.
9. **Discussion**: notebook và `REPORT_SEG301_COMPLETE.md` giải thích cơ chế, giới hạn và kết quả.

Lưu ý: **vocabulary của ba variants `no_stopwords`, `stemmed`, `lemmatized` bằng nhau** trong bản chạy này. Đây là giới hạn phương pháp với ngôn ngữ Việt, không chứng minh stemming/lemmatization tiếng Việt thành công.

## Xem kết quả

- Notebook: `SEG301_Vietnamese_Preprocessing_Lab.ipynb` (Run All từ đúng cwd `Preprocessing`).
- Bảng so sánh strategies: `08_strategy_comparison.csv`.
- TF-IDF/BM25 sample Top-10: `09_tfidf_bm25_top10.csv`; Top-5: `11_top5_rankings_all_strategies.csv` và `12_top5_comparison.csv`.
- Metric 50 documents: `crawl_eval/metric_summary.csv` / `per_query_metrics.csv` / `top5_rankings.csv`.
- Diagnostic 887 documents: `crawl_eval/full_887_diagnostic_summary.csv` / `full_887_diagnostic_per_query.csv`.
- Báo cáo giải thích: `REPORT_SEG301_COMPLETE.md` và các report trong `crawl_eval/`.
- Truy vết filtering: `06_raw_clean_examples.csv`, `15_cleaning_page_audit.csv`, `16_quarantine_pages.csv`, `27_source_cleaning_decisions.csv`.

## Khoa học và provenance

- 50 documents / 40 queries được xây dựa trên chính nội dung corpus; queries không phải bộ độc lập. 2.000 nhãn là AI-assisted; **không công bố chúng là gold labels do con người chấm**.
- P@5/R@5 của fixed subset không suy rộng cho tập 887/889. 887 diagnostic cũng không thay thế Recall@5 thật vì phần lớn corpus không nhãn.
- `source_site` hoặc score của ranker không được dùng làm relevance ground truth.
- `24_auxiliary_corpus_RESEARCH_READY.csv` tương thích với trường `context` của pipeline mining sau này nhưng không có query/positive/negative. Không gắn nhãn training giả cho nó.
- Không commit kho SQLite crawl gốc cùng repo này nếu dữ liệu đang được quản lý trong `Crawler/`; giữ metadata URL và tôn trọng giấy phép/điều khoản website.
- Bộ kiểm định tự động không tương đương kiểm chứng nội dung dữ kiện 100% bằng con người.

Xem thêm `GITHUB_RELEASE_CHECKLIST.md` và tài liệu học liệu DOCX đi kèm bên ngoài repo.
