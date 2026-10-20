# FNAC-HNM | SEG301 Preprocessing

**Hai mục đích tách biệt:**

1. **Bài SEG301:** `data/preprocessing/04_crawled_auxiliary_corpus.csv` (889 documents) và `data/preprocessing/models/` (các chỉ mục TF-IDF/BM25 cho 4 preprocessing strategies). Notebook: `SEG301_Vietnamese_Preprocessing_Lab.ipynb`.
2. **Dữ liệu FNAC-HNM:** `data/preprocessing/24_auxiliary_corpus_RESEARCH_READY.csv` (**887 documents**). Dùng các cột **`doc_id`, `context`** và metadata `source_site`, `domain_label`, `url`, `title`, `crawled_at`. Đây là *unlabeled auxiliary documents*, **không** có `question`, `positive`, `negative`, `qid`, `qrels`.

## Tại sao chỉ có `models/`?

`models/` là chỉ mục cho **889 documents của bài lab**, không phải model Dense Retrieval. Đã bỏ `models_research/` vì index chỉ trên 887 documents **không phải** mining index của OURS-WEB. Trong training, xây BM25 và dense candidate mining trên **VIRE TRAIN documents + 887 auxiliary documents**, giữ VIRE TEST corpus bất biến. Lệnh `search --corpus research` có thể dựng chỉ mục tạm trong RAM để xem thử, nhưng không lưu thêm thư mục.

## Cách xem kết quả

- **Dataset nghiên cứu:** mở `data/preprocessing/24_auxiliary_corpus_RESEARCH_READY.csv` bằng Excel/VS Code.
- **Hai document cách ly:** `25_pending_manual_review.csv`.
- **Báo cáo bài thầy:** `REPORT_SEG301_COMPLETE.md`, notebook có output.
- **Báo cáo chất lượng:** `QUALITY_GATE.md`, `RESEARCH_RELEASE_STATUS.md`.
- **Index bài thực hành:** `models/` chỉ dành cho corpus 889 dòng.

## Cách chạy lại từ nguồn

Đặt `VIRE_Crawler/` và repo này ngang hàng rồi chạy:

```powershell
python -m pip install -r requirements-preprocessing.txt
python preprocess_merged.py --db "..\VIRE_Crawler\data\vire_web_corpus_merged.db" --tokenizer underthesea
python complete_lab.py --out ".\data\preprocessing" --tokenizer underthesea
python verify_research_release.py --out ".\data\preprocessing"
```

Để kiểm tra chức năng search trong tập 887 không cần index riêng:

```powershell
python preprocess_merged.py search --corpus research --method bm25 --query "đăng ký internet"
```

## Tương thích với CSConDa / EduCoQA

- VIRE dùng `question` + `context` (+ qrels/known positives) làm dữ liệu có nhãn.
- Corpus crawl chỉ có **documents** nên cột `context` tương thích làm *candidate documents*, không được nối các hàng này trực tiếp thành các training pairs có nhãn.
- Khi mining, chuẩn hóa VIRE context thành một corpus có `doc_id`; thêm auxiliary bằng `pd.concat([vire_docs, aux[['doc_id','context']]])`, đảm bảo unique IDs và loại trùng văn bản; positive IDs/qrels VIRE vẫn phải giữ ánh xạ nhất quán.
- Chỉ mine negatives cho TRAIN queries, lọc known positives và near duplicates; không đánh giá trên corpus đã mở rộng nếu đang so sánh benchmark VIRE.

## Dữ liệu và giới hạn

Dataset chính có **887** tài liệu đã qua kiểm tra tự động, 2 tài liệu HCMUS không chắc mã ngành được tách riêng. Không nên tuyên bố kiểm định dữ kiện thủ công 100%. Stemming/lemmatization trong lab là minh họa tiếng Anh; không áp dụng Porter/WordNet lên tiếng Việt. `underthesea` giữ từ ghép có `_`.

Tests: `python -m unittest discover -s tests -v`.

## Tích hợp với notebook FNAC-HNM (đúng schema VIRE)

Trong handbook cũ, Cell 4A có `MIN_TEXT_CHARS = 300` và cấp lại `web_{i}`. **Không nên dùng nguyên hai bước đó với corpus này:** 127/887 đoạn sạch ngắn hơn 300 ký tự (chủ yếu MoMo) và file đã có `doc_id` ổn định.

Thay phần mở rộng mining corpus sau khi Cell 4 tạo VIRE `corpus` bằng:

```python
from integration_fnac_hnm import extend_mining_corpus
# Nếu notebook ở Kaggle: copy integration_fnac_hnm.py và upload CSV rồi trỏ đường dẫn đúng.
if USE_CRAWL_AUX:
    mining_corpus = extend_mining_corpus(corpus, '/kaggle/input/.../24_auxiliary_corpus_RESEARCH_READY.csv')
else:
    mining_corpus = corpus[['doc_id','context']].copy()
```

`corpus` là tập VIRE gốc đã mang `doc_id` trỏ đúng đến positives/qrels. **Không thay thế** `queries`, `pairs`, `qrels`, `split` hay evaluation corpus.
