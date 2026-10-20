# Kiểm định chất lượng Text Cleaning cho FNAC-HNM

## Đầu vào và quyết định lọc
- Nguồn: `..\VIRE_Crawler\data\vire_web_corpus_merged.db`
- Số passages nguồn: **1022**
- Corpus được chấp nhận: **889** retrieval documents
- Trang/trích đoạn được trích xuất: 479
- Trang/trích đoạn cách ly: 543
- Tỷ lệ khớp nhiễu đặc trưng còn lại: **0**
- Đoạn có dấu hiệu cần rà soát riêng (không tự xóa): **2**

## Số lượng theo website
| Site | Documents |
|---|---:|
| fpt_telecom_support | 88 |
| hcmus_admissions | 395 |
| mobifone_5g_faq | 12 |
| momo_help | 384 |
| uel_admissions_faq | 10 |

## HCMUS 2026: tách hàng và kiểm định nguồn
- Chunk kết thúc bằng số thứ tự đứng một mình: **0**.
- Hàng tuyển sinh dài hơn max_words (giữ nguyên một hàng; cần kiểm tra token model): **0**.
- Trang bảng thiếu STT trong bản crawl: **0**.
- Tên ngành/chỉ tiêu xuất hiện với nhiều mã ngành giữa hai URL: **1**.
- Xem `21_hcmus_table_integrity.csv`, `22_hcmus_chunk_integrity.csv` và `23_hcmus_code_discrepancies.csv`.
- Bảng raw SQLite đã được kiểm tra đủ 39 STT ở mỗi URL. Nếu bảng khác schema, parser sẽ quarantine thay vì suy đoán.

## Lý do loại/cách ly
| Lý do | Số bản ghi |
|---|---:|
| exact_duplicate | 62 |
| hcmus_admission_method_directory_navigation | 1 |
| hcmus_landing_navigation | 1 |
| mobi_non_faq_marketing_or_category_page | 13 |
| momo_answer_too_short_or_empty | 2 |
| momo_category_listing_without_answer | 61 |
| momo_listing_home | 1 |
| near_duplicate | 3 |
| secondary_duplicate_tail_passage | 463 |
| short_segment | 1 |
| too_short | 1 |
| uel_no_safe_qa_cards | 1 |

## Quy tắc có kiểm soát
- MoMo: chỉ giữ câu trả lời có tiêu đề khớp, điểm bắt đầu và điểm kết thúc (rating); bỏ trang chuyên mục và related articles.
- UEL: phân tách từng FAQ theo dấu hiệu câu hỏi và ranh giới Tags: Sao chép; loại breadcrumb, yêu cầu đáp án đủ dài.
- MobiFone: chỉ giữ trang Câu hỏi thường gặp và tách từng cặp hỏi–đáp đánh số. Các landing/catalog khác được cách ly.
- HCMUS: tách riêng từng hàng mã ngành/chỉ tiêu trong hai bảng 2026; giữ đủ phương thức/tổ hợp, dedup bản trùng tuyệt đối giữa URL, xử lý số mục bị treo; phần nội dung khác chia theo câu.
- FPT: bỏ footer đúng mẫu, không sửa thông tin sản phẩm, mức cước, quy định, URL hoặc phủ định.
- Các đoạn phụ cuối bài của crawler (passage_index > 0 khi có bản chính) được cách ly, không nhân đôi thành tài liệu.

## Kiểm định và các giới hạn
- Kết quả quét nhiễu đặc trưng bằng 5 nhóm mẫu regex không thể chứng minh toàn bộ 100% tài liệu sạch.
- Kiểm tra thủ công `18_manual_review_sample.csv`, `19_review_flags.csv`, và khi cần cả `16_quarantine_pages.csv` trước khi chốt cho training.
- Điền `20_review_form_TEMPLATE.csv` thành bản `review_decisions.csv`, cột decision=keep/drop, rồi chạy `--reviews review_decisions.csv` để loại những bản ghi được xác nhận không phù hợp và rebuild index.
- Một số bài tuyển sinh có bảng được website flatten thành văn bản: phải xem thủ công những bảng/điểm/mức phí quan trọng.
- Có thể còn lỗi chính tả từ website nguồn. Không sửa từ tùy ý vì có thể làm lệch ý nghĩa.
- Khi nạp vào dense retriever, kiểm tra tỷ lệ bị cắt tại MAX_SEQ_LENGTH bởi tokenizer của model thật.
- Các trang bị cách ly vẫn lưu nguồn trong file audit; không giả định chúng là câu trả lời rác tuyệt đối.

## Tình trạng
**PASS kiểm tra kỹ thuật tự động**, **PENDING kiểm tra mẫu thủ công** trước khi dùng nghiên cứu chính thức.
