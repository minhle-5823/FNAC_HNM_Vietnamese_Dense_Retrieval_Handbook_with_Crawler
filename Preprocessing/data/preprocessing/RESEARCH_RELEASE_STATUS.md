# FNAC-HNM auxiliary corpus: audited automated release

- Full preprocessing corpus: **889** records (`04_...csv`).
- High-confidence subset: **887** records (`24_auxiliary_corpus_RESEARCH_READY.csv`).
- Flagged, unreviewed records withheld: **2** (`25_pending_manual_review.csv`).
- HCMUS 2026 tables from SQLite: **39/39 rows for each URL**, including `74401a1`, `75202a1`.
- Known residual boilerplate patterns: **0**.
- No standalone research index is stored: OURS-WEB must mine over VIRE TRAIN documents plus this auxiliary corpus.
- These are reproducible structural checks, not proof of complete factual validation.
- **Use `24_...RESEARCH_READY.csv` in mining experiments**, never treat its unlabeled documents as ground-truth negatives.
