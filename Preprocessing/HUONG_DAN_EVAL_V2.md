# SEG301 Evaluation v2 (40 queries / 50 documents / 2.000 nhãn)

Run in `Preprocessing/`, all source texts from 5 crawled websites:

```powershell
python evaluate_crawled_corpus.py evaluate --tokenizer underthesea
python evaluate_887_diagnostic.py --tokenizer underthesea
python complete_lab.py --out data/preprocessing --tokenizer underthesea
```

`prepare` is only for intentionally creating a NEW unlabeled benchmark; **do not run prepare for the existing fixed 40-query benchmark**.

- P@5/Recall@5 are calculated on the fully judged 50-document subset only.
- Results on all 887 are known-positive partial-label diagnostics, NOT Recall@5.
- AI-assisted labels are preliminary; the README / reports must say so.
- If morphology outputs are required, separately download NLTK English resources and run `english_morphology_experiment.py`.
