# Checklist before uploading to GitHub

- [x] Source and files imported from the user's current `Preprocessing.zip`.
- [x] `python -m unittest discover -s tests -v`: 41 passed, 1 skipped in independent environment lacking `underthesea`.
- [x] `python verify_research_release.py --out data/preprocessing`: PASS, 889/887/2 and no persisted research-only index.
- [x] SHA-256 for the full (889) and research-ready (887) CSVs matches manifest.
- [x] 40 queries / 50 documents / 2,000 labeled pairs; P@5/R@5 calculated only within fixed 50-document set.
- [x] `tabulate` dependency, README and notebook stale paths/commands corrected.
- [ ] On the user's Windows computer: rerun `RUN_EVAL_V2_WINDOWS.bat` and verify metrics exactly match saved CSV (the audit environment lacks `underthesea`).
- [ ] If requested by instructor, download NLTK English resources and run `python english_morphology_experiment.py` to produce POS-aware morphology outputs.
- [ ] For gold standard claims, have independent reviewers adjudicate/verify the AI-assisted relevance labels; current labels are preliminary.
- [ ] Read manual-review samples; automatic quality gates are not proof of absolute semantic accuracy.

No original merged SQLite DB is bundled in this `Preprocessing` repo; it belongs to sibling `Crawler/data/`.
