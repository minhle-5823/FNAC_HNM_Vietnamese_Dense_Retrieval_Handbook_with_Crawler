# FPT Telecom: Hi FPT Help Center

- Seed URL: https://fpt.vn/hi-fpt/huong-dan-su-dung
- Why changed: `/ho-tro/cau-hoi-thuong-gap` returned HTTP 500 (verified by user).
- Scope: `/hi-fpt/huong-dan-su-dung` + one category segment + category/article slug.
- Page types: `help_listing`, `help_category`, `help_detail`.
- Listing/category pages are link-only, not training documents.
- `fpt_hifpt` extractor retains only the article H1 and answer (before page feedback, related items and footer).
- FPT uses some href paths with `/hi-fpt//huong-dan-su-dung/`; the FPT-only link cleaner collapses repeated slashes.
- No change to the other four site crawlers, database schema, or group merging code.
- Run `python main.py crawl --site fpt_telecom_support --reset` after this seed change. The reset affects only the FPT DB.
- Respect robots.txt; anti-bot blocks are not bypassed.
- Before merging, check `python main.py validate --site fpt_telecom_support`; valid documents/passages must be >0.
- Preprocessing must also be updated to recognize Hi FPT instead of the old FAQ URL.
