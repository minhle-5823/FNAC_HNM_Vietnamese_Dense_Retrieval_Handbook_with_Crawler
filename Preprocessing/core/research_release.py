"""Conservative release gate for an *unlabeled* FNAC-HNM auxiliary corpus.

This is an audit filter, NOT relevance labeling. It preserves the full lab
corpus and separates records flagged by deterministic checks for human review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from core.source_cleaning import RESIDUAL_PATTERNS

EXPECTED_SITES = {'fpt_telecom_support', 'momo_help', 'mobifone_5g_faq',
                  'uel_admissions_faq', 'hcmus_admissions'}


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            dig.update(block)
    return dig.hexdigest()


def build_research_release(out_dir: Path, decisions: Path | None = None) -> dict:
    """Make stable high-confidence training input without pretending flagged rows are reviewed.

    Keeps full corpus untouched. Separate all unreviewed flagged documents into
    25_pending_manual_review.csv. Nonflagged rows form research-ready subset.
    """
    out_dir = Path(out_dir)
    corpus_path = out_dir / '04_crawled_auxiliary_corpus.csv'
    raw_path = out_dir / '01_export_merged_raw.csv'
    corpus = pd.read_csv(corpus_path, keep_default_na=False, encoding='utf-8-sig')
    flags_path = out_dir / '19_review_flags.csv'
    flags = pd.read_csv(flags_path, keep_default_na=False, encoding='utf-8-sig')
    tables = pd.read_csv(out_dir / '21_hcmus_table_integrity.csv', keep_default_na=False, encoding='utf-8-sig')
    violations = pd.read_csv(out_dir / '17_residual_noise_audit.csv', keep_default_na=False, encoding='utf-8-sig')
    if set(corpus.source_site) != EXPECTED_SITES:
        raise ValueError('Expected all five websites; refusing incomplete research release')
    if corpus.doc_id.duplicated().any() or corpus.context.eq('').any() or corpus.context.duplicated().any():
        raise ValueError('Document IDs, content completeness and uniqueness must be verified')
    if violations.violations.astype(int).sum():
        raise ValueError('Known source boilerplate remains: do not release')
    if len(tables) != 2 or not tables.parsed_rows.astype(int).eq(39).all() or not tables.missing_stt.eq('').all():
        raise ValueError('Both original HCMUS tables must parse as 39 complete rows')
    # Guard against 8/32 being accidentally appended to preceding program rows.
    table_docs = corpus[corpus.extraction_rule == 'hcmus_structured_admission_row']
    for _, row in table_docs.iterrows():
        # Field-safe: `Mã ngành:` may occur once in its own row only.
        if str(row.context).count('Mã ngành:') != 1 or str(row.context).count('Chỉ tiêu:') != 1:
            raise ValueError(f'Admission row conflation suspected in {row.doc_id}')
    codes_text = '\n'.join(table_docs.context.astype(str))
    for code in ('74401a1', '75202a1'):
        if f'Mã ngành: {code} |' not in codes_text:
            raise ValueError(f'Expected alphanumeric code is missing: {code}')
    raw = pd.read_csv(raw_path, keep_default_na=False, encoding='utf-8-sig')
    if int(raw.source_site.nunique()) != 5:
        raise ValueError('Raw snapshot does not have five sites')

    reviewed_keep = set()
    if decisions is not None:
        dec = pd.read_csv(decisions, keep_default_na=False, encoding='utf-8-sig')
        if not {'doc_id', 'decision'}.issubset(dec.columns):
            raise ValueError('Review decisions require doc_id and decision')
        if dec.doc_id.duplicated().any():
            raise ValueError('Duplicate review decisions detected')
        valid = dec.decision.astype(str).str.strip().str.casefold()
        if not valid.isin(['keep', 'drop', '']).all():
            raise ValueError('Only keep/drop/blank decisions allowed')
        reviewed_keep = set(dec.loc[valid.eq('keep'), 'doc_id'])
        if reviewed_keep - set(corpus.doc_id):
            raise ValueError('Review keep doc_id not in current corpus; rerun review template')

    flagged_ids = set(flags.doc_id.astype(str)) & set(corpus.doc_id)
    unresolved = flagged_ids - reviewed_keep
    research = corpus[~corpus.doc_id.isin(unresolved)].copy()
    pending = corpus[corpus.doc_id.isin(unresolved)].copy()
    explanation = (flags.groupby('doc_id')['flag']
                        .apply(lambda ss: ' | '.join(sorted(set(ss.astype(str)))))
                        .to_dict()) if len(flags) else {}
    pending['review_reason'] = pending.doc_id.map(explanation)
    ready_csv = out_dir / '24_auxiliary_corpus_RESEARCH_READY.csv'
    pending_csv = out_dir / '25_pending_manual_review.csv'
    research.to_csv(ready_csv, index=False, encoding='utf-8-sig')
    pending.to_csv(pending_csv, index=False, encoding='utf-8-sig')
    manifest = {
        'release_label': 'automated_high_confidence_subset_not_manual_approval',
        'raw_passages': len(raw),
        'full_lab_corpus_documents': len(corpus),
        'research_ready_documents': len(research),
        'pending_review_documents': len(pending),
        'all_sources_included': sorted(EXPECTED_SITES),
        'research_ready_by_site': research.source_site.value_counts().sort_index().to_dict(),
        'hcmus_source_table_rows': dict(zip(tables.url.astype(str), tables.parsed_rows.astype(int))),
        'known_boilerplate_violations': int(violations.violations.astype(int).sum()),
        'source_code_discrepancies': len(pd.read_csv(out_dir / '23_hcmus_code_discrepancies.csv')),
        'full_corpus_sha256': sha256(corpus_path),
        'research_ready_sha256': sha256(ready_csv),
        'research_index_policy': 'Build BM25/dense over VIRE TRAIN corpus + auxiliary in the training pipeline',
        'limits': [
            'No human review implied for accepted records.',
            'Crawl corpus is UNLABELED; not VIRE training positives/qrels or evaluation TEST.',
            'External factual disagreements between source URLs remain unresolved.',
            'Dense-model tokenizer MAX_SEQ_LENGTH truncation must still be checked downstream.',
        ],
    }
    (out_dir / '26_research_release_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'RESEARCH_RELEASE_STATUS.md').write_text(
        '# FNAC-HNM auxiliary corpus: audited automated release\n\n'
        f'- Full preprocessing corpus: **{len(corpus)}** records (`04_...csv`).\n'
        f'- High-confidence subset: **{len(research)}** records (`24_auxiliary_corpus_RESEARCH_READY.csv`).\n'
        f'- Flagged, unreviewed records withheld: **{len(pending)}** (`25_pending_manual_review.csv`).\n'
        '- HCMUS 2026 tables from SQLite: **39/39 rows for each URL**, including `74401a1`, `75202a1`.\n'
        f'- Known residual boilerplate patterns: **{int(violations.violations.astype(int).sum())}**.\n'
        '- No standalone research index is stored: OURS-WEB must mine over VIRE TRAIN documents plus this auxiliary corpus.\n'
        '- These are reproducible structural checks, not proof of complete factual validation.\n'
        '- **Use `24_...RESEARCH_READY.csv` in mining experiments**, never treat its unlabeled documents as ground-truth negatives.\n',
        encoding='utf-8')
    return manifest

