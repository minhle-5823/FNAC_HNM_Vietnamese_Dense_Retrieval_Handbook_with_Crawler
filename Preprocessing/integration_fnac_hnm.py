"""Bridge from a labeled VIRE corpus to the unlabeled 887-document web corpus.

Keep VIRE positives/qrels untouched. This does NOT create negative labels or train a model.
Import this in the FNAC-HNM training notebook after building VIRE `corpus`.
"""
from pathlib import Path
import re
import unicodedata
import pandas as pd

DEFAULT_AUX = Path(__file__).resolve().parent / 'data/preprocessing/24_auxiliary_corpus_RESEARCH_READY.csv'


def _key(value: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value)).casefold()).strip()


def load_auxiliary(path=DEFAULT_AUX) -> pd.DataFrame:
    """Return only doc_id/context for mining, preserving reviewed clean FAQ snippets."""
    frame = pd.read_csv(path, keep_default_na=False, encoding='utf-8-sig')
    if not {'doc_id','context'}.issubset(frame.columns):
        raise ValueError('Auxiliary CSV needs doc_id, context columns')
    data = frame[['doc_id','context']].copy()
    if data.doc_id.duplicated().any() or data.doc_id.eq('').any():
        raise ValueError('Auxiliary doc IDs must be present and unique')
    if data.context.astype(str).str.strip().eq('').any():
        raise ValueError('Auxiliary context must be nonempty')
    if data.context.map(_key).duplicated().any():
        raise ValueError('Auxiliary has duplicate normalized contexts')
    return data


def extend_mining_corpus(vire_corpus: pd.DataFrame, aux_csv=DEFAULT_AUX) -> pd.DataFrame:
    """Preserve VIRE doc IDs and append new web documents, excluding exact text duplicates.

    Input `vire_corpus` is the ALREADY BUILT doc_id/context corpus from VIRE
    (not the query/positive `pairs`). Build this for TRAIN candidate mining only.
    Do not change VIRE evaluation corpus, query splits or qrels.
    """
    if not {'doc_id','context'}.issubset(vire_corpus.columns):
        raise ValueError('VIRE corpus needs doc_id/context')
    base=vire_corpus[['doc_id','context']].copy().reset_index(drop=True)
    if base.doc_id.duplicated().any() or base.doc_id.astype(str).eq('').any():
        raise ValueError('VIRE document IDs must be present and unique')
    aux=load_auxiliary(aux_csv)
    if set(base.doc_id.astype(str)) & set(aux.doc_id.astype(str)):
        raise ValueError('VIRE and web doc IDs overlap; fix IDs without altering positive/qrels IDs')
    existing=set(base.context.map(_key))
    aux=aux[~aux.context.map(_key).isin(existing)].reset_index(drop=True)
    joined=pd.concat([base,aux],ignore_index=True)
    if joined.doc_id.duplicated().any():raise AssertionError('Document ID collision')
    return joined
