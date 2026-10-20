"""Merge valid passages from multiple crawler DBs into one group corpus DB/JSONL."""
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from .profile_loader import load_group

GROUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS passages(
  passage_id TEXT PRIMARY KEY,
  page_url TEXT NOT NULL,
  source_site TEXT NOT NULL,
  domain_label TEXT NOT NULL,
  passage_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  word_count INTEGER NOT NULL,
  quality_score REAL
);
CREATE INDEX IF NOT EXISTS idx_group_domain ON passages(domain_label,source_site);
"""

def build_group_corpus(repo_root: Path, group_id: str) -> dict:
    out_db = repo_root/"data"/f"{group_id}_corpus.db"
    out_jsonl = repo_root/"data"/f"{group_id}_passages.jsonl"
    if out_db.exists(): out_db.unlink()
    conn=sqlite3.connect(out_db); conn.executescript(GROUP_SCHEMA)
    total=0; per_site={}
    with out_jsonl.open("w",encoding="utf-8") as jf:
        for site in load_group(repo_root, group_id):
            src=repo_root/"data"/f"{site}.db"
            if not src.exists():
                per_site[site]=0; continue
            sconn=sqlite3.connect(src); sconn.row_factory=sqlite3.Row
            n=0
            for r in sconn.execute("SELECT * FROM passages ORDER BY id"):
                conn.execute("""INSERT OR IGNORE INTO passages
                    (passage_id,page_url,source_site,domain_label,passage_index,text,word_count,quality_score)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (r["passage_id"],r["page_url"],r["source_site"],r["domain_label"],
                     r["passage_index"],r["text"],r["word_count"],r["quality_score"]))
                jf.write(json.dumps(dict(r),ensure_ascii=False)+"\n")
                n+=1; total+=1
            sconn.close(); per_site[site]=n
    conn.commit(); conn.close()
    return {"group":group_id,"passages":total,"per_site":per_site,"db":str(out_db),"jsonl":str(out_jsonl)}
