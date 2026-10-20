"""SQLite persistence for raw crawl data + processed retrieval corpus.

REVIEW MAP
----------
SECTION 1: schema
SECTION 2: persistent URL frontier
SECTION 3: RAW storage
SECTION 4: PROCESSED documents/passages
SECTION 5: links/events/robots/runs
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS url_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    depth INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    discovered_from TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    source_site TEXT NOT NULL,
    domain_label TEXT NOT NULL,
    page_type TEXT NOT NULL,
    title TEXT,
    raw_html TEXT,
    extracted_text TEXT,
    depth INTEGER NOT NULL,
    status_code INTEGER NOT NULL,
    response_time REAL,
    crawled_at TEXT NOT NULL,
    raw_hash TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS processed_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_url TEXT UNIQUE NOT NULL,
    source_site TEXT NOT NULL,
    domain_label TEXT NOT NULL,
    title TEXT,
    clean_text TEXT,
    language_score REAL,
    relevance_score REAL,
    quality_score REAL,
    is_valid INTEGER NOT NULL,
    rejection_reason TEXT,
    processed_hash TEXT,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id TEXT UNIQUE NOT NULL,
    page_url TEXT NOT NULL,
    source_site TEXT NOT NULL,
    domain_label TEXT NOT NULL,
    passage_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    quality_score REAL,
    UNIQUE(page_url, passage_index)
);

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    target_url TEXT NOT NULL,
    discovered_depth INTEGER,
    UNIQUE(source_url, target_url)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    depth INTEGER,
    kind TEXT NOT NULL,
    status_code INTEGER,
    message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS robots_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    reason TEXT,
    status_code INTEGER,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    site_id TEXT NOT NULL,
    max_depth INTEGER NOT NULL,
    max_pages INTEGER NOT NULL,
    max_requests INTEGER NOT NULL,
    raw_saved INTEGER DEFAULT 0,
    processed_valid INTEGER DEFAULT 0,
    passages_saved INTEGER DEFAULT 0,
    requests_sent INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    stop_reason TEXT,
    summary_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_url_state_state_depth ON url_state(state, depth, id);
CREATE INDEX IF NOT EXISTS idx_raw_depth ON raw_pages(depth);
CREATE INDEX IF NOT EXISTS idx_processed_valid ON processed_documents(is_valid);
CREATE INDEX IF NOT EXISTS idx_passages_domain ON passages(domain_label, source_site);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_url);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""

class CrawlDB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @staticmethod
    def reset_file(path: Path) -> None:
        for candidate in [path, Path(str(path)+"-wal"), Path(str(path)+"-shm")]:
            if candidate.exists():
                candidate.unlink()

    def close(self):
        self.conn.commit()
        self.conn.close()

    # SECTION 2
    def discover_url(self, url: str, depth: int, source: str | None = None) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO url_state(url,depth,state,discovered_from,updated_at) VALUES (?,?,?,?,?)",
            (url, depth, "queued", source, utc_now()),
        )
        return cur.rowcount > 0

    def mark_url_done(self, url: str, state: str = "visited") -> None:
        self.conn.execute("UPDATE url_state SET state=?,updated_at=? WHERE url=?", (state, utc_now(), url))

    def visited_urls(self):
        return [r["url"] for r in self.conn.execute("SELECT url FROM url_state WHERE state<>'queued' ORDER BY id")]

    def queued_urls(self):
        return [(r["url"], r["depth"]) for r in self.conn.execute("SELECT url,depth FROM url_state WHERE state='queued' ORDER BY depth,id")]

    def existing_hashes(self):
        return [r["processed_hash"] for r in self.conn.execute(
            "SELECT processed_hash FROM processed_documents WHERE is_valid=1 AND processed_hash IS NOT NULL"
        )]

    # SECTION 3 - RAW
    def save_raw_page(self, page: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO raw_pages
            (url,domain,source_site,domain_label,page_type,title,raw_html,extracted_text,
             depth,status_code,response_time,crawled_at,raw_hash,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                page["url"], page["domain"], page["source_site"], page["domain_label"],
                page["page_type"], page.get("title",""), page.get("raw_html",""),
                page.get("extracted_text",""), page["depth"], page["status_code"],
                page.get("response_time"), page.get("crawled_at", utc_now()),
                page.get("raw_hash"), json.dumps(page.get("metadata") or {}, ensure_ascii=False),
            ),
        )

    # SECTION 4 - PROCESSED
    def save_processed(self, page_url: str, source_site: str, domain_label: str, processed) -> int:
        self.conn.execute(
            """INSERT OR REPLACE INTO processed_documents
            (page_url,source_site,domain_label,title,clean_text,language_score,relevance_score,
             quality_score,is_valid,rejection_reason,processed_hash,processed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                page_url, source_site, domain_label, processed.clean_title, processed.clean_text,
                processed.language_score, processed.relevance_score, processed.quality_score,
                1 if processed.valid else 0, processed.rejection_reason,
                processed.processed_hash, utc_now(),
            ),
        )
        self.conn.execute("DELETE FROM passages WHERE page_url=?", (page_url,))
        saved = 0
        if processed.valid:
            import hashlib
            for p in processed.passages:
                pid = hashlib.sha1(f"{page_url}#{p['passage_index']}".encode()).hexdigest()
                self.conn.execute(
                    """INSERT OR REPLACE INTO passages
                    (passage_id,page_url,source_site,domain_label,passage_index,text,word_count,quality_score)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        pid, page_url, source_site, domain_label, p["passage_index"],
                        p["text"], p["word_count"], processed.quality_score,
                    ),
                )
                saved += 1
        return saved

    def iter_raw_pages(self):
        return self.conn.execute("SELECT * FROM raw_pages ORDER BY id")

    def save_link(self, source_url: str, target_url: str, discovered_depth: int) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO links(source_url,target_url,discovered_depth) VALUES (?,?,?)",
            (source_url,target_url,discovered_depth),
        )
        return cur.rowcount > 0

    # SECTION 5
    def event(self, url, depth, kind, status_code=None, message="") -> None:
        self.conn.execute(
            "INSERT INTO events(url,depth,kind,status_code,message,created_at) VALUES (?,?,?,?,?,?)",
            (url,depth,kind,status_code,str(message),utc_now()),
        )
        self.conn.commit()

    def robots_check(self, url, allowed: bool, reason: str, status_code=None) -> None:
        self.conn.execute(
            "INSERT INTO robots_checks(url,allowed,reason,status_code,checked_at) VALUES (?,?,?,?,?)",
            (url,1 if allowed else 0,reason,status_code,utc_now()),
        )
        self.conn.commit()

    def start_run(self, site_id: str, max_depth: int, max_pages: int, max_requests: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawl_runs(started_at,site_id,max_depth,max_pages,max_requests) VALUES (?,?,?,?,?)",
            (utc_now(),site_id,max_depth,max_pages,max_requests),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, summary: dict) -> None:
        self.conn.execute(
            """UPDATE crawl_runs SET finished_at=?,raw_saved=?,processed_valid=?,passages_saved=?,
               requests_sent=?,failed_requests=?,stop_reason=?,summary_json=? WHERE id=?""",
            (
                utc_now(), summary.get("raw_saved",0), summary.get("processed_valid",0),
                summary.get("passages_saved",0), summary.get("requests_sent",0),
                summary.get("failed_requests",0), summary.get("stop_reason",""),
                json.dumps(summary, ensure_ascii=False), run_id,
            ),
        )
        self.conn.commit()
