"""Validate raw crawl integrity + processed retrieval-corpus quality."""
from __future__ import annotations
import sqlite3
from pathlib import Path

def validate_database(db_path: Path) -> dict:
    if not db_path.exists():
        return {"result":"CHECK REQUIRED","reason":"database_not_found","raw_pages":0}

    conn=sqlite3.connect(db_path)
    conn.row_factory=sqlite3.Row

    def one(sql):
        return conn.execute(sql).fetchone()[0]

    result = {
        "raw_pages": one("SELECT COUNT(*) FROM raw_pages"),
        "processed_documents": one("SELECT COUNT(*) FROM processed_documents"),
        "valid_documents": one("SELECT COUNT(*) FROM processed_documents WHERE is_valid=1"),
        "rejected_documents": one("SELECT COUNT(*) FROM processed_documents WHERE is_valid=0"),
        "passages": one("SELECT COUNT(*) FROM passages"),
        "links": one("SELECT COUNT(*) FROM links"),
        "duplicate_raw_urls": one(
            "SELECT COUNT(*) FROM (SELECT url FROM raw_pages GROUP BY url HAVING COUNT(*)>1)"
        ),
        "duplicate_processed_hashes": one("""SELECT COUNT(*) FROM (
            SELECT processed_hash FROM processed_documents
            WHERE is_valid=1 AND processed_hash<>'' GROUP BY processed_hash HAVING COUNT(*)>1)"""),
        "low_vi_valid": one(
            "SELECT COUNT(*) FROM processed_documents WHERE is_valid=1 AND language_score<0.35"
        ),
        "empty_passages": one("SELECT COUNT(*) FROM passages WHERE TRIM(text)=''"),
        "request_errors": one("SELECT COUNT(*) FROM events WHERE kind='request_error'"),
        "http_errors": one("SELECT COUNT(*) FROM events WHERE kind='http_error'"),
        "robots_skips": one("SELECT COUNT(*) FROM events WHERE kind='robots_skip'"),
        "blocks": one(
            "SELECT COUNT(*) FROM events WHERE kind IN ('host_blocked','challenge')"
        ),
    }

    result["by_depth"] = {
        str(r["depth"]): r["n"]
        for r in conn.execute(
            "SELECT depth,COUNT(*) n FROM raw_pages GROUP BY depth ORDER BY depth"
        )
    }
    result["by_page_type"] = {
        str(r["page_type"]): r["n"]
        for r in conn.execute(
            "SELECT page_type,COUNT(*) n FROM raw_pages GROUP BY page_type ORDER BY page_type"
        )
    }
    result["rejection_reasons"] = {
        (r["rejection_reason"] or "unknown"): r["n"]
        for r in conn.execute(
            """SELECT rejection_reason,COUNT(*) n
               FROM processed_documents
               WHERE is_valid=0 GROUP BY rejection_reason"""
        )
    }

    hard = (
        result["raw_pages"]==0
        or result["valid_documents"]==0
        or result["passages"]==0
        or result["duplicate_raw_urls"]>0
        or result["duplicate_processed_hashes"]>0
        or result["low_vi_valid"]>0
        or result["empty_passages"]>0
    )
    result["result"]="CHECK REQUIRED" if hard else "PASS"
    conn.close()
    return result
