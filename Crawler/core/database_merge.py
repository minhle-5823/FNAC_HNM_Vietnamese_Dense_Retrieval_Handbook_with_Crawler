"""Create one lossless, read-only-style SQLite snapshot from a configured site group.

All eight crawler tables are copied. Since integer IDs restart in each site DB,
merged tables allocate fresh IDs and preserve (origin_site, origin_id) to trace a
record back to its original source. Source databases are never modified.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from .database import utc_now
from .profile_loader import load_group, load_site
from .url_tools import normalize_url

TABLES = (
    "url_state", "raw_pages", "processed_documents", "passages", "links",
    "events", "robots_checks", "crawl_runs",
)


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _columns(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    columns = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    if not columns or columns[0]["name"] != "id":
        raise ValueError(f"Source DB table {table!r} has an unsupported schema")
    return columns


def _validate_seeds(conn: sqlite3.Connection, repo_root: Path, site: str) -> None:
    """Don't silently merge legacy data after a profile's seed URL changes."""
    profile = load_site(repo_root, site)
    expected = {
        normalize_url(url, url, profile) for url in profile["seed_urls"]
    }
    existing = {
        row[0] for row in conn.execute(
            "SELECT url FROM url_state WHERE depth=0 AND discovered_from IS NULL"
        )
    }
    if existing != expected:
        raise ValueError(
            f"{site}: crawl DB contains seeds from a different profile. "
            f"Run 'python main.py crawl --site {site} --reset' first, "
            "then re-run merge-dbs."
        )


def merge_group_databases(repo_root: Path, group_id: str = "vire_web_corpus") -> dict:
    """Merge ALL records of configured site DBs, with atomic output replacement.

    Raises if any DB is missing, corrupted or has stale seed URLs. This avoids
    labelling an incomplete or outdated snapshot as a complete five-site merge.
    """
    sites = load_group(repo_root, group_id)
    if not sites or len(sites) != len(set(sites)):
        raise ValueError("Group must contain nonempty, unique site IDs")

    data_dir = repo_root / "data"
    out_db = data_dir / f"{group_id}_merged.db"
    temporary = data_dir / f"{group_id}_merged.db.tmp"
    data_dir.mkdir(parents=True, exist_ok=True)

    src_paths = {site: data_dir / f"{site}.db" for site in sites}
    absent = [site for site, path in src_paths.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing DB for site(s): {', '.join(absent)}")
    if temporary.exists():
        temporary.unlink()

    merged = None
    per_site: dict[str, dict[str, int]] = {}
    total = {table: 0 for table in TABLES}
    merged_at = utc_now()
    try:
        merged = sqlite3.connect(temporary)
        merged.execute("PRAGMA journal_mode=DELETE")
        merged.execute("PRAGMA synchronous=FULL")
        merged.execute("""
            CREATE TABLE merge_sources (
                site_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                merged_at TEXT NOT NULL,
                row_counts_json TEXT NOT NULL
            )
        """)
        merged.execute("""
            CREATE TABLE merge_info (
                group_id TEXT PRIMARY KEY,
                merged_at TEXT NOT NULL,
                site_count INTEGER NOT NULL
            )
        """)
        canonical: dict[str, list[tuple[str, str, int, object]]] = {}

        for site in sites:
            source_path = src_paths[site]
            # Read-only URI avoids creating or mutating individual crawler DBs.
            with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as src:
                src.row_factory = sqlite3.Row
                if src.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError(f"Corrupted source database: {source_path}")
                _validate_seeds(src, repo_root, site)
                site_counts: dict[str, int] = {}
                for table in TABLES:
                    columns = _columns(src, table)
                    signature = [(c["name"], c["type"], c["notnull"], c["dflt_value"])
                                 for c in columns]
                    if table in canonical and signature != canonical[table]:
                        raise ValueError(f"Schema mismatch for {site}.{table}")
                    canonical.setdefault(table, signature)

                    if site == sites[0]:
                        # Original id is replaced with a globally unique id.
                        definitions = [
                            'id INTEGER PRIMARY KEY AUTOINCREMENT',
                            'origin_site TEXT NOT NULL',
                            'origin_id INTEGER NOT NULL',
                        ]
                        for c in columns[1:]:
                            definition = f'{_q(c["name"])} {c["type"] or "TEXT"}'
                            if c["notnull"]:
                                definition += ' NOT NULL'
                            if c["dflt_value"] is not None:
                                definition += f' DEFAULT {c["dflt_value"]}'
                            definitions.append(definition)
                        definitions.append('UNIQUE (origin_site, origin_id)')
                        merged.execute(f'CREATE TABLE {_q(table)} ({", ".join(definitions)})')

                    column_names = [c["name"] for c in columns[1:]]
                    columns_sql = ", ".join(_q(name) for name in column_names)
                    placeholders = ", ".join("?" for _ in range(2 + len(column_names)))
                    insert_sql = (f'INSERT INTO {_q(table)} '
                                  f'(origin_site, origin_id, {columns_sql}) '
                                  f'VALUES ({placeholders})')
                    cursor = src.execute(f'SELECT * FROM {_q(table)} ORDER BY id')
                    count = 0
                    while True:
                        batch = cursor.fetchmany(100)
                        if not batch:
                            break
                        merged.executemany(
                            insert_sql,
                            [(site, row['id'], *(row[name] for name in column_names)) for row in batch]
                        )
                        count += len(batch)
                    site_counts[table] = count
                    total[table] += count
                merged.execute(
                    'INSERT INTO merge_sources VALUES (?, ?, ?, ?)',
                    (site, source_path.name, merged_at, json.dumps(site_counts)),
                )
                per_site[site] = site_counts

        merged.execute('INSERT INTO merge_info VALUES (?, ?, ?)', (group_id, merged_at, len(sites)))
        merged.executescript("""
            CREATE INDEX idx_merge_raw_site ON raw_pages(origin_site, url);
            CREATE INDEX idx_merge_processed_valid ON processed_documents(is_valid, source_site);
            CREATE INDEX idx_merge_passages_site ON passages(source_site, domain_label);
            CREATE INDEX idx_merge_passages_url ON passages(page_url, passage_index);
            CREATE INDEX idx_merge_links_url ON links(origin_site, source_url);
            CREATE INDEX idx_merge_events_site ON events(origin_site, kind);
            CREATE INDEX idx_merge_state ON url_state(origin_site, state, depth);
        """)
        merged.commit()
        if merged.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Merged database failed integrity_check')
        merged.close()
        merged = None
        os.replace(temporary, out_db)
    except Exception:
        if merged is not None:
            merged.close()
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "group": group_id,
        "database": str(out_db),
        "merged_at": merged_at,
        "sites": sites,
        "totals": total,
        "per_site": per_site,
    }
