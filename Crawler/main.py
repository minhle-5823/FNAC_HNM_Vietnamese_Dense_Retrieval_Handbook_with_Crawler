"""CLI for VIRE web-corpus crawler.

Commands:
  sites, inspect, robots, crawl, crawl-all, merge-dbs, validate, build-corpus
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.crawler import UnifiedCrawler
from core.profile_loader import list_sites, load_group, load_site
from core.validator import validate_database
from core.corpus_builder import build_group_corpus
from core.database_merge import merge_group_databases

REPO_ROOT=Path(__file__).resolve().parent

def cmd_sites(_):
    print("Configured site profiles")
    print("="*88)
    for s in list_sites(REPO_ROOT):
        p=load_site(REPO_ROOT,s["id"])
        print(f'{s["id"]:<24} | {p.get("domain_label",""):<17} | {s["name"]}')

def cmd_inspect(a):
    p=load_site(REPO_ROOT,a.site)
    keys=["id","name","topic","domain_label","seed_urls","allowed_domains","crawl","rate_limit","policy","url","page_types","extract","links","preprocess"]
    print(json.dumps({k:p.get(k) for k in keys},ensure_ascii=False,indent=2))

def make(a,site=None):
    p=load_site(REPO_ROOT,site or a.site)
    return UnifiedCrawler(REPO_ROOT,p,max_pages=getattr(a,"max_pages",None),
        max_depth=getattr(a,"max_depth",None),max_requests=getattr(a,"max_requests",None),
        reset=getattr(a,"reset",False))

def cmd_robots(a):
    c=make(a)
    try:
        for r in c.preflight(): print(json.dumps(r,ensure_ascii=False,indent=2))
    finally:c.close()

def cmd_crawl(a):
    c=make(a)
    try:c.run()
    finally:c.close()

def cmd_crawl_all(a):
    sites=load_group(REPO_ROOT,a.group)
    print(f"[GROUP] {a.group}: {', '.join(sites)}")
    successful = []
    for sid in sites:
        print("\n"+"#"*92+f"\n# SITE: {sid}\n"+"#"*92)
        p=load_site(REPO_ROOT,sid)
        if not p.get("enabled",True):
            print("[SKIP] disabled"); continue
        c=UnifiedCrawler(REPO_ROOT,p,max_pages=a.max_pages,max_depth=a.max_depth,
                         max_requests=a.max_requests,reset=a.reset)
        try:
            summary = c.run()
            if summary.get("stop_reason", "").lower().startswith("interrupted"):
                print(f"[SKIP MERGE] {sid} was interrupted")
            else:
                successful.append(sid)
        except Exception as e: print(f"[SITE ERROR] {sid}: {type(e).__name__}: {e}")
        finally:c.close()
    if len(successful) == len(sites):
        print("\n[GROUP] Finished all sites; merging full SQLite databases...")
        try:
            result = merge_group_databases(REPO_ROOT, a.group)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[MERGE ERROR] {type(e).__name__}: {e}")
            raise
    else:
        print("[SKIP MERGE] Not all group sites completed successfully. "
              "Fix any crawl errors and run merge-dbs when ready.")

def cmd_merge_dbs(a):
    print(json.dumps(merge_group_databases(REPO_ROOT, a.group), ensure_ascii=False, indent=2))

def cmd_validate(a):
    print(json.dumps(validate_database(REPO_ROOT/"data"/f"{a.site}.db"),ensure_ascii=False,indent=2))

def cmd_build(a):
    print(json.dumps(build_group_corpus(REPO_ROOT,a.group),ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser(description="VIRE Vietnamese retrieval web-corpus crawler")
    sp=ap.add_subparsers(dest="cmd",required=True)
    sp.add_parser("sites").set_defaults(func=cmd_sites)
    p=sp.add_parser("inspect"); p.add_argument("--site",required=True); p.set_defaults(func=cmd_inspect)
    p=sp.add_parser("robots"); p.add_argument("--site",required=True); p.set_defaults(func=cmd_robots)
    p=sp.add_parser("crawl")
    p.add_argument("--site",required=True); p.add_argument("--max-pages",type=int); p.add_argument("--max-depth",type=int)
    p.add_argument("--max-requests",type=int); p.add_argument("--reset",action="store_true"); p.set_defaults(func=cmd_crawl)
    p=sp.add_parser("crawl-all")
    p.add_argument("--group",default="vire_web_corpus"); p.add_argument("--max-pages",type=int); p.add_argument("--max-depth",type=int)
    p.add_argument("--max-requests",type=int); p.add_argument("--reset",action="store_true"); p.set_defaults(func=cmd_crawl_all)
    p=sp.add_parser("merge-dbs"); p.add_argument("--group",default="vire_web_corpus"); p.set_defaults(func=cmd_merge_dbs)
    p=sp.add_parser("validate"); p.add_argument("--site",required=True); p.set_defaults(func=cmd_validate)
    p=sp.add_parser("build-corpus"); p.add_argument("--group",default="vire_web_corpus"); p.set_defaults(func=cmd_build)
    a=ap.parse_args(); a.func(a)
if __name__=="__main__": main()
