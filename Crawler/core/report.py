"""Write human-readable and machine-readable crawl reports."""
from __future__ import annotations
import json
from pathlib import Path

def write_report(report_dir: Path, profile: dict, summary: dict, validation: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "site": profile["id"],
        "name": profile["name"],
        "summary": summary,
        "validation": validation,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# Kết quả crawl - {profile['name']}",
        "",
        f"- Site ID: `{profile['id']}`",
        f"- Raw pages saved: **{summary.get('raw_saved', 0):,}**",
        f"- Valid documents: **{summary.get('processed_valid', 0):,}**",
        f"- Rejected documents: **{summary.get('processed_rejected', 0):,}**",
        f"- Passages saved: **{summary.get('passages_saved', 0):,}**",
        f"- Requests sent: **{summary.get('requests_sent', 0):,}**",
        f"- Links saved: **{summary.get('links_saved', 0):,}**",
        f"- Failed requests: **{summary.get('failed_requests', 0):,}**",
        f"- Robots skipped: **{summary.get('robots_skipped', 0):,}**",
        f"- Rate-limit responses: **{summary.get('rate_limit_responses', 0):,}**",
        f"- Hosts blocked: **{summary.get('hosts_blocked', 0):,}**",
        f"- Stop reason: **{summary.get('stop_reason', '')}**",
        f"- Validation: **{validation.get('result', '')}**",
        "",
        "## Phân bố depth",
        "",
    ]

    for depth, count in sorted(
        (validation.get("by_depth") or {}).items(),
        key=lambda x: int(x[0]),
    ):
        lines.append(f"- Depth {depth}: {count:,}")

    lines.extend(["", "## Page type", ""])
    for kind, count in sorted((validation.get("by_page_type") or {}).items()):
        lines.append(f"- {kind}: {count:,}")

    lines.extend(["", "## Rejection reasons", ""])
    for reason, count in sorted((validation.get("rejection_reasons") or {}).items()):
        lines.append(f"- {reason}: {count:,}")

    lines.append("")
    (report_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
