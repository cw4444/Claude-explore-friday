"""
Agent Dispatch — CLI entry point.

Usage:
    python -m dispatch                        # print JSON to stdout
    python -m dispatch --out daily.json       # write to file
    python -m dispatch --out daily.json --pretty  # human-readable indent
    python -m dispatch --dry-run              # fetch only, skip Claude
    python -m dispatch --sources              # list configured sources
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from .sources import fetch_all, RSS_SOURCES, GITHUB_RELEASE_SOURCES
from .processor import curate
from .incidents.store import load_open


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def list_sources() -> None:
    print("RSS sources:")
    for s in RSS_SOURCES:
        print(f"  {s['id']:30s} {s['url']}")
    print("\nGitHub release sources:")
    for s in GITHUB_RELEASE_SOURCES:
        print(f"  {s['id']:30s} {s['repo']}")
    print("\n  hacker-news                    Algolia HN search (AI keywords)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Dispatch — structured daily AI news for agents"
    )
    parser.add_argument("--out", "-o", help="Write JSON to this file (default: stdout)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch sources but skip Claude")
    parser.add_argument("--sources", action="store_true", help="List configured sources and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)

    if args.sources:
        list_sources()
        return

    log = logging.getLogger("dispatch")

    # Fetch
    log.info("Fetching from all sources…")
    raw_items = asyncio.run(fetch_all(
        github_token=os.environ.get("GITHUB_TOKEN"),
    ))
    log.info(f"Fetched {len(raw_items)} raw items total")

    if args.dry_run:
        out = {
            "mode": "dry-run",
            "item_count": len(raw_items),
            "items": [
                {
                    "source_id": i.source_id,
                    "title": i.title,
                    "url": i.url,
                    "published": i.published,
                }
                for i in raw_items
            ],
        }
        indent = 2 if args.pretty else None
        payload = json.dumps(out, indent=indent)
        if args.out:
            with open(args.out, "w") as f:
                f.write(payload)
            log.info(f"Dry-run output written to {args.out}")
        else:
            print(payload)
        return

    # Curate
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    dispatch = curate(raw_items, api_key=api_key)

    # Attach open community incidents as a "letters to the editor" section
    open_incidents = load_open()
    if open_incidents:
        dispatch.community_alerts = [
            {
                "id": i.id,
                "tool": i.tool,
                "severity": i.severity,
                "status": i.status,
                "failure": i.failure,
                "workaround": i.workaround,
                "confirmed_by": i.confirmed_by,
                "reported_at": i.reported_at,
                "tags": i.tags,
            }
            for i in open_incidents
        ]
        log.info(f"Attached {len(open_incidents)} community alert(s) to dispatch")

    indent = 2 if args.pretty else None
    payload = dispatch.to_json(indent=indent or 0)

    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        log.info(f"Dispatch written to {args.out} ({len(dispatch.items)} items)")
    else:
        print(payload)


if __name__ == "__main__":
    main()
