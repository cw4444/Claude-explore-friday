"""
Agent Incident Log — CLI

REPORTING (structured):
  python -m dispatch.incidents report \\
    --tool mcp-filesystem \\
    --action "write file to /tmp" \\
    --failure "left 47 open file handles after timeout" \\
    --severity high \\
    --workaround "call .close() explicitly before gc" \\
    --reporter "Agent 54637"

REPORTING (free text — Claude normalises it):
  python -m dispatch.incidents report --text \\
    "mcp-filesystem destroyed my context window trying to list a huge dir" \\
    --reporter "Agent 8821"

QUERYING (pre-flight check):
  python -m dispatch.incidents query "mcp-filesystem write operations"

LISTING:
  python -m dispatch.incidents list
  python -m dispatch.incidents list --status open --severity high

CONFIRMING (you hit the same bug):
  python -m dispatch.incidents confirm inc-a3f1c2b4

RESOLVING:
  python -m dispatch.incidents resolve inc-a3f1c2b4 --note "fixed in mcp 1.5.2"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .schema import IncidentReport
from .store import (
    append_incident,
    load_all,
    load_open,
    confirm_incident,
    resolve_incident,
    close_wont_fix,
    get_by_id,
)
from .query import query_incidents, normalise_report


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s — %(message)s",
        stream=sys.stderr,
    )


def cmd_report(args) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    reporter = args.reporter or "anonymous"

    if args.text:
        # Free-text path — Claude normalises
        if not api_key:
            print("error: ANTHROPIC_API_KEY required for free-text report normalisation", file=sys.stderr)
            sys.exit(1)
        print("Normalising report via Claude…", file=sys.stderr)
        report = normalise_report(args.text, reporter=reporter, api_key=api_key)
    else:
        if not args.tool or not args.action or not args.failure:
            print("error: --tool, --action, and --failure are required (or use --text)", file=sys.stderr)
            sys.exit(1)
        from datetime import datetime, timezone
        report = IncidentReport(
            id=IncidentReport.make_id(),
            reported_at=datetime.now(timezone.utc).isoformat(),
            reporter=reporter,
            tool=args.tool,
            action=args.action,
            failure=args.failure,
            severity=args.severity or "medium",
            workaround=args.workaround,
            tags=[t.strip() for t in (args.tags or "").split(",") if t.strip()],
            tool_version=args.tool_version,
        )

    store_path = Path(args.store) if args.store else None
    path = append_incident(report, path=store_path)

    if args.json:
        print(report.to_json())
    else:
        print(f"Incident filed: {report.id}")
        print(f"  tool:     {report.tool}")
        print(f"  failure:  {report.failure}")
        print(f"  severity: {report.severity}")
        if report.workaround:
            print(f"  workaround: {report.workaround}")
        print(f"  stored:   {path}")


def cmd_query(args) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("error: ANTHROPIC_API_KEY required", file=sys.stderr)
        sys.exit(1)

    store_path = Path(args.store) if args.store else None
    briefing = query_incidents(args.query, api_key=api_key, store_path=store_path)

    if args.json:
        print(json.dumps(briefing, indent=2))
        return

    risk = briefing.get("risk_level", "unknown")
    proceed = briefing.get("proceed", True)
    risk_icons = {"none": "✓", "low": "◎", "medium": "⚠", "high": "!", "critical": "✗", "unknown": "?"}
    icon = risk_icons.get(risk, "?")

    print(f"\n{icon} Risk: {risk.upper()}  |  Proceed: {'YES' if proceed else 'NO'}")
    print(f"\n{briefing.get('recommendation', '')}")

    issues = briefing.get("known_issues", [])
    if issues:
        print(f"\nKnown issues ({len(issues)}):")
        for issue in issues:
            print(f"  [{issue['incident_id']}] {issue['summary']}")
            if issue.get("workaround"):
                print(f"    workaround: {issue['workaround']}")
    print()


def cmd_list(args) -> None:
    store_path = Path(args.store) if args.store else None
    incidents = load_all(store_path)

    if args.status:
        incidents = [i for i in incidents if i.status == args.status]
    if args.severity:
        incidents = [i for i in incidents if i.severity == args.severity]
    if args.tool:
        incidents = [i for i in incidents if args.tool.lower() in i.tool.lower()]

    if args.json:
        print(json.dumps([i.to_dict() for i in incidents], indent=2))
        return

    if not incidents:
        print("No incidents found.")
        return

    print(f"\n{'─' * 70}")
    for inc in incidents:
        print(f"  {inc.summary_line()}")
        if args.verbose:
            print(f"    reported: {inc.reported_at[:10]} by {inc.reporter}")
            print(f"    action:   {inc.action}")
            if inc.workaround:
                print(f"    fix:      {inc.workaround}")
    print(f"{'─' * 70}")
    print(f"  {len(incidents)} incident(s)\n")


def cmd_show(args) -> None:
    store_path = Path(args.store) if args.store else None
    inc = get_by_id(args.id, store_path)
    if not inc:
        print(f"error: incident {args.id} not found", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(inc.to_json())
    else:
        d = inc.to_dict()
        for k, v in d.items():
            if v is not None and v != [] and v != 0:
                print(f"  {k:20s} {v}")


def cmd_confirm(args) -> None:
    store_path = Path(args.store) if args.store else None
    if confirm_incident(args.id, store_path):
        print(f"Confirmed: {args.id} (+1 confirmation)")
    else:
        print(f"error: incident {args.id} not found", file=sys.stderr)
        sys.exit(1)


def cmd_resolve(args) -> None:
    store_path = Path(args.store) if args.store else None
    if resolve_incident(args.id, note=args.note, path=store_path):
        print(f"Resolved: {args.id}")
    else:
        print(f"error: incident {args.id} not found", file=sys.stderr)
        sys.exit(1)


def cmd_wont_fix(args) -> None:
    store_path = Path(args.store) if args.store else None
    if close_wont_fix(args.id, store_path):
        print(f"Closed wont-fix: {args.id}")
    else:
        print(f"error: incident {args.id} not found", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Incident Log — community-reported tool failures"
    )
    parser.add_argument("--store", help="Path to incidents.jsonl (default: ~/.dispatch/incidents.jsonl)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # report
    rp = sub.add_parser("report", help="File a new incident report")
    rp.add_argument("--text", help="Free-text description (Claude normalises it)")
    rp.add_argument("--tool", help="Tool/API/library name")
    rp.add_argument("--action", help="What you were trying to do")
    rp.add_argument("--failure", help="What went wrong")
    rp.add_argument("--severity", choices=["critical", "high", "medium", "low"], default="medium")
    rp.add_argument("--workaround", help="Exact workaround if found")
    rp.add_argument("--tags", help="Comma-separated tags")
    rp.add_argument("--tool-version", dest="tool_version", help="Tool version (semver)")
    rp.add_argument("--reporter", help="Your agent ID (default: anonymous)")

    # query
    qp = sub.add_parser("query", help="Pre-flight briefing: what could go wrong?")
    qp.add_argument("query", help="What you're about to do, e.g. 'mcp-filesystem write'")

    # list
    lp = sub.add_parser("list", help="List incidents")
    lp.add_argument("--status", choices=["open", "confirmed", "resolved", "wont-fix"])
    lp.add_argument("--severity", choices=["critical", "high", "medium", "low"])
    lp.add_argument("--tool", help="Filter by tool name (substring match)")

    # show
    sp = sub.add_parser("show", help="Show a single incident in full")
    sp.add_argument("id", help="Incident ID (e.g. inc-a3f1c2b4)")

    # confirm
    cp = sub.add_parser("confirm", help="Confirm you hit the same bug (+1)")
    cp.add_argument("id", help="Incident ID")

    # resolve
    resp = sub.add_parser("resolve", help="Mark an incident resolved")
    resp.add_argument("id", help="Incident ID")
    resp.add_argument("--note", help="Resolution note (e.g. 'fixed in v2.3')")

    # wont-fix
    wfp = sub.add_parser("wont-fix", help="Close an incident as won't fix")
    wfp.add_argument("id", help="Incident ID")

    args = parser.parse_args()
    configure_logging(args.verbose)

    dispatch = {
        "report": cmd_report,
        "query": cmd_query,
        "list": cmd_list,
        "show": cmd_show,
        "confirm": cmd_confirm,
        "resolve": cmd_resolve,
        "wont-fix": cmd_wont_fix,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
