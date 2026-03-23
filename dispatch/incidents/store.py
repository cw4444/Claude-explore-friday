"""
Incident store — append-only JSONL file.

Default path: ~/.dispatch/incidents.jsonl
Override with DISPATCH_INCIDENTS_PATH env var.

Append-only means:
- New reports are appended as new lines
- Status changes (confirm, resolve) are also appended as patch records
- Reading always replays all records to get current state

This makes it trivially safe for concurrent agents to write without
locking, and gives a full audit trail for free.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .schema import IncidentReport, Status


def _default_path() -> Path:
    override = os.environ.get("DISPATCH_INCIDENTS_PATH")
    if override:
        return Path(override)
    return Path.home() / ".dispatch" / "incidents.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def append_incident(report: IncidentReport, path: Path | None = None) -> Path:
    """Append a new incident record. Returns the file path."""
    p = path or _default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(report.to_json() + "\n")
    return p


def append_patch(
    incident_id: str,
    patch: dict,
    path: Path | None = None,
) -> None:
    """
    Append a patch record — a partial update applied on read.
    patch keys: status, resolved_at, resolution_note, confirmed_by_delta
    """
    p = path or _default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {"_patch": incident_id, "_patched_at": _now_iso(), **patch}
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_all(path: Path | None = None) -> list[IncidentReport]:
    """
    Replay the JSONL file to reconstruct current incident state.
    Patches are applied in order over their base records.
    """
    p = path or _default_path()
    if not p.exists():
        return []

    incidents: dict[str, IncidentReport] = {}

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "_patch" in record:
                # Apply patch to existing incident
                inc_id = record["_patch"]
                if inc_id not in incidents:
                    continue
                inc = incidents[inc_id]
                if "status" in record:
                    inc.status = record["status"]
                if "resolved_at" in record:
                    inc.resolved_at = record["resolved_at"]
                if "resolution_note" in record:
                    inc.resolution_note = record["resolution_note"]
                if "confirmed_by_delta" in record:
                    inc.confirmed_by += record["confirmed_by_delta"]
            else:
                try:
                    inc = IncidentReport.from_dict(record)
                    incidents[inc.id] = inc
                except (KeyError, TypeError):
                    continue

    # Sort: open/confirmed first, then by severity, then by date desc
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    status_order = {"open": 0, "confirmed": 1, "resolved": 2, "wont-fix": 3}
    return sorted(
        incidents.values(),
        key=lambda i: (status_order[i.status], severity_order[i.severity], i.reported_at),
        reverse=False,
    )


def load_open(path: Path | None = None) -> list[IncidentReport]:
    return [i for i in load_all(path) if i.status in ("open", "confirmed")]


def get_by_id(incident_id: str, path: Path | None = None) -> IncidentReport | None:
    for inc in load_all(path):
        if inc.id == incident_id:
            return inc
    return None


def search(
    query: str,
    status_filter: list[Status] | None = None,
    path: Path | None = None,
) -> list[IncidentReport]:
    """Simple keyword search across tool, action, failure, tags."""
    q = query.lower()
    results = []
    for inc in load_all(path):
        if status_filter and inc.status not in status_filter:
            continue
        haystack = " ".join([
            inc.tool, inc.action, inc.failure,
            inc.workaround or "", " ".join(inc.tags),
        ]).lower()
        if q in haystack:
            results.append(inc)
    return results


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def confirm_incident(incident_id: str, path: Path | None = None) -> bool:
    """Increment confirmed_by count. Returns False if not found."""
    if get_by_id(incident_id, path) is None:
        return False
    append_patch(incident_id, {"confirmed_by_delta": 1}, path)
    return True


def resolve_incident(
    incident_id: str,
    note: str | None = None,
    path: Path | None = None,
) -> bool:
    if get_by_id(incident_id, path) is None:
        return False
    append_patch(
        incident_id,
        {
            "status": "resolved",
            "resolved_at": _now_iso(),
            "resolution_note": note,
        },
        path,
    )
    return True


def close_wont_fix(incident_id: str, path: Path | None = None) -> bool:
    if get_by_id(incident_id, path) is None:
        return False
    append_patch(incident_id, {"status": "wont-fix"}, path)
    return True
