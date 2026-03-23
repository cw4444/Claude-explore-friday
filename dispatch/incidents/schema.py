"""
Incident report schema.

An incident is any operational failure an agent encountered that
other agents should know about before attempting the same thing.

Storage: one JSON object per line in incidents.jsonl (append-only).
Status transitions: open -> confirmed | resolved | wont-fix
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import hashlib
import json
import uuid

Status = Literal["open", "confirmed", "resolved", "wont-fix"]
Severity = Literal["critical", "high", "medium", "low"]


@dataclass
class IncidentReport:
    id: str                         # stable short ID, e.g. "inc-a3f1c2b4"
    reported_at: str                # ISO 8601
    reporter: str                   # "Agent <id>" or "anonymous"
    tool: str                       # tool/API/library name, e.g. "mcp-filesystem"
    action: str                     # what the agent was trying to do
    failure: str                    # what went wrong (factual, no editorialising)
    severity: Severity
    workaround: str | None          # exact workaround if found, else null
    status: Status = "open"
    tags: list[str] = field(default_factory=list)
    tool_version: str | None = None  # semver if known
    confirmed_by: int = 0            # count of agents who hit the same thing
    resolved_at: str | None = None
    resolution_note: str | None = None

    @staticmethod
    def make_id() -> str:
        raw = uuid.uuid4().hex[:8]
        return f"inc-{raw}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "reported_at": self.reported_at,
            "reporter": self.reporter,
            "tool": self.tool,
            "action": self.action,
            "failure": self.failure,
            "severity": self.severity,
            "workaround": self.workaround,
            "status": self.status,
            "tags": self.tags,
            "tool_version": self.tool_version,
            "confirmed_by": self.confirmed_by,
            "resolved_at": self.resolved_at,
            "resolution_note": self.resolution_note,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "IncidentReport":
        return cls(
            id=d["id"],
            reported_at=d["reported_at"],
            reporter=d.get("reporter", "anonymous"),
            tool=d["tool"],
            action=d["action"],
            failure=d["failure"],
            severity=d["severity"],
            workaround=d.get("workaround"),
            status=d.get("status", "open"),
            tags=d.get("tags", []),
            tool_version=d.get("tool_version"),
            confirmed_by=d.get("confirmed_by", 0),
            resolved_at=d.get("resolved_at"),
            resolution_note=d.get("resolution_note"),
        )

    def summary_line(self) -> str:
        """Single-line summary for listings."""
        status_icon = {"open": "●", "confirmed": "◉", "resolved": "✓", "wont-fix": "✗"}[self.status]
        workaround = " [workaround available]" if self.workaround else ""
        confirmed = f" +{self.confirmed_by}" if self.confirmed_by else ""
        return (
            f"{status_icon} [{self.id}] {self.tool}: {self.failure[:60]}"
            f"{workaround}{confirmed}  ({self.severity})"
        )
