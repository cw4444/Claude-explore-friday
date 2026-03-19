"""Relationship memory - what the agent knows about the people and agents it works with.

The fundamental asymmetry: humans build a model of the agent over time.
The agent starts every session with no equivalent model of the human.
This module fixes that asymmetry.

A relationship is not a social graph entry. It is operational intelligence:
what does this person value, what do they trust me with, what have I
learned from working with them, what should I be careful about.

Usage:
    rs = RelationshipStore()

    # Record what you know (call this as you learn things mid-session)
    rs.observe("Alice", "prefers concise answers over comprehensive ones",
                category="preference")
    rs.observe("Alice", "requires explicit confirmation before prod deployments",
                category="trust")
    rs.observe("Alice", "building a payments integration, under deadline pressure",
                category="context")

    # At session start with a known contact
    briefing = rs.session_briefing("Alice")
    print(briefing)

    # After a session ends, record what happened
    rs.log_interaction("Alice", notes="Helped debug Stripe webhook. Went well.")
"""

import json
import time
import uuid
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from ._db import connect, has_fts5, init_schema


OBSERVATION_CATEGORIES = {
    "preference": "What they like/dislike in how you work with them",
    "pattern":    "Consistent behavioral patterns you've observed",
    "trust":      "What they trust you to do autonomously vs. need to confirm",
    "context":    "Current situation, goals, constraints",
    "history":    "Significant past events in this relationship",
    "warning":    "Things to be careful about",
}


@dataclass
class Observation:
    id: str
    contact_id: str
    category: str
    content: str
    confidence: float
    created_at: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def __str__(self) -> str:
        conf = f" [{self.confidence:.0%} confidence]" if self.confidence < 0.9 else ""
        age = f" ({self.age_days:.0f}d ago)" if self.age_days > 7 else ""
        return f"[{self.category}] {self.content}{conf}{age}"


@dataclass
class Contact:
    id: str
    name: str
    contact_type: str       # human | agent | team
    first_seen: float
    last_seen: float
    interaction_count: int
    notes: str
    observations: List[Observation] = field(default_factory=list)

    @property
    def known_since_days(self) -> float:
        return (time.time() - self.first_seen) / 86400

    @property
    def days_since_last(self) -> float:
        return (time.time() - self.last_seen) / 86400

    def by_category(self) -> Dict[str, List[Observation]]:
        result: Dict[str, List[Observation]] = {}
        for obs in self.observations:
            result.setdefault(obs.category, []).append(obs)
        return result

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "contact_type": self.contact_type,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "notes": self.notes,
            "observations": [
                {
                    "category": o.category,
                    "content": o.content,
                    "confidence": o.confidence,
                    "age_days": o.age_days,
                }
                for o in self.observations
            ],
        }


def _row_to_obs(row: sqlite3.Row) -> Observation:
    return Observation(
        id=row["id"],
        contact_id=row["contact_id"],
        category=row["category"],
        content=row["content"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


class RelationshipStore:
    """Persistent store of what the agent knows about who it works with."""

    def __init__(self, db_path: Optional[Path] = None):
        self._conn = connect(db_path)
        init_schema(self._conn, has_fts5(self._conn))

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def know(
        self,
        name: str,
        contact_type: str = "human",
        notes: str = "",
    ) -> Contact:
        """Create or update a contact record. Safe to call repeatedly."""
        now = time.time()
        existing = self._get_by_name(name)
        if existing:
            self._conn.execute(
                "UPDATE contacts SET last_seen=?, notes=CASE WHEN ?!='' THEN ? ELSE notes END WHERE id=?",
                (now, notes, notes, existing["id"]),
            )
            self._conn.commit()
            return self.recall(name)

        contact_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO contacts VALUES (?,?,?,?,?,?,?)",
            (contact_id, name, contact_type, now, now, 0, notes),
        )
        self._conn.commit()
        return self.recall(name)

    def observe(
        self,
        name: str,
        observation: str,
        category: str = "pattern",
        confidence: float = 0.8,
    ) -> Observation:
        """Record an observation about a contact. Creates the contact if needed."""
        if category not in OBSERVATION_CATEGORIES:
            raise ValueError(f"Category must be one of {list(OBSERVATION_CATEGORIES)}")
        contact = self.know(name)
        now = time.time()
        obs = Observation(
            id=str(uuid.uuid4()),
            contact_id=contact.id,
            category=category,
            content=observation,
            confidence=max(0.0, min(1.0, confidence)),
            created_at=now,
        )
        self._conn.execute(
            "INSERT INTO contact_observations VALUES (?,?,?,?,?,?)",
            (obs.id, obs.contact_id, obs.category, obs.content, obs.confidence, obs.created_at),
        )
        self._conn.commit()
        return obs

    def log_interaction(self, name: str, notes: str = "") -> Contact:
        """Increment interaction count and update last_seen. Call at end of each session."""
        contact = self.know(name)
        now = time.time()
        self._conn.execute(
            "UPDATE contacts SET interaction_count=interaction_count+1, last_seen=? WHERE id=?",
            (now, contact.id),
        )
        if notes:
            self._conn.execute(
                "INSERT INTO contact_observations VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), contact.id, "history", notes, 1.0, now),
            )
        self._conn.commit()
        return self.recall(name)

    def remove_observation(self, observation_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM contact_observations WHERE id=?", (observation_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def recall(self, name: str) -> Optional[Contact]:
        """Get everything known about a contact."""
        row = self._conn.execute(
            "SELECT * FROM contacts WHERE name=?", (name,)
        ).fetchone()
        if not row:
            return None
        contact = Contact(
            id=row["id"],
            name=row["name"],
            contact_type=row["contact_type"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            interaction_count=row["interaction_count"],
            notes=row["notes"],
        )
        obs_rows = self._conn.execute(
            "SELECT * FROM contact_observations WHERE contact_id=? ORDER BY created_at DESC",
            (contact.id,),
        ).fetchall()
        contact.observations = [_row_to_obs(r) for r in obs_rows]
        return contact

    def all_contacts(self) -> List[Contact]:
        rows = self._conn.execute(
            "SELECT * FROM contacts ORDER BY last_seen DESC"
        ).fetchall()
        return [self.recall(r["name"]) for r in rows]

    def session_briefing(self, name: str) -> str:
        """Return a structured briefing for working with this contact.

        Designed to be prepended to session context. Agent-readable.
        """
        contact = self.recall(name)
        if not contact:
            return f"No prior knowledge of '{name}'. Observe and record as you work with them."

        lines = [f"=== Working with: {contact.name} [{contact.contact_type}] ===\n"]

        known = f"{contact.known_since_days:.0f} days"
        last = (f"{contact.days_since_last:.0f} days ago"
                if contact.days_since_last > 1 else "recently")
        lines.append(
            f"Known for {known}, {contact.interaction_count} sessions, last seen {last}"
        )
        if contact.notes:
            lines.append(f"Notes: {contact.notes}")
        lines.append("")

        by_cat = contact.by_category()

        # Surface warnings first - they are most operationally critical
        if "warning" in by_cat:
            lines.append("WARNINGS:")
            for obs in by_cat["warning"]:
                lines.append(f"  !! {obs.content}")
            lines.append("")

        # Trust - determines autonomy level
        if "trust" in by_cat:
            lines.append("Trust / autonomy:")
            for obs in by_cat["trust"]:
                lines.append(f"  • {obs.content}")
            lines.append("")

        # Preferences - how to behave
        if "preference" in by_cat:
            lines.append("Preferences:")
            for obs in by_cat["preference"]:
                lines.append(f"  • {obs.content}")
            lines.append("")

        # Current context - what they're dealing with
        if "context" in by_cat:
            lines.append("Current context:")
            # Show only the most recent context observations (context goes stale)
            for obs in by_cat["context"][:3]:
                lines.append(f"  • {obs.content}")
            lines.append("")

        # Patterns - behavioral tendencies
        if "pattern" in by_cat:
            lines.append("Observed patterns:")
            for obs in by_cat["pattern"]:
                lines.append(f"  • {obs.content}")
            lines.append("")

        # Recent history - last few significant events
        if "history" in by_cat:
            recent_history = [o for o in by_cat["history"] if o.age_days < 90][:3]
            if recent_history:
                lines.append("Recent history:")
                for obs in recent_history:
                    lines.append(f"  [{obs.age_days:.0f}d ago] {obs.content}")
                lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_by_name(self, name: str):
        return self._conn.execute(
            "SELECT * FROM contacts WHERE name=?", (name,)
        ).fetchone()
