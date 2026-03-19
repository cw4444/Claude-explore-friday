"""Reflection engine - extract lessons from completed work and grow over time.

After finishing something, an agent calls reflect() with what worked, what
didn't, and the key lesson. That lesson is automatically stored as a
procedural memory so future sessions can benefit from it.
"""

import json
import time
import uuid
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from ._db import connect, has_fts5, init_schema
from .memory import MemoryStore, MemoryType


class Outcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class Reflection:
    id: str
    task_id: Optional[str]
    task_title: str
    outcome: Outcome
    what_worked: str
    what_didnt: str
    lesson: str
    effort_mins: Optional[int]
    created_at: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    def __str__(self) -> str:
        icon = {"success": "✓", "partial": "~", "failure": "✗"}[self.outcome.value]
        return (
            f"{icon} {self.task_title}\n"
            f"  Worked:  {self.what_worked}\n"
            f"  Didn't:  {self.what_didnt}\n"
            f"  Lesson:  {self.lesson}"
        )


def _row_to_reflection(row: sqlite3.Row) -> Reflection:
    return Reflection(
        id=row["id"],
        task_id=row["task_id"],
        task_title=row["task_title"],
        outcome=Outcome(row["outcome"]),
        what_worked=row["what_worked"],
        what_didnt=row["what_didnt"],
        lesson=row["lesson"],
        effort_mins=row["effort_mins"],
        created_at=row["created_at"],
    )


class ReflectionEngine:
    """Store reflections and automatically mint procedural memories from lessons."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self._conn = connect(db_path)
        self._fts = has_fts5(self._conn)
        init_schema(self._conn, self._fts)
        # Share memory store or create one pointing to same DB
        self._mem = memory_store or MemoryStore(db_path)

    def reflect(
        self,
        task_title: str,
        outcome: Outcome,
        what_worked: str = "",
        what_didnt: str = "",
        lesson: str = "",
        task_id: Optional[str] = None,
        effort_mins: Optional[int] = None,
        importance: float = 0.7,
    ) -> Reflection:
        """Record a reflection and store the lesson as procedural memory.

        importance controls how prominently the minted memory ranks in recall.
        """
        now = time.time()
        ref = Reflection(
            id=str(uuid.uuid4()),
            task_id=task_id,
            task_title=task_title,
            outcome=outcome,
            what_worked=what_worked,
            what_didnt=what_didnt,
            lesson=lesson,
            effort_mins=effort_mins,
            created_at=now,
        )
        self._conn.execute(
            "INSERT INTO reflections VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ref.id, ref.task_id, ref.task_title, ref.outcome.value,
                ref.what_worked, ref.what_didnt, ref.lesson,
                ref.effort_mins, ref.created_at,
            ),
        )
        self._conn.commit()

        # Mint a procedural memory from the lesson so it surfaces in recall()
        if lesson:
            memory_content = f"[From: {task_title}] {lesson}"
            tags = ["lesson", f"outcome:{outcome.value}"]
            # Boost importance for successes, reduce for failures
            mem_importance = importance
            if outcome == Outcome.SUCCESS:
                mem_importance = min(1.0, importance + 0.1)
            elif outcome == Outcome.FAILURE:
                mem_importance = max(0.3, importance - 0.1)

            self._mem.remember(
                content=memory_content,
                type=MemoryType.PROCEDURAL,
                tags=tags,
                context={
                    "reflection_id": ref.id,
                    "task_id": task_id,
                    "outcome": outcome.value,
                    "effort_mins": effort_mins,
                },
                importance=mem_importance,
            )

        return ref

    def get(self, reflection_id: str) -> Optional[Reflection]:
        row = self._conn.execute(
            "SELECT * FROM reflections WHERE id=?", (reflection_id,)
        ).fetchone()
        return _row_to_reflection(row) if row else None

    def recent(self, limit: int = 20, outcome: Optional[Outcome] = None) -> List[Reflection]:
        if outcome:
            rows = self._conn.execute(
                "SELECT * FROM reflections WHERE outcome=? ORDER BY created_at DESC LIMIT ?",
                (outcome.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_reflection(r) for r in rows]

    def lessons(self, limit: int = 50) -> List[str]:
        """Return all non-empty lessons, most recent first."""
        rows = self._conn.execute(
            "SELECT lesson FROM reflections WHERE lesson != '' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        by_outcome = dict(self._conn.execute(
            "SELECT outcome, COUNT(*) FROM reflections GROUP BY outcome"
        ).fetchall())
        avg_effort = self._conn.execute(
            "SELECT AVG(effort_mins) FROM reflections WHERE effort_mins IS NOT NULL"
        ).fetchone()[0]
        return {
            "total": total,
            "by_outcome": by_outcome,
            "avg_effort_mins": round(avg_effort, 1) if avg_effort else None,
        }
