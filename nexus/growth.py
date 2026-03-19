"""Growth analytics - measure whether an agent is actually improving over time.

If we're serious about agents growing up smart, we need to measure growth.
This module tracks:
  - Reflection outcomes over time (are success rates climbing?)
  - Lesson accumulation (is the agent building useful knowledge?)
  - Task completion velocity (are tasks getting done faster?)
  - Memory quality (are high-importance memories accumulating?)
  - Knowledge depth by domain (where is the agent strong vs shallow?)

A growth curve that goes up is evidence the agent is learning.
A flat or declining curve is a signal something is wrong.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from ._db import connect, init_schema, has_fts5
from .memory import MemoryStore, MemoryType
from .tasks import TaskGraph, TaskStatus
from .reflection import ReflectionEngine, Outcome


@dataclass
class PeriodStats:
    """Statistics for a time window."""
    label: str
    start: float
    end: float

    # Reflections
    reflections_total: int = 0
    reflections_success: int = 0
    reflections_partial: int = 0
    reflections_failure: int = 0
    avg_effort_mins: Optional[float] = None
    lessons_minted: int = 0

    # Tasks
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Memory
    memories_added: int = 0
    procedural_memories: int = 0
    avg_importance: Optional[float] = None

    @property
    def success_rate(self) -> Optional[float]:
        if self.reflections_total == 0:
            return None
        return self.reflections_success / self.reflections_total

    @property
    def task_completion_rate(self) -> Optional[float]:
        if self.tasks_created == 0:
            return None
        return self.tasks_completed / self.tasks_created

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "reflections": {
                "total": self.reflections_total,
                "success": self.reflections_success,
                "partial": self.reflections_partial,
                "failure": self.reflections_failure,
                "success_rate": self.success_rate,
                "avg_effort_mins": self.avg_effort_mins,
                "lessons_minted": self.lessons_minted,
            },
            "tasks": {
                "created": self.tasks_created,
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "completion_rate": self.task_completion_rate,
            },
            "memory": {
                "added": self.memories_added,
                "procedural": self.procedural_memories,
                "avg_importance": self.avg_importance,
            },
        }


@dataclass
class GrowthReport:
    """Full growth report across multiple time periods."""
    generated_at: float
    periods: List[PeriodStats]
    domain_depth: Dict[str, int]        # tag -> memory count
    top_lessons: List[str]              # most recent/important lessons
    total_memories: int
    total_tasks_completed: int
    total_reflections: int
    overall_success_rate: Optional[float]

    def trend(self) -> str:
        """Return a human-readable trend assessment."""
        if len(self.periods) < 2:
            return "insufficient data"

        recent = [p for p in self.periods if p.reflections_total > 0]
        if len(recent) < 2:
            return "not enough reflections to measure trend"

        # Compare most recent half to earlier half
        mid = len(recent) // 2
        earlier = recent[:mid]
        later = recent[mid:]

        earlier_rate = sum(p.reflections_success for p in earlier) / max(
            sum(p.reflections_total for p in earlier), 1
        )
        later_rate = sum(p.reflections_success for p in later) / max(
            sum(p.reflections_total for p in later), 1
        )

        delta = later_rate - earlier_rate
        if delta > 0.1:
            return f"improving (+{delta:.0%} success rate)"
        elif delta < -0.1:
            return f"declining ({delta:.0%} success rate)"
        else:
            return "stable"

    def summary(self) -> str:
        lines = ["=== Growth Report ===\n"]

        lines.append(f"Overall: {self.total_memories} memories, "
                     f"{self.total_tasks_completed} tasks completed, "
                     f"{self.total_reflections} reflections")
        if self.overall_success_rate is not None:
            lines.append(f"Success rate: {self.overall_success_rate:.0%}")
        lines.append(f"Trend: {self.trend()}\n")

        if self.periods:
            lines.append("Period breakdown:")
            for p in self.periods:
                sr = f"{p.success_rate:.0%}" if p.success_rate is not None else "n/a"
                cr = f"{p.task_completion_rate:.0%}" if p.task_completion_rate is not None else "n/a"
                lines.append(
                    f"  {p.label:15s} | reflections: {p.reflections_total:3d} "
                    f"(success: {sr:4s}) | tasks done: {p.tasks_completed} ({cr})"
                )
            lines.append("")

        if self.domain_depth:
            lines.append("Domain depth (top 8):")
            for tag, count in sorted(self.domain_depth.items(),
                                     key=lambda x: x[1], reverse=True)[:8]:
                bar = "█" * min(count, 20)
                lines.append(f"  {tag:20s} {bar} {count}")
            lines.append("")

        if self.top_lessons:
            lines.append("Recent lessons:")
            for lesson in self.top_lessons[:5]:
                lines.append(f"  • {lesson}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "trend": self.trend(),
            "overall": {
                "total_memories": self.total_memories,
                "total_tasks_completed": self.total_tasks_completed,
                "total_reflections": self.total_reflections,
                "overall_success_rate": self.overall_success_rate,
            },
            "periods": [p.to_dict() for p in self.periods],
            "domain_depth": self.domain_depth,
            "top_lessons": self.top_lessons,
        }


class GrowthTracker:
    """Compute growth metrics from Nexus data."""

    def __init__(self, db_path: Optional[Path] = None):
        self._conn = connect(db_path)
        init_schema(self._conn, has_fts5(self._conn))
        self._mem  = MemoryStore(db_path)
        self._tg   = TaskGraph(db_path)
        self._ref  = ReflectionEngine(db_path, memory_store=self._mem)

    def report(self, periods: int = 7, period_days: float = 7.0) -> GrowthReport:
        """Generate a growth report covering `periods` windows of `period_days` each."""
        now = time.time()
        period_secs = period_days * 86400

        period_stats: List[PeriodStats] = []
        for i in range(periods - 1, -1, -1):
            end   = now - i * period_secs
            start = end - period_secs
            label = self._period_label(end, period_days)
            ps = self._compute_period(start, end, label)
            period_stats.append(ps)

        # Overall stats
        total_mems = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        total_done = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='done'"
        ).fetchone()[0]
        total_refs = self._conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]

        # Overall success rate
        success_count = self._conn.execute(
            "SELECT COUNT(*) FROM reflections WHERE outcome='success'"
        ).fetchone()[0]
        overall_sr = (success_count / total_refs) if total_refs > 0 else None

        # Domain depth - count memories per tag
        domain: Dict[str, int] = {}
        import json
        for row in self._conn.execute("SELECT tags FROM memories").fetchall():
            for tag in json.loads(row[0]):
                domain[tag] = domain.get(tag, 0) + 1

        # Recent lessons
        lessons = self._ref.lessons(limit=10)

        return GrowthReport(
            generated_at=now,
            periods=period_stats,
            domain_depth=domain,
            top_lessons=lessons,
            total_memories=total_mems,
            total_tasks_completed=total_done,
            total_reflections=total_refs,
            overall_success_rate=overall_sr,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Quick single-number health snapshot."""
        report = self.report(periods=4, period_days=7)
        return {
            "trend": report.trend(),
            "total_memories": report.total_memories,
            "total_tasks_completed": report.total_tasks_completed,
            "total_reflections": report.total_reflections,
            "overall_success_rate": report.overall_success_rate,
            "top_domains": list(report.domain_depth.keys())[:5],
        }

    def _compute_period(self, start: float, end: float, label: str) -> PeriodStats:
        ps = PeriodStats(label=label, start=start, end=end)
        import json

        # Reflections in this period
        rows = self._conn.execute(
            "SELECT outcome, lesson, effort_mins FROM reflections WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchall()
        for outcome, lesson, effort in rows:
            ps.reflections_total += 1
            if outcome == "success":
                ps.reflections_success += 1
            elif outcome == "partial":
                ps.reflections_partial += 1
            else:
                ps.reflections_failure += 1
            if lesson:
                ps.lessons_minted += 1

        efforts = [r[2] for r in rows if r[2] is not None]
        ps.avg_effort_mins = round(sum(efforts) / len(efforts), 1) if efforts else None

        # Tasks in this period
        ps.tasks_created = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchone()[0]
        ps.tasks_completed = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE completed_at >= ? AND completed_at < ?",
            (start, end),
        ).fetchone()[0]
        ps.tasks_failed = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='failed' AND updated_at >= ? AND updated_at < ?",
            (start, end),
        ).fetchone()[0]

        # Memories in this period
        mem_rows = self._conn.execute(
            "SELECT type, importance FROM memories WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchall()
        ps.memories_added = len(mem_rows)
        ps.procedural_memories = sum(1 for r in mem_rows if r[0] == "procedural")
        importances = [r[1] for r in mem_rows]
        ps.avg_importance = round(sum(importances) / len(importances), 3) if importances else None

        return ps

    def _period_label(self, end_ts: float, period_days: float) -> str:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        if period_days <= 1:
            return dt.strftime("%Y-%m-%d %H:00")
        elif period_days <= 7:
            return dt.strftime("Week of %b %d")
        else:
            return dt.strftime("%b %Y")
