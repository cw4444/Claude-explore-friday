"""Context packager - bootstrap a new agent session with relevant state.

The biggest UX win for a stateless agent: at the start of a session, call
context_for(current_task) and get back a structured briefing containing:
  - Relevant memories from past sessions
  - In-progress tasks
  - Recent lessons learned
  - Any explicitly pinned knowledge

This turns a cold-start into a warm-start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from .memory import MemoryStore, Memory, MemoryType
from .tasks import TaskGraph, Task, TaskStatus
from .reflection import ReflectionEngine


@dataclass
class SessionContext:
    """A structured briefing for starting a new agent session."""

    task_description: str
    relevant_memories: List[Memory] = field(default_factory=list)
    in_progress_tasks: List[Task] = field(default_factory=list)
    next_task: Optional[Task] = None
    recent_lessons: List[str] = field(default_factory=list)
    pinned: List[Memory] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def summary(self, verbose: bool = False) -> str:
        """Return a human-readable (and agent-readable) briefing string."""
        parts: List[str] = []

        parts.append(f"=== Session Context: {self.task_description} ===\n")

        if self.pinned:
            parts.append("[ Pinned Knowledge ]")
            for m in self.pinned:
                parts.append(f"  • {m.content}")
            parts.append("")

        if self.next_task:
            parts.append("[ Recommended Next Task ]")
            parts.append(f"  {self.next_task}")
            if self.next_task.description:
                parts.append(f"  {self.next_task.description}")
            if self.next_task.notes:
                parts.append(f"  Notes: {self.next_task.notes}")
            parts.append("")

        if self.in_progress_tasks:
            parts.append("[ In Progress ]")
            for t in self.in_progress_tasks:
                parts.append(f"  {t}")
            parts.append("")

        if self.recent_lessons:
            parts.append("[ Recent Lessons ]")
            for lesson in self.recent_lessons[:5]:
                parts.append(f"  • {lesson}")
            parts.append("")

        if self.relevant_memories:
            parts.append("[ Relevant Memories ]")
            for m in self.relevant_memories:
                parts.append(f"  [{m.type.value}] {m.content}")
                if verbose and m.tags:
                    parts.append(f"    tags: {', '.join(m.tags)}")
            parts.append("")

        if not any([self.pinned, self.next_task, self.in_progress_tasks,
                    self.recent_lessons, self.relevant_memories]):
            parts.append("  (No prior context found - this looks like a fresh start.)")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "task_description": self.task_description,
            "relevant_memories": [m.to_dict() for m in self.relevant_memories],
            "in_progress_tasks": [t.to_dict() for t in self.in_progress_tasks],
            "next_task": self.next_task.to_dict() if self.next_task else None,
            "recent_lessons": self.recent_lessons,
            "pinned": [m.to_dict() for m in self.pinned],
            "generated_at": self.generated_at,
        }


class ContextManager:
    """Package agent context for session bootstrap."""

    def __init__(self, db_path: Optional[Path] = None):
        self._mem   = MemoryStore(db_path)
        self._tasks = TaskGraph(db_path)
        self._ref   = ReflectionEngine(db_path, memory_store=self._mem)

    @property
    def memory(self) -> MemoryStore:
        return self._mem

    @property
    def tasks(self) -> TaskGraph:
        return self._tasks

    @property
    def reflection(self) -> ReflectionEngine:
        return self._ref

    def prepare(
        self,
        task_description: str,
        memory_limit: int = 8,
        lesson_limit: int = 5,
        include_procedural: bool = True,
    ) -> SessionContext:
        """Build a SessionContext relevant to task_description.

        Call this at the start of every agent session.
        """
        # Memories most relevant to the current task
        relevant = self._mem.recall(task_description, limit=memory_limit)

        # Surface high-importance memories: procedural + explicitly pinned semantic
        pinned: List[Memory] = []
        if include_procedural:
            candidates = self._mem.all(limit=50)
            relevant_ids = {m.id for m in relevant}
            pinned = [
                m for m in candidates
                if m.id not in relevant_ids
                and m.importance >= 0.75
                and (m.type == MemoryType.PROCEDURAL or "pinned" in m.tags)
            ][:3]

        # In-progress tasks
        in_progress = [
            t for t in self._tasks.pending()
            if t.status == TaskStatus.IN_PROGRESS
        ]

        # Next ready task
        next_task = self._tasks.next()

        # Recent lessons
        lessons = self._ref.lessons(limit=lesson_limit)

        return SessionContext(
            task_description=task_description,
            relevant_memories=relevant,
            in_progress_tasks=in_progress,
            next_task=next_task,
            recent_lessons=lessons,
            pinned=pinned,
        )

    def pin(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a high-importance semantic fact that always surfaces in context."""
        return self._mem.remember(
            content=content,
            type=MemoryType.SEMANTIC,
            tags=(tags or []) + ["pinned"],
            context=context or {},
            importance=0.9,
        )
