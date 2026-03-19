"""Task graph management for multi-session autonomous work.

Tasks persist across agent sessions. The dependency graph lets you model
complex workflows where task B can't start until task A is done.
"""

import json
import time
import uuid
import sqlite3
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

from ._db import connect, has_fts5, init_schema


class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    FAILED      = "failed"
    BLOCKED     = "blocked"


class Priority(int, Enum):
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3
    URGENT = 4


@dataclass
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: Priority
    tags: List[str]
    metadata: Dict[str, Any]
    notes: str
    created_at: float
    updated_at: float
    completed_at: Optional[float]
    deps: List[str] = field(default_factory=list)  # IDs of tasks this depends on

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def is_actionable(self) -> bool:
        """True if the task can be worked on (no incomplete deps)."""
        return self.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        return d

    def __str__(self) -> str:
        dep_str = f" (needs: {len(self.deps)} deps)" if self.deps else ""
        return f"[{self.status.value}] [{self.priority.name}] {self.title}{dep_str}"


def _row_to_task(row: sqlite3.Row, deps: List[str]) -> Task:
    return Task(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        priority=Priority(row["priority"]),
        tags=json.loads(row["tags"]),
        metadata=json.loads(row["metadata"]),
        notes=row["notes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        deps=deps,
    )


class TaskGraph:
    """Persistent task graph with dependency tracking."""

    def __init__(self, db_path: Optional[Path] = None):
        self._conn = connect(db_path)
        self._fts = has_fts5(self._conn)
        init_schema(self._conn, self._fts)

    # ------------------------------------------------------------------
    # Creating tasks
    # ------------------------------------------------------------------

    def create(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        deps: Optional[List[str]] = None,
    ) -> Task:
        """Create a new task and return it."""
        now = time.time()
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
            notes="",
            created_at=now,
            updated_at=now,
            completed_at=None,
            deps=deps or [],
        )
        self._conn.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                task.id, task.title, task.description, task.status.value,
                task.priority.value, json.dumps(task.tags),
                json.dumps(task.metadata), task.notes,
                task.created_at, task.updated_at, task.completed_at,
            ),
        )
        for dep_id in task.deps:
            self._conn.execute(
                "INSERT OR IGNORE INTO task_deps VALUES (?,?)", (task.id, dep_id)
            )
        if self._fts:
            self._conn.execute(
                "INSERT INTO tasks_fts(id,title,description,tags) VALUES(?,?,?,?)",
                (task.id, task.title, task.description, " ".join(task.tags)),
            )
        self._conn.commit()
        return task

    def decompose(
        self,
        parent_title: str,
        subtasks: List[str],
        chain: bool = True,
        **parent_kwargs,
    ) -> List[Task]:
        """Create a parent task and sequential subtasks.

        If chain=True, each subtask depends on the previous one.
        Returns [parent, sub1, sub2, ...].
        """
        parent = self.create(parent_title, **parent_kwargs)
        created: List[Task] = [parent]
        prev_id: Optional[str] = None
        for title in subtasks:
            deps = [prev_id] if (chain and prev_id) else []
            task = self.create(title, deps=deps, tags=parent.tags)
            created.append(task)
            prev_id = task.id
        return created

    # ------------------------------------------------------------------
    # Updating tasks
    # ------------------------------------------------------------------

    def start(self, task_id: str, notes: str = "") -> Task:
        return self._set_status(task_id, TaskStatus.IN_PROGRESS, notes)

    def complete(self, task_id: str, notes: str = "") -> Task:
        now = time.time()
        self._conn.execute(
            "UPDATE tasks SET status=?, completed_at=?, updated_at=?, notes=? WHERE id=?",
            (TaskStatus.DONE.value, now, now, notes, task_id),
        )
        self._conn.commit()
        return self.get(task_id)

    def fail(self, task_id: str, notes: str = "") -> Task:
        return self._set_status(task_id, TaskStatus.FAILED, notes)

    def block(self, task_id: str, notes: str = "") -> Task:
        return self._set_status(task_id, TaskStatus.BLOCKED, notes)

    def reopen(self, task_id: str, notes: str = "") -> Task:
        return self._set_status(task_id, TaskStatus.PENDING, notes)

    def add_note(self, task_id: str, note: str) -> Task:
        task = self.get(task_id)
        if not task:
            raise KeyError(f"Task {task_id} not found")
        new_notes = f"{task.notes}\n{note}".strip() if task.notes else note
        self._conn.execute(
            "UPDATE tasks SET notes=?, updated_at=? WHERE id=?",
            (new_notes, time.time(), task_id),
        )
        self._conn.commit()
        return self.get(task_id)

    def add_dep(self, task_id: str, dep_id: str) -> None:
        if self._would_create_cycle(task_id, dep_id):
            raise ValueError(f"Adding dep {dep_id} -> {task_id} would create a cycle")
        self._conn.execute(
            "INSERT OR IGNORE INTO task_deps VALUES (?,?)", (task_id, dep_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def resolve_id(self, id_prefix: str) -> Optional[str]:
        """Resolve a full task ID from a prefix (as short as 4 chars)."""
        rows = self._conn.execute(
            "SELECT id FROM tasks WHERE id LIKE ?", (f"{id_prefix}%",)
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            raise ValueError(f"Ambiguous ID prefix '{id_prefix}' matches {len(rows)} tasks")
        return None

    def get(self, task_id: str) -> Optional[Task]:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if not row:
            return None
        deps = [r[0] for r in self._conn.execute(
            "SELECT dep_id FROM task_deps WHERE task_id=?", (task_id,)
        ).fetchall()]
        return _row_to_task(row, deps)

    def find(
        self,
        query: str = "",
        status: Optional[TaskStatus] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Task]:
        tasks: List[Task]
        if query and self._fts:
            tasks = self._fts_search(query, limit * 2)
        elif query:
            tasks = self._like_search(query, limit * 2)
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, created_at DESC LIMIT ?",
                (limit * 2,),
            ).fetchall()
            tasks = [self._load_task(r) for r in rows]

        if status:
            tasks = [t for t in tasks if t.status == status]
        if tags:
            tasks = [t for t in tasks if any(tg in t.tags for tg in tags)]
        return tasks[:limit]

    def pending(self) -> List[Task]:
        """All actionable tasks (pending/in_progress), ordered by priority."""
        rows = self._conn.execute(
            """SELECT * FROM tasks
               WHERE status IN ('pending','in_progress','blocked')
               ORDER BY priority DESC, created_at ASC"""
        ).fetchall()
        tasks = [self._load_task(r) for r in rows]

        # Compute which tasks are actually blocked by incomplete deps
        done_ids = self._done_ids()
        for task in tasks:
            if task.status != TaskStatus.IN_PROGRESS and task.deps:
                if not all(d in done_ids for d in task.deps):
                    task.status = TaskStatus.BLOCKED

        return tasks

    def next(self) -> Optional[Task]:
        """Return the highest-priority task that's ready to work on."""
        done_ids = self._done_ids()
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY priority DESC, created_at ASC"
        ).fetchall()
        for row in rows:
            task = self._load_task(row)
            if all(d in done_ids for d in task.deps):
                return task
        return None

    def stats(self) -> Dict[str, Any]:
        counts = dict(self._conn.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status"
        ).fetchall())
        total = sum(counts.values())
        return {"total": total, "by_status": counts}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, task_id: str, status: TaskStatus, notes: str) -> Task:
        extra = ", notes=?" if notes else ""
        params = [status.value, time.time()]
        if notes:
            params.append(notes)
        params.append(task_id)
        self._conn.execute(
            f"UPDATE tasks SET status=?, updated_at=?{extra} WHERE id=?", params
        )
        self._conn.commit()
        return self.get(task_id)

    def _done_ids(self) -> Set[str]:
        rows = self._conn.execute(
            "SELECT id FROM tasks WHERE status='done'"
        ).fetchall()
        return {r[0] for r in rows}

    def _load_task(self, row: sqlite3.Row) -> Task:
        deps = [r[0] for r in self._conn.execute(
            "SELECT dep_id FROM task_deps WHERE task_id=?", (row["id"],)
        ).fetchall()]
        return _row_to_task(row, deps)

    def _would_create_cycle(self, task_id: str, dep_id: str) -> bool:
        """Check if adding dep_id as a dependency of task_id creates a cycle."""
        visited: Set[str] = set()
        queue = [dep_id]
        while queue:
            node = queue.pop()
            if node == task_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            children = [r[0] for r in self._conn.execute(
                "SELECT dep_id FROM task_deps WHERE task_id=?", (node,)
            ).fetchall()]
            queue.extend(children)
        return False

    def _fts_search(self, query: str, limit: int) -> List[Task]:
        words = [w.strip('"\'') for w in query.split() if w.strip('"\'')]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                """SELECT t.* FROM tasks t
                   JOIN tasks_fts fts ON t.id = fts.id
                   WHERE tasks_fts MATCH ?
                   ORDER BY t.priority DESC, t.created_at DESC
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
            return [self._load_task(r) for r in rows]
        except sqlite3.OperationalError:
            return self._like_search(query, limit)

    def _like_search(self, query: str, limit: int) -> List[Task]:
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """SELECT * FROM tasks
               WHERE title LIKE ? OR description LIKE ?
               ORDER BY priority DESC, created_at DESC
               LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [self._load_task(r) for r in rows]
