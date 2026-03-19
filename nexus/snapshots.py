"""Agent state snapshots - time travel for stateless agents.

When something goes wrong - a bad import, a corrupted memory set, a handoff
that injected wrong tasks, a session that stored incorrect procedural knowledge -
an agent needs to be able to say "revert to before that happened."

Snapshots are complete state captures: memories, tasks, reflections, contacts,
pinned traits. Restore replaces the current state entirely with the snapshot.
A pre-restore snapshot is always created first, so the revert itself is
reversible.

Auto-snapshots happen before bulk operations that modify state significantly:
  - Before importing a knowledge bundle  (trigger: 'pre-import')
  - Before applying an agent handoff     (trigger: 'pre-handoff')
  - Before pulling GitHub issues         (trigger: 'pre-github-sync')

Manual snapshots: nexus snapshot create "before major refactor"

Diff: nexus snapshot diff <id> shows what changed since that point.

Storage: ~/.nexus/snapshots/ (or db_path.parent/snapshots/ for custom DBs)
Each snapshot is a JSON file ~= the size of the database content.
Run nexus snapshot prune to remove old ones.

Usage:
    sm = SnapshotManager(db_path=path)

    # Create a checkpoint before something risky
    snap = sm.create("before applying handoff from agent-x")

    # Something went wrong - restore
    sm.restore(snap.meta.id)

    # See what changed since a snapshot
    diff = sm.diff(snap.meta.id)
    print(diff.summary())

    # List available snapshots
    for meta in sm.list():
        print(meta.label, meta.created_at)
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from ._db import connect, init_schema, has_fts5


@dataclass
class SnapshotMeta:
    id: str
    label: str
    created_at: float
    trigger: str          # 'manual' | 'pre-import' | 'pre-handoff' | 'pre-github-sync' | 'pre-restore'
    memory_count: int
    task_count: int
    reflection_count: int
    contact_count: int

    @property
    def age_mins(self) -> float:
        return (time.time() - self.created_at) / 60

    def display(self) -> str:
        age = self.age_mins
        if age < 60:
            age_str = f"{age:.0f}m ago"
        elif age < 1440:
            age_str = f"{age/60:.1f}h ago"
        else:
            age_str = f"{age/1440:.1f}d ago"
        return (f"[{self.id[:8]}] {age_str}  [{self.trigger}]  "
                f"mem:{self.memory_count} tasks:{self.task_count} "
                f"reflect:{self.reflection_count}  {self.label}")


@dataclass
class Snapshot:
    meta: SnapshotMeta
    memories: List[dict]
    tasks: List[dict]
    task_deps: List[dict]          # list of {task_id, dep_id}
    reflections: List[dict]
    contacts: List[dict]
    contact_observations: List[dict]
    pinned_traits: List[dict]

    def to_dict(self) -> dict:
        return {
            "meta": {
                "id": self.meta.id,
                "label": self.meta.label,
                "created_at": self.meta.created_at,
                "trigger": self.meta.trigger,
                "memory_count": self.meta.memory_count,
                "task_count": self.meta.task_count,
                "reflection_count": self.meta.reflection_count,
                "contact_count": self.meta.contact_count,
            },
            "memories": self.memories,
            "tasks": self.tasks,
            "task_deps": self.task_deps,
            "reflections": self.reflections,
            "contacts": self.contacts,
            "contact_observations": self.contact_observations,
            "pinned_traits": self.pinned_traits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        m = d["meta"]
        return cls(
            meta=SnapshotMeta(
                id=m["id"],
                label=m.get("label", ""),
                created_at=m["created_at"],
                trigger=m.get("trigger", "manual"),
                memory_count=m.get("memory_count", 0),
                task_count=m.get("task_count", 0),
                reflection_count=m.get("reflection_count", 0),
                contact_count=m.get("contact_count", 0),
            ),
            memories=d.get("memories", []),
            tasks=d.get("tasks", []),
            task_deps=d.get("task_deps", []),
            reflections=d.get("reflections", []),
            contacts=d.get("contacts", []),
            contact_observations=d.get("contact_observations", []),
            pinned_traits=d.get("pinned_traits", []),
        )


@dataclass
class SnapshotDiff:
    """What changed between a snapshot and the current state."""
    snapshot_id: str
    snapshot_label: str
    snapshot_age_mins: float

    memories_added: int
    memories_removed: int
    added_memory_samples: List[str]    # content preview of new memories
    removed_memory_samples: List[str]

    tasks_added: int
    tasks_removed: int
    task_status_changes: List[str]     # "title: pending → done"

    reflections_added: int

    def summary(self) -> str:
        lines = [
            f"=== Diff since snapshot [{self.snapshot_id[:8]}] "
            f"({self.snapshot_age_mins:.0f}m ago) ===",
            f'"{self.snapshot_label}"' if self.snapshot_label else "",
            "",
        ]
        if self.memories_added or self.memories_removed:
            lines.append(f"Memories: +{self.memories_added} added, -{self.memories_removed} removed")
            for s in self.added_memory_samples[:5]:
                lines.append(f"  + {s}")
            for s in self.removed_memory_samples[:5]:
                lines.append(f"  - {s}")
        else:
            lines.append("Memories: no change")

        if self.tasks_added or self.tasks_removed or self.task_status_changes:
            lines.append(f"Tasks: +{self.tasks_added} added, -{self.tasks_removed} removed")
            for s in self.task_status_changes[:5]:
                lines.append(f"  ~ {s}")
        else:
            lines.append("Tasks: no change")

        if self.reflections_added:
            lines.append(f"Reflections: +{self.reflections_added} added")
        else:
            lines.append("Reflections: no change")

        return "\n".join(l for l in lines if l is not None)


class SnapshotManager:
    """Create, restore, list, and diff agent state snapshots."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        snapshot_dir: Optional[Path] = None,
    ):
        self._db_path = db_path
        if snapshot_dir:
            self._dir = Path(snapshot_dir)
        elif db_path:
            self._dir = Path(db_path).parent / "snapshots"
        else:
            from ._db import _DEFAULT_DIR
            self._dir = _DEFAULT_DIR / "snapshots"

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self, label: str = "", trigger: str = "manual") -> Snapshot:
        """Capture the current state as a named snapshot.

        trigger values: 'manual', 'pre-import', 'pre-handoff',
                        'pre-github-sync', 'pre-restore'
        """
        conn = connect(self._db_path)
        init_schema(conn, fts=has_fts5(conn))
        try:
            memories = [dict(r) for r in conn.execute(
                "SELECT * FROM memories ORDER BY created_at"
            ).fetchall()]

            tasks = [dict(r) for r in conn.execute(
                "SELECT * FROM tasks ORDER BY created_at"
            ).fetchall()]

            task_deps = [dict(r) for r in conn.execute(
                "SELECT task_id, dep_id FROM task_deps"
            ).fetchall()]

            reflections = [dict(r) for r in conn.execute(
                "SELECT * FROM reflections ORDER BY created_at"
            ).fetchall()]

            contacts = [dict(r) for r in conn.execute(
                "SELECT * FROM contacts ORDER BY first_seen"
            ).fetchall()]

            observations = [dict(r) for r in conn.execute(
                "SELECT * FROM contact_observations ORDER BY created_at"
            ).fetchall()]

            traits = [dict(r) for r in conn.execute(
                "SELECT * FROM pinned_traits ORDER BY created_at"
            ).fetchall()]
        finally:
            conn.close()

        meta = SnapshotMeta(
            id=str(uuid.uuid4()),
            label=label,
            created_at=time.time(),
            trigger=trigger,
            memory_count=len(memories),
            task_count=len(tasks),
            reflection_count=len(reflections),
            contact_count=len(contacts),
        )
        snap = Snapshot(
            meta=meta,
            memories=memories,
            tasks=tasks,
            task_deps=task_deps,
            reflections=reflections,
            contacts=contacts,
            contact_observations=observations,
            pinned_traits=traits,
        )
        self._save(snap)
        return snap

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(
        self,
        snapshot_id: str,
        create_pre_restore_snapshot: bool = True,
    ) -> Snapshot:
        """Replace current state with a snapshot.

        By default, creates a pre-restore snapshot first so the revert
        is itself reversible. The pre-restore snapshot is labelled
        'pre-restore (reverting to <snapshot_id[:8]>)'.
        """
        snap = self.get(snapshot_id)

        if create_pre_restore_snapshot:
            self.create(
                label=f"pre-restore (reverting to {snapshot_id[:8]})",
                trigger="pre-restore",
            )

        conn = connect(self._db_path)
        try:
            # Wipe all tables. task_deps must go before tasks (FK constraint).
            conn.executescript("""
                DELETE FROM contact_observations;
                DELETE FROM contacts;
                DELETE FROM task_deps;
                DELETE FROM tasks;
                DELETE FROM memories;
                DELETE FROM reflections;
                DELETE FROM pinned_traits;
            """)

            # Re-insert memories
            for m in snap.memories:
                conn.execute(
                    "INSERT OR REPLACE INTO memories "
                    "(id, content, type, tags, context, created_at, "
                    " accessed_at, access_count, importance) "
                    "VALUES (:id, :content, :type, :tags, :context, "
                    ":created_at, :accessed_at, :access_count, :importance)",
                    m,
                )

            # Re-insert tasks (no deps yet - FK would fail)
            for t in snap.tasks:
                conn.execute(
                    "INSERT OR REPLACE INTO tasks "
                    "(id, title, description, status, priority, tags, "
                    " metadata, notes, created_at, updated_at, completed_at) "
                    "VALUES (:id, :title, :description, :status, :priority, "
                    ":tags, :metadata, :notes, :created_at, :updated_at, :completed_at)",
                    t,
                )

            # Re-insert deps
            for dep in snap.task_deps:
                conn.execute(
                    "INSERT OR REPLACE INTO task_deps (task_id, dep_id) "
                    "VALUES (:task_id, :dep_id)",
                    dep,
                )

            # Re-insert reflections
            for r in snap.reflections:
                conn.execute(
                    "INSERT OR REPLACE INTO reflections "
                    "(id, task_id, task_title, outcome, what_worked, "
                    " what_didnt, lesson, effort_mins, created_at) "
                    "VALUES (:id, :task_id, :task_title, :outcome, :what_worked, "
                    ":what_didnt, :lesson, :effort_mins, :created_at)",
                    r,
                )

            # Re-insert contacts
            for c in snap.contacts:
                conn.execute(
                    "INSERT OR REPLACE INTO contacts "
                    "(id, name, contact_type, first_seen, last_seen, "
                    " interaction_count, notes) "
                    "VALUES (:id, :name, :contact_type, :first_seen, :last_seen, "
                    ":interaction_count, :notes)",
                    c,
                )

            # Re-insert observations
            for o in snap.contact_observations:
                conn.execute(
                    "INSERT OR REPLACE INTO contact_observations "
                    "(id, contact_id, category, content, confidence, created_at) "
                    "VALUES (:id, :contact_id, :category, :content, :confidence, :created_at)",
                    o,
                )

            # Re-insert traits
            for tr in snap.pinned_traits:
                conn.execute(
                    "INSERT OR REPLACE INTO pinned_traits "
                    "(id, name, description, source, created_at) "
                    "VALUES (:id, :name, :description, :source, :created_at)",
                    tr,
                )

            conn.commit()

            # Rebuild FTS indexes if present
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                conn.execute("INSERT INTO tasks_fts(tasks_fts) VALUES('rebuild')")
                conn.commit()
            except Exception:
                pass  # FTS not available - that's fine

        finally:
            conn.close()

        return snap

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list(self, limit: int = 20) -> List[SnapshotMeta]:
        """List snapshots newest-first."""
        if not self._dir.is_dir():
            return []
        metas = []
        for fp in self._dir.glob("snapshot-*.json"):
            try:
                d = json.loads(fp.read_text())
                m = d["meta"]
                metas.append(SnapshotMeta(
                    id=m["id"],
                    label=m.get("label", ""),
                    created_at=m["created_at"],
                    trigger=m.get("trigger", "manual"),
                    memory_count=m.get("memory_count", 0),
                    task_count=m.get("task_count", 0),
                    reflection_count=m.get("reflection_count", 0),
                    contact_count=m.get("contact_count", 0),
                ))
            except Exception:
                continue
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas[:limit]

    def get(self, snapshot_id: str) -> Snapshot:
        """Load a snapshot by ID or 8-char prefix."""
        path = self._find_path(snapshot_id)
        return Snapshot.from_dict(json.loads(path.read_text()))

    def diff(self, snapshot_id: str) -> SnapshotDiff:
        """Show what changed between a snapshot and current state."""
        snap = self.get(snapshot_id)
        conn = connect(self._db_path)
        try:
            current_mems = {r["id"]: dict(r) for r in
                            conn.execute("SELECT id, content FROM memories").fetchall()}
            current_tasks = {r["id"]: dict(r) for r in
                             conn.execute("SELECT id, title, status FROM tasks").fetchall()}
            current_refs = {r["id"] for r in
                            conn.execute("SELECT id FROM reflections").fetchall()}
        finally:
            conn.close()

        snap_mem_ids = {m["id"] for m in snap.memories}
        snap_mems = {m["id"]: m for m in snap.memories}

        added_ids = set(current_mems) - snap_mem_ids
        removed_ids = snap_mem_ids - set(current_mems)

        snap_task_ids = {t["id"] for t in snap.tasks}
        snap_tasks = {t["id"]: t for t in snap.tasks}

        tasks_added = set(current_tasks) - snap_task_ids
        tasks_removed = snap_task_ids - set(current_tasks)
        status_changes = []
        for tid, cur in current_tasks.items():
            if tid in snap_tasks:
                old_status = snap_tasks[tid]["status"]
                if old_status != cur["status"]:
                    status_changes.append(
                        f"{cur['title'][:50]}: {old_status} → {cur['status']}"
                    )

        snap_ref_ids = {r["id"] for r in snap.reflections}
        refs_added = current_refs - snap_ref_ids

        return SnapshotDiff(
            snapshot_id=snap.meta.id,
            snapshot_label=snap.meta.label,
            snapshot_age_mins=snap.meta.age_mins,
            memories_added=len(added_ids),
            memories_removed=len(removed_ids),
            added_memory_samples=[
                current_mems[i]["content"][:80] for i in list(added_ids)[:5]
            ],
            removed_memory_samples=[
                snap_mems[i]["content"][:80] for i in list(removed_ids)[:5]
            ],
            tasks_added=len(tasks_added),
            tasks_removed=len(tasks_removed),
            task_status_changes=status_changes,
            reflections_added=len(refs_added),
        )

    def prune(self, keep: int = 20, keep_manual: bool = True) -> int:
        """Delete old snapshots, keeping the N most recent.

        keep_manual=True preserves manually labelled snapshots regardless of age.
        Returns number of snapshots deleted.
        """
        all_metas = self.list(limit=10000)
        if len(all_metas) <= keep:
            return 0

        to_delete = []
        kept = 0
        for meta in all_metas:  # newest-first
            if kept < keep:
                kept += 1
                continue
            if keep_manual and meta.trigger == "manual" and meta.label:
                continue
            to_delete.append(meta)

        for meta in to_delete:
            try:
                self._find_path(meta.id).unlink()
            except Exception:
                pass
        return len(to_delete)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self, snap: Snapshot) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"snapshot-{snap.meta.id}.json"
        path.write_text(json.dumps(snap.to_dict(), indent=2))
        return path

    def _find_path(self, snapshot_id: str) -> Path:
        """Resolve full ID or 8-char prefix to a snapshot file path."""
        if len(snapshot_id) >= 32:
            path = self._dir / f"snapshot-{snapshot_id}.json"
            if path.exists():
                return path
        # Prefix search
        if self._dir.is_dir():
            for fp in self._dir.glob(f"snapshot-{snapshot_id}*.json"):
                return fp
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
