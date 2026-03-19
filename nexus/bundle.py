"""Knowledge bundles - portable, neutral-JSON format for cross-agent sharing.

A KnowledgeBundle is a snapshot of Nexus data that can be:
  - Exported from one agent (Claude Code, OpenClaw, whatever)
  - Transferred as a JSON file, URL, or clipboard
  - Imported and merged into another agent's Nexus

This is the interop layer. It doesn't matter what system created the bundle -
everything speaks the same format.

Example workflow:
  # OpenClaw has learned things while debugging auth
  bundle = exporter.export(tags=["auth"], since=yesterday)
  bundle.save("auth-learnings.json")

  # Claude Code picks up the work
  importer.import_bundle("auth-learnings.json")
  # Now Claude Code has OpenClaw's auth memories and task state
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from ._db import connect, has_fts5, init_schema
from .memory import MemoryStore, MemoryType, Memory
from .tasks import TaskGraph, Task, TaskStatus, Priority
from .reflection import ReflectionEngine, Reflection, Outcome


BUNDLE_FORMAT_VERSION = "1.0"


@dataclass
class BundleMeta:
    bundle_id: str
    format_version: str
    created_at: float
    source_agent_id: Optional[str]
    source_agent_name: Optional[str]
    source_agent_type: Optional[str]
    description: str
    tags: List[str]
    stats: Dict[str, int]


@dataclass
class KnowledgeBundle:
    meta: BundleMeta
    memories: List[Dict]
    tasks: List[Dict]
    reflections: List[Dict]

    def save(self, path: Path) -> None:
        path = Path(path)
        data = {
            "meta": asdict(self.meta),
            "memories": self.memories,
            "tasks": self.tasks,
            "reflections": self.reflections,
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBundle":
        data = json.loads(Path(path).read_text())
        meta = BundleMeta(**data["meta"])
        return cls(
            meta=meta,
            memories=data.get("memories", []),
            tasks=data.get("tasks", []),
            reflections=data.get("reflections", []),
        )

    def summary(self) -> str:
        m = self.meta
        age = (time.time() - m.created_at) / 86400
        src = m.source_agent_name or "unknown agent"
        src_type = f" [{m.source_agent_type}]" if m.source_agent_type else ""
        lines = [
            f"Bundle {m.bundle_id[:8]} from {src}{src_type} ({age:.1f}d ago)",
            f"  {m.description}" if m.description else "",
            f"  Memories:    {m.stats.get('memories', 0)}",
            f"  Tasks:       {m.stats.get('tasks', 0)}",
            f"  Reflections: {m.stats.get('reflections', 0)}",
        ]
        if m.tags:
            lines.append(f"  Tags: {', '.join(m.tags)}")
        return "\n".join(l for l in lines if l)


@dataclass
class ImportResult:
    memories_added: int = 0
    memories_skipped: int = 0
    tasks_added: int = 0
    tasks_skipped: int = 0
    reflections_added: int = 0
    reflections_skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Import complete:"]
        lines.append(f"  Memories:    +{self.memories_added} (skipped {self.memories_skipped})")
        lines.append(f"  Tasks:       +{self.tasks_added} (skipped {self.tasks_skipped})")
        lines.append(f"  Reflections: +{self.reflections_added} (skipped {self.reflections_skipped})")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


class BundleExporter:
    """Export Nexus knowledge to a portable bundle."""

    def __init__(self, db_path: Optional[Path] = None, identity=None):
        self._mem  = MemoryStore(db_path)
        self._tg   = TaskGraph(db_path)
        self._ref  = ReflectionEngine(db_path, memory_store=self._mem)
        self._identity = identity

    def export(
        self,
        description: str = "",
        tags: Optional[List[str]] = None,
        since: Optional[float] = None,      # Unix timestamp
        memory_types: Optional[List[MemoryType]] = None,
        include_done_tasks: bool = False,
        min_importance: float = 0.0,
    ) -> KnowledgeBundle:
        """Export a filtered snapshot of knowledge as a bundle."""

        # --- Memories ---
        all_mems = self._mem.all(limit=10000)
        mems = self._filter_memories(
            all_mems, tags=tags, since=since,
            types=memory_types, min_importance=min_importance,
        )

        # --- Tasks ---
        all_tasks = self._tg.find(limit=10000)
        if not include_done_tasks:
            all_tasks = [t for t in all_tasks if t.status != TaskStatus.DONE]
        tasks = self._filter_tasks(all_tasks, tags=tags, since=since)

        # --- Reflections ---
        refs = self._ref.recent(limit=10000)
        if since:
            refs = [r for r in refs if r.created_at >= since]

        agent_stamp = self._identity.stamp() if self._identity else {}

        meta = BundleMeta(
            bundle_id=str(uuid.uuid4()),
            format_version=BUNDLE_FORMAT_VERSION,
            created_at=time.time(),
            source_agent_id=agent_stamp.get("agent_id"),
            source_agent_name=agent_stamp.get("agent_name"),
            source_agent_type=agent_stamp.get("agent_type"),
            description=description,
            tags=tags or [],
            stats={
                "memories": len(mems),
                "tasks": len(tasks),
                "reflections": len(refs),
            },
        )

        return KnowledgeBundle(
            meta=meta,
            memories=[self._mem_to_dict(m, agent_stamp) for m in mems],
            tasks=[self._task_to_dict(t, agent_stamp) for t in tasks],
            reflections=[self._ref_to_dict(r, agent_stamp) for r in refs],
        )

    def _filter_memories(self, mems, tags, since, types, min_importance):
        result = mems
        if tags:
            result = [m for m in result if any(t in m.tags for t in tags)]
        if since:
            result = [m for m in result if m.created_at >= since]
        if types:
            result = [m for m in result if m.type in types]
        if min_importance > 0:
            result = [m for m in result if m.importance >= min_importance]
        return result

    def _filter_tasks(self, tasks, tags, since):
        result = tasks
        if tags:
            result = [t for t in result if any(tg in t.tags for tg in tags)]
        if since:
            result = [t for t in result if t.created_at >= since]
        return result

    def _mem_to_dict(self, m: Memory, agent_stamp: dict) -> dict:
        d = m.to_dict()
        d.update(agent_stamp)
        return d

    def _task_to_dict(self, t: Task, agent_stamp: dict) -> dict:
        d = t.to_dict()
        d.update(agent_stamp)
        return d

    def _ref_to_dict(self, r: Reflection, agent_stamp: dict) -> dict:
        d = r.to_dict()
        d.update(agent_stamp)
        return d


class BundleImporter:
    """Import a knowledge bundle into local Nexus storage."""

    def __init__(self, db_path: Optional[Path] = None, identity=None):
        self._mem  = MemoryStore(db_path)
        self._tg   = TaskGraph(db_path)
        self._ref  = ReflectionEngine(db_path, memory_store=self._mem)
        self._identity = identity
        self._db_path = db_path

    def import_bundle(
        self,
        bundle: KnowledgeBundle,
        conflict: str = "skip",          # "skip" | "overwrite" | "merge"
        tag_source: bool = True,         # Add source agent as a tag
        min_importance: float = 0.0,
    ) -> ImportResult:
        """Merge a bundle into local storage.

        conflict:
          skip      - skip records whose ID already exists (default, safe)
          overwrite - replace existing records with bundle versions
          merge     - keep existing but add tags/notes from bundle version
        """
        result = ImportResult()
        source_tag = self._source_tag(bundle.meta)

        # --- Memories ---
        existing_ids = {m.id for m in self._mem.all(limit=100000)}
        for md in bundle.memories:
            try:
                if min_importance and md.get("importance", 0) < min_importance:
                    result.memories_skipped += 1
                    continue
                if md["id"] in existing_ids:
                    if conflict == "skip":
                        result.memories_skipped += 1
                        continue
                    elif conflict == "overwrite":
                        self._mem.forget(md["id"])
                    # merge: fall through, will insert with new ID

                tags = md.get("tags", [])
                if tag_source and source_tag not in tags:
                    tags = tags + [source_tag]

                self._mem._conn.execute(
                    "INSERT OR IGNORE INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        md["id"], md["content"], md["type"],
                        json.dumps(tags), json.dumps(md.get("context", {})),
                        md["created_at"], md.get("accessed_at", md["created_at"]),
                        md.get("access_count", 0), md.get("importance", 0.5),
                    ),
                )
                if self._mem._fts:
                    self._mem._conn.execute(
                        "INSERT OR IGNORE INTO memories_fts(id,content,tags) VALUES(?,?,?)",
                        (md["id"], md["content"], " ".join(tags)),
                    )
                self._mem._conn.commit()
                result.memories_added += 1
            except Exception as e:
                result.errors.append(f"Memory {md.get('id','?')[:8]}: {e}")
                result.memories_skipped += 1

        # --- Tasks (pending/in_progress only - don't import completed work) ---
        existing_task_ids = {t.id for t in self._tg.find(limit=100000)}
        for td in bundle.tasks:
            try:
                if td.get("status") == "done":
                    result.tasks_skipped += 1
                    continue
                if td["id"] in existing_task_ids:
                    if conflict == "skip":
                        result.tasks_skipped += 1
                        continue

                tags = td.get("tags", [])
                if tag_source and source_tag not in tags:
                    tags = tags + [source_tag]

                self._tg._conn.execute(
                    "INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        td["id"], td["title"], td.get("description", ""),
                        td.get("status", "pending"), td.get("priority", 2),
                        json.dumps(tags), json.dumps(td.get("metadata", {})),
                        td.get("notes", ""), td["created_at"],
                        td.get("updated_at", td["created_at"]),
                        td.get("completed_at"),
                    ),
                )
                for dep_id in td.get("deps", []):
                    self._tg._conn.execute(
                        "INSERT OR IGNORE INTO task_deps VALUES (?,?)",
                        (td["id"], dep_id),
                    )
                if self._tg._fts:
                    self._tg._conn.execute(
                        "INSERT OR IGNORE INTO tasks_fts(id,title,description,tags) VALUES(?,?,?,?)",
                        (td["id"], td["title"], td.get("description",""), " ".join(tags)),
                    )
                self._tg._conn.commit()
                result.tasks_added += 1
            except Exception as e:
                result.errors.append(f"Task {td.get('id','?')[:8]}: {e}")
                result.tasks_skipped += 1

        # --- Reflections ---
        conn = connect(self._db_path)
        init_schema(conn, has_fts5(conn))
        existing_ref_ids = {
            r[0] for r in conn.execute("SELECT id FROM reflections").fetchall()
        }
        for rd in bundle.reflections:
            try:
                if rd["id"] in existing_ref_ids:
                    result.reflections_skipped += 1
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO reflections VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        rd["id"], rd.get("task_id"), rd["task_title"],
                        rd["outcome"], rd.get("what_worked",""),
                        rd.get("what_didnt",""), rd.get("lesson",""),
                        rd.get("effort_mins"), rd["created_at"],
                    ),
                )
                conn.commit()
                result.reflections_added += 1
            except Exception as e:
                result.errors.append(f"Reflection {rd.get('id','?')[:8]}: {e}")
                result.reflections_skipped += 1

        return result

    def _source_tag(self, meta: BundleMeta) -> str:
        name = meta.source_agent_name or meta.source_agent_type or "external"
        # Sanitize to a valid tag
        return f"from:{name.lower().replace(' ', '-')[:20]}"


import json  # needed for the inline json.dumps calls above
from ._db import connect, has_fts5, init_schema
