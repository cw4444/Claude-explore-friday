"""Three-tier memory system for AI agents.

Memory types:
  episodic   - Things that happened ("I debugged the auth module, found JWT expiry issue")
  semantic   - Facts about the world ("prod DB is at db.prod.example.com")
  procedural - How to do things ("to deploy: run ./scripts/deploy.sh --env=prod")
"""

import json
import time
import uuid
import sqlite3
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from ._db import connect, has_fts5, init_schema


class MemoryType(str, Enum):
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class Memory:
    id: str
    content: str
    type: MemoryType
    tags: List[str]
    context: Dict[str, Any]
    created_at: float
    accessed_at: float
    access_count: int
    importance: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    def __str__(self) -> str:
        tag_str = f"  [{', '.join(self.tags)}]" if self.tags else ""
        return f"[{self.type.value}]{tag_str} {self.content}"


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        type=MemoryType(row["type"]),
        tags=json.loads(row["tags"]),
        context=json.loads(row["context"]),
        created_at=row["created_at"],
        accessed_at=row["accessed_at"],
        access_count=row["access_count"],
        importance=row["importance"],
    )


class MemoryStore:
    """Persistent memory store backed by SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self._conn = connect(db_path)
        self._fts = has_fts5(self._conn)
        init_schema(self._conn, self._fts)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        type: MemoryType = MemoryType.SEMANTIC,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Memory:
        """Store a new memory. Returns the created Memory."""
        now = time.time()
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            type=type,
            tags=tags or [],
            context=context or {},
            created_at=now,
            accessed_at=now,
            access_count=0,
            importance=max(0.0, min(1.0, importance)),
        )
        self._conn.execute(
            """INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                memory.id, memory.content, memory.type.value,
                json.dumps(memory.tags), json.dumps(memory.context),
                memory.created_at, memory.accessed_at,
                memory.access_count, memory.importance,
            ),
        )
        if self._fts:
            self._conn.execute(
                "INSERT INTO memories_fts(id,content,tags) VALUES(?,?,?)",
                (memory.id, memory.content, " ".join(memory.tags)),
            )
        self._conn.commit()
        return memory

    def update_importance(self, memory_id: str, importance: float) -> None:
        self._conn.execute(
            "UPDATE memories SET importance=? WHERE id=?",
            (max(0.0, min(1.0, importance)), memory_id),
        )
        self._conn.commit()

    def forget(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if it existed."""
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        if self._fts:
            self._conn.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[Memory]:
        """Find memories relevant to a query string."""
        if self._fts:
            memories = self._fts_search(query, limit * 3)
        else:
            memories = self._like_search(query, limit * 3)

        if type:
            memories = [m for m in memories if m.type == type]
        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]
        if min_importance > 0:
            memories = [m for m in memories if m.importance >= min_importance]

        memories = memories[:limit]

        # Bump access stats
        now = time.time()
        for m in memories:
            self._conn.execute(
                "UPDATE memories SET accessed_at=?, access_count=access_count+1 WHERE id=?",
                (now, m.id),
            )
        self._conn.commit()
        return memories

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def all(
        self,
        type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Memory]:
        if type:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE type=? ORDER BY importance DESC, created_at DESC LIMIT ?",
                (type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        memories = [_row_to_memory(r) for r in rows]
        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]
        return memories

    def stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = dict(
            self._conn.execute(
                "SELECT type, COUNT(*) FROM memories GROUP BY type"
            ).fetchall()
        )
        top_tags: Dict[str, int] = {}
        for row in self._conn.execute("SELECT tags FROM memories").fetchall():
            for tag in json.loads(row[0]):
                top_tags[tag] = top_tags.get(tag, 0) + 1
        sorted_tags = sorted(top_tags.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "total": total,
            "by_type": by_type,
            "top_tags": dict(sorted_tags),
            "fts_enabled": self._fts,
        }

    # ------------------------------------------------------------------
    # Internal search helpers
    # ------------------------------------------------------------------

    def _fts_search(self, query: str, limit: int) -> List[Memory]:
        words = [w.strip('"\'') for w in query.split() if w.strip('"\'')]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = self._conn.execute(
                """SELECT m.* FROM memories m
                   JOIN memories_fts fts ON m.id = fts.id
                   WHERE memories_fts MATCH ?
                   ORDER BY m.importance DESC, m.accessed_at DESC
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
            return [_row_to_memory(r) for r in rows]
        except sqlite3.OperationalError:
            return self._like_search(query, limit)

    def _like_search(self, query: str, limit: int) -> List[Memory]:
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE content LIKE ? OR tags LIKE ?
               ORDER BY importance DESC, accessed_at DESC
               LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]
