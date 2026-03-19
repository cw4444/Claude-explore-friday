"""Database utilities - SQLite with FTS5 fallback."""

import sqlite3
from pathlib import Path
from typing import Optional


_DEFAULT_DIR = Path.home() / ".nexus"


def default_db_path() -> Path:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DIR / "nexus.db"


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_test USING fts5(x)")
        conn.execute("DROP TABLE _fts5_test")
        return True
    except sqlite3.OperationalError:
        return False


def init_schema(conn: sqlite3.Connection, fts: bool) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT    PRIMARY KEY,
            content     TEXT    NOT NULL,
            type        TEXT    NOT NULL CHECK(type IN ('episodic','semantic','procedural')),
            tags        TEXT    NOT NULL DEFAULT '[]',
            context     TEXT    NOT NULL DEFAULT '{}',
            created_at  REAL    NOT NULL,
            accessed_at REAL    NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            importance  REAL    NOT NULL DEFAULT 0.5
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id           TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            description  TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'pending'
                         CHECK(status IN ('pending','in_progress','done','failed','blocked')),
            priority     INTEGER NOT NULL DEFAULT 2,
            tags         TEXT NOT NULL DEFAULT '[]',
            metadata     TEXT NOT NULL DEFAULT '{}',
            notes        TEXT NOT NULL DEFAULT '',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL,
            completed_at REAL
        );

        CREATE TABLE IF NOT EXISTS task_deps (
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            dep_id  TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            PRIMARY KEY (task_id, dep_id)
        );

        CREATE TABLE IF NOT EXISTS reflections (
            id           TEXT PRIMARY KEY,
            task_id      TEXT,
            task_title   TEXT NOT NULL,
            outcome      TEXT NOT NULL CHECK(outcome IN ('success','partial','failure')),
            what_worked  TEXT NOT NULL DEFAULT '',
            what_didnt   TEXT NOT NULL DEFAULT '',
            lesson       TEXT NOT NULL DEFAULT '',
            effort_mins  INTEGER,
            created_at   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contacts (
            id                TEXT PRIMARY KEY,
            name              TEXT NOT NULL UNIQUE,
            contact_type      TEXT NOT NULL DEFAULT 'human'
                              CHECK(contact_type IN ('human','agent','team')),
            first_seen        REAL NOT NULL,
            last_seen         REAL NOT NULL,
            interaction_count INTEGER NOT NULL DEFAULT 0,
            notes             TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS contact_observations (
            id          TEXT PRIMARY KEY,
            contact_id  TEXT NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
            category    TEXT NOT NULL
                        CHECK(category IN ('preference','pattern','trust','context','history','warning')),
            content     TEXT NOT NULL,
            confidence  REAL NOT NULL DEFAULT 0.8,
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pinned_traits (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'observed',
            created_at  REAL NOT NULL
        );
    """)

    if fts:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(id UNINDEXED, content, tags, tokenize='porter ascii');

            CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts
            USING fts5(id UNINDEXED, title, description, tags, tokenize='porter ascii');
        """)

    conn.commit()
