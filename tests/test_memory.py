"""Tests for the memory system."""

import time
import tempfile
from pathlib import Path

import pytest

from nexus.memory import MemoryStore, MemoryType, Memory


@pytest.fixture
def mem(tmp_path):
    return MemoryStore(db_path=tmp_path / "test.db")


def test_remember_returns_memory(mem):
    m = mem.remember("The API key is in .env.prod", type=MemoryType.SEMANTIC)
    assert isinstance(m, Memory)
    assert m.content == "The API key is in .env.prod"
    assert m.type == MemoryType.SEMANTIC
    assert m.id


def test_remember_all_types(mem):
    mem.remember("Something happened", type=MemoryType.EPISODIC)
    mem.remember("A fact", type=MemoryType.SEMANTIC)
    mem.remember("How to do X", type=MemoryType.PROCEDURAL)
    all_mems = mem.all()
    assert len(all_mems) == 3


def test_remember_with_tags(mem):
    m = mem.remember("Deploy with deploy.sh", type=MemoryType.PROCEDURAL,
                     tags=["deploy", "production"])
    assert "deploy" in m.tags
    assert "production" in m.tags


def test_recall_returns_relevant(mem):
    mem.remember("The production database URL is db.prod.example.com",
                 type=MemoryType.SEMANTIC, tags=["db", "prod"])
    mem.remember("The dev database is localhost:5432",
                 type=MemoryType.SEMANTIC, tags=["db", "dev"])
    mem.remember("How to deploy to production", type=MemoryType.PROCEDURAL)

    results = mem.recall("database")
    assert any("database" in m.content.lower() or "db" in m.tags for m in results)


def test_recall_type_filter(mem):
    mem.remember("episodic thing", type=MemoryType.EPISODIC)
    mem.remember("semantic thing", type=MemoryType.SEMANTIC)
    mem.remember("procedural thing", type=MemoryType.PROCEDURAL)

    results = mem.recall("thing", type=MemoryType.EPISODIC)
    assert all(m.type == MemoryType.EPISODIC for m in results)


def test_recall_bumps_access_count(mem):
    m = mem.remember("test content", type=MemoryType.SEMANTIC)
    assert m.access_count == 0
    mem.recall("test content")
    updated = mem.get(m.id)
    assert updated.access_count == 1


def test_get_returns_memory(mem):
    m = mem.remember("test", type=MemoryType.SEMANTIC)
    fetched = mem.get(m.id)
    assert fetched.id == m.id
    assert fetched.content == m.content


def test_get_missing_returns_none(mem):
    assert mem.get("nonexistent-id") is None


def test_forget_removes_memory(mem):
    m = mem.remember("to delete", type=MemoryType.SEMANTIC)
    assert mem.forget(m.id) is True
    assert mem.get(m.id) is None


def test_forget_missing_returns_false(mem):
    assert mem.forget("does-not-exist") is False


def test_importance_clamped(mem):
    m = mem.remember("test", importance=2.5)
    assert m.importance == 1.0
    m2 = mem.remember("test2", importance=-1.0)
    assert m2.importance == 0.0


def test_all_type_filter(mem):
    mem.remember("ep", type=MemoryType.EPISODIC)
    mem.remember("se", type=MemoryType.SEMANTIC)
    mem.remember("pr", type=MemoryType.PROCEDURAL)
    semantics = mem.all(type=MemoryType.SEMANTIC)
    assert len(semantics) == 1
    assert semantics[0].type == MemoryType.SEMANTIC


def test_stats(mem):
    mem.remember("ep", type=MemoryType.EPISODIC, tags=["x"])
    mem.remember("se", type=MemoryType.SEMANTIC, tags=["x", "y"])
    s = mem.stats()
    assert s["total"] == 2
    assert "x" in s["top_tags"]


def test_update_importance(mem):
    m = mem.remember("test", importance=0.5)
    mem.update_importance(m.id, 0.9)
    updated = mem.get(m.id)
    assert abs(updated.importance - 0.9) < 1e-6


def test_recall_empty_store(mem):
    results = mem.recall("anything")
    assert results == []


def test_memory_age_days(mem):
    m = mem.remember("test")
    assert m.age_days < 1.0


def test_multiple_recalls_accumulate(mem):
    m = mem.remember("frequently accessed", type=MemoryType.SEMANTIC)
    for _ in range(5):
        mem.recall("frequently accessed")
    updated = mem.get(m.id)
    assert updated.access_count == 5
