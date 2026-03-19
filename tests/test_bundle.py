"""Tests for knowledge bundle export/import."""

import time
import json
import pytest

from nexus.bundle import BundleExporter, BundleImporter, KnowledgeBundle
from nexus.memory import MemoryType
from nexus.tasks import TaskStatus, Priority
from nexus.reflection import Outcome
from nexus.identity import get_or_create_identity
from nexus.context import ContextManager


@pytest.fixture
def source_cm(tmp_path):
    return ContextManager(db_path=tmp_path / "source.db")


@pytest.fixture
def dest_cm(tmp_path):
    return ContextManager(db_path=tmp_path / "dest.db")


@pytest.fixture
def source_identity(tmp_path):
    return get_or_create_identity(name="openclaw-1", agent_type="openclaw",
                                   path=tmp_path / "src_identity.json")


@pytest.fixture
def exporter(tmp_path, source_identity):
    return BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)


@pytest.fixture
def importer(tmp_path):
    return BundleImporter(db_path=tmp_path / "dest.db")


def _seed_source(cm: ContextManager):
    """Seed source DB with test data."""
    cm.memory.remember("prod DB at db.prod.example.com", type=MemoryType.SEMANTIC,
                        tags=["db", "prod"], importance=0.8)
    cm.memory.remember("JWT tokens expire after 1h", type=MemoryType.SEMANTIC,
                        tags=["auth"], importance=0.7)
    cm.memory.remember("Always mock HTTP in tests", type=MemoryType.PROCEDURAL,
                        tags=["testing"], importance=0.85)
    cm.tasks.create("Fix auth bug", tags=["auth"])
    cm.tasks.create("Deploy to prod", tags=["deploy"])
    cm.reflection.reflect("Fixed login bug", Outcome.SUCCESS,
                           lesson="Check session expiry first")


def test_export_creates_bundle(tmp_path, source_identity):
    cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export(description="Test bundle")
    assert bundle.meta.bundle_id
    assert bundle.meta.source_agent_name == "openclaw-1"
    assert bundle.meta.source_agent_type == "openclaw"
    # 3 explicit + 1 minted from reflection lesson = 4
    assert len(bundle.memories) == 4
    assert len(bundle.tasks) == 2
    assert len(bundle.reflections) == 1


def test_export_tag_filter(tmp_path, source_identity):
    cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export(tags=["auth"])
    # Only auth-tagged items
    assert all(
        any("auth" in (m.get("tags") or []) for _ in [m])
        for m in bundle.memories
    )


def test_export_min_importance(tmp_path, source_identity):
    cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export(min_importance=0.8)
    # Should only include memories with importance >= 0.8
    assert all(m["importance"] >= 0.8 for m in bundle.memories)


def test_export_excludes_done_tasks_by_default(tmp_path, source_identity):
    cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(cm)
    tasks = list(cm.tasks.find(limit=100))
    cm.tasks.complete(tasks[0].id)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()
    assert all(t["status"] != "done" for t in bundle.tasks)


def test_save_and_load_bundle(tmp_path, source_identity):
    cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()

    path = tmp_path / "bundle.json"
    bundle.save(path)
    assert path.exists()

    loaded = KnowledgeBundle.load(path)
    assert loaded.meta.bundle_id == bundle.meta.bundle_id
    assert len(loaded.memories) == len(bundle.memories)
    assert len(loaded.tasks) == len(bundle.tasks)


def test_import_memories(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(src_cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()

    dest_cm = ContextManager(db_path=tmp_path / "dest.db")
    importer = BundleImporter(db_path=tmp_path / "dest.db")
    result = importer.import_bundle(bundle)

    assert result.memories_added == 4  # 3 explicit + 1 lesson mint
    assert result.tasks_added == 2
    assert result.reflections_added == 1

    # Memories are in dest
    mems = dest_cm.memory.all()
    assert len(mems) == 4


def test_import_tags_with_source(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    src_cm.memory.remember("a fact", type=MemoryType.SEMANTIC)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()

    dest_cm = ContextManager(db_path=tmp_path / "dest.db")
    importer = BundleImporter(db_path=tmp_path / "dest.db")
    importer.import_bundle(bundle, tag_source=True)

    mems = dest_cm.memory.all()
    assert any("from:openclaw-1" in m.tags for m in mems)


def test_import_skip_conflict(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    src_cm.memory.remember("shared fact", type=MemoryType.SEMANTIC)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()

    dest_cm = ContextManager(db_path=tmp_path / "dest.db")
    importer = BundleImporter(db_path=tmp_path / "dest.db")

    result1 = importer.import_bundle(bundle)
    assert result1.memories_added == 1

    result2 = importer.import_bundle(bundle, conflict="skip")
    assert result2.memories_added == 0
    assert result2.memories_skipped == 1


def test_import_does_not_import_done_tasks(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    t = src_cm.tasks.create("Already done")
    src_cm.tasks.complete(t.id)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export(include_done_tasks=True)
    # Bundle has the task, but import should skip done tasks
    importer = BundleImporter(db_path=tmp_path / "dest.db")
    result = importer.import_bundle(bundle)
    assert result.tasks_added == 0
    assert result.tasks_skipped == 1


def test_bundle_summary(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    _seed_source(src_cm)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export(description="Test bundle")
    summary = bundle.summary()
    assert "openclaw-1" in summary
    assert "Memories" in summary
    assert "Tasks" in summary


def test_roundtrip_preserves_content(tmp_path, source_identity):
    src_cm = ContextManager(db_path=tmp_path / "source.db")
    src_cm.memory.remember("Critical: always validate input",
                            type=MemoryType.PROCEDURAL, tags=["security"], importance=0.9)
    exporter = BundleExporter(db_path=tmp_path / "source.db", identity=source_identity)
    bundle = exporter.export()

    path = tmp_path / "bundle.json"
    bundle.save(path)
    loaded = KnowledgeBundle.load(path)

    dest_cm = ContextManager(db_path=tmp_path / "dest.db")
    importer = BundleImporter(db_path=tmp_path / "dest.db")
    importer.import_bundle(loaded)

    mems = dest_cm.memory.all()
    assert any("validate input" in m.content for m in mems)
    assert any(m.importance >= 0.9 for m in mems)
