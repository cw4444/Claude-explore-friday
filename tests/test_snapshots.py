"""Tests for agent state snapshots (time travel)."""

import json
import pytest
import time
from pathlib import Path

from nexus.snapshots import SnapshotManager, Snapshot, SnapshotMeta, SnapshotDiff
from nexus.context import ContextManager
from nexus.memory import MemoryType
from nexus.tasks import Priority, TaskStatus


@pytest.fixture
def sm(tmp_path):
    return SnapshotManager(
        db_path=tmp_path / "nexus.db",
        snapshot_dir=tmp_path / "snapshots",
    )


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "nexus.db")


@pytest.fixture
def sm_cm(tmp_path):
    db = tmp_path / "nexus.db"
    return (
        SnapshotManager(db_path=db, snapshot_dir=tmp_path / "snapshots"),
        ContextManager(db_path=db),
    )


def _seed(cm, n_memories=3, n_tasks=2):
    for i in range(n_memories):
        cm.memory.remember(f"Memory {i}", type=MemoryType.SEMANTIC, tags=[f"tag{i}"])
    for i in range(n_tasks):
        cm.tasks.create(f"Task {i}", priority=Priority.MEDIUM)


# ------------------------------------------------------------------
# SnapshotMeta
# ------------------------------------------------------------------

def test_snapshot_meta_display():
    meta = SnapshotMeta(
        id="abc", label="before refactor", created_at=time.time() - 120,
        trigger="manual", memory_count=5, task_count=3,
        reflection_count=1, contact_count=0,
    )
    d = meta.display()
    assert "before refactor" in d
    assert "manual" in d
    assert "5" in d


# ------------------------------------------------------------------
# Create
# ------------------------------------------------------------------

def test_create_empty_db(sm):
    snap = sm.create("empty state")
    assert snap.meta.label == "empty state"
    assert snap.meta.memory_count == 0
    assert snap.meta.task_count == 0
    assert len(snap.memories) == 0
    assert len(snap.tasks) == 0


def test_create_captures_memories(sm_cm):
    sm, cm = sm_cm
    _seed(cm, n_memories=4, n_tasks=0)
    snap = sm.create("after seeding")
    assert snap.meta.memory_count == 4
    assert len(snap.memories) == 4


def test_create_captures_tasks(sm_cm):
    sm, cm = sm_cm
    _seed(cm, n_memories=0, n_tasks=3)
    snap = sm.create()
    assert snap.meta.task_count == 3
    assert len(snap.tasks) == 3


def test_create_captures_task_deps(sm_cm):
    sm, cm = sm_cm
    tasks = cm.tasks.decompose("Parent", ["Step 1", "Step 2"])
    snap = sm.create()
    assert len(snap.task_deps) > 0


def test_create_saves_to_disk(sm_cm):
    sm, cm = sm_cm
    snap = sm.create("disk test")
    path = sm._dir / f"snapshot-{snap.meta.id}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["meta"]["label"] == "disk test"


def test_create_trigger_stored(sm):
    snap = sm.create("test", trigger="pre-import")
    assert snap.meta.trigger == "pre-import"


# ------------------------------------------------------------------
# List
# ------------------------------------------------------------------

def test_list_empty(sm):
    assert sm.list() == []


def test_list_returns_newest_first(sm_cm):
    sm, cm = sm_cm
    s1 = sm.create("first")
    time.sleep(0.01)
    s2 = sm.create("second")
    metas = sm.list()
    assert metas[0].id == s2.meta.id
    assert metas[1].id == s1.meta.id


def test_list_limit(sm):
    for i in range(5):
        sm.create(f"snap {i}")
    assert len(sm.list(limit=3)) == 3


# ------------------------------------------------------------------
# Get
# ------------------------------------------------------------------

def test_get_by_full_id(sm_cm):
    sm, cm = sm_cm
    _seed(cm)
    snap = sm.create("test")
    loaded = sm.get(snap.meta.id)
    assert loaded.meta.id == snap.meta.id
    assert len(loaded.memories) == len(snap.memories)


def test_get_by_prefix(sm_cm):
    sm, cm = sm_cm
    snap = sm.create("prefix test")
    loaded = sm.get(snap.meta.id[:8])
    assert loaded.meta.id == snap.meta.id


def test_get_missing_raises(sm):
    with pytest.raises(FileNotFoundError):
        sm.get("nonexistent-id")


# ------------------------------------------------------------------
# Restore
# ------------------------------------------------------------------

def test_restore_reverts_memories(sm_cm):
    sm, cm = sm_cm
    cm.memory.remember("Original memory", type=MemoryType.SEMANTIC)
    snap = sm.create("before changes")
    mem_id = cm.memory.all()[0].id

    # Add a new memory and delete original
    cm.memory.remember("New bad memory", type=MemoryType.SEMANTIC)
    cm.memory.forget(mem_id)

    sm.restore(snap.meta.id, create_pre_restore_snapshot=False)

    memories = cm.memory.all()
    contents = [m.content for m in memories]
    assert "Original memory" in contents
    assert "New bad memory" not in contents


def test_restore_reverts_tasks(sm_cm):
    sm, cm = sm_cm
    t = cm.tasks.create("Original task")
    snap = sm.create("before task changes")

    cm.tasks.create("Unwanted task")
    cm.tasks.complete(t.id)

    sm.restore(snap.meta.id, create_pre_restore_snapshot=False)

    tasks = cm.tasks.find(limit=100)
    assert len(tasks) == 1
    assert tasks[0].title == "Original task"
    assert tasks[0].status == TaskStatus.PENDING


def test_restore_reverts_to_empty(sm_cm):
    sm, cm = sm_cm
    snap = sm.create("empty")
    _seed(cm)

    sm.restore(snap.meta.id, create_pre_restore_snapshot=False)

    assert cm.memory.all() == []
    assert cm.tasks.find(limit=100) == []


def test_restore_creates_pre_restore_snapshot(sm_cm):
    sm, cm = sm_cm
    _seed(cm)
    snap = sm.create("checkpoint")

    # Modify state
    cm.memory.remember("New memory after checkpoint", type=MemoryType.SEMANTIC)

    # Restore - should create a pre-restore snapshot automatically
    sm.restore(snap.meta.id, create_pre_restore_snapshot=True)

    metas = sm.list()
    triggers = [m.trigger for m in metas]
    assert "pre-restore" in triggers


def test_restore_is_reversible(sm_cm):
    sm, cm = sm_cm
    cm.memory.remember("State A", type=MemoryType.SEMANTIC)
    snap_a = sm.create("state A")

    cm.memory.remember("State B addition", type=MemoryType.SEMANTIC)
    snap_b = sm.create("state B")

    # Restore to A
    sm.restore(snap_a.meta.id)

    # State should be A
    contents = [m.content for m in cm.memory.all()]
    assert "State A" in contents
    assert "State B addition" not in contents

    # Restore to B using the pre-restore snapshot
    metas = sm.list()
    pre_restore = next(m for m in metas if m.trigger == "pre-restore")
    sm.restore(pre_restore.id, create_pre_restore_snapshot=False)

    # State should be B again
    contents = [m.content for m in cm.memory.all()]
    assert "State B addition" in contents


def test_restore_preserves_task_deps(sm_cm):
    sm, cm = sm_cm
    cm.tasks.decompose("Big task", ["Step 1", "Step 2", "Step 3"])
    snap = sm.create("with deps")

    sm.restore(snap.meta.id, create_pre_restore_snapshot=False)

    # Dependency chain should be intact - 4 tasks (parent + 3 subs)
    tasks = cm.tasks.find(limit=100)
    assert len(tasks) == 4

    # Only the parent (no deps) should be available as "next"
    next_task = cm.tasks.next()
    assert next_task is not None
    assert next_task.title == "Big task"


# ------------------------------------------------------------------
# Diff
# ------------------------------------------------------------------

def test_diff_no_change(sm_cm):
    sm, cm = sm_cm
    _seed(cm, n_memories=2, n_tasks=1)
    snap = sm.create("baseline")

    d = sm.diff(snap.meta.id)
    assert d.memories_added == 0
    assert d.memories_removed == 0
    assert d.tasks_added == 0
    assert d.reflections_added == 0


def test_diff_added_memories(sm_cm):
    sm, cm = sm_cm
    _seed(cm, n_memories=1)
    snap = sm.create("before")

    cm.memory.remember("New memory 1", type=MemoryType.SEMANTIC)
    cm.memory.remember("New memory 2", type=MemoryType.SEMANTIC)

    d = sm.diff(snap.meta.id)
    assert d.memories_added == 2
    assert d.memories_removed == 0
    assert len(d.added_memory_samples) == 2


def test_diff_removed_memories(sm_cm):
    sm, cm = sm_cm
    cm.memory.remember("Will be deleted", type=MemoryType.SEMANTIC)
    snap = sm.create("before")
    mem = cm.memory.all()[0]
    cm.memory.forget(mem.id)

    d = sm.diff(snap.meta.id)
    assert d.memories_removed == 1


def test_diff_task_status_changes(sm_cm):
    sm, cm = sm_cm
    t = cm.tasks.create("Track me")
    snap = sm.create("before completion")

    cm.tasks.complete(t.id)
    d = sm.diff(snap.meta.id)
    assert d.task_status_changes
    assert "pending" in d.task_status_changes[0]
    assert "done" in d.task_status_changes[0]


def test_diff_summary_readable(sm_cm):
    sm, cm = sm_cm
    _seed(cm)
    snap = sm.create("baseline")
    cm.memory.remember("Extra memory", type=MemoryType.SEMANTIC)

    d = sm.diff(snap.meta.id)
    summary = d.summary()
    assert "Diff" in summary
    assert "Memories" in summary


# ------------------------------------------------------------------
# Prune
# ------------------------------------------------------------------

def test_prune_removes_old(sm):
    for i in range(10):
        sm.create(f"snap {i}")
    # keep_manual=False to actually delete labelled manual snapshots
    deleted = sm.prune(keep=5, keep_manual=False)
    assert deleted == 5
    assert len(sm.list()) == 5


def test_prune_keeps_manual(sm):
    for i in range(5):
        sm.create(f"auto {i}", trigger="pre-import")
    sm.create("important manual checkpoint", trigger="manual")
    sm.prune(keep=3, keep_manual=True)
    metas = sm.list()
    labels = [m.label for m in metas]
    assert "important manual checkpoint" in labels


def test_prune_nothing_to_delete(sm):
    sm.create("only one")
    assert sm.prune(keep=5) == 0


# ------------------------------------------------------------------
# Serialisation roundtrip
# ------------------------------------------------------------------

def test_snapshot_roundtrip(sm_cm):
    sm, cm = sm_cm
    _seed(cm)
    snap = sm.create("roundtrip")
    loaded = sm.get(snap.meta.id)
    assert loaded.meta.id == snap.meta.id
    assert len(loaded.memories) == len(snap.memories)
    assert len(loaded.tasks) == len(snap.tasks)


# ------------------------------------------------------------------
# Auto-snapshot integration
# ------------------------------------------------------------------

def test_bundle_import_auto_snapshots(tmp_path):
    """BundleImporter should auto-snapshot before modifying state."""
    from nexus.bundle import BundleExporter, BundleImporter

    db = tmp_path / "nexus.db"
    cm = ContextManager(db_path=db)
    cm.memory.remember("Pre-import memory", type=MemoryType.SEMANTIC)

    exporter = BundleExporter(db_path=db)
    bundle = exporter.export()

    # Clear and import into a fresh DB
    db2 = tmp_path / "nexus2.db"
    sm = SnapshotManager(db_path=db2, snapshot_dir=tmp_path / "snapshots")
    importer = BundleImporter(db_path=db2)
    importer.import_bundle(bundle)

    # A pre-import snapshot should now exist
    metas = sm.list()
    assert any(m.trigger == "pre-import" for m in metas)


def test_handoff_apply_auto_snapshots(tmp_path):
    """HandoffManager.apply() should auto-snapshot before modifying state."""
    from nexus.handoff import HandoffManager

    db = tmp_path / "nexus.db"
    cm = ContextManager(db_path=db)
    sm = SnapshotManager(db_path=db, snapshot_dir=tmp_path / "snapshots")

    hm = HandoffManager(db_path=db)
    h = hm.create(summary="Test handoff", next_steps=["Do X"], include_bundle=False)
    hm.apply(h, cm)

    metas = sm.list()
    assert any(m.trigger == "pre-handoff" for m in metas)
