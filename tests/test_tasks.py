"""Tests for the task graph system."""

import pytest

from nexus.tasks import TaskGraph, Task, TaskStatus, Priority


@pytest.fixture
def tg(tmp_path):
    return TaskGraph(db_path=tmp_path / "test.db")


def test_create_task(tg):
    t = tg.create("Write tests")
    assert isinstance(t, Task)
    assert t.title == "Write tests"
    assert t.status == TaskStatus.PENDING
    assert t.id


def test_create_with_priority(tg):
    t = tg.create("Urgent thing", priority=Priority.URGENT)
    assert t.priority == Priority.URGENT


def test_create_with_tags(tg):
    t = tg.create("Tagged task", tags=["auth", "backend"])
    assert "auth" in t.tags


def test_start_task(tg):
    t = tg.create("Task to start")
    tg.start(t.id)
    updated = tg.get(t.id)
    assert updated.status == TaskStatus.IN_PROGRESS


def test_complete_task(tg):
    t = tg.create("Task to complete")
    tg.complete(t.id, notes="Done!")
    updated = tg.get(t.id)
    assert updated.status == TaskStatus.DONE
    assert updated.completed_at is not None
    assert updated.notes == "Done!"


def test_fail_task(tg):
    t = tg.create("Task that fails")
    tg.fail(t.id, notes="Couldn't do it")
    updated = tg.get(t.id)
    assert updated.status == TaskStatus.FAILED


def test_reopen_task(tg):
    t = tg.create("Task to reopen")
    tg.complete(t.id)
    tg.reopen(t.id)
    updated = tg.get(t.id)
    assert updated.status == TaskStatus.PENDING


def test_add_note(tg):
    t = tg.create("Task with notes")
    tg.add_note(t.id, "First note")
    tg.add_note(t.id, "Second note")
    updated = tg.get(t.id)
    assert "First note" in updated.notes
    assert "Second note" in updated.notes


def test_get_missing_returns_none(tg):
    assert tg.get("nonexistent") is None


def test_decompose_creates_chain(tg):
    tasks = tg.decompose(
        "Big task",
        subtasks=["Step 1", "Step 2", "Step 3"],
        chain=True,
    )
    assert len(tasks) == 4  # parent + 3 subtasks
    # Step 2 depends on Step 1
    assert tasks[1].id in tasks[2].deps
    # Step 3 depends on Step 2
    assert tasks[2].id in tasks[3].deps
    # Step 1 has no deps
    assert tasks[1].deps == []


def test_decompose_parallel(tg):
    tasks = tg.decompose(
        "Parallel task",
        subtasks=["A", "B", "C"],
        chain=False,
    )
    for t in tasks[1:]:
        assert t.deps == []


def test_next_returns_ready_task(tg):
    t1 = tg.create("Task 1")
    t2 = tg.create("Task 2", deps=[t1.id])
    next_t = tg.next()
    # t1 should be next since t2 depends on it
    assert next_t.id == t1.id


def test_next_skips_blocked(tg):
    t1 = tg.create("Blocker", priority=Priority.LOW)
    t2 = tg.create("Blocked", deps=[t1.id], priority=Priority.HIGH)
    t3 = tg.create("Ready", priority=Priority.MEDIUM)
    next_t = tg.next()
    # t1 is ready (no deps), t3 is ready. t1 has lower priority.
    # t3 should NOT be next since t1 (the blocker) has no deps and needs doing
    # Actually: next() returns highest priority with no incomplete deps
    # t1 (LOW priority, no deps) and t3 (MEDIUM priority, no deps) -> t3 wins
    assert next_t.id == t3.id


def test_next_after_complete(tg):
    t1 = tg.create("Step 1")
    t2 = tg.create("Step 2", deps=[t1.id])
    tg.complete(t1.id)
    next_t = tg.next()
    assert next_t.id == t2.id


def test_cycle_detection(tg):
    t1 = tg.create("A")
    t2 = tg.create("B", deps=[t1.id])
    with pytest.raises(ValueError, match="cycle"):
        tg.add_dep(t1.id, t2.id)


def test_self_cycle_detection(tg):
    t = tg.create("Self")
    with pytest.raises(ValueError, match="cycle"):
        tg.add_dep(t.id, t.id)


def test_find_by_status(tg):
    tg.create("Pending 1")
    t2 = tg.create("Task 2")
    tg.complete(t2.id)
    pending = tg.find(status=TaskStatus.PENDING)
    assert all(t.status == TaskStatus.PENDING for t in pending)
    assert len(pending) == 1


def test_stats(tg):
    tg.create("T1")
    t2 = tg.create("T2")
    tg.complete(t2.id)
    s = tg.stats()
    assert s["total"] == 2
    assert s["by_status"].get("done") == 1
    assert s["by_status"].get("pending") == 1


def test_pending_includes_in_progress(tg):
    t1 = tg.create("Pending")
    t2 = tg.create("Started")
    tg.start(t2.id)
    pending = tg.pending()
    ids = [t.id for t in pending]
    assert t1.id in ids
    assert t2.id in ids


def test_task_is_actionable(tg):
    t = tg.create("New task")
    assert t.is_actionable()
    tg.complete(t.id)
    updated = tg.get(t.id)
    assert not updated.is_actionable()
