"""Tests for the context manager."""

import pytest

from nexus.context import ContextManager, SessionContext
from nexus.memory import MemoryType
from nexus.tasks import TaskStatus, Priority
from nexus.reflection import Outcome


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "test.db")


def test_prepare_empty_returns_context(cm):
    ctx = cm.prepare("Fix the auth bug")
    assert isinstance(ctx, SessionContext)
    assert ctx.task_description == "Fix the auth bug"
    assert ctx.relevant_memories == []
    assert ctx.in_progress_tasks == []
    assert ctx.next_task is None
    assert ctx.recent_lessons == []


def test_prepare_includes_relevant_memories(cm):
    cm.memory.remember("JWT tokens expire after 1 hour",
                       type=MemoryType.SEMANTIC, tags=["auth", "jwt"])
    cm.memory.remember("Database is at db.prod.example.com",
                       type=MemoryType.SEMANTIC, tags=["db"])
    ctx = cm.prepare("Fix the authentication token bug")
    assert len(ctx.relevant_memories) >= 1
    contents = [m.content for m in ctx.relevant_memories]
    assert any("JWT" in c or "token" in c.lower() or "auth" in c.lower()
               for c in contents)


def test_prepare_includes_in_progress_tasks(cm):
    t = cm.tasks.create("In-progress task")
    cm.tasks.start(t.id)
    ctx = cm.prepare("Do some work")
    assert len(ctx.in_progress_tasks) == 1
    assert ctx.in_progress_tasks[0].id == t.id


def test_prepare_next_task_is_ready(cm):
    t1 = cm.tasks.create("Blocked task")
    t2 = cm.tasks.create("Ready task", priority=Priority.HIGH)
    cm.tasks.create("Depends on T1", deps=[t1.id])
    ctx = cm.prepare("Get next work item")
    # Should return the highest priority task with no incomplete deps
    assert ctx.next_task is not None
    assert ctx.next_task.id in [t1.id, t2.id]


def test_prepare_includes_lessons(cm):
    cm.reflection.reflect(
        task_title="Past work",
        outcome=Outcome.SUCCESS,
        lesson="Always validate input at the boundary",
    )
    ctx = cm.prepare("Write new API endpoint")
    assert len(ctx.recent_lessons) >= 1
    assert any("validate" in l.lower() for l in ctx.recent_lessons)


def test_prepare_includes_pinned_procedural(cm):
    cm.memory.remember(
        "CRITICAL: Always run tests before pushing",
        type=MemoryType.PROCEDURAL,
        tags=["testing", "pinned"],
        importance=0.9,
    )
    ctx = cm.prepare("Implement new feature")
    # Should appear in either relevant_memories or pinned
    all_contents = (
        [m.content for m in ctx.relevant_memories] +
        [m.content for m in ctx.pinned]
    )
    assert any("tests" in c.lower() for c in all_contents)


def test_summary_is_string(cm):
    ctx = cm.prepare("Test task")
    s = ctx.summary()
    assert isinstance(s, str)
    assert "Test task" in s


def test_summary_empty_context(cm):
    ctx = cm.prepare("New thing")
    s = ctx.summary()
    assert "fresh start" in s.lower() or "no prior" in s.lower()


def test_to_dict(cm):
    cm.memory.remember("test fact", type=MemoryType.SEMANTIC)
    ctx = cm.prepare("test")
    d = ctx.to_dict()
    assert "task_description" in d
    assert "relevant_memories" in d
    assert "in_progress_tasks" in d
    assert "recent_lessons" in d
    assert "generated_at" in d


def test_pin_creates_high_importance_memory(cm):
    m = cm.pin("The staging server is at staging.example.com",
                tags=["infrastructure"])
    assert m.importance >= 0.9
    assert "pinned" in m.tags


def test_pin_surfaces_in_context(cm):
    cm.pin("CRITICAL: Never delete prod data directly",
           tags=["safety", "prod"])
    ctx = cm.prepare("Perform database maintenance")
    all_mems = ctx.relevant_memories + ctx.pinned
    assert any("prod" in m.content.lower() or "data" in m.content.lower()
               for m in all_mems)


def test_shared_components(cm):
    """ContextManager exposes memory, tasks, and reflection."""
    assert cm.memory is not None
    assert cm.tasks is not None
    assert cm.reflection is not None
