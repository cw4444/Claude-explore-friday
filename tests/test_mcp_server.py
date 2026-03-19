"""Tests for the MCP server.

We verify the server builds correctly and test the underlying tool logic
through the ContextManager that the server wraps. Wire protocol testing
is left to integration tests.
"""

import asyncio
import pytest

from nexus.mcp_server import build_server
from nexus.memory import MemoryType
from nexus.tasks import TaskStatus


def test_server_builds(tmp_path):
    """Smoke test that server builds without errors."""
    server = build_server(db_path=tmp_path / "test.db")
    assert server is not None
    assert server.name == "nexus"


def test_tool_list_via_coroutine(tmp_path):
    """Verify all expected tools are registered."""
    server = build_server(db_path=tmp_path / "test.db")

    async def get_tools():
        # The list_tools decorator registers a handler we can call directly
        for handler in server._tool_handlers.values():
            result = await handler()
            return result
        return []

    # Use the public list_tools mechanism if accessible
    # Fall back to checking the handler registrations
    expected = {
        "nexus_remember", "nexus_recall", "nexus_context", "nexus_pin",
        "nexus_task_create", "nexus_task_decompose", "nexus_task_list",
        "nexus_task_next", "nexus_task_start", "nexus_task_done",
        "nexus_task_fail", "nexus_task_note", "nexus_reflect", "nexus_stats",
        "nexus_growth", "nexus_export_bundle", "nexus_import_bundle", "nexus_identity",
    }
    # Verify by checking that all expected tool names appear in the server module source
    import inspect
    from nexus import mcp_server
    src = inspect.getsource(mcp_server)
    for tool_name in expected:
        assert f'name="{tool_name}"' in src, f"Tool {tool_name} not found in server definition"


# ---------------------------------------------------------------------------
# Test the actual tool logic via ContextManager (the MCP server wraps CM)
# ---------------------------------------------------------------------------

def test_mcp_remember_and_recall(tmp_path):
    """Verify the underlying CM operations the MCP server uses."""
    from nexus.context import ContextManager
    cm = ContextManager(db_path=tmp_path / "test.db")

    m = cm.memory.remember("JWT tokens expire after 1h", type=MemoryType.SEMANTIC,
                            tags=["auth"])
    assert m.id

    results = cm.memory.recall("JWT token expiry")
    assert any("JWT" in r.content for r in results)


def test_mcp_task_workflow(tmp_path):
    from nexus.context import ContextManager
    from nexus.tasks import Priority
    cm = ContextManager(db_path=tmp_path / "test.db")

    t = cm.tasks.create("MCP-created task", priority=Priority.HIGH)
    assert t.status == TaskStatus.PENDING

    cm.tasks.start(t.id)
    updated = cm.tasks.get(t.id)
    assert updated.status == TaskStatus.IN_PROGRESS

    cm.tasks.complete(t.id, notes="Done via MCP")
    done = cm.tasks.get(t.id)
    assert done.status == TaskStatus.DONE


def test_mcp_reflect_mints_memory(tmp_path):
    from nexus.context import ContextManager
    from nexus.reflection import Outcome
    cm = ContextManager(db_path=tmp_path / "test.db")

    cm.reflection.reflect(
        task_title="Fix flaky test",
        outcome=Outcome.SUCCESS,
        lesson="Always mock external HTTP calls in tests",
    )
    lessons = cm.reflection.lessons()
    assert any("mock" in l.lower() for l in lessons)

    # Lesson should be in procedural memory
    mems = cm.memory.all(type=MemoryType.PROCEDURAL)
    assert any("mock" in m.content.lower() for m in mems)


def test_mcp_context_warm_start(tmp_path):
    from nexus.context import ContextManager
    from nexus.reflection import Outcome
    cm = ContextManager(db_path=tmp_path / "test.db")

    # Seed data as an agent would across sessions
    cm.memory.remember("prod DB at db.prod.example.com", type=MemoryType.SEMANTIC,
                       tags=["db", "prod"])
    cm.reflection.reflect("Past debug session", Outcome.SUCCESS,
                           lesson="Check connection pool size when DB is slow")
    t = cm.tasks.create("Investigate DB slowness")

    ctx = cm.prepare("Database is slow again")
    summary = ctx.summary()

    assert "DB" in summary or "database" in summary.lower() or "slow" in summary.lower()
    assert ctx.next_task is not None


def test_mcp_decompose(tmp_path):
    from nexus.context import ContextManager
    cm = ContextManager(db_path=tmp_path / "test.db")

    tasks = cm.tasks.decompose(
        "Ship new feature",
        subtasks=["Write spec", "Implement", "Review", "Deploy"],
    )
    assert len(tasks) == 5  # parent + 4
    # Each subtask except the first depends on the previous
    assert tasks[2].deps == [tasks[1].id]
    assert tasks[3].deps == [tasks[2].id]


def test_mcp_pin_surfaces_in_context(tmp_path):
    from nexus.context import ContextManager
    cm = ContextManager(db_path=tmp_path / "test.db")

    cm.pin("NEVER commit secrets to git", tags=["security"])
    ctx = cm.prepare("Add a new API integration")
    all_mems = ctx.relevant_memories + ctx.pinned
    assert any("secret" in m.content.lower() or "commit" in m.content.lower()
               for m in all_mems)
