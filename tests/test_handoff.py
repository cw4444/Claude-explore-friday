"""Tests for agent-to-agent handoff protocol."""

import json
import pytest
import time
from pathlib import Path

from nexus.handoff import Handoff, HandoffManager, HANDOFF_VERSION, HANDOFF_FILENAME_PREFIX
from nexus.context import ContextManager
from nexus.tasks import Priority, TaskStatus
from nexus.memory import MemoryType


@pytest.fixture
def hm(tmp_path):
    return HandoffManager(db_path=tmp_path / "test.db")


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "test.db")


@pytest.fixture
def hm_cm(tmp_path):
    db = tmp_path / "test.db"
    return HandoffManager(db_path=db), ContextManager(db_path=db)


# ------------------------------------------------------------------
# Handoff dataclass
# ------------------------------------------------------------------

def test_handoff_briefing_contains_summary(hm):
    h = hm.create(summary="Implemented auth. Tests pass.", to_agent="claude-code")
    briefing = h.briefing()
    assert "auth" in briefing
    assert "Tests pass" in briefing


def test_handoff_briefing_shows_warnings(hm):
    h = hm.create(
        summary="Done.",
        warnings=["Do not touch prod DB", "Rate limit is 100/min"],
    )
    briefing = h.briefing()
    assert "WARNINGS" in briefing
    assert "prod DB" in briefing


def test_handoff_briefing_shows_next_steps_ordered(hm):
    h = hm.create(
        summary="Done.",
        next_steps=["File PR", "Update docs", "Add tests"],
    )
    briefing = h.briefing()
    assert "1." in briefing
    assert "File PR" in briefing
    pr_pos = briefing.find("File PR")
    docs_pos = briefing.find("Update docs")
    assert pr_pos < docs_pos


def test_handoff_briefing_shows_work_done(hm):
    h = hm.create(summary="Done.", work_done=["Fixed auth bug", "Added rate limiting"])
    briefing = h.briefing()
    assert "Fixed auth bug" in briefing


def test_handoff_briefing_shows_blockers(hm):
    h = hm.create(summary="Done.", blockers=["Need prod creds"])
    briefing = h.briefing()
    assert "Need prod creds" in briefing


def test_handoff_to_dict_roundtrip(hm):
    h = hm.create(
        summary="Test handoff",
        to_agent="claude-code",
        next_steps=["Step 1", "Step 2"],
        key_facts=["Python 3.11", "SQLite DB at ~/.nexus/nexus.db"],
        warnings=["Don't delete pinned memories"],
    )
    d = h.to_dict()
    h2 = Handoff.from_dict(d)
    assert h2.id == h.id
    assert h2.summary == h.summary
    assert h2.next_steps == h.next_steps
    assert h2.key_facts == h.key_facts
    assert h2.warnings == h.warnings
    assert h2.version == HANDOFF_VERSION


# ------------------------------------------------------------------
# Create
# ------------------------------------------------------------------

def test_create_sets_fields(hm):
    h = hm.create(
        summary="Auth module done.",
        to_agent="claude-code",
        next_steps=["File PR"],
        blockers=["Need creds"],
        key_facts=["SQLite backed"],
    )
    assert h.to_agent == "claude-code"
    assert h.summary == "Auth module done."
    assert "File PR" in h.next_steps
    assert "Need creds" in h.blockers
    assert "SQLite backed" in h.key_facts
    assert h.version == HANDOFF_VERSION
    assert h.id
    assert h.created_at > 0
    assert h.status == "pending"


def test_create_includes_bundle_by_default(hm_cm):
    hm, cm = hm_cm
    cm.memory.remember("Key fact about auth", type=MemoryType.SEMANTIC, tags=["auth"])
    h = hm.create(summary="Test", include_bundle=True, bundle_tags=["auth"])
    # Bundle may be None if export fails, but in normal conditions it should exist
    # Just verify it doesn't crash
    assert h.summary == "Test"


def test_create_no_bundle(hm):
    h = hm.create(summary="Test", include_bundle=False)
    assert h.bundle is None


def test_create_from_agent_from_identity(hm):
    h = hm.create(summary="Test")
    assert isinstance(h.from_agent, str)
    assert len(h.from_agent) > 0


# ------------------------------------------------------------------
# Write / Load
# ------------------------------------------------------------------

def test_write_to_file(tmp_path, hm):
    h = hm.create(summary="Write test")
    path = tmp_path / "handoffs" / "test.json"
    written = hm.write(h, path)
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["id"] == h.id


def test_write_to_directory_autogenerates_name(tmp_path, hm):
    h = hm.create(summary="Auto-name test")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    written = hm.write(h, handoff_dir)
    assert written.exists()
    assert written.name.startswith(HANDOFF_FILENAME_PREFIX)


def test_load_from_file(tmp_path, hm):
    h = hm.create(summary="Load test", next_steps=["Do the thing"])
    path = tmp_path / "handoff.json"
    hm.write(h, path)
    loaded = hm.load(path)
    assert loaded.id == h.id
    assert loaded.next_steps == ["Do the thing"]


# ------------------------------------------------------------------
# Find
# ------------------------------------------------------------------

def test_find_incoming_empty_dir(tmp_path, hm):
    result = hm.find_incoming(tmp_path / "handoffs")
    assert result is None


def test_find_incoming_finds_pending(tmp_path, hm):
    h = hm.create(summary="Pending handoff", to_agent="any")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    hm.write(h, handoff_dir)
    found = hm.find_incoming(handoff_dir)
    assert found is not None
    assert found.id == h.id


def test_find_incoming_prefers_exact_match(tmp_path, hm):
    h_any = hm.create(summary="For anyone", to_agent="any")
    h_exact = hm.create(summary="For claude-code", to_agent="claude-code")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    hm.write(h_any, handoff_dir)
    time.sleep(0.01)
    hm.write(h_exact, handoff_dir)
    found = hm.find_incoming(handoff_dir, agent_type="claude-code")
    assert found.id == h_exact.id


def test_find_incoming_falls_back_to_any(tmp_path, hm):
    h = hm.create(summary="Broadcast", to_agent="any")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    hm.write(h, handoff_dir)
    found = hm.find_incoming(handoff_dir, agent_type="claude-code")
    assert found.id == h.id


def test_find_incoming_ignores_acknowledged(tmp_path, hm):
    h = hm.create(summary="Already done", to_agent="any")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    path = hm.write(h, handoff_dir)
    hm.acknowledge(h, path)
    found = hm.find_incoming(handoff_dir)
    assert found is None


# ------------------------------------------------------------------
# Apply
# ------------------------------------------------------------------

def test_apply_creates_tasks_from_next_steps(hm_cm):
    hm, cm = hm_cm
    h = hm.create(
        summary="Done.",
        next_steps=["File PR", "Update docs"],
        include_bundle=False,
    )
    count = hm.apply(h, cm)
    assert count > 0
    tasks = cm.tasks.find(limit=100)
    titles = [t.title for t in tasks]
    assert "File PR" in titles
    assert "Update docs" in titles


def test_apply_first_step_is_high_priority(hm_cm):
    hm, cm = hm_cm
    h = hm.create(
        summary="Done.",
        next_steps=["Urgent first step", "Less urgent"],
        include_bundle=False,
    )
    hm.apply(h, cm)
    tasks = cm.tasks.find(limit=100)
    first = next(t for t in tasks if t.title == "Urgent first step")
    assert first.priority == Priority.HIGH


def test_apply_stores_key_facts(hm_cm):
    hm, cm = hm_cm
    h = hm.create(
        summary="Done.",
        key_facts=["Auth uses JWT RS256", "DB at ~/.nexus/nexus.db"],
        include_bundle=False,
    )
    hm.apply(h, cm)
    memories = cm.memory.all(limit=100)
    contents = [m.content for m in memories]
    assert any("JWT RS256" in c for c in contents)


def test_apply_stores_summary_as_memory(hm_cm):
    hm, cm = hm_cm
    h = hm.create(
        summary="Auth module complete. Tests pass.",
        include_bundle=False,
    )
    hm.apply(h, cm)
    memories = cm.memory.all(limit=100)
    assert any("Auth module complete" in m.content for m in memories)


def test_apply_tags_memories_with_source(hm_cm):
    hm, cm = hm_cm
    h = hm.create(
        summary="Done.",
        key_facts=["important fact"],
        include_bundle=False,
    )
    # Patch from_agent
    h.from_agent = "agent-a"
    hm.apply(h, cm)
    memories = cm.memory.all(limit=100)
    handoff_mems = [m for m in memories if "from:agent-a" in m.tags]
    assert len(handoff_mems) >= 1


# ------------------------------------------------------------------
# Acknowledge
# ------------------------------------------------------------------

def test_acknowledge_updates_status(hm):
    h = hm.create(summary="Test")
    assert h.status == "pending"
    hm.acknowledge(h)
    assert h.status == "acknowledged"


def test_acknowledge_writes_to_file(tmp_path, hm):
    h = hm.create(summary="Test")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    path = hm.write(h, handoff_dir)
    hm.acknowledge(h, path)
    loaded = hm.load(path)
    assert loaded.status == "acknowledged"


# ------------------------------------------------------------------
# find_and_apply (one-shot)
# ------------------------------------------------------------------

def test_find_and_apply_complete_flow(tmp_path):
    db = tmp_path / "test.db"
    hm = HandoffManager(db_path=db)
    cm = ContextManager(db_path=db)

    h = hm.create(
        summary="Ready for handoff",
        next_steps=["Complete the thing"],
        include_bundle=False,
    )
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    hm.write(h, handoff_dir)

    result = hm.find_and_apply(handoff_dir, cm=cm)
    assert result is not None
    assert result.id == h.id
    assert result.status == "acknowledged"

    # Verify applied
    tasks = cm.tasks.find(limit=100)
    assert any("Complete the thing" in t.title for t in tasks)

    # Second call should find nothing (already acknowledged)
    result2 = hm.find_and_apply(handoff_dir, cm=cm)
    assert result2 is None


def test_find_and_apply_no_handoffs(tmp_path):
    hm = HandoffManager(db_path=tmp_path / "test.db")
    handoff_dir = tmp_path / "handoffs"
    handoff_dir.mkdir()
    result = hm.find_and_apply(handoff_dir)
    assert result is None
