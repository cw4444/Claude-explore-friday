"""Tests for the skills hub / bootstrap knowledge system."""

import json
import pytest
from pathlib import Path

from nexus.bootstrap import BootstrapManager, BootstrapResult
from nexus.context import ContextManager
from nexus.memory import MemoryType


@pytest.fixture
def bm(tmp_path):
    return BootstrapManager(db_path=tmp_path / "nexus.db")


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "nexus.db")


@pytest.fixture
def bm_cm(tmp_path):
    db = tmp_path / "nexus.db"
    return BootstrapManager(db_path=db), ContextManager(db_path=db)


# ------------------------------------------------------------------
# BootstrapResult
# ------------------------------------------------------------------

def test_result_summary_with_additions():
    r = BootstrapResult(packs_applied=["nexus-basics"], entries_added=8, entries_skipped=0)
    s = r.summary()
    assert "8" in s
    assert "nexus-basics" in s


def test_result_summary_all_skipped():
    r = BootstrapResult(packs_applied=[], entries_added=0, entries_skipped=5)
    s = r.summary()
    assert "No bootstrap" in s or "Skipped" in s or "0" in s


# ------------------------------------------------------------------
# list_packs
# ------------------------------------------------------------------

def test_list_packs_returns_all_builtin(bm):
    packs = bm.list_packs()
    assert "nexus-basics" in packs
    assert "agent-patterns" in packs
    assert "human-collab" in packs
    assert "failure-modes" in packs


def test_list_packs_has_descriptions(bm):
    packs = bm.list_packs()
    for name, desc in packs.items():
        assert len(desc) > 0


def test_pack_entries_nonempty(bm):
    for pack_name in bm.list_packs():
        entries = bm.pack_entries(pack_name)
        assert len(entries) > 0, f"Pack {pack_name} has no entries"


def test_pack_entries_have_required_fields(bm):
    for pack_name in bm.list_packs():
        for entry in bm.pack_entries(pack_name):
            assert "content" in entry
            assert "importance" in entry
            assert 0.0 <= entry["importance"] <= 1.0
            assert len(entry["content"]) > 20  # not trivially short


# ------------------------------------------------------------------
# apply
# ------------------------------------------------------------------

def test_apply_installs_all_packs(bm_cm):
    bm, cm = bm_cm
    result = bm.apply(cm)
    assert result.entries_added > 0
    assert len(result.packs_applied) == 4  # all 4 packs

    memories = cm.memory.all(limit=1000)
    bootstrap_mems = [m for m in memories if "bootstrap" in m.tags]
    assert len(bootstrap_mems) == result.entries_added


def test_apply_installs_as_procedural(bm_cm):
    bm, cm = bm_cm
    bm.apply(cm)
    memories = cm.memory.all(limit=1000)
    bootstrap_mems = [m for m in memories if "bootstrap" in m.tags]
    for m in bootstrap_mems:
        assert m.type == MemoryType.PROCEDURAL


def test_apply_tags_with_pack_name(bm_cm):
    bm, cm = bm_cm
    bm.apply(cm, packs=["nexus-basics"])
    memories = cm.memory.all(limit=1000)
    pack_tagged = [m for m in memories if "pack:nexus-basics" in m.tags]
    assert len(pack_tagged) > 0


def test_apply_specific_pack(bm_cm):
    bm, cm = bm_cm
    result = bm.apply(cm, packs=["nexus-basics"])
    assert result.packs_applied == ["nexus-basics"]
    all_mems = cm.memory.all(limit=1000)
    assert not any("pack:agent-patterns" in m.tags for m in all_mems)


def test_apply_is_idempotent(bm_cm):
    bm, cm = bm_cm
    r1 = bm.apply(cm)
    r2 = bm.apply(cm)
    assert r2.entries_added == 0
    assert r2.entries_skipped == r1.entries_added


def test_apply_skips_existing(bm_cm):
    bm, cm = bm_cm
    # Install once
    r1 = bm.apply(cm)
    total = r1.entries_added

    # Install again - all should be skipped
    r2 = bm.apply(cm)
    assert r2.entries_added == 0
    assert r2.entries_skipped == total


def test_apply_partial_install_fills_gap(bm_cm):
    bm, cm = bm_cm
    # Install two packs
    bm.apply(cm, packs=["nexus-basics", "agent-patterns"])

    # Now install all
    r2 = bm.apply(cm)
    # Should only add the remaining two packs
    assert r2.entries_added > 0
    assert "human-collab" in r2.packs_applied or "failure-modes" in r2.packs_applied


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------

def test_status_empty_db(bm_cm):
    bm, cm = bm_cm
    status = bm.status(cm)
    for pack_name, info in status.items():
        assert info["installed"] == 0
        assert info["missing"] == info["total"]
        assert not info["complete"]


def test_status_after_full_install(bm_cm):
    bm, cm = bm_cm
    bm.apply(cm)
    status = bm.status(cm)
    for pack_name, info in status.items():
        assert info["installed"] == info["total"]
        assert info["missing"] == 0
        assert info["complete"]


def test_status_partial_install(bm_cm):
    bm, cm = bm_cm
    bm.apply(cm, packs=["nexus-basics"])
    status = bm.status(cm)
    assert status["nexus-basics"]["complete"]
    assert not status["agent-patterns"]["complete"]


# ------------------------------------------------------------------
# contribute
# ------------------------------------------------------------------

def test_contribute_saves_to_file(bm):
    bm.contribute(
        "When debugging, read the error message before changing anything.",
        pack="agent-patterns",
    )
    contribs = bm.list_contributions()
    assert len(contribs) == 1
    assert "read the error message" in contribs[0]["content"]


def test_contribute_stores_in_memory(bm_cm):
    bm, cm = bm_cm
    bm.contribute(
        "Always verify the environment before starting a long operation.",
        pack="failure-modes",
        cm=cm,
    )
    mems = cm.memory.all(limit=100)
    assert any("verify the environment" in m.content for m in mems)


def test_contribute_tags_as_contributed(bm_cm):
    bm, cm = bm_cm
    bm.contribute("Test contribution.", pack="nexus-basics", cm=cm)
    mems = cm.memory.all(limit=100)
    contrib_mem = next(m for m in mems if "Test contribution" in m.content)
    assert "contributed" in contrib_mem.tags


def test_contribute_clamps_importance(bm):
    entry = bm.contribute("Test.", importance=1.5)
    assert entry["importance"] <= 1.0
    entry2 = bm.contribute("Test2.", importance=-0.5)
    assert entry2["importance"] >= 0.0


def test_contribute_multiple(bm):
    bm.contribute("Lesson 1.", pack="agent-patterns")
    bm.contribute("Lesson 2.", pack="agent-patterns")
    bm.contribute("Lesson 3.", pack="human-collab")
    contribs = bm.list_contributions()
    assert len(contribs) == 3


# ------------------------------------------------------------------
# Contributions merged into apply
# ------------------------------------------------------------------

def test_contributed_entries_appear_in_apply(bm_cm):
    bm, cm = bm_cm
    lesson = "Always commit before a risky refactor - rollback is much easier."
    bm.contribute(lesson, pack="agent-patterns")

    # Apply to a fresh context (simulate new agent)
    fresh_cm = ContextManager(db_path=bm._db_path)
    result = bm.apply(fresh_cm)

    mems = fresh_cm.memory.all(limit=1000)
    assert any(lesson in m.content for m in mems)


def test_contributed_entries_dont_duplicate(bm_cm):
    bm, cm = bm_cm
    lesson = "Unique lesson that won't conflict."
    bm.contribute(lesson, pack="agent-patterns", cm=cm)  # installs to memory

    # apply should skip it since it's already in memory
    result = bm.apply(cm)
    mems = [m for m in cm.memory.all(limit=1000) if lesson in m.content]
    assert len(mems) == 1  # not duplicated


def test_contribute_to_unknown_pack(bm):
    # Should create a new pack dynamically
    bm.contribute("New pack lesson.", pack="my-custom-pack")
    packs = bm.list_packs()
    assert "my-custom-pack" in packs


# ------------------------------------------------------------------
# Content quality (smoke tests on the actual knowledge)
# ------------------------------------------------------------------

def test_nexus_basics_mentions_context(bm):
    entries = bm.pack_entries("nexus-basics")
    contents = " ".join(e["content"] for e in entries).lower()
    assert "context" in contents


def test_failure_modes_mentions_injection(bm):
    entries = bm.pack_entries("failure-modes")
    contents = " ".join(e["content"] for e in entries).lower()
    assert "injection" in contents or "external" in contents


def test_human_collab_mentions_intent(bm):
    entries = bm.pack_entries("human-collab")
    contents = " ".join(e["content"] for e in entries).lower()
    assert "intent" in contents


def test_all_entries_are_actionable(bm):
    """Entries should be concrete enough to act on (basic length check)."""
    for pack_name in bm.list_packs():
        for entry in bm.pack_entries(pack_name):
            assert len(entry["content"]) >= 50, (
                f"Entry in {pack_name} too short to be useful: {entry['content'][:40]}"
            )
