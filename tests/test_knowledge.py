"""Tests for the tiered knowledge system (KnowledgeStore)."""

import json
import time
import pytest
from pathlib import Path

from nexus.knowledge import (
    KnowledgeStore,
    KnowledgeEntry,
    ApplyResult,
    TIER_TRUST,
    TIERS,
    DEFAULT_APPLY_TIERS,
    _scope_matches,
    _detect_scope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """KnowledgeStore backed by a temp dir (no canon loaded)."""
    s = KnowledgeStore(db_path=tmp_path / "nexus.db")
    # Monkeypatch _load_canon to return nothing (tests don't need canon)
    s._load_canon = lambda: []
    return s


@pytest.fixture
def cm(tmp_path):
    from nexus.context import ContextManager
    return ContextManager(db_path=tmp_path / "nexus.db")


@pytest.fixture
def store_with_cm(tmp_path):
    db_path = tmp_path / "nexus.db"
    s = KnowledgeStore(db_path=db_path)
    s._load_canon = lambda: []
    from nexus.context import ContextManager
    c = ContextManager(db_path=db_path)
    return s, c


@pytest.fixture
def human_reviewer():
    return {"id": "human-1", "agent_type": "human", "name": "Alice"}


@pytest.fixture
def agent_reviewer():
    return {"id": "agent-1", "agent_type": "agent", "name": "Bob-agent"}


# ---------------------------------------------------------------------------
# KnowledgeEntry.trust_score
# ---------------------------------------------------------------------------

class TestTrustScore:
    def test_canon_baseline(self):
        e = KnowledgeEntry(
            id="x", content="c", tier="canon", importance=1.0, tags=[],
            source_pack=None,
            author={"id": "a", "agent_type": "system", "name": "a"},
            created_at=0.0, updated_at=0.0, version=1,
            applies_to=[], reviews=[], rejected=False, rejection_reason="",
        )
        assert e.trust_score == 1.0

    def test_quarantine_baseline(self):
        e = KnowledgeEntry(
            id="x", content="c", tier="quarantine", importance=0.5, tags=[],
            source_pack=None,
            author={"id": "a", "agent_type": "agent", "name": "a"},
            created_at=0.0, updated_at=0.0, version=1,
            applies_to=[], reviews=[], rejected=False, rejection_reason="",
        )
        assert e.trust_score == 0.2

    def test_human_promote_adds_delta(self):
        e = KnowledgeEntry(
            id="x", content="c", tier="field", importance=0.8, tags=[],
            source_pack=None,
            author={"id": "a", "agent_type": "agent", "name": "a"},
            created_at=0.0, updated_at=0.0, version=1,
            applies_to=[],
            reviews=[{"reviewer_agent_type": "human", "verdict": "promote"}],
            rejected=False, rejection_reason="",
        )
        # field baseline 0.6 + human promote 0.15 = 0.75
        assert abs(e.trust_score - 0.75) < 1e-9

    def test_agent_reject_subtracts_delta(self):
        e = KnowledgeEntry(
            id="x", content="c", tier="field", importance=0.8, tags=[],
            source_pack=None,
            author={"id": "a", "agent_type": "agent", "name": "a"},
            created_at=0.0, updated_at=0.0, version=1,
            applies_to=[],
            reviews=[{"reviewer_agent_type": "agent", "verdict": "reject"}],
            rejected=False, rejection_reason="",
        )
        # field baseline 0.6 - agent reject 0.1 = 0.5
        assert abs(e.trust_score - 0.5) < 1e-9

    def test_trust_clamped_to_zero_to_one(self):
        reviews = [{"reviewer_agent_type": "human", "verdict": "reject"}] * 10
        e = KnowledgeEntry(
            id="x", content="c", tier="field", importance=0.8, tags=[],
            source_pack=None,
            author={"id": "a", "agent_type": "agent", "name": "a"},
            created_at=0.0, updated_at=0.0, version=1,
            applies_to=[], reviews=reviews, rejected=False, rejection_reason="",
        )
        assert e.trust_score == 0.0


# ---------------------------------------------------------------------------
# Scope matching
# ---------------------------------------------------------------------------

class TestScopeMatching:
    def test_universal_always_matches(self):
        assert _scope_matches([], ["repo:a/b"]) is True
        assert _scope_matches([], []) is True

    def test_repo_match(self):
        assert _scope_matches(["repo:a/b"], ["repo:a/b", "org:a"]) is True

    def test_org_match(self):
        assert _scope_matches(["org:a"], ["repo:a/b", "org:a"]) is True

    def test_no_match(self):
        assert _scope_matches(["repo:x/y"], ["repo:a/b", "org:a"]) is False

    def test_empty_scope_universal_still_matches(self):
        assert _scope_matches([], []) is True

    def test_scoped_entry_no_scope_context_no_match(self):
        # Scoped entry with non-empty applies_to, current scope is empty
        assert _scope_matches(["repo:a/b"], []) is False


# ---------------------------------------------------------------------------
# KnowledgeStore - contribute
# ---------------------------------------------------------------------------

class TestContribute:
    def test_contribute_creates_entry(self, store):
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        entry = store.contribute("Test lesson", author=author)
        assert entry.id
        assert entry.tier == "field"
        assert entry.content == "Test lesson"
        assert "contributed" in entry.tags
        assert entry.version == 1

    def test_contribute_quarantine_tier(self, store):
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        entry = store.contribute("Unverified tip", tier="quarantine", author=author)
        assert entry.tier == "quarantine"

    def test_contribute_persists(self, store):
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        store.contribute("Persistent lesson", author=author)
        entries = store.all()
        assert any(e.content == "Persistent lesson" for e in entries)

    def test_contribute_with_cm_installs_provisional(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        store.contribute("My lesson", author=author, cm=cm)
        mems = cm.memory.all(limit=1000)
        provisional = [m for m in mems if "knowledge:provisional" in m.tags]
        assert len(provisional) == 1
        assert provisional[0].content == "My lesson"

    def test_contribute_without_cm_no_provisional(self, store):
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        store.contribute("Lesson", author=author)
        # No cm provided - should not raise
        entries = store.all()
        assert len(entries) == 1

    def test_contribute_custom_applies_to(self, store):
        author = {"id": "a1", "agent_type": "agent", "name": "Agent A"}
        entry = store.contribute(
            "Repo-specific tip",
            applies_to=["repo:owner/project"],
            author=author,
        )
        assert entry.applies_to == ["repo:owner/project"]


# ---------------------------------------------------------------------------
# KnowledgeStore - get / all
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_all_returns_contributed(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Entry 1", author=author)
        store.contribute("Entry 2", author=author)
        assert len(store.all()) == 2

    def test_all_filters_by_tier(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("field entry", tier="field", author=author)
        store.contribute("quarantine entry", tier="quarantine", author=author)
        assert len(store.all(tier="field")) == 1
        assert len(store.all(tier="quarantine")) == 1

    def test_all_excludes_rejected_by_default(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad entry", author=author)
        store.reject(entry.id, "bad content")
        assert len(store.all()) == 0
        assert len(store.all(include_rejected=True)) == 1

    def test_get_by_full_id(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Entry", author=author)
        fetched = store.get(entry.id)
        assert fetched is not None
        assert fetched.id == entry.id

    def test_get_by_prefix(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Entry", author=author)
        fetched = store.get(entry.id[:8])
        assert fetched is not None
        assert fetched.id == entry.id

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent-id") is None


# ---------------------------------------------------------------------------
# KnowledgeStore - review (auto-promotion / rejection)
# ---------------------------------------------------------------------------

class TestReview:
    def test_review_adds_record(self, store, agent_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="quarantine", author=author)
        updated = store.review(entry.id, verdict="promote", reviewer=agent_reviewer)
        assert updated is not None
        assert len(updated.reviews) == 1
        assert updated.reviews[0]["verdict"] == "promote"

    def test_one_human_promotes_quarantine_to_field(self, store, human_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="quarantine", author=author)
        updated = store.review(entry.id, verdict="promote", reviewer=human_reviewer)
        assert updated.tier == "field"
        assert updated.version == 2

    def test_two_agent_promotes_quarantine_to_field(self, store, agent_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="quarantine", author=author)
        store.review(entry.id, verdict="promote", reviewer=agent_reviewer)
        updated = store.review(entry.id, verdict="promote",
                               reviewer={"id": "b", "agent_type": "agent", "name": "C"})
        assert updated.tier == "field"

    def test_one_agent_promote_does_not_auto_promote(self, store, agent_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="quarantine", author=author)
        updated = store.review(entry.id, verdict="promote", reviewer=agent_reviewer)
        assert updated.tier == "quarantine"  # still quarantine, not enough reviews

    def test_one_human_rejects_quarantine(self, store, human_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad lesson", tier="quarantine", author=author)
        updated = store.review(entry.id, verdict="reject", reviewer=human_reviewer,
                               note="Contains bad advice")
        assert updated.rejected is True
        assert updated.rejection_reason == "Contains bad advice"

    def test_two_agent_rejects_quarantine(self, store, agent_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad lesson", tier="quarantine", author=author)
        store.review(entry.id, verdict="reject", reviewer=agent_reviewer)
        updated = store.review(entry.id, verdict="reject",
                               reviewer={"id": "b", "agent_type": "agent", "name": "C"})
        assert updated.rejected is True

    def test_explicit_promote_non_quarantine(self, store, human_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Good lesson", tier="field", author=author)
        updated = store.review(entry.id, verdict="promote",
                               reviewer=human_reviewer, to_tier="warning")
        assert updated.tier == "warning"
        assert updated.version == 2

    def test_review_invalid_verdict(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", author=author)
        with pytest.raises(ValueError, match="verdict must be"):
            store.review(entry.id, verdict="approve")

    def test_review_canon_returns_none(self, store):
        # Canon entries are read-only; no local entry to find
        result = store.review("nonexistent-canonical-id", verdict="promote")
        assert result is None

    def test_auto_promote_uses_to_tier_if_specified(self, store, human_reviewer):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="quarantine", author=author)
        updated = store.review(entry.id, verdict="promote",
                               reviewer=human_reviewer, to_tier="warning")
        assert updated.tier == "warning"


# ---------------------------------------------------------------------------
# KnowledgeStore - promote / reject
# ---------------------------------------------------------------------------

class TestPromoteReject:
    def test_promote_changes_tier(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Lesson", tier="field", author=author)
        updated = store.promote(entry.id, to_tier="warning")
        assert updated.tier == "warning"
        assert updated.version == 2

    def test_promote_nonexistent_returns_none(self, store):
        assert store.promote("nonexistent", to_tier="warning") is None

    def test_reject_marks_entry(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad entry", author=author)
        updated = store.reject(entry.id, reason="Incorrect")
        assert updated.rejected is True
        assert updated.rejection_reason == "Incorrect"

    def test_reject_nonexistent_returns_none(self, store):
        assert store.reject("nonexistent") is None

    def test_reject_hides_from_all(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Entry", author=author)
        store.reject(entry.id)
        assert store.all() == []


# ---------------------------------------------------------------------------
# KnowledgeStore - apply
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_adds_to_memory(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Remember this", tier="field", author=author)
        result = store.apply(cm, scope=[])
        assert result.entries_added == 1
        mems = cm.memory.all(limit=1000)
        assert any("Remember this" in m.content for m in mems)

    def test_apply_is_idempotent(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Lesson", tier="field", author=author)
        store.apply(cm, scope=[])
        result2 = store.apply(cm, scope=[])
        assert result2.entries_added == 0
        assert result2.entries_skipped == 1

    def test_apply_quarantine_excluded_by_default(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Unverified", tier="quarantine", author=author)
        result = store.apply(cm, scope=[])
        assert result.entries_added == 0
        assert result.entries_excluded >= 1

    def test_apply_quarantine_with_explicit_tier(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Unverified", tier="quarantine", author=author)
        result = store.apply(cm, tiers=["quarantine"], scope=[])
        assert result.entries_added == 1

    def test_apply_rejected_excluded(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad", tier="field", author=author)
        store.reject(entry.id, "wrong")
        result = store.apply(cm, scope=[])
        assert result.entries_added == 0

    def test_apply_scoped_procedure_matches(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute(
            "Repo procedure",
            tier="procedure",
            applies_to=["repo:owner/project"],
            author=author,
        )
        result = store.apply(cm, scope=["repo:owner/project", "org:owner"])
        assert result.entries_added == 1

    def test_apply_scoped_procedure_no_match_skipped(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute(
            "Repo procedure",
            tier="procedure",
            applies_to=["repo:other/project"],
            author=author,
        )
        result = store.apply(cm, scope=["repo:owner/project", "org:owner"])
        assert result.entries_added == 0
        assert result.entries_excluded >= 1

    def test_apply_min_trust_filters(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Field lesson", tier="field", author=author)  # trust 0.6
        result = store.apply(cm, min_trust=0.9, scope=[])
        assert result.entries_added == 0

    def test_apply_updates_versioned_entry(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Original content", tier="field", author=author)

        # Apply v1
        store.apply(cm, scope=[])

        # Promote and update the entry (version bump)
        updated = store.promote(entry.id, to_tier="warning")
        assert updated.version == 2

        # Manually update content as a new version to test update path
        # Simulate version bump by modifying local entry
        local = store._load_local_by_id(entry.id)
        local.content = "Updated content"
        local.version = 3
        store._update_local(local)

        result2 = store.apply(cm, scope=[])
        assert result2.entries_updated == 1

    def test_apply_result_summary(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Lesson A", tier="field", author=author)
        store.contribute("Lesson B", tier="warning", author=author)
        result = store.apply(cm, scope=[])
        summary = result.summary()
        assert "Added" in summary
        assert "2" in summary


# ---------------------------------------------------------------------------
# KnowledgeStore - status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_shows_per_tier(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Field lesson", tier="field", author=author)
        status = store.status(cm)
        assert "field" in status
        assert status["field"]["total"] == 1
        assert status["field"]["installed"] == 0

    def test_status_after_apply(self, store_with_cm):
        store, cm = store_with_cm
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Lesson", tier="field", author=author)
        store.apply(cm, scope=[])
        status = store.status(cm)
        assert status["field"]["installed"] == 1
        assert status["field"]["complete"] is True


# ---------------------------------------------------------------------------
# KnowledgeStore - inspect
# ---------------------------------------------------------------------------

class TestInspect:
    def test_inspect_quarantine_only(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        store.contribute("Field", tier="field", author=author)
        store.contribute("Quarantine", tier="quarantine", author=author)
        entries = store.inspect(tier="quarantine")
        assert len(entries) == 1
        assert entries[0].tier == "quarantine"

    def test_inspect_excludes_rejected(self, store):
        author = {"id": "a", "agent_type": "agent", "name": "A"}
        entry = store.contribute("Bad", tier="quarantine", author=author)
        store.reject(entry.id, "bad")
        entries = store.inspect(tier="quarantine")
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# KnowledgeEntry serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_round_trip(self):
        e = KnowledgeEntry(
            id="abc-123",
            content="Test",
            tier="field",
            importance=0.8,
            tags=["test"],
            source_pack="my-pack",
            author={"id": "a", "agent_type": "agent", "name": "A"},
            created_at=1000.0,
            updated_at=1001.0,
            version=3,
            applies_to=["repo:x/y"],
            reviews=[{"reviewer_agent_type": "human", "verdict": "promote",
                      "id": "r1", "reviewer_id": "h1", "reviewer_name": "H",
                      "at": 1000.0, "to_tier": None, "note": ""}],
            rejected=False,
            rejection_reason="",
        )
        d = e.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.id == e.id
        assert restored.tier == e.tier
        assert restored.version == e.version
        assert restored.applies_to == e.applies_to
        assert len(restored.reviews) == 1
        assert abs(restored.trust_score - 0.75) < 1e-9  # field + human promote


# ---------------------------------------------------------------------------
# Canon loading
# ---------------------------------------------------------------------------

class TestCanon:
    def test_canon_loaded_by_default(self, tmp_path):
        """Real store (with canon.json) should return canon entries."""
        store = KnowledgeStore(db_path=tmp_path / "nexus.db")
        entries = store.all(tier="canon")
        assert len(entries) > 0, "canon.json should have entries"
        for e in entries:
            assert e.tier == "canon"
            assert e.trust_score == 1.0

    def test_canon_entries_not_modifiable(self, tmp_path):
        """promote() on a canon entry returns None (only local entries mutable)."""
        store = KnowledgeStore(db_path=tmp_path / "nexus.db")
        canons = store.all(tier="canon")
        assert len(canons) > 0
        result = store.promote(canons[0].id, to_tier="field")
        assert result is None  # Can't promote canon - it's read-only

    def test_tiers_summary(self, tmp_path):
        store = KnowledgeStore(db_path=tmp_path / "nexus.db")
        summary = store.tiers_summary()
        assert "canon" in summary
        assert summary["canon"] > 0
