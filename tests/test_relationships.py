"""Tests for relationship memory."""

import time
import pytest

from nexus.relationships import RelationshipStore, OBSERVATION_CATEGORIES


@pytest.fixture
def rs(tmp_path):
    return RelationshipStore(db_path=tmp_path / "test.db")


def test_know_creates_contact(rs):
    c = rs.know("Alice", contact_type="human")
    assert c.name == "Alice"
    assert c.contact_type == "human"
    assert c.interaction_count == 0


def test_know_is_idempotent(rs):
    rs.know("Alice")
    rs.know("Alice")
    contacts = rs.all_contacts()
    assert len(contacts) == 1


def test_observe_creates_contact(rs):
    obs = rs.observe("Bob", "prefers concise answers", category="preference")
    assert obs.category == "preference"
    assert "concise" in obs.content
    contact = rs.recall("Bob")
    assert len(contact.observations) == 1


def test_observe_all_categories(rs):
    for cat in OBSERVATION_CATEGORIES:
        rs.observe("Alice", f"test {cat}", category=cat)
    contact = rs.recall("Alice")
    observed_cats = {o.category for o in contact.observations}
    assert observed_cats == set(OBSERVATION_CATEGORIES.keys())


def test_invalid_category_raises(rs):
    with pytest.raises(ValueError):
        rs.observe("Alice", "something", category="invalid_category")


def test_recall_returns_none_for_unknown(rs):
    assert rs.recall("Nobody") is None


def test_recall_includes_observations(rs):
    rs.observe("Carol", "always wants code examples", category="preference")
    rs.observe("Carol", "trusts autonomous staging deploys", category="trust")
    contact = rs.recall("Carol")
    assert len(contact.observations) == 2
    cats = {o.category for o in contact.observations}
    assert "preference" in cats
    assert "trust" in cats


def test_log_interaction_increments(rs):
    rs.know("Dave")
    rs.log_interaction("Dave", notes="Fixed auth bug together")
    rs.log_interaction("Dave")
    contact = rs.recall("Dave")
    assert contact.interaction_count == 2


def test_log_interaction_stores_history(rs):
    rs.log_interaction("Eve", notes="Reviewed PR #42")
    contact = rs.recall("Eve")
    history_obs = [o for o in contact.observations if o.category == "history"]
    assert len(history_obs) == 1
    assert "PR #42" in history_obs[0].content


def test_session_briefing_unknown(rs):
    briefing = rs.session_briefing("Stranger")
    assert "No prior knowledge" in briefing


def test_session_briefing_known(rs):
    rs.observe("Frank", "prefers bullet points over prose", category="preference")
    rs.observe("Frank", "never approve prod deploys without review", category="warning")
    rs.observe("Frank", "building payments integration", category="context")
    rs.observe("Frank", "trusts autonomous staging", category="trust")

    briefing = rs.session_briefing("Frank")
    assert "Frank" in briefing
    assert "WARNINGS" in briefing
    assert "prod deploy" in briefing.lower() or "prod deploys" in briefing.lower()
    assert "preference" in briefing.lower() or "Preference" in briefing
    assert "payments" in briefing


def test_session_briefing_warnings_first(rs):
    rs.observe("Grace", "likes detailed explanations", category="preference")
    rs.observe("Grace", "CRITICAL: has destructive migration in prod", category="warning")
    briefing = rs.session_briefing("Grace")
    warning_pos = briefing.find("WARNINGS")
    pref_pos = briefing.find("Preferences")
    assert warning_pos < pref_pos


def test_confidence_stored(rs):
    rs.observe("Heidi", "probably prefers morning sessions", category="pattern",
               confidence=0.4)
    contact = rs.recall("Heidi")
    obs = contact.observations[0]
    assert abs(obs.confidence - 0.4) < 0.01


def test_remove_observation(rs):
    obs = rs.observe("Ivan", "likes morning sessions", category="pattern")
    assert rs.remove_observation(obs.id)
    contact = rs.recall("Ivan")
    assert len(contact.observations) == 0


def test_all_contacts(rs):
    rs.know("Alice")
    rs.know("Bob")
    rs.know("Carol")
    contacts = rs.all_contacts()
    assert len(contacts) == 3
    names = {c.name for c in contacts}
    assert {"Alice", "Bob", "Carol"} == names


def test_by_category(rs):
    rs.observe("Judy", "prefers X", category="preference")
    rs.observe("Judy", "trusts Y", category="trust")
    rs.observe("Judy", "does Z", category="pattern")
    contact = rs.recall("Judy")
    by_cat = contact.by_category()
    assert "preference" in by_cat
    assert "trust" in by_cat
    assert "pattern" in by_cat


def test_context_observations_limited_to_recent(rs):
    """Context observations should only show 3 most recent."""
    for i in range(5):
        rs.observe("Ken", f"context observation {i}", category="context")
    briefing = rs.session_briefing("Ken")
    # Should show at most 3 context entries
    count = briefing.count("context observation")
    assert count <= 3


def test_known_since_days(rs):
    c = rs.know("Laura")
    assert c.known_since_days < 1.0


def test_update_notes(rs):
    rs.know("Mike", notes="original notes")
    rs.know("Mike", notes="updated notes")
    contact = rs.recall("Mike")
    assert "updated" in contact.notes
