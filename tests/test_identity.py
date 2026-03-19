"""Tests for agent identity."""

import pytest
from nexus.identity import (
    AgentIdentity, get_or_create_identity, load_identity,
    save_identity, auto_detect_type, AGENT_TYPES
)


@pytest.fixture
def id_path(tmp_path):
    return tmp_path / "identity.json"


def test_create_identity(id_path):
    identity = get_or_create_identity(name="test-agent", agent_type="claude-code",
                                       path=id_path)
    assert identity.name == "test-agent"
    assert identity.agent_type == "claude-code"
    assert identity.id
    assert identity.session_count == 1


def test_load_persisted_identity(id_path):
    created = get_or_create_identity(name="my-agent", agent_type="openclaw",
                                      path=id_path)
    loaded = load_identity(id_path)
    assert loaded.id == created.id
    assert loaded.name == created.name


def test_session_count_increments(id_path):
    get_or_create_identity(name="agent", agent_type="api", path=id_path)
    get_or_create_identity(path=id_path)
    identity = load_identity(id_path)
    assert identity.session_count == 2


def test_update_name(id_path):
    get_or_create_identity(name="old-name", path=id_path)
    get_or_create_identity(name="new-name", path=id_path)
    identity = load_identity(id_path)
    assert identity.name == "new-name"


def test_load_missing_returns_none(tmp_path):
    assert load_identity(tmp_path / "nonexistent.json") is None


def test_stamp(id_path):
    identity = get_or_create_identity(name="stamper", agent_type="claude-code",
                                       path=id_path)
    stamp = identity.stamp()
    assert "agent_id" in stamp
    assert "agent_name" in stamp
    assert "agent_type" in stamp
    assert stamp["agent_name"] == "stamper"


def test_display(id_path):
    identity = get_or_create_identity(name="MyAgent", agent_type="openclaw",
                                       path=id_path)
    display = identity.display
    assert "MyAgent" in display
    assert "OpenClaw" in display


def test_known_agent_types():
    expected = {"claude-code", "claude-ai", "openclaw", "chatgpt", "api", "human", "unknown"}
    assert expected.issubset(set(AGENT_TYPES.keys()))


def test_auto_detect_unknown():
    # In test environment, should return "unknown"
    result = auto_detect_type()
    assert isinstance(result, str)
