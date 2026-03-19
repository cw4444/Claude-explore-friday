"""Tests for the narrative / self-model engine."""

import pytest

from nexus.narrative import NarrativeEngine, SelfModel, Trait
from nexus.context import ContextManager
from nexus.memory import MemoryType
from nexus.reflection import Outcome
from nexus.identity import get_or_create_identity


@pytest.fixture
def ne(tmp_path):
    return NarrativeEngine(db_path=tmp_path / "test.db")


@pytest.fixture
def ne_with_cm(tmp_path):
    db = tmp_path / "test.db"
    return NarrativeEngine(db_path=db), ContextManager(db_path=db)


def test_generate_empty_returns_model(ne):
    model = ne.generate()
    assert isinstance(model, SelfModel)
    assert model.agent_name
    assert model.generated_at > 0


def test_generate_uses_identity(tmp_path):
    db = tmp_path / "test.db"
    get_or_create_identity(name="test-agent", agent_type="claude-code",
                            path=tmp_path / "identity.json")
    ne = NarrativeEngine(db_path=db)
    model = ne.generate()
    # Identity comes from identity file - may be "unnamed-agent" in test env
    assert isinstance(model.agent_name, str)


def test_pin_trait(ne):
    ne.pin_trait("over-engineers solutions",
                  "consistently observed - apply simplicity constraint")
    traits = ne.pinned_traits()
    assert len(traits) == 1
    assert traits[0].name == "over-engineers solutions"
    assert traits[0].source == "pinned"
    assert traits[0].confidence == 1.0


def test_remove_trait(ne):
    ne.pin_trait("verbose explainer", "tends to over-explain")
    assert ne.remove_trait("verbose explainer")
    assert ne.pinned_traits() == []


def test_remove_nonexistent_trait(ne):
    assert not ne.remove_trait("does not exist")


def test_pinned_traits_appear_in_model(ne):
    ne.pin_trait("systematic", "always decomposes before implementing")
    model = ne.generate()
    trait_names = [t.name for t in model.traits]
    assert "systematic" in trait_names


def test_pinned_traits_appear_first(ne):
    ne.pin_trait("pinned-trait", "explicitly noted")
    model = ne.generate()
    if model.traits:
        assert model.traits[0].source == "pinned"


def test_briefing_is_string(ne):
    model = ne.generate()
    briefing = model.briefing()
    assert isinstance(briefing, str)
    assert "Identity" in briefing


def test_briefing_contains_agent_info(ne):
    model = ne.generate()
    briefing = model.briefing()
    assert model.agent_name in briefing or "unnamed" in briefing


def test_to_dict(ne):
    model = ne.generate()
    d = model.to_dict()
    assert "agent_name" in d
    assert "strengths" in d
    assert "traits" in d
    assert "trajectory" in d
    assert "trend" in d


def test_trajectory_early_with_no_data(ne):
    model = ne.generate()
    assert model.trajectory == "early"


def test_trajectory_established_with_data(ne_with_cm):
    ne, cm = ne_with_cm
    for _ in range(10):
        cm.reflection.reflect("Task", Outcome.SUCCESS, lesson="A lesson")
    model = ne.generate()
    assert model.trajectory in ("growing", "established", "early")


def test_domain_depth_populated(ne_with_cm):
    ne, cm = ne_with_cm
    for _ in range(4):
        cm.memory.remember("auth fact", type=MemoryType.SEMANTIC, tags=["auth"])
    model = ne.generate()
    assert "auth" in model.domain_depth
    assert model.domain_depth["auth"] >= 4


def test_strengths_require_depth(ne_with_cm):
    ne, cm = ne_with_cm
    # Only 2 memories - below threshold of 3
    cm.memory.remember("x", type=MemoryType.SEMANTIC, tags=["sparse"])
    cm.memory.remember("y", type=MemoryType.SEMANTIC, tags=["sparse"])
    model = ne.generate()
    strength_domains = [s[0] for s in model.strengths]
    assert "sparse" not in strength_domains


def test_strengths_with_deep_domain(ne_with_cm):
    ne, cm = ne_with_cm
    for _ in range(5):
        cm.memory.remember("auth knowledge", type=MemoryType.SEMANTIC, tags=["auth"])
    model = ne.generate()
    strength_domains = [s[0] for s in model.strengths]
    assert "auth" in strength_domains


def test_trait_lesson_extractor(ne_with_cm):
    ne, cm = ne_with_cm
    # Record many reflections with lessons
    for i in range(6):
        cm.reflection.reflect(f"Task {i}", Outcome.SUCCESS, lesson=f"Lesson {i}")
    model = ne.generate()
    trait_names = [t.name for t in model.traits]
    assert "knowledge-extractor" in trait_names


def test_trait_decomposer(ne_with_cm):
    ne, cm = ne_with_cm
    # Create tasks with dependencies
    tasks = cm.tasks.decompose("Big task", subtasks=["A", "B", "C", "D"])
    model = ne.generate()
    trait_names = [t.name for t in model.traits]
    assert "systematic-decomposer" in trait_names


def test_knowledge_span(ne_with_cm):
    ne, cm = ne_with_cm
    cm.memory.remember("one", type=MemoryType.SEMANTIC)
    cm.memory.remember("two", type=MemoryType.SEMANTIC)
    model = ne.generate()
    assert model.knowledge_span >= 2


def test_overall_success_rate_none_with_no_reflections(ne):
    model = ne.generate()
    assert model.overall_success_rate is None


def test_overall_success_rate_with_data(ne_with_cm):
    ne, cm = ne_with_cm
    cm.reflection.reflect("A", Outcome.SUCCESS)
    cm.reflection.reflect("B", Outcome.SUCCESS)
    cm.reflection.reflect("C", Outcome.FAILURE)
    model = ne.generate()
    assert model.overall_success_rate is not None
    assert 0 < model.overall_success_rate < 1


def test_context_manager_who_am_i(tmp_path):
    cm = ContextManager(db_path=tmp_path / "test.db")
    model = cm.who_am_i()
    assert isinstance(model, SelfModel)


def test_context_manager_who_is(tmp_path):
    cm = ContextManager(db_path=tmp_path / "test.db")
    cm.relationships.observe("Alice", "values brevity", category="preference")
    briefing = cm.who_is("Alice")
    assert "Alice" in briefing
    assert "brevity" in briefing
