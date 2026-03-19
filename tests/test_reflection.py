"""Tests for the reflection engine."""

import pytest

from nexus.reflection import ReflectionEngine, Reflection, Outcome
from nexus.memory import MemoryStore, MemoryType


@pytest.fixture
def re(tmp_path):
    return ReflectionEngine(db_path=tmp_path / "test.db")


def test_reflect_returns_reflection(re):
    ref = re.reflect(
        task_title="Fix auth bug",
        outcome=Outcome.SUCCESS,
        lesson="Check JWT expiry first",
    )
    assert isinstance(ref, Reflection)
    assert ref.task_title == "Fix auth bug"
    assert ref.outcome == Outcome.SUCCESS
    assert ref.id


def test_reflect_mints_procedural_memory(re):
    re.reflect(
        task_title="Fix payment bug",
        outcome=Outcome.SUCCESS,
        lesson="Always validate currency code before charging",
    )
    # The lesson should appear as procedural memory
    mems = re._mem.recall("validate currency", type=MemoryType.PROCEDURAL)
    assert len(mems) >= 1
    assert "validate currency" in mems[0].content.lower()


def test_reflect_no_lesson_no_memory(re):
    re.reflect(
        task_title="Some task",
        outcome=Outcome.SUCCESS,
        what_worked="stuff",
        # no lesson
    )
    procedural = re._mem.all(type=MemoryType.PROCEDURAL)
    assert len(procedural) == 0


def test_reflect_stores_all_fields(re):
    ref = re.reflect(
        task_title="Complex task",
        outcome=Outcome.PARTIAL,
        what_worked="Breaking into subtasks",
        what_didnt="Initial design",
        lesson="Design before coding",
        effort_mins=90,
    )
    fetched = re.get(ref.id)
    assert fetched.what_worked == "Breaking into subtasks"
    assert fetched.what_didnt == "Initial design"
    assert fetched.lesson == "Design before coding"
    assert fetched.effort_mins == 90


def test_get_missing_returns_none(re):
    assert re.get("nonexistent") is None


def test_recent(re):
    re.reflect("Task 1", Outcome.SUCCESS)
    re.reflect("Task 2", Outcome.FAILURE)
    re.reflect("Task 3", Outcome.PARTIAL)
    refs = re.recent(limit=10)
    assert len(refs) == 3


def test_recent_outcome_filter(re):
    re.reflect("Task 1", Outcome.SUCCESS)
    re.reflect("Task 2", Outcome.FAILURE)
    re.reflect("Task 3", Outcome.SUCCESS)
    successes = re.recent(outcome=Outcome.SUCCESS)
    assert len(successes) == 2
    assert all(r.outcome == Outcome.SUCCESS for r in successes)


def test_lessons(re):
    re.reflect("T1", Outcome.SUCCESS, lesson="Lesson A")
    re.reflect("T2", Outcome.FAILURE, lesson="Lesson B")
    re.reflect("T3", Outcome.SUCCESS)  # no lesson
    lessons = re.lessons()
    assert "Lesson A" in lessons
    assert "Lesson B" in lessons
    assert len(lessons) == 2


def test_stats(re):
    re.reflect("T1", Outcome.SUCCESS, effort_mins=30)
    re.reflect("T2", Outcome.FAILURE, effort_mins=60)
    re.reflect("T3", Outcome.SUCCESS, effort_mins=45)
    s = re.stats()
    assert s["total"] == 3
    assert s["by_outcome"]["success"] == 2
    assert s["by_outcome"]["failure"] == 1
    assert s["avg_effort_mins"] == 45.0


def test_failure_reduces_memory_importance(re):
    re.reflect(
        task_title="Failed approach",
        outcome=Outcome.FAILURE,
        lesson="Don't do X",
        importance=0.7,
    )
    mems = re._mem.all(type=MemoryType.PROCEDURAL)
    assert len(mems) == 1
    # failure should reduce importance below base
    assert mems[0].importance < 0.7


def test_success_boosts_memory_importance(re):
    re.reflect(
        task_title="Successful approach",
        outcome=Outcome.SUCCESS,
        lesson="Always do Y",
        importance=0.7,
    )
    mems = re._mem.all(type=MemoryType.PROCEDURAL)
    assert len(mems) == 1
    # success should boost importance above base
    assert mems[0].importance > 0.7


def test_reflection_age_days(re):
    ref = re.reflect("Test", Outcome.SUCCESS)
    assert ref.age_days < 1.0
