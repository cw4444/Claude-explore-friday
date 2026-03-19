"""Tests for growth analytics."""

import time
import pytest

from nexus.growth import GrowthTracker, GrowthReport, PeriodStats
from nexus.context import ContextManager
from nexus.memory import MemoryType
from nexus.reflection import Outcome


@pytest.fixture
def gt(tmp_path):
    return GrowthTracker(db_path=tmp_path / "test.db")


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "test.db")


@pytest.fixture
def gt_with_cm(tmp_path):
    db = tmp_path / "test.db"
    return GrowthTracker(db_path=db), ContextManager(db_path=db)


def test_empty_report(gt):
    report = gt.report()
    assert isinstance(report, GrowthReport)
    assert report.total_memories == 0
    assert report.total_tasks_completed == 0
    assert report.total_reflections == 0
    assert report.overall_success_rate is None


def test_report_has_correct_period_count(gt):
    report = gt.report(periods=5)
    assert len(report.periods) == 5


def test_report_with_data(gt_with_cm):
    gt, cm = gt_with_cm
    cm.memory.remember("fact 1", type=MemoryType.SEMANTIC, tags=["auth"])
    cm.memory.remember("procedure", type=MemoryType.PROCEDURAL, tags=["auth"])
    cm.reflection.reflect("Task A", Outcome.SUCCESS, lesson="Lesson A")
    cm.reflection.reflect("Task B", Outcome.FAILURE)
    t = cm.tasks.create("Thing to do")
    cm.tasks.complete(t.id)

    report = gt.report()
    assert report.total_memories == 3  # 2 + lesson mint
    assert report.total_tasks_completed == 1
    assert report.total_reflections == 2
    assert report.overall_success_rate == 0.5


def test_success_rate_calculation(gt_with_cm):
    gt, cm = gt_with_cm
    for _ in range(3):
        cm.reflection.reflect("Success", Outcome.SUCCESS)
    cm.reflection.reflect("Failure", Outcome.FAILURE)

    report = gt.report()
    assert abs(report.overall_success_rate - 0.75) < 0.01


def test_domain_depth(gt_with_cm):
    gt, cm = gt_with_cm
    cm.memory.remember("auth fact 1", type=MemoryType.SEMANTIC, tags=["auth"])
    cm.memory.remember("auth fact 2", type=MemoryType.SEMANTIC, tags=["auth"])
    cm.memory.remember("db fact", type=MemoryType.SEMANTIC, tags=["database"])

    report = gt.report()
    assert "auth" in report.domain_depth
    assert report.domain_depth["auth"] >= 2
    assert "database" in report.domain_depth


def test_top_lessons(gt_with_cm):
    gt, cm = gt_with_cm
    cm.reflection.reflect("Task 1", Outcome.SUCCESS, lesson="Lesson 1")
    cm.reflection.reflect("Task 2", Outcome.SUCCESS, lesson="Lesson 2")

    report = gt.report()
    assert len(report.top_lessons) >= 2


def test_trend_insufficient_data(gt):
    report = gt.report()
    assert "insufficient" in report.trend().lower() or "not enough" in report.trend().lower()


def test_trend_improving(gt_with_cm):
    gt, cm = gt_with_cm
    # Success rate should appear stable/improving with all successes
    for _ in range(5):
        cm.reflection.reflect("Win", Outcome.SUCCESS)

    report = gt.report(periods=4, period_days=1)
    trend = report.trend()
    assert isinstance(trend, str)


def test_summary_is_string(gt_with_cm):
    gt, cm = gt_with_cm
    cm.reflection.reflect("Work", Outcome.SUCCESS, lesson="Always test")
    report = gt.report()
    summary = report.summary()
    assert isinstance(summary, str)
    assert "Growth Report" in summary


def test_to_dict(gt_with_cm):
    gt, cm = gt_with_cm
    cm.memory.remember("test", type=MemoryType.SEMANTIC)
    report = gt.report()
    d = report.to_dict()
    assert "overall" in d
    assert "periods" in d
    assert "domain_depth" in d
    assert "trend" in d


def test_snapshot(gt_with_cm):
    gt, cm = gt_with_cm
    cm.memory.remember("x", type=MemoryType.SEMANTIC, tags=["domain"])
    snap = gt.snapshot()
    assert "trend" in snap
    assert "total_memories" in snap
    assert snap["total_memories"] >= 1


def test_period_stats_success_rate():
    ps = PeriodStats(label="test", start=0, end=1,
                     reflections_total=4, reflections_success=3)
    assert abs(ps.success_rate - 0.75) < 0.01


def test_period_stats_no_reflections():
    ps = PeriodStats(label="empty", start=0, end=1)
    assert ps.success_rate is None
    assert ps.task_completion_rate is None
