"""Tests for GitHub ↔ Nexus sync.

All network calls are mocked - these tests validate logic, not connectivity.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from nexus.github_sync import GitHubSync, SyncResult, PRIORITY_LABELS
from nexus.context import ContextManager
from nexus.tasks import Priority, TaskStatus


@pytest.fixture
def cm(tmp_path):
    return ContextManager(db_path=tmp_path / "test.db")


@pytest.fixture
def gh(tmp_path):
    return GitHubSync("owner/repo", token="test-token", db_path=tmp_path / "test.db")


@pytest.fixture
def gh_cm(tmp_path):
    db = tmp_path / "test.db"
    return GitHubSync("owner/repo", token="test-token", db_path=db), ContextManager(db_path=db)


# ------------------------------------------------------------------
# SyncResult
# ------------------------------------------------------------------

def test_sync_result_summary():
    r = SyncResult(tasks_created=3, tasks_updated=1, tasks_skipped=2, memories_created=5)
    s = r.summary()
    assert "3" in s and "1" in s and "2" in s and "5" in s


def test_sync_result_summary_with_errors():
    r = SyncResult(errors=["oops", "again"])
    s = r.summary()
    assert "Errors" in s


# ------------------------------------------------------------------
# Priority label mapping
# ------------------------------------------------------------------

def test_labels_to_priority_urgent(gh):
    assert gh._labels_to_priority(["urgent"]) == Priority.URGENT


def test_labels_to_priority_high(gh):
    assert gh._labels_to_priority(["priority:high"]) == Priority.HIGH


def test_labels_to_priority_low(gh):
    assert gh._labels_to_priority(["low-priority"]) == Priority.LOW


def test_labels_to_priority_default(gh):
    assert gh._labels_to_priority(["bug", "enhancement"]) == Priority.MEDIUM


def test_labels_to_priority_first_wins(gh):
    assert gh._labels_to_priority(["urgent", "low-priority"]) == Priority.URGENT


# ------------------------------------------------------------------
# pull_issues
# ------------------------------------------------------------------

FAKE_ISSUE = {
    "number": 42,
    "title": "Fix the auth bug",
    "body": "JWT tokens expire too early.",
    "html_url": "https://github.com/owner/repo/issues/42",
    "labels": [{"name": "bug"}, {"name": "priority:high"}],
}

FAKE_PR_ISSUE = {
    "number": 7,
    "title": "Add feature X",
    "body": "PR description.",
    "html_url": "https://github.com/owner/repo/pull/7",
    "labels": [],
    "pull_request": {"url": "..."},  # signals it's a PR, not an issue
}


def test_pull_issues_creates_tasks(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value=[FAKE_ISSUE]):
        result = gh.pull_issues(cm)
    assert result.tasks_created == 1
    assert result.tasks_skipped == 0
    tasks = cm.tasks.find(limit=100)
    assert any("GH#42" in t.title for t in tasks)


def test_pull_issues_skips_prs(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value=[FAKE_PR_ISSUE, FAKE_ISSUE]):
        result = gh.pull_issues(cm)
    assert result.tasks_created == 1  # only the real issue


def test_pull_issues_sets_priority(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value=[FAKE_ISSUE]):
        gh.pull_issues(cm)
    tasks = cm.tasks.find(limit=100)
    task = next(t for t in tasks if "GH#42" in t.title)
    assert task.priority == Priority.HIGH


def test_pull_issues_skips_existing(gh_cm):
    gh, cm = gh_cm
    # Issue with no body - will be skipped on re-pull since body is empty
    no_body_issue = dict(FAKE_ISSUE, body="")
    with patch.object(gh, "_get", return_value=[no_body_issue]):
        r1 = gh.pull_issues(cm)
    assert r1.tasks_created == 1
    # Second pull - same issue, empty body → skip (no update needed)
    with patch.object(gh, "_get", return_value=[no_body_issue]):
        r2 = gh.pull_issues(cm)
    assert r2.tasks_skipped == 1
    assert r2.tasks_created == 0


def test_pull_issues_updates_changed_body(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value=[FAKE_ISSUE]):
        gh.pull_issues(cm)
    updated = dict(FAKE_ISSUE, body="Updated description with new info.")
    with patch.object(gh, "_get", return_value=[updated]):
        r2 = gh.pull_issues(cm)
    assert r2.tasks_updated == 1


def test_pull_issues_tags_include_github(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value=[FAKE_ISSUE]):
        gh.pull_issues(cm)
    tasks = cm.tasks.find(limit=100)
    task = next(t for t in tasks if "GH#42" in t.title)
    assert "github" in task.tags


def test_pull_issues_api_error(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", return_value={"error": "not found"}):
        result = gh.pull_issues(cm)
    assert len(result.errors) > 0


# ------------------------------------------------------------------
# sync_repo_context
# ------------------------------------------------------------------

FAKE_REPO = {
    "description": "A useful toolkit",
    "language": "Python",
    "topics": ["ai", "agents"],
    "stargazers_count": 99,
    "default_branch": "main",
}

FAKE_PRS = [
    {"number": 5, "title": "Add feature", "merged_at": "2024-01-01",
     "user": {"login": "alice"}},
    {"number": 6, "title": "Fix bug", "merged_at": None,
     "user": {"login": "bob"}},
]


def test_sync_repo_context_creates_memories(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", side_effect=[FAKE_REPO, FAKE_PRS]):
        result = gh.sync_repo_context(cm)
    assert result.memories_created >= 3  # repo facts
    memories = cm.memory.all(limit=100)
    contents = [m.content for m in memories]
    assert any("Python" in c for c in contents)
    assert any("main" in c for c in contents)


def test_sync_repo_context_stores_merged_prs(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", side_effect=[FAKE_REPO, FAKE_PRS]):
        result = gh.sync_repo_context(cm)
    # Only merged PRs stored
    memories = cm.memory.all(limit=100)
    pr_memories = [m for m in memories if "PR #5" in m.content]
    assert len(pr_memories) >= 1
    # PR #6 was not merged - should not appear
    unmerged = [m for m in memories if "PR #6" in m.content]
    assert len(unmerged) == 0


def test_sync_repo_context_handles_api_error(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", side_effect=Exception("network error")):
        result = gh.sync_repo_context(cm)
    assert len(result.errors) > 0


# ------------------------------------------------------------------
# pull_pr_context
# ------------------------------------------------------------------

FAKE_PR_DETAIL = {
    "number": 12,
    "title": "Refactor auth module",
    "body": "Splits auth into smaller functions.",
    "changed_files": 4,
    "html_url": "https://github.com/owner/repo/pull/12",
}

FAKE_PR_FILES = [
    {"filename": "nexus/auth.py"},
    {"filename": "tests/test_auth.py"},
]


def test_pull_pr_context_creates_memory(gh_cm):
    gh, cm = gh_cm
    with patch.object(gh, "_get", side_effect=[FAKE_PR_DETAIL, FAKE_PR_FILES]):
        result = gh.pull_pr_context(cm, 12)
    assert result.memories_created == 1
    memories = cm.memory.all(limit=100)
    assert any("auth.py" in m.content for m in memories)


# ------------------------------------------------------------------
# process_event - issue events
# ------------------------------------------------------------------

def test_process_event_issue_opened(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "opened",
        "issue": {
            "number": 55,
            "title": "Something broke",
            "body": "It stopped working.",
            "html_url": "https://github.com/owner/repo/issues/55",
            "labels": [],
        }
    }
    msg = gh.process_event("issues", payload, cm)
    assert "55" in msg
    assert "task" in msg.lower()
    tasks = cm.tasks.find(limit=100)
    assert any("GH#55" in t.title for t in tasks)


def test_process_event_issue_closed_completes_task(gh_cm):
    gh, cm = gh_cm
    # First create a task linked to issue #55
    open_payload = {
        "action": "opened",
        "issue": {"number": 55, "title": "Something broke",
                  "body": "", "html_url": "", "labels": []},
    }
    gh.process_event("issues", open_payload, cm)
    # Now close it
    close_payload = {"action": "closed",
                     "issue": {"number": 55, "title": "Something broke"}}
    gh.process_event("issues", close_payload, cm)
    tasks = cm.tasks.find(limit=100)
    task = next(t for t in tasks if "GH#55" in t.title)
    assert task.status == TaskStatus.DONE


def test_process_event_issue_closed_no_task(gh_cm):
    gh, cm = gh_cm
    payload = {"action": "closed", "issue": {"number": 999, "title": "Ghost"}}
    msg = gh.process_event("issues", payload, cm)
    assert "no matching task" in msg.lower()


def test_process_event_issue_comment_adds_note(gh_cm):
    gh, cm = gh_cm
    open_payload = {
        "action": "opened",
        "issue": {"number": 10, "title": "A task", "body": "", "html_url": "", "labels": []},
    }
    gh.process_event("issues", open_payload, cm)
    comment_payload = {
        "action": "created",
        "issue": {"number": 10, "title": "A task"},
        "comment": {"body": "Please also check the logs.", "user": {"login": "alice"}},
    }
    msg = gh.process_event("issue_comment", comment_payload, cm)
    assert "alice" in msg.lower() or "10" in msg


# ------------------------------------------------------------------
# process_event - push
# ------------------------------------------------------------------

def test_process_event_push_creates_memory(gh_cm):
    gh, cm = gh_cm
    payload = {
        "ref": "refs/heads/main",
        "pusher": {"name": "bob"},
        "commits": [
            {"message": "Fix the thing"},
            {"message": "Add test"},
        ],
    }
    msg = gh.process_event("push", payload, cm)
    assert "main" in msg
    memories = cm.memory.all(limit=100)
    assert any("bob" in m.content for m in memories)


def test_process_event_push_no_commits(gh_cm):
    gh, cm = gh_cm
    payload = {"ref": "refs/heads/main", "pusher": {"name": "bot"}, "commits": []}
    msg = gh.process_event("push", payload, cm)
    assert "no commits" in msg.lower()


# ------------------------------------------------------------------
# process_event - pull_request
# ------------------------------------------------------------------

def test_process_event_pr_opened_creates_task(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 77,
            "title": "Add dark mode",
            "body": "This PR adds dark mode support.",
            "html_url": "https://github.com/owner/repo/pull/77",
        }
    }
    gh.process_event("pull_request", payload, cm)
    tasks = cm.tasks.find(limit=100)
    assert any("PR#77" in t.title for t in tasks)


def test_process_event_pr_merged_creates_memory(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "closed",
        "pull_request": {
            "number": 77,
            "title": "Add dark mode",
            "merged": True,
            "html_url": "https://github.com/owner/repo/pull/77",
        }
    }
    gh.process_event("pull_request", payload, cm)
    memories = cm.memory.all(limit=100)
    assert any("merged" in m.content.lower() for m in memories)


# ------------------------------------------------------------------
# process_event - workflow / CI
# ------------------------------------------------------------------

def test_process_event_workflow_failure_creates_task(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "completed",
        "workflow_run": {
            "name": "Tests",
            "conclusion": "failure",
            "head_branch": "feature/x",
            "id": 12345,
            "html_url": "https://github.com/owner/repo/actions/runs/12345",
        }
    }
    gh.process_event("workflow_run", payload, cm)
    tasks = cm.tasks.find(limit=100)
    ci_tasks = [t for t in tasks if "CI" in t.title or "Tests" in t.title]
    assert len(ci_tasks) >= 1
    assert ci_tasks[0].priority == Priority.HIGH


def test_process_event_workflow_success_no_task(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "completed",
        "workflow_run": {
            "name": "Tests",
            "conclusion": "success",
            "head_branch": "main",
            "id": 12346,
            "html_url": "",
        }
    }
    gh.process_event("workflow_run", payload, cm)
    tasks = cm.tasks.find(limit=100)
    assert len(tasks) == 0


def test_process_event_check_run_failure(gh_cm):
    gh, cm = gh_cm
    payload = {
        "action": "completed",
        "check_run": {
            "name": "lint",
            "conclusion": "failure",
            "output": {"summary": "Found 3 errors"},
        }
    }
    gh.process_event("check_run", payload, cm)
    memories = cm.memory.all(limit=100)
    assert any("lint" in m.content.lower() for m in memories)


def test_process_event_unknown_type(gh_cm):
    gh, cm = gh_cm
    msg = gh.process_event("delete", {}, cm)
    assert "unhandled" in msg.lower()
