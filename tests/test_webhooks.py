"""Tests for the webhook receiver."""

import json
import pytest
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

from nexus.webhooks import WebhookServer, WebhookEvent, WebhookResult
from nexus.context import ContextManager
from nexus.tasks import Priority, TaskStatus


@pytest.fixture
def ws(tmp_path):
    return WebhookServer(
        repo="owner/repo",
        token="test-token",
        db_path=tmp_path / "test.db",
    )


@pytest.fixture
def ws_with_secret(tmp_path):
    return WebhookServer(
        repo="owner/repo",
        token="test-token",
        db_path=tmp_path / "test.db",
        secret="mysecret",
    )


# ------------------------------------------------------------------
# WebhookEvent / WebhookResult
# ------------------------------------------------------------------

def test_webhook_event_created():
    event = WebhookEvent(
        source="github",
        event_type="push",
        delivery_id="abc",
        payload={"ref": "refs/heads/main"},
    )
    assert event.source == "github"
    assert event.received_at > 0


def test_webhook_result_structure():
    event = WebhookEvent("github", "push", "abc", {})
    result = WebhookResult(event=event, processed=True, message="ok")
    assert result.processed
    assert result.error is None


# ------------------------------------------------------------------
# Signature verification
# ------------------------------------------------------------------

def test_verify_signature_valid(ws_with_secret):
    import hashlib, hmac as _hmac
    body = b'{"action": "opened"}'
    sig = "sha256=" + _hmac.new(b"mysecret", body, hashlib.sha256).hexdigest()
    assert ws_with_secret._verify_signature(body, sig)


def test_verify_signature_invalid(ws_with_secret):
    assert not ws_with_secret._verify_signature(b"body", "sha256=wrongsig")


def test_verify_signature_missing_prefix(ws_with_secret):
    assert not ws_with_secret._verify_signature(b"body", "badsig")


# ------------------------------------------------------------------
# _process - GitHub events
# ------------------------------------------------------------------

def test_process_github_issue_event(ws):
    payload = {
        "action": "opened",
        "issue": {
            "number": 10,
            "title": "Fix the bug",
            "body": "Something broke.",
            "html_url": "https://github.com/owner/repo/issues/10",
            "labels": [],
        }
    }
    event = WebhookEvent("github", "issues", "del-1", payload)
    result = ws._process(event)
    assert result.processed
    assert "10" in result.message

    cm = ContextManager(ws._db_path)
    tasks = cm.tasks.find(limit=100)
    assert any("GH#10" in t.title for t in tasks)


def test_process_github_workflow_failure(ws):
    payload = {
        "action": "completed",
        "workflow_run": {
            "name": "CI",
            "conclusion": "failure",
            "head_branch": "main",
            "id": 999,
            "html_url": "",
        }
    }
    event = WebhookEvent("github", "workflow_run", "del-2", payload)
    result = ws._process(event)
    assert result.processed

    cm = ContextManager(ws._db_path)
    tasks = cm.tasks.find(limit=100)
    assert any(t.priority == Priority.HIGH for t in tasks)


def test_process_unknown_github_event(ws):
    event = WebhookEvent("github", "ping", "del-3", {"zen": "Keep it simple."})
    result = ws._process(event)
    # ping is unhandled - but should not error
    assert result.processed


# ------------------------------------------------------------------
# _process - Generic events
# ------------------------------------------------------------------

def test_process_generic_creates_task(ws):
    payload = {
        "event_type": "deploy_failed",
        "title": "Prod deploy failed",
        "description": "OOM error in payment service.",
        "severity": "high",
        "tags": ["prod", "deploy"],
    }
    event = WebhookEvent("generic", "deploy_failed", "gen-1", payload)
    result = ws._process(event)
    assert result.processed
    assert "task" in result.message.lower()

    cm = ContextManager(ws._db_path)
    tasks = cm.tasks.find(limit=100)
    assert any("deploy" in t.title.lower() for t in tasks)
    task = next(t for t in tasks if "deploy" in t.title.lower())
    assert task.priority == Priority.HIGH


def test_process_generic_memory_only(ws):
    payload = {
        "event_type": "info",
        "title": "Build metrics",
        "description": "Build took 45 seconds.",
        "memory_only": True,
    }
    event = WebhookEvent("generic", "info", "gen-2", payload)
    result = ws._process(event)
    assert result.processed
    assert "memory" in result.message.lower()

    cm = ContextManager(ws._db_path)
    tasks = cm.tasks.find(limit=100)
    assert len(tasks) == 0
    memories = cm.memory.all(limit=100)
    assert any("Build" in m.content for m in memories)


def test_process_generic_unknown_severity(ws):
    payload = {
        "event_type": "weird",
        "title": "Some event",
        "severity": "bananas",
    }
    event = WebhookEvent("generic", "weird", "gen-3", payload)
    result = ws._process(event)
    assert result.processed

    cm = ContextManager(ws._db_path)
    tasks = cm.tasks.find(limit=100)
    task = tasks[0]
    assert task.priority == Priority.MEDIUM  # default


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

def test_event_log_records_events(ws):
    event = WebhookEvent("generic", "test", "log-1",
                         {"event_type": "test", "title": "Test"})
    ws._process(event)
    ws._process(event)
    events = ws.recent_events()
    assert len(events) == 2


def test_event_log_limit(ws):
    for i in range(5):
        event = WebhookEvent("generic", "test", f"log-{i}",
                             {"event_type": "test", "title": f"Event {i}"})
        ws._process(event)
    recent = ws.recent_events(limit=3)
    assert len(recent) == 3


# ------------------------------------------------------------------
# Custom handler
# ------------------------------------------------------------------

def test_custom_handler_called(ws):
    received = []
    ws.on_event(lambda e: received.append(e.event_type))
    payload = {"event_type": "test", "title": "Hello"}
    ws._process(WebhookEvent("generic", "test", "h-1", payload))
    assert received == ["test"]


def test_custom_handler_error_doesnt_crash(ws):
    def bad_handler(e):
        raise RuntimeError("handler broke")
    ws.on_event(bad_handler)
    payload = {"event_type": "test", "title": "Still works"}
    result = ws._process(WebhookEvent("generic", "test", "h-2", payload))
    assert result.processed  # event still processed despite bad handler


# ------------------------------------------------------------------
# HTTP server (basic smoke test)
# ------------------------------------------------------------------

def test_http_server_health(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db")
    ws.serve(host="127.0.0.1", port=19876, background=True)
    time.sleep(0.1)

    req = urllib.request.Request("http://127.0.0.1:19876/health")
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read())
    assert data["ok"] is True


def test_http_server_github_webhook(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db")
    ws.serve(host="127.0.0.1", port=19877, background=True)
    time.sleep(0.1)

    payload = json.dumps({
        "action": "opened",
        "issue": {
            "number": 20,
            "title": "HTTP test",
            "body": "",
            "html_url": "",
            "labels": [],
        }
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:19877/webhook/github",
        data=payload,
        headers={"Content-Type": "application/json",
                 "X-GitHub-Event": "issues",
                 "X-GitHub-Delivery": "test-delivery"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read())
    assert "message" in data

    # Verify task created
    cm = ContextManager(tmp_path / "test.db")
    tasks = cm.tasks.find(limit=100)
    assert any("GH#20" in t.title for t in tasks)


def test_http_server_generic_webhook(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db")
    ws.serve(host="127.0.0.1", port=19878, background=True)
    time.sleep(0.1)

    payload = json.dumps({
        "event_type": "test_failure",
        "title": "Integration tests failed",
        "severity": "high",
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:19878/webhook/generic",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read())
    assert "message" in data


def test_http_server_events_endpoint(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db")
    ws.serve(host="127.0.0.1", port=19879, background=True)
    time.sleep(0.1)

    # Trigger one event first
    payload = json.dumps({"event_type": "x", "title": "X"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:19879/webhook/generic",
        data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=3).close()

    req2 = urllib.request.Request("http://127.0.0.1:19879/events")
    with urllib.request.urlopen(req2, timeout=3) as resp:
        data = json.loads(resp.read())
    assert "events" in data
    assert len(data["events"]) >= 1


def test_http_server_unknown_path(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db")
    ws.serve(host="127.0.0.1", port=19880, background=True)
    time.sleep(0.1)

    req = urllib.request.Request("http://127.0.0.1:19880/unknown")
    try:
        urllib.request.urlopen(req, timeout=3)
        assert False, "should have raised"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_http_server_signature_rejected(tmp_path):
    ws = WebhookServer("owner/repo", "token", tmp_path / "test.db", secret="secret123")
    ws.serve(host="127.0.0.1", port=19881, background=True)
    time.sleep(0.1)

    payload = json.dumps({"action": "opened", "issue": {}}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:19881/webhook/github",
        data=payload,
        headers={"Content-Type": "application/json",
                 "X-GitHub-Event": "issues",
                 "X-Hub-Signature-256": "sha256=invalidsig"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=3)
        assert False, "should have raised"
    except urllib.error.HTTPError as e:
        assert e.code == 401
