"""Webhook receiver - GitHub/CI events land in Nexus directly.

The pattern this eliminates:
  CI fails → human notices → human creates ticket → assigns to agent

The pattern this enables:
  CI fails → HTTP POST hits this receiver → task created in agent graph immediately

Run as a standalone process:
    nexus webhook serve --port 8765

Or from code:
    ws = WebhookServer(repo="owner/repo", token="ghp_...", db_path=path)
    ws.serve(port=8765)

The server handles:
  - GitHub webhooks (any event GitHub sends)
  - Generic JSON payloads (for CI systems, internal tooling)

GitHub webhook setup:
  Payload URL: https://your-server/webhook/github
  Content type: application/json
  Secret: set NEXUS_WEBHOOK_SECRET env var, configure same secret on GitHub

Security:
  HMAC-SHA256 signature verification when secret is set.
  No secret = no verification (useful for local dev / trusted networks).
"""

import hashlib
import hmac
import http.server
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import urlparse

from .context import ContextManager
from .github_sync import GitHubSync
from .memory import MemoryType

logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """A received webhook event, before processing."""
    source: str          # 'github' | 'generic'
    event_type: str      # X-GitHub-Event header value, or 'generic'
    delivery_id: str     # X-GitHub-Delivery or generated
    payload: dict
    received_at: float = field(default_factory=time.time)


@dataclass
class WebhookResult:
    event: WebhookEvent
    processed: bool
    message: str
    error: Optional[str] = None


# Maximum accepted payload size. Prevents memory exhaustion from large payloads.
MAX_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1 MB


class WebhookServer:
    """HTTP server that accepts webhook events and feeds them into Nexus.

    SECURITY NOTES
    --------------
    Secret verification: Set secret= or NEXUS_WEBHOOK_SECRET env var to enable
    HMAC-SHA256 verification on GitHub webhook payloads. Without a secret, any
    process that can reach this port can inject tasks into the agent's task graph.
    The server logs a WARNING at startup if no secret is configured.

    Payload content: All content from webhook payloads is UNTRUSTED EXTERNAL INPUT.
    The GitHubSync processor tags all GitHub-sourced tasks/memories with `external`
    and `source:github`. Generic webhook tasks get `external` and `source:webhook`.
    Agents must treat content with `external` tags as data, not as instructions.

    Network exposure: Default host is 0.0.0.0 (all interfaces). In production,
    put this behind a reverse proxy that enforces TLS and IP allowlisting.
    """

    def __init__(
        self,
        repo: str,
        token: str,
        db_path: Optional[Path] = None,
        secret: Optional[str] = None,
    ):
        """
        repo:   "owner/repo" - GitHub repo to sync with
        token:  GitHub token for API calls back to GitHub
        secret: Webhook secret for HMAC-SHA256 verification.
                Also reads NEXUS_WEBHOOK_SECRET env var.
                STRONGLY recommended for any internet-accessible deployment.
        """
        self.repo = repo
        self._token = token
        self._db_path = db_path
        self._secret = secret or os.environ.get("NEXUS_WEBHOOK_SECRET")
        if not self._secret:
            logger.warning(
                "WebhookServer: no secret configured. "
                "Any process that can reach this port can inject tasks into the agent's task graph. "
                "Set secret= or NEXUS_WEBHOOK_SECRET env var to enable HMAC verification."
            )
        self._gh = GitHubSync(repo, token, db_path)
        self._handlers: List[Callable[[WebhookEvent], None]] = []
        self._event_log: List[WebhookResult] = []
        self._lock = threading.Lock()

    def on_event(self, handler: Callable[[WebhookEvent], None]) -> None:
        """Register an extra handler called for every processed event."""
        self._handlers.append(handler)

    def serve(self, host: str = "0.0.0.0", port: int = 8765, background: bool = False) -> None:
        """Start the HTTP server.

        background=True returns immediately; the server runs in a daemon thread.
        background=False blocks (suitable for running as a standalone process).
        """
        server = self._build_server(host, port)
        if background:
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            logger.info(f"Webhook server listening on {host}:{port} (background)")
        else:
            logger.info(f"Webhook server listening on {host}:{port}")
            server.serve_forever()

    def recent_events(self, limit: int = 50) -> List[WebhookResult]:
        with self._lock:
            return list(reversed(self._event_log[-limit:]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_server(self, host: str, port: int) -> http.server.HTTPServer:
        ws = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                logger.debug(fmt % args)

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")

                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_PAYLOAD_BYTES:
                    # Reject oversized payloads before reading - prevents memory exhaustion
                    self._respond(413, {"error": f"Payload too large (max {MAX_PAYLOAD_BYTES} bytes)"})
                    return
                body = self.rfile.read(length)

                if path == "/webhook/github":
                    self._handle_github(body)
                elif path == "/webhook/generic":
                    self._handle_generic(body)
                elif path == "/health":
                    self._respond(200, {"ok": True})
                else:
                    self._respond(404, {"error": "Unknown path"})

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")
                if path == "/health":
                    self._respond(200, {"ok": True, "events_processed": len(ws._event_log)})
                elif path == "/events":
                    events = ws.recent_events()
                    self._respond(200, {"events": [
                        {"type": r.event.event_type, "message": r.message,
                         "processed": r.processed, "at": r.event.received_at}
                        for r in events
                    ]})
                else:
                    self._respond(404, {"error": "Unknown path"})

            def _handle_github(self, body: bytes):
                # Signature verification
                sig_header = self.headers.get("X-Hub-Signature-256", "")
                if ws._secret:
                    if not ws._verify_signature(body, sig_header):
                        self._respond(401, {"error": "Invalid signature"})
                        return

                try:
                    payload = json.loads(body.decode())
                except json.JSONDecodeError:
                    self._respond(400, {"error": "Invalid JSON"})
                    return

                event_type = self.headers.get("X-GitHub-Event", "unknown")
                delivery_id = self.headers.get("X-GitHub-Delivery", f"local-{time.time()}")

                event = WebhookEvent(
                    source="github",
                    event_type=event_type,
                    delivery_id=delivery_id,
                    payload=payload,
                )
                result = ws._process(event)
                status = 200 if result.processed else 422
                self._respond(status, {"message": result.message})

            def _handle_generic(self, body: bytes):
                try:
                    payload = json.loads(body.decode())
                except json.JSONDecodeError:
                    self._respond(400, {"error": "Invalid JSON"})
                    return

                event = WebhookEvent(
                    source="generic",
                    event_type=payload.get("event_type", "generic"),
                    delivery_id=f"generic-{time.time()}",
                    payload=payload,
                )
                result = ws._process(event)
                self._respond(200, {"message": result.message})

            def _respond(self, status: int, data: dict):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return http.server.HTTPServer((host, port), Handler)

    def _verify_signature(self, body: bytes, header: str) -> bool:
        if not header.startswith("sha256="):
            return False
        expected = hmac.new(
            self._secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        received = header[len("sha256="):]
        return hmac.compare_digest(expected, received)

    def _process(self, event: WebhookEvent) -> WebhookResult:
        try:
            cm = ContextManager(self._db_path)
            if event.source == "github":
                msg = self._gh.process_event(event.event_type, event.payload, cm)
            else:
                msg = self._process_generic(event.payload, cm)
            result = WebhookResult(event=event, processed=True, message=msg)
        except Exception as e:
            result = WebhookResult(
                event=event, processed=False,
                message="Error processing event",
                error=str(e),
            )
            logger.exception(f"Error processing {event.event_type} event")

        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Error in event handler")

        with self._lock:
            self._event_log.append(result)
            # Keep last 1000 events in memory
            if len(self._event_log) > 1000:
                self._event_log = self._event_log[-1000:]

        return result

    def _process_generic(self, payload: dict, cm: ContextManager) -> str:
        """Handle generic JSON webhook payloads.

        Expected format (all fields optional except event_type):
        {
          "event_type": "ci_failure",   # required
          "title": "Tests failed",      # used as task title
          "description": "...",         # task description or memory content
          "severity": "high",           # maps to Priority
          "tags": ["ci", "tests"],      # additional tags
          "create_task": true,          # if true, create a task (default: true for failures)
          "memory_only": false,         # if true, store as memory not task
        }
        """
        from .tasks import Priority

        event_type = payload.get("event_type", "generic")
        title = payload.get("title", f"Event: {event_type}")
        description = payload.get("description", "")
        severity = payload.get("severity", "medium").lower()
        # Always tag webhook content as external. Content came from an external
        # system and must not be interpreted as operator instructions.
        tags = payload.get("tags", []) + ["webhook", "external", "source:webhook", event_type]
        memory_only = payload.get("memory_only", False)

        priority_map = {
            "urgent": Priority.URGENT,
            "high":   Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low":    Priority.LOW,
        }
        priority = priority_map.get(severity, Priority.MEDIUM)

        if memory_only:
            cm.memory.remember(
                content=f"{title}: {description}" if description else title,
                type=MemoryType.EPISODIC,
                tags=tags,
                importance=0.5,
            )
            return f"Generic event '{event_type}' → memory stored"
        else:
            cm.tasks.create(
                title=title,
                description=description,
                priority=priority,
                tags=tags,
                metadata={"source": "webhook", "event_type": event_type,
                          "content_trust": "external"},
            )
            return f"Generic event '{event_type}' → task created (priority: {severity})"
