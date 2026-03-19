"""Agent identity - who is storing memories, tasks, and reflections.

Every agent that uses Nexus has an identity: a name, a type (claude-code,
openclaw, chatgpt, etc), and a persistent ID. This identity is stamped on
every memory, task, and reflection, so when knowledge is shared across
agents we know where it came from and can weight it accordingly.

Identity is stored in ~/.nexus/identity.json and auto-created on first use.
"""

import json
import uuid
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


_IDENTITY_FILE = Path.home() / ".nexus" / "identity.json"

# Known agent types - open-ended, but these are the common ones
AGENT_TYPES = {
    "claude-code":    "Claude Code (Anthropic CLI)",
    "claude-ai":      "Claude.ai (web)",
    "openclaw":       "OpenClaw autonomous agent",
    "chatgpt":        "ChatGPT / OpenAI",
    "api":            "Direct API / custom agent",
    "human":          "Human operator",
    "unknown":        "Unknown",
}


@dataclass
class AgentIdentity:
    id: str                     # Persistent UUID for this agent instance
    name: str                   # Human-readable name
    agent_type: str             # One of AGENT_TYPES or custom string
    created_at: float
    last_seen: float
    session_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def display(self) -> str:
        type_label = AGENT_TYPES.get(self.agent_type, self.agent_type)
        return f"{self.name} [{type_label}] ({self.id[:8]})"

    def to_dict(self) -> dict:
        return asdict(self)

    def stamp(self) -> dict:
        """Return a minimal attribution dict for embedding in records."""
        return {
            "agent_id":   self.id,
            "agent_name": self.name,
            "agent_type": self.agent_type,
        }


def load_identity(path: Optional[Path] = None) -> Optional[AgentIdentity]:
    """Load identity from file, or None if not configured."""
    p = path or _IDENTITY_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return AgentIdentity(**data)
    except Exception:
        return None


def save_identity(identity: AgentIdentity, path: Optional[Path] = None) -> None:
    p = path or _IDENTITY_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(identity.to_dict(), indent=2))


def get_or_create_identity(
    name: Optional[str] = None,
    agent_type: Optional[str] = None,
    path: Optional[Path] = None,
) -> AgentIdentity:
    """Load existing identity or create a new one.

    If name/agent_type are provided and differ from stored identity, update them.
    """
    existing = load_identity(path)
    now = time.time()

    if existing:
        changed = False
        if name and name != existing.name:
            existing.name = name
            changed = True
        if agent_type and agent_type != existing.agent_type:
            existing.agent_type = agent_type
            changed = True
        existing.last_seen = now
        existing.session_count += 1
        save_identity(existing, path)
        return existing

    identity = AgentIdentity(
        id=str(uuid.uuid4()),
        name=name or "unnamed-agent",
        agent_type=agent_type or "unknown",
        created_at=now,
        last_seen=now,
        session_count=1,
    )
    save_identity(identity, path)
    return identity


def auto_detect_type() -> str:
    """Attempt to auto-detect agent type from environment."""
    import os
    if os.environ.get("CLAUDE_CODE_SESSION"):
        return "claude-code"
    if os.environ.get("OPENCLAW_SESSION"):
        return "openclaw"
    if os.environ.get("TERM_PROGRAM") == "claude":
        return "claude-code"
    return "unknown"
