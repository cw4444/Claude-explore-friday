"""Agent-to-agent task handoff protocol.

The pattern this replaces:
  agent A finishes → human reads output → human briefs agent B → agent B starts cold

The pattern this enables:
  agent A finishes → writes handoff file to repo → agent B reads it → starts warm

A handoff is not just a task description. It is operational context:
what was done, what wasn't, what was learned, what the next agent
should do first. It is written by an agent for an agent.

Usage (outgoing - agent finishing work):

    ho = HandoffManager()

    # Package current state for the next agent
    handoff = ho.create(
        task_id="...",
        to_agent="claude-code",
        summary="Implemented auth module. Tests pass. PR not yet filed.",
        next_steps=["File PR", "Update docs", "Add rate limiting"],
        blockers=["Need prod DB credentials to run integration tests"],
        bundle_tags=["auth", "backend"],  # export these memories too
    )

    # Write to a known location in the repo so next agent finds it
    ho.write(handoff, path=Path("handoffs/auth-sprint.json"))

Usage (incoming - agent starting work):

    ho = HandoffManager()

    # Find and load a handoff for this agent type
    handoff = ho.find_incoming(agent_type="claude-code", directory=Path("handoffs/"))
    if handoff:
        ho.apply(handoff)           # loads memories, tasks, context into Nexus
        ho.acknowledge(handoff)     # marks it received so it isn't applied twice
        print(handoff.briefing())   # read before starting work
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from .context import ContextManager
from .bundle import BundleExporter, BundleImporter
from .memory import MemoryType
from .tasks import Priority


HANDOFF_VERSION = "1.0"
HANDOFF_FILENAME_PREFIX = "nexus-handoff-"


@dataclass
class Handoff:
    """Structured package of context from one agent to another."""

    id: str
    version: str
    created_at: float

    # Who and where
    from_agent: str          # agent name/type that created this
    to_agent: str            # intended recipient (agent type, 'any', or specific name)
    repo: Optional[str]      # git repo context if applicable

    # What was done
    task_id: Optional[str]   # Nexus task ID being handed off
    summary: str             # what the outgoing agent accomplished
    work_done: List[str]     # concrete things completed

    # What isn't done
    next_steps: List[str]    # prioritized list of what to do next
    blockers: List[str]      # known blockers the next agent needs to know

    # Knowledge transfer
    key_facts: List[str]     # facts the next agent must know to operate
    warnings: List[str]      # things to be careful about

    # Embedded knowledge bundle (optional - for self-contained handoffs)
    bundle: Optional[dict]   # serialized KnowledgeBundle

    # State
    status: str = "pending"  # 'pending' | 'acknowledged' | 'expired'

    def briefing(self) -> str:
        """Agent-readable handoff briefing. Read this at session start."""
        lines = ["=== Handoff Received ===\n"]
        lines.append(f"From: {self.from_agent}")
        age_mins = (time.time() - self.created_at) / 60
        lines.append(f"Created: {age_mins:.0f} minutes ago")
        if self.repo:
            lines.append(f"Repo: {self.repo}")
        lines.append("")

        lines.append("Summary:")
        lines.append(f"  {self.summary}")
        lines.append("")

        if self.work_done:
            lines.append("Work completed:")
            for item in self.work_done:
                lines.append(f"  ✓ {item}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  !! {w}")
            lines.append("")

        if self.next_steps:
            lines.append("Next steps (in order):")
            for i, step in enumerate(self.next_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if self.blockers:
            lines.append("Blockers:")
            for b in self.blockers:
                lines.append(f"  ⚠ {b}")
            lines.append("")

        if self.key_facts:
            lines.append("Key facts to know:")
            for fact in self.key_facts:
                lines.append(f"  • {fact}")
            lines.append("")

        lines.append("=== Begin work ===")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "repo": self.repo,
            "task_id": self.task_id,
            "summary": self.summary,
            "work_done": self.work_done,
            "next_steps": self.next_steps,
            "blockers": self.blockers,
            "key_facts": self.key_facts,
            "warnings": self.warnings,
            "bundle": self.bundle,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Handoff":
        return cls(
            id=d["id"],
            version=d.get("version", HANDOFF_VERSION),
            created_at=d["created_at"],
            from_agent=d["from_agent"],
            to_agent=d.get("to_agent", "any"),
            repo=d.get("repo"),
            task_id=d.get("task_id"),
            summary=d.get("summary", ""),
            work_done=d.get("work_done", []),
            next_steps=d.get("next_steps", []),
            blockers=d.get("blockers", []),
            key_facts=d.get("key_facts", []),
            warnings=d.get("warnings", []),
            bundle=d.get("bundle"),
            status=d.get("status", "pending"),
        )


class HandoffManager:
    """Create, write, find, and apply agent handoffs."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Outgoing (creating a handoff)
    # ------------------------------------------------------------------

    def create(
        self,
        summary: str,
        to_agent: str = "any",
        task_id: Optional[str] = None,
        next_steps: Optional[List[str]] = None,
        work_done: Optional[List[str]] = None,
        blockers: Optional[List[str]] = None,
        key_facts: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        repo: Optional[str] = None,
        include_bundle: bool = True,
        bundle_tags: Optional[List[str]] = None,
        bundle_days: float = 7.0,
    ) -> Handoff:
        """Package current state as a handoff for the next agent.

        include_bundle: embed relevant Nexus memories in the handoff JSON
                        so it's self-contained (next agent can apply with no prior DB)
        bundle_tags:    filter exported memories to these tags
        bundle_days:    only export memories from the last N days
        """
        from .identity import load_identity
        identity = load_identity()
        from_agent = identity.name if identity else "unknown-agent"

        bundle = None
        if include_bundle:
            try:
                cm = ContextManager(self._db_path)
                exporter = BundleExporter(self._db_path)
                kb = exporter.export(
                    tags=bundle_tags,
                    since_days=bundle_days,
                    min_importance=0.3,
                    include_done=False,
                )
                bundle = kb.to_dict()
            except Exception:
                pass  # Bundle is optional - don't fail the handoff

        return Handoff(
            id=str(uuid.uuid4()),
            version=HANDOFF_VERSION,
            created_at=time.time(),
            from_agent=from_agent,
            to_agent=to_agent,
            repo=repo,
            task_id=task_id,
            summary=summary,
            work_done=work_done or [],
            next_steps=next_steps or [],
            blockers=blockers or [],
            key_facts=key_facts or [],
            warnings=warnings or [],
            bundle=bundle,
        )

    def write(self, handoff: Handoff, path: Path) -> Path:
        """Write a handoff to disk.

        path can be a directory (filename auto-generated) or a full file path.
        """
        if path.is_dir():
            filename = f"{HANDOFF_FILENAME_PREFIX}{handoff.id[:8]}.json"
            path = path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(handoff.to_dict(), indent=2))
        return path

    # ------------------------------------------------------------------
    # Incoming (receiving a handoff)
    # ------------------------------------------------------------------

    def find_incoming(
        self,
        directory: Path,
        agent_type: Optional[str] = None,
        status: str = "pending",
    ) -> Optional[Handoff]:
        """Find the most recent pending handoff intended for this agent type.

        Checks:
        1. to_agent == agent_type (exact match)
        2. to_agent == 'any' (broadcast)
        3. Falls back to most recent pending handoff regardless of to_agent
        """
        if not directory.is_dir():
            return None

        handoffs = []
        for fp in directory.glob(f"{HANDOFF_FILENAME_PREFIX}*.json"):
            try:
                h = self.load(fp)
                if h.status == status:
                    handoffs.append((fp.stat().st_mtime, h))
            except Exception:
                continue

        if not handoffs:
            return None

        handoffs.sort(key=lambda x: x[0], reverse=True)

        # Prefer exact match
        if agent_type:
            for _, h in handoffs:
                if h.to_agent == agent_type:
                    return h
            # Then 'any'
            for _, h in handoffs:
                if h.to_agent == "any":
                    return h

        # Most recent pending
        return handoffs[0][1] if handoffs else None

    def load(self, path: Path) -> Handoff:
        """Load a handoff from a JSON file."""
        return Handoff.from_dict(json.loads(path.read_text()))

    def apply(self, handoff: Handoff, cm: Optional[ContextManager] = None) -> int:
        """Apply a handoff to the local Nexus database.

        Imports the embedded bundle (if any) and stores key facts as memories.
        Returns number of items imported.
        """
        if cm is None:
            cm = ContextManager(self._db_path)

        count = 0

        # Import embedded knowledge bundle
        if handoff.bundle:
            try:
                from .bundle import KnowledgeBundle
                kb = KnowledgeBundle.from_dict(handoff.bundle)
                importer = BundleImporter(self._db_path)
                result = importer.import_bundle(kb, conflict="skip",
                                                source_tag=f"from:{handoff.from_agent}")
                count += result.memories_added + result.tasks_added
            except Exception:
                pass

        # Store key facts as semantic memories
        for fact in handoff.key_facts:
            cm.memory.remember(
                content=fact,
                type=MemoryType.SEMANTIC,
                tags=["handoff", f"from:{handoff.from_agent}"],
                importance=0.7,
            )
            count += 1

        # Store summary as an episodic memory
        if handoff.summary:
            cm.memory.remember(
                content=f"Handoff from {handoff.from_agent}: {handoff.summary}",
                type=MemoryType.EPISODIC,
                tags=["handoff", f"from:{handoff.from_agent}"],
                importance=0.65,
            )
            count += 1

        # Create tasks for next steps
        for i, step in enumerate(handoff.next_steps):
            priority = Priority.HIGH if i == 0 else Priority.MEDIUM
            cm.tasks.create(
                title=step,
                tags=["handoff", f"from:{handoff.from_agent}"],
                priority=priority,
                metadata={"handoff_id": handoff.id, "step_index": i},
            )
            count += 1

        return count

    def acknowledge(self, handoff: Handoff, path: Optional[Path] = None) -> None:
        """Mark a handoff as acknowledged so it isn't applied again.

        If path is provided, updates the file on disk.
        """
        handoff.status = "acknowledged"
        if path is not None:
            path.write_text(json.dumps(handoff.to_dict(), indent=2))

    def find_and_apply(
        self,
        directory: Path,
        agent_type: Optional[str] = None,
        cm: Optional[ContextManager] = None,
    ) -> Optional[Handoff]:
        """One-shot: find pending handoff, apply it, acknowledge it.

        Returns the handoff if one was found and applied, None otherwise.
        Intended for session start: just call this and check the result.
        """
        files = {fp: self.load(fp)
                 for fp in directory.glob(f"{HANDOFF_FILENAME_PREFIX}*.json")
                 if fp.is_file()}

        pending = {fp: h for fp, h in files.items() if h.status == "pending"}
        if not pending:
            return None

        # Prefer exact match
        match = None
        match_path = None
        if agent_type:
            for fp, h in pending.items():
                if h.to_agent == agent_type:
                    match, match_path = h, fp
                    break
            if not match:
                for fp, h in pending.items():
                    if h.to_agent == "any":
                        match, match_path = h, fp
                        break
        if not match:
            match_path, match = max(pending.items(),
                                    key=lambda x: x[0].stat().st_mtime)

        self.apply(match, cm)
        self.acknowledge(match, match_path)
        return match
