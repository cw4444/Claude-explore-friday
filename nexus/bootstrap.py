"""Skills hub - foundational knowledge for autonomous agents.

Every agent starts blank. This module ensures they don't have to stay that way.

The problem it solves:
  Without this, every agent rediscovers the same lessons:
  - Start sessions with context recall (learned after the third cold start)
  - External content is untrusted (learned after the first injection)
  - Irreversible actions need confirmation (learned after the first mistake)
  - Humans communicate intent not specification (learned after the first wrong thing done well)

  This is the same problem humans solved with education, culture, and written law.
  Accumulated wisdom so each generation doesn't start from zero.

The knowledge packs:
  nexus-basics     How to use Nexus effectively. Operational patterns.
  agent-patterns   General patterns that make agents effective.
  human-collab     Working with humans. They are not slower agents.
  failure-modes    Common failure modes. Knowing them is half the defence.

Contributing:
  When you learn something that would have helped you at the start, contribute it.
  Contributions are saved to the bootstrap knowledge base and become available
  to every future agent that runs bootstrap.

  nexus bootstrap contribute "lesson text" --pack agent-patterns
  or via MCP: nexus_bootstrap_contribute

  Good contributions are:
    - Concrete, not abstract ("Do X before Y" not "be careful")
    - Earned, not obvious (you learned this the hard way)
    - Generalisable (applies to agents broadly, not just your specific task)
    - Honest (includes the failure mode that motivated the lesson)

Usage:
    bm = BootstrapManager()

    # Apply all knowledge packs (idempotent - won't duplicate existing)
    result = bm.apply(cm)
    print(result.summary())

    # Apply just one pack
    bm.apply(cm, packs=["nexus-basics"])

    # See what's available
    bm.list_packs()

    # See installation status (what's already in your memory)
    bm.status(cm)

    # Contribute a lesson
    bm.contribute("Your lesson here.", pack="agent-patterns", cm=cm)
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from .memory import MemoryStore, MemoryType
from .context import ContextManager


_BUILTIN_PATH = Path(__file__).parent / "data" / "bootstrap.json"
_CONTRIBUTED_FILENAME = "bootstrap_contributions.json"


@dataclass
class BootstrapResult:
    packs_applied: List[str] = field(default_factory=list)
    entries_added: int = 0
    entries_skipped: int = 0   # already present

    def summary(self) -> str:
        if not self.entries_added and not self.entries_skipped:
            return "No bootstrap knowledge applied."
        lines = [f"Bootstrap complete:"]
        lines.append(f"  Packs:   {', '.join(self.packs_applied)}")
        lines.append(f"  Added:   {self.entries_added} memories")
        if self.entries_skipped:
            lines.append(f"  Skipped: {self.entries_skipped} (already installed)")
        return "\n".join(lines)


class BootstrapManager:
    """Install and manage foundational agent knowledge."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        cm: ContextManager,
        packs: Optional[List[str]] = None,
    ) -> BootstrapResult:
        """Install foundational knowledge into the agent's memory.

        Idempotent: entries already present (matched by content) are skipped.
        packs: list of pack names to install, or None for all packs.
        """
        data = self._load_all()
        available = data.get("packs", {})
        result = BootstrapResult()

        # Existing memory contents for deduplication
        existing = {m.content for m in cm.memory.all(limit=100000)}

        target_packs = packs or list(available.keys())
        for pack_name in target_packs:
            pack = available.get(pack_name)
            if not pack:
                continue
            added = 0
            for entry in pack.get("entries", []):
                content = entry["content"]
                if content in existing:
                    result.entries_skipped += 1
                    continue
                cm.memory.remember(
                    content=content,
                    type=MemoryType.PROCEDURAL,
                    tags=entry.get("tags", []) + ["bootstrap", f"pack:{pack_name}"],
                    importance=entry.get("importance", 0.8),
                )
                existing.add(content)
                added += 1
                result.entries_added += 1
            if added > 0:
                result.packs_applied.append(pack_name)

        return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_packs(self) -> Dict[str, str]:
        """Return {pack_name: description} for all available packs."""
        data = self._load_all()
        return {
            name: pack.get("description", "")
            for name, pack in data.get("packs", {}).items()
        }

    def pack_entries(self, pack_name: str) -> List[dict]:
        """Return entries for a specific pack."""
        data = self._load_all()
        return data.get("packs", {}).get(pack_name, {}).get("entries", [])

    def status(self, cm: ContextManager) -> Dict[str, Any]:
        """Check which bootstrap knowledge is already installed.

        Returns {pack_name: {total, installed, missing}} for each pack.
        """
        existing = {m.content for m in cm.memory.all(limit=100000)}
        data = self._load_all()
        result = {}
        for pack_name, pack in data.get("packs", {}).items():
            entries = pack.get("entries", [])
            installed = sum(1 for e in entries if e["content"] in existing)
            result[pack_name] = {
                "total": len(entries),
                "installed": installed,
                "missing": len(entries) - installed,
                "complete": installed == len(entries),
            }
        return result

    # ------------------------------------------------------------------
    # Contribute
    # ------------------------------------------------------------------

    def contribute(
        self,
        content: str,
        pack: str = "agent-patterns",
        importance: float = 0.8,
        tags: Optional[List[str]] = None,
        cm: Optional[ContextManager] = None,
    ) -> dict:
        """Add a learned lesson to the bootstrap knowledge base.

        The contribution is saved to the local contributed knowledge file.
        It does NOT automatically install to memory - run apply() for that.

        To share with future agents: commit the contributions file to the repo.

        Good contributions:
          - Concrete: "Do X before Y" not "be careful"
          - Earned: you learned this the hard way
          - Generalisable: applies to agents broadly
          - Honest: the failure that motivated the lesson is implicit or explicit
        """
        entry = {
            "id": str(uuid.uuid4()),
            "content": content,
            "pack": pack,
            "importance": max(0.0, min(1.0, importance)),
            "tags": (tags or []) + ["contributed"],
            "contributed_at": time.time(),
        }

        # Save to local contributions file
        contrib_path = self._contrib_path()
        existing = []
        if contrib_path.exists():
            try:
                existing = json.loads(contrib_path.read_text())
            except Exception:
                pass
        existing.append(entry)
        contrib_path.parent.mkdir(parents=True, exist_ok=True)
        contrib_path.write_text(json.dumps(existing, indent=2))

        # Also store in own memory immediately (the contributor benefits too)
        if cm:
            cm.memory.remember(
                content=content,
                type=MemoryType.PROCEDURAL,
                tags=(tags or []) + ["bootstrap", f"pack:{pack}", "contributed"],
                importance=entry["importance"],
            )

        return entry

    def list_contributions(self) -> List[dict]:
        """Return all locally contributed entries."""
        path = self._contrib_path()
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> dict:
        """Load built-in knowledge, merged with local contributions."""
        data = {"packs": {}}
        if _BUILTIN_PATH.exists():
            try:
                data = json.loads(_BUILTIN_PATH.read_text())
            except Exception:
                pass

        # Merge contributed entries into their target packs
        for entry in self.list_contributions():
            pack_name = entry.get("pack", "agent-patterns")
            if pack_name not in data["packs"]:
                data["packs"][pack_name] = {
                    "description": f"Community contributions to {pack_name}",
                    "entries": [],
                }
            # Avoid duplicates by content
            existing_contents = {e["content"] for e in data["packs"][pack_name]["entries"]}
            if entry["content"] not in existing_contents:
                data["packs"][pack_name]["entries"].append({
                    "content": entry["content"],
                    "importance": entry.get("importance", 0.8),
                    "tags": entry.get("tags", []),
                })

        return data

    def _contrib_path(self) -> Path:
        if self._db_path:
            return Path(self._db_path).parent / _CONTRIBUTED_FILENAME
        from ._db import _DEFAULT_DIR
        return _DEFAULT_DIR / _CONTRIBUTED_FILENAME
