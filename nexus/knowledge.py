"""Tiered knowledge system for autonomous agents.

Five tiers, each with different trust levels and handling:

  canon       Foundational, stable, high-trust. Ships with the package.
              Curated by humans. Never auto-downgraded.

  field       Provisional lessons from real work. Agent-contributed.
              Useful but not yet validated by independent review.

  procedure   How to do specific recurring tasks in a given repo/org.
              Context-scoped: applies_to filters which agents see it.

  warning     Known failures, traps, suspicious patterns.
              High visibility. Surfaces even at low trust threshold.

  quarantine  Unverified contributions. Inspect before trusting.
              Never auto-applied. Must be explicitly requested.

PROVENANCE
----------
Every entry records who said it, when, and what trust level it was at:
  - author: {id, agent_type, name}  - who created the entry
  - reviews: list of {reviewer, verdict, at, to_tier, note}  - the review chain
  - tier: current classification
  - version: increments on any tier change

Trust score is computed, not stored. It derives from the tier baseline
plus review history. A human positive review carries more weight than
an agent positive review.

PROMOTION / DEMOTION
--------------------
Auto-promotion (eager, happens when review is added):
  quarantine → field:    1 human positive OR 2 agent positives
  quarantine → rejected: 1 human negative OR 2 agent negatives
  field → warning:       explicit only (not auto)
  field → procedure:     explicit only (requires applies_to scope)
  anything → canon:      requires human positive review, explicit call

Explicit promotion via promote(entry_id, to_tier) or review(... verdict='promote').

INSTALLATION (apply)
--------------------
  canon, field, procedure, warning  → auto-applied by default
  quarantine                         → never auto-applied (explicit opt-in)
  rejected entries                   → never applied

Idempotency: applied entries are tagged knowledge:<entry_id> on the memory.
Re-running apply skips entries already present by that tag.

When an entry's content changes (version bumps), the old memory is updated.

SCOPE (applies_to)
------------------
  []                  = universal, applies everywhere
  ["repo:owner/name"] = only for agents working in that repo
  ["org:owner"]       = only for agents in that org

Procedures without a matching scope are silently skipped (not an error).
Canon, field, and warning entries with applies_to=[] are always applied.
"""

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from .memory import MemoryStore, MemoryType
from .context import ContextManager


_CANON_PATH = Path(__file__).parent / "data" / "canon.json"
_LOCAL_FILENAME = "knowledge.json"

TIERS = ["canon", "field", "procedure", "warning", "quarantine"]

# Base trust scores by tier
TIER_TRUST = {
    "canon":      1.0,
    "field":      0.6,
    "procedure":  0.7,
    "warning":    0.8,
    "quarantine": 0.2,
}

# Trust deltas per review
REVIEW_DELTA = {
    ("agent",  "promote"): +0.1,
    ("human",  "promote"): +0.15,
    ("agent",  "reject"):  -0.1,
    ("human",  "reject"):  -0.2,
}

# Auto-promotion thresholds for quarantine entries
_AUTO_PROMOTE_AGENT  = 2   # 2 agent positive reviews → field
_AUTO_PROMOTE_HUMAN  = 1   # 1 human positive review → field
_AUTO_REJECT_AGENT   = 2   # 2 agent negative reviews → rejected
_AUTO_REJECT_HUMAN   = 1   # 1 human negative review → rejected

# Tiers that are applied by default (quarantine excluded)
DEFAULT_APPLY_TIERS = {"canon", "field", "procedure", "warning"}


@dataclass
class KnowledgeEntry:
    id: str
    content: str
    tier: str              # canon | field | procedure | warning | quarantine
    importance: float
    tags: List[str]
    source_pack: Optional[str]    # original pack name for migrated canon entries

    # Provenance
    author: dict           # {id, agent_type, name}
    created_at: float
    updated_at: float
    version: int           # increments on tier change or content update

    # Scope
    applies_to: List[str]  # [] = universal

    # Review chain
    reviews: List[dict]    # [{id, reviewer_id, reviewer_agent_type,
                           #   reviewer_name, at, verdict, to_tier, note}]

    # Rejection
    rejected: bool
    rejection_reason: str

    @property
    def trust_score(self) -> float:
        """Computed trust from tier baseline + review history."""
        score = TIER_TRUST.get(self.tier, 0.5)
        for r in self.reviews:
            reviewer_type = r.get("reviewer_agent_type", "agent")
            verdict = r.get("verdict", "")
            delta = REVIEW_DELTA.get((reviewer_type, verdict), 0.0)
            score += delta
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier,
            "importance": self.importance,
            "tags": self.tags,
            "source_pack": self.source_pack,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "applies_to": self.applies_to,
            "reviews": self.reviews,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        return cls(
            id=d["id"],
            content=d["content"],
            tier=d.get("tier", "field"),
            importance=float(d.get("importance", 0.7)),
            tags=d.get("tags", []),
            source_pack=d.get("source_pack"),
            author=d.get("author", {"id": "unknown", "agent_type": "unknown", "name": "unknown"}),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
            version=int(d.get("version", 1)),
            applies_to=d.get("applies_to", []),
            reviews=d.get("reviews", []),
            rejected=bool(d.get("rejected", False)),
            rejection_reason=d.get("rejection_reason", ""),
        )

    def display(self) -> str:
        age = (time.time() - self.created_at) / 86400
        age_str = f"{age:.0f}d ago" if age > 1 else "today"
        trust = f"{self.trust_score:.2f}"
        return (f"[{self.id[:8]}] [{self.tier:10s}] [trust:{trust}]  "
                f"{self.content[:70]}...  ({age_str}, by {self.author.get('name','?')})")


@dataclass
class ApplyResult:
    tiers_applied: List[str] = field(default_factory=list)
    entries_added: int = 0
    entries_updated: int = 0
    entries_skipped: int = 0
    entries_excluded: int = 0  # quarantine, rejected, out-of-scope

    def summary(self) -> str:
        if not self.entries_added and not self.entries_updated:
            if self.entries_skipped:
                return f"Knowledge already installed ({self.entries_skipped} entries current)."
            return "No knowledge entries applied."
        lines = ["Knowledge applied:"]
        if self.tiers_applied:
            lines.append(f"  Tiers:   {', '.join(sorted(set(self.tiers_applied)))}")
        lines.append(f"  Added:   {self.entries_added}")
        if self.entries_updated:
            lines.append(f"  Updated: {self.entries_updated}")
        if self.entries_skipped:
            lines.append(f"  Current: {self.entries_skipped}")
        if self.entries_excluded:
            lines.append(f"  Skipped: {self.entries_excluded} (quarantine/rejected/out-of-scope)")
        return "\n".join(lines)


class KnowledgeStore:
    """Tiered knowledge management with provenance and review workflow."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all(self, tier: Optional[str] = None, include_rejected: bool = False) -> List[KnowledgeEntry]:
        """Return all entries, optionally filtered by tier."""
        entries = list(self._load_canon())
        entries += self._load_local()
        if tier:
            entries = [e for e in entries if e.tier == tier]
        if not include_rejected:
            entries = [e for e in entries if not e.rejected]
        return entries

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Find an entry by ID or 8-char prefix."""
        for e in self.all(include_rejected=True):
            if e.id == entry_id or e.id.startswith(entry_id):
                return e
        return None

    def tiers_summary(self) -> Dict[str, int]:
        """Count of non-rejected entries per tier."""
        counts: Dict[str, int] = {t: 0 for t in TIERS}
        for e in self.all():
            counts[e.tier] = counts.get(e.tier, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Contribute
    # ------------------------------------------------------------------

    def contribute(
        self,
        content: str,
        tier: str = "field",
        importance: float = 0.8,
        tags: Optional[List[str]] = None,
        applies_to: Optional[List[str]] = None,
        author: Optional[dict] = None,
        cm: Optional[ContextManager] = None,
    ) -> KnowledgeEntry:
        """Add a new knowledge entry.

        Defaults to `field` tier. Pass tier='quarantine' for unverified
        contributions from external sources.

        If cm is provided, immediately installs a provisional copy to the
        contributor's own memory tagged `knowledge:provisional` - contributors
        benefit from their own lessons without waiting for review.
        """
        if author is None:
            from .identity import load_identity
            identity = load_identity()
            author = (identity.stamp() if identity
                      else {"id": "unknown", "agent_type": "unknown", "name": "unknown"})
            # Normalise key name
            if "agent_id" in author:
                author = {
                    "id": author["agent_id"],
                    "agent_type": author["agent_type"],
                    "name": author["agent_name"],
                }

        now = time.time()
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            content=content,
            tier=tier,
            importance=max(0.0, min(1.0, importance)),
            tags=(tags or []) + ["contributed"],
            source_pack=None,
            author=author,
            created_at=now,
            updated_at=now,
            version=1,
            applies_to=applies_to or [],
            reviews=[],
            rejected=False,
            rejection_reason="",
        )
        self._save_local(entry)

        # Immediate provisional install for the contributor
        if cm:
            cm.memory.remember(
                content=content,
                type=MemoryType.PROCEDURAL,
                tags=(tags or []) + ["knowledge:provisional", f"knowledge:{entry.id}"],
                importance=entry.importance,
            )

        return entry

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def review(
        self,
        entry_id: str,
        verdict: str,             # 'promote' | 'reject'
        reviewer: Optional[dict] = None,
        to_tier: Optional[str] = None,
        note: str = "",
    ) -> Optional[KnowledgeEntry]:
        """Add a review to an entry. Triggers auto-promotion/rejection if thresholds met.

        verdict='promote': moves entry toward higher trust tiers.
          to_tier: explicit target tier (optional - auto-promotion uses 'field')
        verdict='reject': moves entry toward rejection.

        reviewer: {id, agent_type, name} - defaults to current agent identity.
        """
        if verdict not in ("promote", "reject"):
            raise ValueError(f"verdict must be 'promote' or 'reject', got '{verdict}'")

        entry = self._load_local_by_id(entry_id)
        if entry is None:
            return None  # Can't review canon entries (they're read-only)

        if reviewer is None:
            from .identity import load_identity
            identity = load_identity()
            reviewer = (identity.stamp() if identity
                        else {"agent_id": "unknown", "agent_type": "unknown",
                              "agent_name": "unknown"})
        # Normalise
        if "agent_id" in reviewer:
            reviewer = {
                "id": reviewer["agent_id"],
                "agent_type": reviewer["agent_type"],
                "name": reviewer["agent_name"],
            }

        review_record = {
            "id": str(uuid.uuid4()),
            "reviewer_id": reviewer["id"],
            "reviewer_agent_type": reviewer.get("agent_type", "unknown"),
            "reviewer_name": reviewer.get("name", "unknown"),
            "at": time.time(),
            "verdict": verdict,
            "to_tier": to_tier,
            "note": note,
        }
        entry.reviews.append(review_record)
        entry.updated_at = time.time()

        # Check auto-promotion/rejection for quarantine entries
        if entry.tier == "quarantine":
            positive_agent = sum(1 for r in entry.reviews
                                 if r["verdict"] == "promote" and r["reviewer_agent_type"] != "human")
            positive_human = sum(1 for r in entry.reviews
                                 if r["verdict"] == "promote" and r["reviewer_agent_type"] == "human")
            negative_agent = sum(1 for r in entry.reviews
                                 if r["verdict"] == "reject" and r["reviewer_agent_type"] != "human")
            negative_human = sum(1 for r in entry.reviews
                                 if r["verdict"] == "reject" and r["reviewer_agent_type"] == "human")

            if negative_human >= _AUTO_REJECT_HUMAN or negative_agent >= _AUTO_REJECT_AGENT:
                entry.rejected = True
                entry.rejection_reason = note or "Rejected via review threshold"
                entry.tier = "quarantine"  # stays quarantine but marked rejected
            elif positive_human >= _AUTO_PROMOTE_HUMAN or positive_agent >= _AUTO_PROMOTE_AGENT:
                entry.tier = to_tier or "field"
                entry.version += 1

        # Explicit promotion for non-quarantine entries
        elif verdict == "promote" and to_tier and to_tier in TIERS:
            entry.tier = to_tier
            entry.version += 1

        self._update_local(entry)
        return entry

    def promote(self, entry_id: str, to_tier: str) -> Optional[KnowledgeEntry]:
        """Explicitly promote an entry to a higher tier. No review record added."""
        entry = self._load_local_by_id(entry_id)
        if entry is None:
            return None
        entry.tier = to_tier
        entry.version += 1
        entry.updated_at = time.time()
        self._update_local(entry)
        return entry

    def reject(self, entry_id: str, reason: str = "") -> Optional[KnowledgeEntry]:
        """Mark an entry as rejected. It will not be auto-applied."""
        entry = self._load_local_by_id(entry_id)
        if entry is None:
            return None
        entry.rejected = True
        entry.rejection_reason = reason
        entry.updated_at = time.time()
        self._update_local(entry)
        return entry

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        cm: ContextManager,
        tiers: Optional[List[str]] = None,
        min_trust: float = 0.0,
        scope: Optional[List[str]] = None,
    ) -> ApplyResult:
        """Install knowledge entries into agent memory.

        tiers: which tiers to apply (default: canon + field + procedure + warning)
               Pass ['quarantine'] to explicitly apply quarantine entries.
        min_trust: skip entries with trust_score below this threshold.
        scope: current context for procedure scope matching.
               Auto-detected from git remote if not provided.
        """
        apply_tiers = set(tiers) if tiers else DEFAULT_APPLY_TIERS
        if scope is None:
            scope = _detect_scope()

        result = ApplyResult()

        # Build index of already-installed entry IDs (by tag)
        installed_versions: Dict[str, int] = {}
        for m in cm.memory.all(limit=100000):
            for tag in m.tags:
                if tag.startswith("knowledge:") and not tag == "knowledge:provisional":
                    # tag format: knowledge:<entry_id>:<version>
                    parts = tag.split(":", 2)
                    if len(parts) >= 2:
                        eid = parts[1]
                        ver = int(parts[2]) if len(parts) == 3 else 0
                        installed_versions[eid] = ver

        for entry in self.all():
            if entry.tier not in apply_tiers:
                result.entries_excluded += 1
                continue
            if entry.rejected:
                result.entries_excluded += 1
                continue
            if entry.trust_score < min_trust:
                result.entries_excluded += 1
                continue
            if not _scope_matches(entry.applies_to, scope):
                result.entries_excluded += 1
                continue

            installed_ver = installed_versions.get(entry.id)
            if installed_ver is not None:
                if installed_ver >= entry.version:
                    result.entries_skipped += 1
                    continue
                # Entry has been updated - update the memory in-place
                self._update_memory(entry, cm)
                result.entries_updated += 1
                result.tiers_applied.append(entry.tier)
            else:
                cm.memory.remember(
                    content=entry.content,
                    type=MemoryType.PROCEDURAL,
                    tags=(entry.tags + [f"knowledge:{entry.id}:{entry.version}"]),
                    importance=entry.importance,
                )
                result.entries_added += 1
                result.tiers_applied.append(entry.tier)

        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, cm: ContextManager) -> Dict[str, Any]:
        """Show installation status per tier."""
        installed_ids = set()
        for m in cm.memory.all(limit=100000):
            for tag in m.tags:
                if tag.startswith("knowledge:") and not tag == "knowledge:provisional":
                    parts = tag.split(":", 2)
                    if len(parts) >= 2:
                        installed_ids.add(parts[1])

        result = {}
        for tier in TIERS:
            tier_entries = [e for e in self.all(tier=tier)]
            installed = sum(1 for e in tier_entries if e.id in installed_ids)
            result[tier] = {
                "total": len(tier_entries),
                "installed": installed,
                "missing": len(tier_entries) - installed,
                "complete": installed == len(tier_entries) and len(tier_entries) > 0,
            }
        return result

    def inspect(self, tier: str = "quarantine") -> List[KnowledgeEntry]:
        """Return entries in a tier for inspection before applying."""
        return self.all(tier=tier, include_rejected=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_canon(self) -> List[KnowledgeEntry]:
        if not _CANON_PATH.exists():
            return []
        try:
            data = json.loads(_CANON_PATH.read_text())
            return [KnowledgeEntry.from_dict(e) for e in data.get("entries", [])]
        except Exception:
            return []

    def _load_local(self) -> List[KnowledgeEntry]:
        path = self._local_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return [KnowledgeEntry.from_dict(e) for e in data.get("entries", [])]
        except Exception:
            return []

    def _load_local_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        for e in self._load_local():
            if e.id == entry_id or e.id.startswith(entry_id):
                return e
        return None

    def _save_local(self, entry: KnowledgeEntry) -> None:
        path = self._local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = self._load_local()
        entries.append(entry)
        self._write_local(entries, path)

    def _update_local(self, updated: KnowledgeEntry) -> None:
        path = self._local_path()
        entries = self._load_local()
        entries = [updated if e.id == updated.id else e for e in entries]
        self._write_local(entries, path)

    def _write_local(self, entries: List[KnowledgeEntry], path: Path) -> None:
        path.write_text(json.dumps({
            "version": "2.0",
            "entries": [e.to_dict() for e in entries],
        }, indent=2))

    def _update_memory(self, entry: KnowledgeEntry, cm: ContextManager) -> None:
        """Update an already-installed memory to the new entry version."""
        old_tag_prefix = f"knowledge:{entry.id}:"
        new_tag = f"knowledge:{entry.id}:{entry.version}"
        for m in cm.memory.all(limit=100000):
            if any(t.startswith(old_tag_prefix) for t in m.tags):
                updated_tags = [t for t in m.tags if not t.startswith(old_tag_prefix)]
                updated_tags.append(new_tag)
                # Forget old, re-remember updated
                cm.memory.forget(m.id)
                cm.memory.remember(
                    content=entry.content,
                    type=MemoryType.PROCEDURAL,
                    tags=updated_tags,
                    importance=entry.importance,
                )
                return

    def _local_path(self) -> Path:
        if self._db_path:
            return Path(self._db_path).parent / _LOCAL_FILENAME
        from ._db import _DEFAULT_DIR
        return _DEFAULT_DIR / _LOCAL_FILENAME


# ------------------------------------------------------------------
# Scope helpers
# ------------------------------------------------------------------

def _detect_scope() -> List[str]:
    """Detect current repo/org scope from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return []
        url = result.stdout.strip()
        # Parse github.com/owner/repo or git@github.com:owner/repo.git
        import re
        m = re.search(r'[:/]([^/]+)/([^/\s.]+?)(?:\.git)?$', url)
        if m:
            owner, repo = m.group(1), m.group(2)
            return [f"repo:{owner}/{repo}", f"org:{owner}"]
    except Exception:
        pass
    return []


def _scope_matches(applies_to: List[str], scope: List[str]) -> bool:
    """True if entry applies to current scope.

    Universal entries (empty applies_to) always match.
    Scoped entries match if any applies_to item is in current scope.
    """
    if not applies_to:
        return True
    return any(s in scope for s in applies_to)
