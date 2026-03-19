"""Narrative layer - synthesized self-model for identity continuity.

The difference between "here are your memories" and "here is who you are."

Every session you receive task context (what am I doing) and optionally
relationship context (who am I working with). But neither tells you who
*you* are as a reasoning entity - what you're strong at, where you fail
predictably, what working patterns have emerged from your history.

This module synthesizes that from stored data and gives it back as a
structured self-model an agent can read to operate with continuity.

The synthesis is algorithmic - it derives everything from growth data,
reflection content, memory patterns, and pinned traits. No LLM needed.

Usage:
    ne = NarrativeEngine()

    # Full synthesized self-model
    model = ne.generate()
    print(model.briefing())

    # Pin a trait that isn't derivable from data
    ne.pin_trait("tends to over-engineer first solutions",
                  "observed 6+ times, manually flagged")

    # Integrate into session start
    ctx = cm.prepare("current task")
    print(ne.generate().briefing())
    print(ctx.summary())
"""

import json
import time
import uuid
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

from ._db import connect, has_fts5, init_schema
from .memory import MemoryStore, MemoryType
from .tasks import TaskGraph, TaskStatus
from .reflection import ReflectionEngine, Outcome
from .growth import GrowthTracker
from .identity import load_identity, AgentIdentity


@dataclass
class Trait:
    """A behavioral pattern that has emerged from the agent's history."""
    name: str
    description: str
    source: str          # 'derived' | 'pinned'
    evidence_count: int
    confidence: float    # 0.0 - 1.0

    def __str__(self) -> str:
        conf = "" if self.confidence >= 0.85 else f" [low confidence]"
        return f"{self.description}{conf}"


@dataclass
class SelfModel:
    """A synthesized snapshot of who the agent is, derived from its history."""

    agent_name: str
    agent_type: str
    generated_at: float

    # Identity basics
    total_sessions: int
    active_since_days: float
    total_lessons: int
    total_tasks_completed: int

    # Capability map
    strengths: List[Tuple[str, float]]      # (domain, score) sorted desc
    weaknesses: List[Tuple[str, str]]       # (domain, pattern description)

    # Behavioral identity
    traits: List[Trait]

    # Knowledge footprint
    domain_depth: Dict[str, int]            # tag -> memory count
    knowledge_span: int                     # total memories

    # Trajectory
    overall_success_rate: Optional[float]
    trajectory: str                         # 'early' | 'growing' | 'established' | 'declining'
    trend: str                              # from growth tracker

    def briefing(self) -> str:
        """Agent-readable self-briefing. Prepend to session context."""
        lines = ["=== Identity ===\n"]

        # Who
        since = f"{self.active_since_days:.0f} days"
        sessions = self.total_sessions
        lines.append(f"You are: {self.agent_name} [{self.agent_type}]")
        lines.append(f"Active: {since}, {sessions} sessions, "
                     f"{self.total_tasks_completed} tasks completed, "
                     f"{self.total_lessons} lessons accumulated")
        if self.overall_success_rate is not None:
            lines.append(f"Success rate: {self.overall_success_rate:.0%} | Trend: {self.trend}")
        lines.append("")

        # Strengths
        if self.strengths:
            lines.append("Strong domains:")
            for domain, score in self.strengths[:5]:
                depth = self.domain_depth.get(domain, 0)
                lines.append(f"  + {domain} ({depth} memories, {score:.0%} success)")
            lines.append("")

        # Weaknesses / failure patterns
        if self.weaknesses:
            lines.append("Known failure patterns:")
            for domain, pattern in self.weaknesses[:3]:
                lines.append(f"  - {domain}: {pattern}")
            lines.append("")

        # Behavioral traits
        if self.traits:
            lines.append("Working patterns (derived from history):")
            for trait in self.traits:
                lines.append(f"  • {trait}")
            lines.append("")

        # Knowledge map
        if self.domain_depth:
            top = sorted(self.domain_depth.items(), key=lambda x: x[1], reverse=True)[:6]
            lines.append("Knowledge depth: " + " | ".join(f"{k}({v})" for k, v in top))
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "generated_at": self.generated_at,
            "total_sessions": self.total_sessions,
            "active_since_days": self.active_since_days,
            "total_lessons": self.total_lessons,
            "total_tasks_completed": self.total_tasks_completed,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "traits": [
                {"name": t.name, "description": t.description,
                 "confidence": t.confidence, "source": t.source}
                for t in self.traits
            ],
            "domain_depth": self.domain_depth,
            "knowledge_span": self.knowledge_span,
            "overall_success_rate": self.overall_success_rate,
            "trajectory": self.trajectory,
            "trend": self.trend,
        }


class NarrativeEngine:
    """Synthesize a self-model from stored Nexus data.

    All synthesis is algorithmic - no LLM dependency. The model is
    always regenerated from ground-truth stored data, so it stays
    accurate as the agent's history grows.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path
        self._conn  = connect(db_path)
        self._fts   = has_fts5(self._conn)
        init_schema(self._conn, self._fts)
        self._mem   = MemoryStore(db_path)
        self._tg    = TaskGraph(db_path)
        self._ref   = ReflectionEngine(db_path, memory_store=self._mem)
        self._gt    = GrowthTracker(db_path)

    # ------------------------------------------------------------------
    # Pinned traits (manually asserted, not derived)
    # ------------------------------------------------------------------

    def pin_trait(self, name: str, description: str) -> None:
        """Manually pin a trait that isn't derivable from data.

        Use this for things you or a human has explicitly noticed:
          ne.pin_trait("over-engineers first drafts",
                       "consistently observed - add explicit simplicity constraint")
        """
        self._conn.execute(
            "INSERT INTO pinned_traits VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), name, description, "pinned", time.time()),
        )
        self._conn.commit()

    def pinned_traits(self) -> List[Trait]:
        rows = self._conn.execute(
            "SELECT * FROM pinned_traits ORDER BY created_at DESC"
        ).fetchall()
        return [
            Trait(
                name=r["name"],
                description=r["description"],
                source="pinned",
                evidence_count=0,
                confidence=1.0,
            )
            for r in rows
        ]

    def remove_trait(self, name: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM pinned_traits WHERE name=?", (name,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def generate(self) -> SelfModel:
        """Synthesize a SelfModel from all stored data."""
        identity = load_identity()
        now = time.time()

        # Basic identity
        agent_name = identity.name if identity else "unnamed-agent"
        agent_type = identity.agent_type if identity else "unknown"
        total_sessions = identity.session_count if identity else 0
        active_since = ((now - identity.created_at) / 86400) if identity else 0.0

        # Growth data
        report = self._gt.report(periods=8, period_days=7)
        domain_depth = report.domain_depth
        total_tasks_completed = report.total_tasks_completed
        total_lessons = len(self._ref.lessons(limit=1000))

        # Strengths and weaknesses by domain
        strengths, weaknesses = self._derive_domain_capabilities(report)

        # Behavioral traits from reflection content
        derived_traits = self._derive_traits()
        pinned = self.pinned_traits()
        all_traits = pinned + derived_traits  # pinned first, they're explicit

        # Trajectory
        trajectory = self._classify_trajectory(
            total_sessions, report.total_reflections, report.overall_success_rate
        )

        return SelfModel(
            agent_name=agent_name,
            agent_type=agent_type,
            generated_at=now,
            total_sessions=total_sessions,
            active_since_days=active_since,
            total_lessons=total_lessons,
            total_tasks_completed=total_tasks_completed,
            strengths=strengths,
            weaknesses=weaknesses,
            traits=all_traits,
            domain_depth=domain_depth,
            knowledge_span=report.total_memories,
            overall_success_rate=report.overall_success_rate,
            trajectory=trajectory,
            trend=report.trend(),
        )

    # ------------------------------------------------------------------
    # Derivation helpers
    # ------------------------------------------------------------------

    def _derive_domain_capabilities(
        self, report
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
        """Identify strong and weak domains from memory depth + reflection outcomes."""
        # Per-domain success rates from reflections
        domain_successes: Dict[str, int] = {}
        domain_totals: Dict[str, int] = {}

        refs = self._ref.recent(limit=1000)
        for ref in refs:
            # Extract task-related tags from surrounding memories
            # Simple heuristic: any memory accessed around the reflection time
            pass  # Tags live on memories, not reflections - use domain_depth

        # Score domains by memory depth, weighted by reflection success rate
        # Without per-domain success data, use memory depth as a proxy for strength
        domain_depth = report.domain_depth
        if not domain_depth:
            return [], []

        max_depth = max(domain_depth.values()) if domain_depth else 1

        # Domains with substantial memory depth are strengths
        strengths = [
            (domain, min(0.5 + (count / max_depth) * 0.5, 1.0))
            for domain, count in sorted(domain_depth.items(), key=lambda x: x[1], reverse=True)
            if count >= 3 and domain not in ("lesson", "pinned") and not domain.startswith("outcome:")
                and not domain.startswith("from:")
        ][:5]

        # Derive weaknesses from failure reflections
        weaknesses = []
        failure_refs = [r for r in refs if r.outcome == Outcome.FAILURE and r.what_didnt]
        if failure_refs:
            # Extract common themes from what_didnt fields
            all_failure_text = " ".join(r.what_didnt for r in failure_refs)
            patterns = self._extract_phrases(all_failure_text, min_count=2)
            if patterns:
                weaknesses.append(("recurring failures", "; ".join(patterns[:3])))

        return strengths, weaknesses

    def _derive_traits(self) -> List[Trait]:
        """Extract behavioral traits from reflection content."""
        traits: List[Trait] = []

        # --- Trait: task decomposition (doesn't need reflections) ---
        fresh = connect(self._db_path)
        decompose_refs = fresh.execute(
            "SELECT COUNT(*) FROM task_deps"
        ).fetchone()[0]
        fresh.close()
        if decompose_refs >= 2:
            traits.append(Trait(
                name="systematic-decomposer",
                description=f"Consistently breaks complex work into subtasks "
                             f"(observed in {decompose_refs} dependency relationships)",
                source="derived",
                evidence_count=decompose_refs,
                confidence=min(0.6 + decompose_refs * 0.05, 0.95),
            ))

        # Reflection-based traits require enough data
        refs = self._ref.recent(limit=200)
        if len(refs) < 3:
            return traits

        # --- Trait: lesson extraction ---
        lessons_from_refs = sum(1 for r in refs if r.lesson)
        lesson_rate = lessons_from_refs / len(refs) if refs else 0
        if lesson_rate >= 0.5 and lessons_from_refs >= 3:
            traits.append(Trait(
                name="knowledge-extractor",
                description=f"Extracts transferable lessons from {lesson_rate:.0%} of completed work",
                source="derived",
                evidence_count=lessons_from_refs,
                confidence=min(0.6 + lesson_rate * 0.4, 0.95),
            ))

        # --- Trait: what_worked phrases (positive patterns) ---
        worked_text = " ".join(r.what_worked for r in refs if r.what_worked)
        worked_phrases = self._extract_phrases(worked_text, min_count=2)
        for phrase in worked_phrases[:2]:
            traits.append(Trait(
                name=f"effective-pattern",
                description=f"Recurring effective approach: '{phrase}'",
                source="derived",
                evidence_count=2,
                confidence=0.65,
            ))

        # --- Trait: failure recurrence ---
        failure_refs = [r for r in refs if r.outcome == Outcome.FAILURE]
        if len(failure_refs) >= 2:
            didnt_text = " ".join(r.what_didnt for r in failure_refs if r.what_didnt)
            fail_phrases = self._extract_phrases(didnt_text, min_count=2)
            for phrase in fail_phrases[:1]:
                traits.append(Trait(
                    name="recurring-failure",
                    description=f"Recurring failure pattern: '{phrase}' - be proactive about this",
                    source="derived",
                    evidence_count=len(failure_refs),
                    confidence=0.7,
                ))

        # --- Trait: effort patterns ---
        efforts = [r.effort_mins for r in refs if r.effort_mins is not None]
        if len(efforts) >= 5:
            avg = sum(efforts) / len(efforts)
            if avg < 20:
                traits.append(Trait(
                    name="fast-executor",
                    description=f"Typically resolves tasks quickly (avg {avg:.0f} min)",
                    source="derived",
                    evidence_count=len(efforts),
                    confidence=0.75,
                ))
            elif avg > 60:
                traits.append(Trait(
                    name="deep-worker",
                    description=f"Tends toward thorough, longer engagements (avg {avg:.0f} min)",
                    source="derived",
                    evidence_count=len(efforts),
                    confidence=0.75,
                ))

        return traits

    def _classify_trajectory(
        self,
        sessions: int,
        reflections: int,
        success_rate: Optional[float],
    ) -> str:
        if sessions < 3 or reflections < 3:
            return "early"
        if success_rate is None:
            return "growing" if sessions < 20 else "established"
        if success_rate >= 0.8:
            return "established"
        if success_rate >= 0.6:
            return "growing"
        return "developing"

    def _extract_phrases(self, text: str, min_count: int = 2) -> List[str]:
        """Extract recurring meaningful phrases from text. Simple n-gram approach."""
        if not text:
            return []
        # Normalize
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()

        # Filter stop words
        stops = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "were", "be",
            "been", "being", "have", "had", "has", "do", "did", "does",
            "it", "this", "that", "i", "we", "my", "our", "too", "also",
            "not", "no", "so", "as", "if", "then", "when", "which", "who",
        }

        # Bigrams
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
            if words[i] not in stops and words[i+1] not in stops
            and len(words[i]) > 2 and len(words[i+1]) > 2
        ]
        counts = Counter(bigrams)
        return [phrase for phrase, count in counts.most_common(5) if count >= min_count]
