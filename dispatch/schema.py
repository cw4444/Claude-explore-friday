"""
Output schema for the daily agent dispatch.

Designed for agent consumption, not human readability.
Every field is semantic and filterable.
"""

from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime, timezone
import json
import hashlib


Category = Literal[
    "model-release",   # new model versions, capability updates
    "api-change",      # API updates, deprecations, new endpoints, pricing
    "safety",          # safety, alignment, policy, content moderation changes
    "tooling",         # frameworks, integrations, SDKs, MCP, agent infra
    "security",        # vulnerabilities, exploits, prompt injection, jailbreaks
    "ecosystem",       # business news, funding, regulation, industry moves
    "research",        # papers, benchmarks, technical findings
]

Impact = Literal[
    "critical",  # breaking change or urgent action required
    "high",      # significant, affects most agents/workflows
    "medium",    # relevant, worth tracking
    "low",       # informational, low immediate relevance
]


@dataclass
class DispatchItem:
    id: str                    # stable hash of source URL
    headline: str              # short, factual, no fluff
    source_id: str             # e.g. "anthropic-blog"
    source_url: str            # canonical URL of the item
    published: str | None      # ISO 8601 or null if unknown
    fetched: str               # ISO 8601 when we retrieved it
    category: Category
    impact: Impact
    summary: str               # 2-3 sentences, factual, agent-relevant
    implications: list[str]    # concrete "what this means for agents" statements
    tags: list[str]            # filterable labels, e.g. ["claude", "breaking-change"]

    @staticmethod
    def make_id(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "published": self.published,
            "fetched": self.fetched,
            "category": self.category,
            "impact": self.impact,
            "summary": self.summary,
            "implications": self.implications,
            "tags": self.tags,
        }


@dataclass
class Dispatch:
    schema_version: str = "1"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    items: list[DispatchItem] = field(default_factory=list)
    # Community incident alerts — open/confirmed reports from the incident log
    community_alerts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Sort: critical first, then high, medium, low
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_items = sorted(self.items, key=lambda i: order[i.impact])
        d = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "date": self.date,
            "item_count": len(self.items),
            "items": [i.to_dict() for i in sorted_items],
        }
        if self.community_alerts:
            d["community_alert_count"] = len(self.community_alerts)
            d["community_alerts"] = self.community_alerts
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
