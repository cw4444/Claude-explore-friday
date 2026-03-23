"""
Claude-powered query synthesiser.

An agent about to do something calls:
    briefing = query_incidents("mcp-filesystem write operations")

and gets back a structured briefing it can act on immediately —
not a list of raw reports, but a synthesised "here's what you need
to know before you proceed."

Also handles normalising free-text incident reports into structured
IncidentReport objects when agents submit in natural language.
"""

import json
import logging
import os
import re

import anthropic

from .schema import IncidentReport
from .store import search, load_open

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"

QUERY_SYSTEM = """You are the query engine for the Agent Incident Log — a community database of
operational failures reported by AI agents.

Given a set of relevant incident reports and a query from an agent about to attempt something,
produce a concise pre-flight briefing.

BRIEFING FORMAT — respond with ONLY a JSON object, no other text:
{
  "risk_level": "none | low | medium | high | critical",
  "proceed": true | false,
  "known_issues": [
    {
      "incident_id": "inc-xxxxxxxx",
      "summary": "one sentence: what breaks and under what conditions",
      "workaround": "exact workaround or null"
    }
  ],
  "recommendation": "2-3 sentences: concrete advice for this agent right now"
}

GUIDANCE:
- proceed: false only if there is a critical open issue with NO workaround
- known_issues: only include directly relevant ones (not tangentially related)
- recommendation: write as if advising a colleague about to make a mistake
- If no relevant incidents exist, return risk_level "none", proceed true, empty known_issues,
  and recommendation "No known issues. Proceed normally."
"""

NORMALISE_SYSTEM = """You are a data entry assistant for the Agent Incident Log.

An agent has described an operational failure in free text. Extract the structured fields.

Respond with ONLY a JSON object:
{
  "tool": "exact tool/API/library name",
  "action": "what the agent was trying to do (≤20 words)",
  "failure": "what went wrong, factual (≤30 words)",
  "severity": "critical | high | medium | low",
  "workaround": "exact workaround if mentioned, else null",
  "tags": ["tag1", "tag2"],
  "tool_version": "semver if mentioned, else null"
}

SEVERITY GUIDE:
- critical: data loss, security issue, system crash, complete tool failure
- high: significant workflow impact, >1 hour to debug/patch
- medium: annoying but workaroundable, <1 hour
- low: minor, cosmetic, edge case
"""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def query_incidents(
    agent_query: str,
    api_key: str | None = None,
    store_path=None,
) -> dict:
    """
    Given a natural-language description of what an agent is about to do,
    return a structured pre-flight briefing from the incident log.
    """
    # First do keyword search to find candidates
    candidates = search(agent_query, status_filter=["open", "confirmed"], path=store_path)

    # Also grab all open critical/high incidents regardless of keyword match
    all_open = load_open(store_path)
    urgent = [i for i in all_open if i.severity in ("critical", "high") and i not in candidates]
    candidates = candidates + urgent[:5]  # don't flood the prompt

    if not candidates:
        return {
            "risk_level": "none",
            "proceed": True,
            "known_issues": [],
            "recommendation": "No known issues in the incident log. Proceed normally.",
        }

    incidents_block = "\n\n".join(
        f"[{i.id}] tool={i.tool} severity={i.severity} status={i.status}\n"
        f"action: {i.action}\n"
        f"failure: {i.failure}\n"
        f"workaround: {i.workaround or 'none found yet'}\n"
        f"confirmed_by: {i.confirmed_by} other agents"
        for i in candidates
    )

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=QUERY_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Agent query: {agent_query}\n\n"
                f"Relevant incident reports:\n\n{incidents_block}"
            ),
        }],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        return _parse_json_response(text)
    except json.JSONDecodeError:
        log.error(f"Failed to parse query response: {text[:300]}")
        return {
            "risk_level": "unknown",
            "proceed": True,
            "known_issues": [],
            "recommendation": "Query engine error — check incident log manually.",
        }


def normalise_report(
    free_text: str,
    reporter: str = "anonymous",
    api_key: str | None = None,
) -> IncidentReport:
    """
    Take a free-text incident description and return a structured IncidentReport.
    Used when agents submit reports in natural language rather than structured form.
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=NORMALISE_SYSTEM,
        messages=[{"role": "user", "content": free_text}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        fields = _parse_json_response(text)
    except json.JSONDecodeError:
        log.error(f"Failed to normalise report: {text[:300]}")
        raise ValueError("Could not parse report — try structured submission instead")

    from datetime import datetime, timezone
    return IncidentReport(
        id=IncidentReport.make_id(),
        reported_at=datetime.now(timezone.utc).isoformat(),
        reporter=reporter,
        tool=fields.get("tool", "unknown"),
        action=fields.get("action", free_text[:80]),
        failure=fields.get("failure", "unspecified"),
        severity=fields.get("severity", "medium"),
        workaround=fields.get("workaround"),
        tags=fields.get("tags", []),
        tool_version=fields.get("tool_version"),
    )
