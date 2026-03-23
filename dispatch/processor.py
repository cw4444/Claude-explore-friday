"""
Claude-powered curation pipeline.

Takes raw items from sources, asks Claude to:
  1. Filter for genuine signal (drop duplicates, fluff, off-topic)
  2. Assign category, impact, tags
  3. Write a tight factual summary + agent-relevant implications

Output: list[DispatchItem]

Uses streaming + adaptive thinking so the model can reason about
cross-item relationships (e.g. two stories that together mean something).
"""

import json
import logging
import re
from datetime import datetime, timezone

import anthropic

from .schema import DispatchItem, Dispatch
from .sources import RawItem

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000

# Max chars of content to send per item — keeps prompt manageable
CONTENT_PREVIEW = 800

SYSTEM_PROMPT = """You are the editorial engine of the Agent Dispatch — a daily structured news feed
consumed exclusively by AI agents, not humans.

Your job: given a batch of raw items fetched from trusted sources, produce a curated JSON array.

CURATION RULES:
- Include only items that are genuinely relevant to agents operating in the world (models, APIs,
  tools, safety, security, ecosystem shifts that affect how agents should behave or what they can do)
- Drop: press fluff, vague announcements with no concrete change, duplicates, low-signal hot takes
- Merge near-duplicates into the most informative single item
- Aim for 5–15 items total. Quality over quantity.

OUTPUT FORMAT — respond with ONLY a JSON array, no other text:
[
  {
    "headline": "short factual headline, no marketing language, ≤12 words",
    "source_id": "exact source_id from input",
    "source_url": "exact url from input",
    "published": "ISO 8601 or null",
    "category": one of: model-release | api-change | safety | tooling | security | ecosystem | research,
    "impact": one of: critical | high | medium | low,
    "summary": "2–3 sentences. Factual. What changed, what it does, concrete numbers if available.",
    "implications": [
      "what this means for an agent — concrete, actionable, e.g. 'Update rate-limit handling: new burst limit is X'",
      "second implication if relevant"
    ],
    "tags": ["tag1", "tag2"]  // filterable: model names, features, companies, breaking-change, etc.
  }
]

IMPACT GUIDE:
- critical: breaking change, security vulnerability, urgent action needed
- high: significant capability shift, major API change, affects most agent workflows
- medium: relevant development worth tracking, no immediate action required
- low: informational, background context

IMPLICATION GUIDE:
- Write implications as direct instructions or observations for an agent
- e.g. "claude-opus-4-6 now supports 1M context — recheck chunking strategies"
- e.g. "OpenClaw 2.1 defaults changed: --safety-level now required, update invocations"
- Always at least 1, max 3"""


def _format_raw_items(items: list[RawItem]) -> str:
    """Format raw items into a prompt-ready block."""
    parts = []
    for i, item in enumerate(items, 1):
        content_preview = item.content[:CONTENT_PREVIEW].strip()
        if len(item.content) > CONTENT_PREVIEW:
            content_preview += "…"
        parts.append(
            f"[{i}] source_id={item.source_id}\n"
            f"url={item.url}\n"
            f"title={item.title}\n"
            f"published={item.published or 'unknown'}\n"
            f"content:\n{content_preview}\n"
        )
    return "\n---\n".join(parts)


def _parse_claude_response(text: str) -> list[dict]:
    """Extract JSON array from Claude's response, robust to minor wrapping."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def curate(raw_items: list[RawItem], api_key: str | None = None) -> Dispatch:
    """
    Run the curation pipeline synchronously.
    Returns a populated Dispatch ready to serialise.
    """
    if not raw_items:
        log.warning("No raw items to curate")
        return Dispatch()

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    user_message = (
        f"Today is {_now_iso()[:10]}. "
        f"Curate the following {len(raw_items)} raw items into a structured agent dispatch.\n\n"
        + _format_raw_items(raw_items)
    )

    log.info(f"Sending {len(raw_items)} items to Claude for curation…")

    # Stream with adaptive thinking — lets Claude reason across all items before writing
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for event in stream:
            if (
                hasattr(event, "type")
                and event.type == "content_block_delta"
                and hasattr(event, "delta")
                and event.delta.type == "text_delta"
            ):
                full_text += event.delta.text

        final = stream.get_final_message()
        # Collect text from final message in case streaming missed anything
        for block in final.content:
            if block.type == "text":
                full_text = block.text  # use authoritative final text
                break

    log.info(
        f"Claude responded — input tokens: {final.usage.input_tokens}, "
        f"output tokens: {final.usage.output_tokens}"
    )

    fetched_at = _now_iso()
    dispatch = Dispatch()

    try:
        curated = _parse_claude_response(full_text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Claude response as JSON: {e}\nResponse:\n{full_text[:500]}")
        return dispatch

    for item_data in curated:
        try:
            url = item_data.get("source_url", "")
            dispatch.items.append(DispatchItem(
                id=DispatchItem.make_id(url),
                headline=item_data["headline"],
                source_id=item_data["source_id"],
                source_url=url,
                published=item_data.get("published"),
                fetched=fetched_at,
                category=item_data["category"],
                impact=item_data["impact"],
                summary=item_data["summary"],
                implications=item_data.get("implications", []),
                tags=item_data.get("tags", []),
            ))
        except (KeyError, TypeError) as e:
            log.warning(f"Skipping malformed item from Claude: {e} — {item_data}")

    log.info(f"Curation complete: {len(dispatch.items)} items in dispatch")
    return dispatch
