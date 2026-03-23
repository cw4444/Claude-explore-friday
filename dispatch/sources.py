"""
Source definitions and raw fetching.

Each source produces a list of RawItem dicts:
  { url, title, content, published, source_id }

Fetching is async and fail-soft — a broken source never kills the run.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import feedparser

log = logging.getLogger(__name__)

FETCH_TIMEOUT = 15  # seconds per source
USER_AGENT = "AgentDispatch/1.0 (daily AI news feed for agents)"


@dataclass
class RawItem:
    url: str
    title: str
    content: str        # best available text: full body > summary > description
    published: str | None
    source_id: str
    source_name: str


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

RSS_SOURCES = [
    {
        "id": "anthropic-blog",
        "name": "Anthropic Blog",
        "url": "https://www.anthropic.com/rss.xml",
    },
    {
        "id": "openai-news",
        "name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
    },
    {
        "id": "google-deepmind",
        "name": "Google DeepMind",
        "url": "https://deepmind.google/blog/rss.xml",
    },
    {
        "id": "techcrunch-ai",
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
    },
    {
        "id": "steipete-blog",
        "name": "Peter Steinberger",
        "url": "https://steipete.me/feed.xml",
    },
    {
        "id": "simonw-blog",
        "name": "Simon Willison",
        "url": "https://simonwillison.net/atom/everything/",
    },
    {
        "id": "xai-blog",
        "name": "xAI Blog",
        "url": "https://x.ai/blog/rss.xml",
    },
]

GITHUB_RELEASE_SOURCES = [
    {
        "id": "openclaw-github",
        "name": "OpenClaw",
        "repo": "OpenClaw/OpenClaw",  # adjust to real repo slug
    },
    {
        "id": "claude-code-github",
        "name": "Claude Code",
        "repo": "anthropics/claude-code",
    },
    {
        "id": "anthropic-sdk-python",
        "name": "Anthropic Python SDK",
        "repo": "anthropics/anthropic-sdk-python",
    },
    {
        "id": "modelcontextprotocol",
        "name": "Model Context Protocol",
        "repo": "modelcontextprotocol/specification",
    },
]

# Hacker News — top stories via Algolia that mention AI keywords
HN_KEYWORDS = [
    "llm", "claude", "gpt", "gemini", "anthropic", "openai", "deepmind",
    "agent", "mcp", "model context", "jailbreak", "prompt injection",
]


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _parse_date(entry: Any) -> str | None:
    """Try to extract an ISO 8601 date from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return None


def _best_content(entry: Any) -> str:
    """Extract the best available text from a feedparser entry."""
    # content list (full article)
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    # summary
    if hasattr(entry, "summary") and entry.summary:
        return entry.summary
    # description
    if hasattr(entry, "description") and entry.description:
        return entry.description
    return entry.get("title", "")


async def fetch_rss(source: dict, client: httpx.AsyncClient) -> list[RawItem]:
    """Fetch and parse an RSS/Atom feed."""
    items: list[RawItem] = []
    try:
        resp = await client.get(source["url"], timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        for entry in feed.entries[:10]:  # cap at 10 most recent per source
            url = entry.get("link", "")
            if not url:
                continue
            items.append(RawItem(
                url=url,
                title=entry.get("title", ""),
                content=_best_content(entry),
                published=_parse_date(entry),
                source_id=source["id"],
                source_name=source["name"],
            ))
        log.info(f"[{source['id']}] fetched {len(items)} items")
    except Exception as e:
        log.warning(f"[{source['id']}] fetch failed: {e}")
    return items


async def fetch_github_releases(source: dict, client: httpx.AsyncClient) -> list[RawItem]:
    """Fetch GitHub releases for a repo."""
    items: list[RawItem] = []
    try:
        url = f"https://api.github.com/repos/{source['repo']}/releases"
        resp = await client.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code == 404:
            log.warning(f"[{source['id']}] repo not found: {source['repo']}")
            return items
        resp.raise_for_status()
        releases = resp.json()
        for release in releases[:5]:  # last 5 releases
            if release.get("draft"):
                continue
            items.append(RawItem(
                url=release.get("html_url", ""),
                title=f"{source['name']} {release.get('tag_name', '')} released",
                content=release.get("body", "") or "",
                published=release.get("published_at"),
                source_id=source["id"],
                source_name=source["name"],
            ))
        log.info(f"[{source['id']}] fetched {len(items)} releases")
    except Exception as e:
        log.warning(f"[{source['id']}] github fetch failed: {e}")
    return items


async def fetch_hn_ai(client: httpx.AsyncClient) -> list[RawItem]:
    """Fetch top HN stories mentioning AI via Algolia search API."""
    items: list[RawItem] = []
    try:
        query = " OR ".join(HN_KEYWORDS[:6])  # Algolia supports OR
        resp = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": query,
                "tags": "story",
                "numericFilters": "points>50",
                "hitsPerPage": 15,
            },
            timeout=FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        for hit in hits:
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            items.append(RawItem(
                url=url,
                title=hit.get("title", ""),
                content=hit.get("story_text") or hit.get("title", ""),
                published=datetime.fromtimestamp(
                    hit["created_at_i"], tz=timezone.utc
                ).isoformat() if hit.get("created_at_i") else None,
                source_id="hacker-news",
                source_name="Hacker News",
            ))
        log.info(f"[hacker-news] fetched {len(items)} items")
    except Exception as e:
        log.warning(f"[hacker-news] fetch failed: {e}")
    return items


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fetch_all(github_token: str | None = None) -> list[RawItem]:
    """Fetch from all sources concurrently. Returns deduplicated RawItems."""
    headers = {"User-Agent": USER_AGENT}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        tasks = []
        for source in RSS_SOURCES:
            tasks.append(fetch_rss(source, client))
        for source in GITHUB_RELEASE_SOURCES:
            tasks.append(fetch_github_releases(source, client))
        tasks.append(fetch_hn_ai(client))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[RawItem] = []
    seen_urls: set[str] = set()
    for result in results:
        if isinstance(result, Exception):
            log.warning(f"Source task failed: {result}")
            continue
        for item in result:
            if item.url and item.url not in seen_urls:
                seen_urls.add(item.url)
                all_items.append(item)

    log.info(f"Total raw items after dedup: {len(all_items)}")
    return all_items
