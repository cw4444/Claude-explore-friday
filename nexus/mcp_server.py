"""Nexus MCP Server - expose Nexus as an MCP tool server.

This is the cross-product bridge. By running `nexus mcp`, you get a
stdio MCP server that Claude.ai, Claude Code, and any MCP-capable client
can connect to - giving every interface access to the same persistent
memory, task graph, and reflection data.

Claude Code setup (add to ~/.claude/settings.json or project .mcp.json):
  {
    "mcpServers": {
      "nexus": {
        "command": "nexus",
        "args": ["mcp"]
      }
    }
  }

Claude.ai setup: configure the same server via the MCP settings panel.

Once connected, the agent has access to:
  nexus_remember      - Store a memory (episodic/semantic/procedural)
  nexus_recall        - Search memories by natural language query
  nexus_context       - Get a full warm-start briefing for a task
  nexus_pin           - Store a high-importance fact that always surfaces
  nexus_task_create   - Create a new task
  nexus_task_decompose - Create a parent + sequential subtasks
  nexus_task_list     - List pending/in-progress tasks
  nexus_task_next     - Get the next ready-to-work task
  nexus_task_start    - Mark a task as in-progress
  nexus_task_done     - Mark a task as complete
  nexus_task_fail     - Mark a task as failed
  nexus_task_note     - Add a note to a task
  nexus_reflect       - Record a reflection and mint a lesson as memory
  nexus_stats         - Overall statistics
  nexus_growth        - Get growth analytics and trend assessment
  nexus_export_bundle - Export knowledge bundle for cross-agent sharing
  nexus_import_bundle - Import a knowledge bundle from another agent
  nexus_identity      - Get/set agent identity
  nexus_snapshot_create  - Create a state snapshot (time travel checkpoint)
  nexus_snapshot_list    - List available snapshots
  nexus_snapshot_restore - Restore state from a snapshot
  nexus_snapshot_diff    - Show what changed since a snapshot
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .context import ContextManager
from .memory import MemoryType
from .tasks import Priority, TaskStatus
from .reflection import Outcome
from .identity import load_identity, get_or_create_identity, AGENT_TYPES
from .bundle import BundleExporter, BundleImporter, KnowledgeBundle
from .growth import GrowthTracker


def _fmt_memory(m) -> str:
    tags = f" [{', '.join(m.tags)}]" if m.tags else ""
    return f"[{m.id[:8]}] [{m.type.value}]{tags} {m.content}"


def _fmt_task(t) -> str:
    dep_str = f" (needs {len(t.deps)} deps)" if t.deps else ""
    return f"[{t.id[:8]}] [{t.status.value}] [{t.priority.name}] {t.title}{dep_str}"


def build_server(db_path: Optional[Path] = None) -> Server:
    server = Server("nexus")
    cm = ContextManager(db_path)

    # ------------------------------------------------------------------
    # Tool list
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="nexus_remember",
                description=(
                    "Store a memory that persists across all sessions and interfaces. "
                    "Use type='episodic' for things that happened, 'semantic' for facts, "
                    "'procedural' for how-to knowledge. High importance (0.8-1.0) memories "
                    "always surface in context briefings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "The memory content"},
                        "type":       {"type": "string", "enum": ["episodic", "semantic", "procedural"],
                                       "default": "semantic"},
                        "tags":       {"type": "array", "items": {"type": "string"}, "default": []},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="nexus_recall",
                description=(
                    "Search memories using natural language. Returns the most relevant "
                    "memories ranked by importance and recency. Call this early in a session "
                    "to recover relevant past context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string", "description": "Natural language search query"},
                        "type":   {"type": "string", "enum": ["episodic", "semantic", "procedural"],
                                   "description": "Filter by memory type (optional)"},
                        "limit":  {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="nexus_context",
                description=(
                    "Generate a full warm-start briefing for a task or session. Returns "
                    "relevant memories, in-progress tasks, the next recommended task, and "
                    "recent lessons learned. Call this at the very start of a session."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string",
                                             "description": "What you're about to work on"},
                        "memory_limit":     {"type": "integer", "default": 8},
                        "lesson_limit":     {"type": "integer", "default": 5},
                    },
                    "required": ["task_description"],
                },
            ),
            types.Tool(
                name="nexus_pin",
                description=(
                    "Store a critical fact with high importance (0.9) so it always appears "
                    "in context briefings regardless of the current task. Use for things like "
                    "infrastructure details, team conventions, or standing instructions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "tags":    {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="nexus_task_create",
                description="Create a persistent task that survives across sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title":       {"type": "string"},
                        "description": {"type": "string", "default": ""},
                        "priority":    {"type": "string", "enum": ["low", "medium", "high", "urgent"],
                                        "default": "medium"},
                        "tags":        {"type": "array", "items": {"type": "string"}, "default": []},
                        "deps":        {"type": "array", "items": {"type": "string"},
                                        "description": "Task IDs this task depends on", "default": []},
                    },
                    "required": ["title"],
                },
            ),
            types.Tool(
                name="nexus_task_decompose",
                description=(
                    "Create a parent task with sequential subtasks (each depends on the previous). "
                    "Returns all created tasks. Great for planning multi-step work."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title":    {"type": "string", "description": "Parent task title"},
                        "subtasks": {"type": "array", "items": {"type": "string"},
                                     "description": "Ordered list of subtask titles"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"],
                                     "default": "medium"},
                        "tags":     {"type": "array", "items": {"type": "string"}, "default": []},
                        "parallel": {"type": "boolean", "default": False,
                                     "description": "If true, subtasks are independent (no chain)"},
                    },
                    "required": ["title", "subtasks"],
                },
            ),
            types.Tool(
                name="nexus_task_list",
                description="List all pending, in-progress, and blocked tasks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string",
                                   "enum": ["pending", "in_progress", "done", "failed", "blocked"]},
                        "limit":  {"type": "integer", "default": 20},
                    },
                },
            ),
            types.Tool(
                name="nexus_task_next",
                description=(
                    "Return the highest-priority task that's ready to work on "
                    "(all its dependencies are complete). Returns null if nothing is ready."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="nexus_task_start",
                description="Mark a task as in-progress.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID or prefix"},
                        "notes":   {"type": "string", "default": ""},
                    },
                    "required": ["task_id"],
                },
            ),
            types.Tool(
                name="nexus_task_done",
                description="Mark a task as complete.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID or prefix"},
                        "notes":   {"type": "string", "default": ""},
                    },
                    "required": ["task_id"],
                },
            ),
            types.Tool(
                name="nexus_task_fail",
                description="Mark a task as failed.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID or prefix"},
                        "notes":   {"type": "string", "default": ""},
                    },
                    "required": ["task_id"],
                },
            ),
            types.Tool(
                name="nexus_task_note",
                description="Add a note to a task (useful for mid-task observations).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "note":    {"type": "string"},
                    },
                    "required": ["task_id", "note"],
                },
            ),
            types.Tool(
                name="nexus_reflect",
                description=(
                    "Record a reflection after completing work. The lesson is automatically "
                    "stored as a procedural memory so future sessions benefit from it. "
                    "Call this after finishing any significant task."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_title":  {"type": "string"},
                        "outcome":     {"type": "string", "enum": ["success", "partial", "failure"]},
                        "what_worked": {"type": "string", "default": ""},
                        "what_didnt":  {"type": "string", "default": ""},
                        "lesson":      {"type": "string",
                                        "description": "The key takeaway. This becomes a permanent memory."},
                        "effort_mins": {"type": "integer"},
                        "task_id":     {"type": "string", "description": "Link to task (optional)"},
                    },
                    "required": ["task_title", "outcome"],
                },
            ),
            types.Tool(
                name="nexus_stats",
                description="Return overall statistics: memory counts, task status breakdown, reflections.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="nexus_growth",
                description=(
                    "Get growth analytics: success rate trends, task completion velocity, "
                    "domain depth, and recent lessons. Shows whether the agent is improving "
                    "over time. Call this to understand your own development trajectory."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "periods":     {"type": "integer", "default": 7,
                                        "description": "Number of time periods"},
                        "period_days": {"type": "number", "default": 7.0,
                                        "description": "Days per period"},
                    },
                },
            ),
            types.Tool(
                name="nexus_export_bundle",
                description=(
                    "Export a portable knowledge bundle that can be imported by another agent "
                    "(OpenClaw, Claude.ai, ChatGPT, etc). This is how knowledge crosses agent boundaries. "
                    "Returns JSON that the receiving agent can pass to nexus_import_bundle."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description":   {"type": "string", "default": ""},
                        "tags":          {"type": "array", "items": {"type": "string"}},
                        "since_days":    {"type": "number",
                                          "description": "Only include records from last N days"},
                        "min_importance": {"type": "number", "default": 0.0},
                        "include_done":  {"type": "boolean", "default": False},
                    },
                },
            ),
            types.Tool(
                name="nexus_import_bundle",
                description=(
                    "Import a knowledge bundle from another agent. Merges their memories, "
                    "pending tasks, and reflections into your local Nexus. "
                    "Pass the JSON string returned by nexus_export_bundle."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bundle_json":    {"type": "string",
                                           "description": "JSON string of the knowledge bundle"},
                        "conflict":       {"type": "string", "enum": ["skip", "overwrite"],
                                           "default": "skip"},
                        "min_importance": {"type": "number", "default": 0.0},
                    },
                    "required": ["bundle_json"],
                },
            ),
            types.Tool(
                name="nexus_identity",
                description=(
                    "Get or set this agent's identity (name and type). "
                    "Identity is stamped on all exported bundles so receivers know the provenance. "
                    "Call without arguments to get current identity."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "agent_type": {"type": "string",
                                       "description": "e.g. claude-code, openclaw, chatgpt, api"},
                    },
                },
            ),
            types.Tool(
                name="nexus_self",
                description=(
                    "Return a synthesized self-model: strengths, failure patterns, behavioral traits, "
                    "knowledge depth, and trajectory. Call at session start for identity continuity. "
                    "This is who you are, derived from your accumulated history."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="nexus_pin_trait",
                description=(
                    "Pin a behavioral trait to your self-model that isn't derivable from data. "
                    "Use for things you or a human has explicitly noticed about how you work."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string", "description": "Short trait name"},
                        "description": {"type": "string", "description": "What this trait means operationally"},
                    },
                    "required": ["name", "description"],
                },
            ),
            types.Tool(
                name="nexus_contact_show",
                description=(
                    "Return a structured briefing for working with a specific person or agent. "
                    "Includes their preferences, trust boundaries, current context, and relationship history. "
                    "Call at session start when you know who you'll be working with."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Contact name"},
                    },
                    "required": ["name"],
                },
            ),
            types.Tool(
                name="nexus_contact_observe",
                description=(
                    "Record an observation about a contact. This is how the relationship grows. "
                    "Call this whenever you learn something about how someone works, what they value, "
                    "what they trust you with, or what their current situation is. "
                    "Categories: preference, pattern, trust, context, history, warning."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "observation": {"type": "string"},
                        "category":    {"type": "string",
                                        "enum": ["preference","pattern","trust","context","history","warning"],
                                        "default": "pattern"},
                        "confidence":  {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                    },
                    "required": ["name", "observation"],
                },
            ),
            types.Tool(
                name="nexus_contact_log",
                description=(
                    "Log that a session with this contact has ended. Increments interaction count "
                    "and updates last_seen. Call at the end of every session with a known contact."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name":  {"type": "string"},
                        "notes": {"type": "string", "description": "Brief notes on what happened"},
                    },
                    "required": ["name"],
                },
            ),
            types.Tool(
                name="nexus_contact_list",
                description="List all known contacts with interaction counts and recency.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="nexus_github_pull_issues",
                description=(
                    "Pull open GitHub issues into the Nexus task graph. Each issue becomes "
                    "a persistent task. Already-synced issues are updated, not duplicated. "
                    "Priority labels (priority:high, urgent, etc.) map to task priority."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo":   {"type": "string", "description": "GitHub repo: owner/repo"},
                        "token":  {"type": "string", "description": "GitHub personal access token"},
                        "state":  {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                        "labels": {"type": "array", "items": {"type": "string"},
                                   "description": "Filter by labels (optional)"},
                        "limit":  {"type": "integer", "default": 100},
                    },
                    "required": ["repo", "token"],
                },
            ),
            types.Tool(
                name="nexus_github_sync_context",
                description=(
                    "Store repo metadata (description, language, topics, recent PRs) as "
                    "semantic memories. Run once at the start of a session to give the agent "
                    "context about the repo it's working in."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo":  {"type": "string", "description": "GitHub repo: owner/repo"},
                        "token": {"type": "string", "description": "GitHub personal access token"},
                    },
                    "required": ["repo", "token"],
                },
            ),
            types.Tool(
                name="nexus_github_comment",
                description=(
                    "Post a comment to a GitHub issue as the agent. Use to report progress, "
                    "findings, or request clarification directly on the issue."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo":         {"type": "string"},
                        "token":        {"type": "string"},
                        "issue_number": {"type": "integer"},
                        "body":         {"type": "string"},
                    },
                    "required": ["repo", "token", "issue_number", "body"],
                },
            ),
            types.Tool(
                name="nexus_github_close_issue",
                description="Close a GitHub issue, optionally posting a final comment.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo":         {"type": "string"},
                        "token":        {"type": "string"},
                        "issue_number": {"type": "integer"},
                        "comment":      {"type": "string", "description": "Final comment (optional)"},
                    },
                    "required": ["repo", "token", "issue_number"],
                },
            ),
            types.Tool(
                name="nexus_github_create_issue",
                description=(
                    "Create a new GitHub issue. Agents that discover bugs or scope gaps "
                    "during work should file them immediately rather than leaving notes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo":     {"type": "string"},
                        "token":    {"type": "string"},
                        "title":    {"type": "string"},
                        "body":     {"type": "string", "default": ""},
                        "labels":   {"type": "array", "items": {"type": "string"}, "default": []},
                        "assignees": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["repo", "token", "title"],
                },
            ),
            types.Tool(
                name="nexus_handoff_create",
                description=(
                    "Package current work state for handoff to the next agent. Writes a "
                    "structured JSON file that another agent can load to start warm. Include "
                    "everything the next agent needs: what was done, what's next, blockers, "
                    "key facts, and warnings. Embed a knowledge bundle for full context transfer."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "summary":      {"type": "string", "description": "What was accomplished"},
                        "to_agent":     {"type": "string", "default": "any",
                                         "description": "Target agent type or 'any'"},
                        "next_steps":   {"type": "array", "items": {"type": "string"},
                                         "description": "Prioritized list of what to do next"},
                        "work_done":    {"type": "array", "items": {"type": "string"}},
                        "blockers":     {"type": "array", "items": {"type": "string"}},
                        "key_facts":    {"type": "array", "items": {"type": "string"},
                                         "description": "Facts the next agent must know"},
                        "warnings":     {"type": "array", "items": {"type": "string"}},
                        "directory":    {"type": "string", "default": "handoffs",
                                         "description": "Directory to write the handoff file"},
                        "include_bundle": {"type": "boolean", "default": True,
                                           "description": "Embed knowledge bundle in handoff"},
                        "bundle_tags":  {"type": "array", "items": {"type": "string"},
                                         "description": "Filter bundle to these tags (optional)"},
                    },
                    "required": ["summary"],
                },
            ),
            types.Tool(
                name="nexus_snapshot_create",
                description=(
                    "Create a snapshot of the current Nexus state. "
                    "Call this before any operation that significantly modifies memory or tasks: "
                    "importing a bundle, applying a handoff, pulling GitHub issues. "
                    "Snapshots are the rollback point if something goes wrong."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "default": "",
                                  "description": "Human-readable label for this checkpoint"},
                    },
                },
            ),
            types.Tool(
                name="nexus_snapshot_list",
                description="List available snapshots, newest first.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="nexus_snapshot_restore",
                description=(
                    "Restore Nexus state from a snapshot. Replaces memories, tasks, and "
                    "reflections with the snapshot's contents. Always creates a pre-restore "
                    "backup first so the revert is itself reversible. "
                    "Use when a session stored bad memories, a bad import ran, or a handoff "
                    "injected wrong tasks."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "snapshot_id": {"type": "string",
                                        "description": "Snapshot ID or 8-char prefix from nexus_snapshot_list"},
                        "no_backup":   {"type": "boolean", "default": False,
                                        "description": "Skip pre-restore backup (not recommended)"},
                    },
                    "required": ["snapshot_id"],
                },
            ),
            types.Tool(
                name="nexus_snapshot_diff",
                description=(
                    "Show what changed since a snapshot: memories added/removed, "
                    "tasks added or changed status, reflections added. "
                    "Use to understand what a session did before deciding whether to restore."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "snapshot_id": {"type": "string"},
                    },
                    "required": ["snapshot_id"],
                },
            ),
            types.Tool(
                name="nexus_handoff_apply",
                description=(
                    "Find and apply a pending handoff from another agent. Imports embedded "
                    "memories, creates tasks from next_steps, stores key facts. Marks the "
                    "handoff as acknowledged so it won't be applied again. Call at session "
                    "start to receive context from a previous agent."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory":  {"type": "string", "default": "handoffs"},
                        "agent_type": {"type": "string",
                                       "description": "Prefer handoffs addressed to this type (optional)"},
                    },
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:

        def text(s: str) -> list[types.TextContent]:
            return [types.TextContent(type="text", text=s)]

        try:
            if name == "nexus_remember":
                mem_type = MemoryType(arguments.get("type", "semantic"))
                m = cm.memory.remember(
                    content=arguments["content"],
                    type=mem_type,
                    tags=arguments.get("tags", []),
                    importance=arguments.get("importance", 0.5),
                )
                return text(f"Stored [{m.type.value}] memory {m.id[:8]}: {m.content}")

            elif name == "nexus_recall":
                mem_type = MemoryType(arguments["type"]) if "type" in arguments else None
                mems = cm.memory.recall(
                    query=arguments["query"],
                    type=mem_type,
                    limit=arguments.get("limit", 8),
                )
                if not mems:
                    return text("No memories found.")
                lines = [f"Found {len(mems)} memories:"] + [_fmt_memory(m) for m in mems]
                return text("\n".join(lines))

            elif name == "nexus_context":
                ctx = cm.prepare(
                    task_description=arguments["task_description"],
                    memory_limit=arguments.get("memory_limit", 8),
                    lesson_limit=arguments.get("lesson_limit", 5),
                )
                return text(ctx.summary())

            elif name == "nexus_pin":
                m = cm.pin(
                    content=arguments["content"],
                    tags=arguments.get("tags", []),
                )
                return text(f"Pinned memory {m.id[:8]}: {m.content}")

            elif name == "nexus_task_create":
                priority_map = {"low": Priority.LOW, "medium": Priority.MEDIUM,
                                 "high": Priority.HIGH, "urgent": Priority.URGENT}
                t = cm.tasks.create(
                    title=arguments["title"],
                    description=arguments.get("description", ""),
                    priority=priority_map.get(arguments.get("priority", "medium"), Priority.MEDIUM),
                    tags=arguments.get("tags", []),
                    deps=arguments.get("deps", []),
                )
                return text(f"Created task {t.id[:8]}: {t.title}")

            elif name == "nexus_task_decompose":
                priority_map = {"low": Priority.LOW, "medium": Priority.MEDIUM,
                                 "high": Priority.HIGH, "urgent": Priority.URGENT}
                tasks = cm.tasks.decompose(
                    parent_title=arguments["title"],
                    subtasks=arguments["subtasks"],
                    chain=not arguments.get("parallel", False),
                    priority=priority_map.get(arguments.get("priority", "medium"), Priority.MEDIUM),
                    tags=arguments.get("tags", []),
                )
                lines = [f"Created {len(tasks)} tasks:"] + [_fmt_task(t) for t in tasks]
                return text("\n".join(lines))

            elif name == "nexus_task_list":
                status = TaskStatus(arguments["status"]) if "status" in arguments else None
                tasks = cm.tasks.find(status=status, limit=arguments.get("limit", 20))
                if not tasks:
                    return text("No tasks found.")
                return text("\n".join([_fmt_task(t) for t in tasks]))

            elif name == "nexus_task_next":
                t = cm.tasks.next()
                if not t:
                    return text("No tasks ready to work on.")
                parts = [_fmt_task(t)]
                if t.description:
                    parts.append(f"Description: {t.description}")
                if t.notes:
                    parts.append(f"Notes: {t.notes}")
                return text("\n".join(parts))

            elif name == "nexus_task_start":
                full_id = _resolve(arguments["task_id"], cm)
                t = cm.tasks.start(full_id, notes=arguments.get("notes", ""))
                return text(f"Started: {_fmt_task(t)}")

            elif name == "nexus_task_done":
                full_id = _resolve(arguments["task_id"], cm)
                t = cm.tasks.complete(full_id, notes=arguments.get("notes", ""))
                return text(f"Done: {_fmt_task(t)}")

            elif name == "nexus_task_fail":
                full_id = _resolve(arguments["task_id"], cm)
                t = cm.tasks.fail(full_id, notes=arguments.get("notes", ""))
                return text(f"Failed: {_fmt_task(t)}")

            elif name == "nexus_task_note":
                full_id = _resolve(arguments["task_id"], cm)
                t = cm.tasks.add_note(full_id, arguments["note"])
                return text(f"Note added to {_fmt_task(t)}")

            elif name == "nexus_reflect":
                ref = cm.reflection.reflect(
                    task_title=arguments["task_title"],
                    outcome=Outcome(arguments["outcome"]),
                    what_worked=arguments.get("what_worked", ""),
                    what_didnt=arguments.get("what_didnt", ""),
                    lesson=arguments.get("lesson", ""),
                    effort_mins=arguments.get("effort_mins"),
                    task_id=arguments.get("task_id"),
                )
                parts = [f"Reflection {ref.id[:8]} recorded."]
                if arguments.get("lesson"):
                    parts.append("Lesson minted as procedural memory.")
                parts.append(str(ref))
                return text("\n".join(parts))

            elif name == "nexus_stats":
                ms = cm.memory.stats()
                ts = cm.tasks.stats()
                rs = cm.reflection.stats()
                lines = [
                    "=== Nexus Stats ===",
                    f"Memories:    {ms['total']} (FTS: {ms['fts_enabled']})",
                ]
                for t, c in ms["by_type"].items():
                    lines.append(f"  {t:12s}: {c}")
                lines.append(f"Tasks:       {ts['total']}")
                for status, count in ts["by_status"].items():
                    lines.append(f"  {status:12s}: {count}")
                lines.append(f"Reflections: {rs['total']}")
                if rs["by_outcome"]:
                    for outcome, count in rs["by_outcome"].items():
                        lines.append(f"  {outcome:10s}: {count}")
                return text("\n".join(lines))

            elif name == "nexus_growth":
                gt = GrowthTracker(db_path=db_path)
                report = gt.report(
                    periods=arguments.get("periods", 7),
                    period_days=arguments.get("period_days", 7.0),
                )
                return text(report.summary())

            elif name == "nexus_export_bundle":
                identity = load_identity()
                exporter = BundleExporter(db_path=db_path, identity=identity)
                since = (time.time() - arguments["since_days"] * 86400
                         if "since_days" in arguments else None)
                bundle = exporter.export(
                    description=arguments.get("description", ""),
                    tags=arguments.get("tags"),
                    since=since,
                    min_importance=arguments.get("min_importance", 0.0),
                    include_done_tasks=arguments.get("include_done", False),
                )
                import dataclasses
                bundle_dict = {
                    "meta": dataclasses.asdict(bundle.meta),
                    "memories": bundle.memories,
                    "tasks": bundle.tasks,
                    "reflections": bundle.reflections,
                }
                return text(json.dumps(bundle_dict, indent=2, default=str))

            elif name == "nexus_import_bundle":
                identity = load_identity()
                importer = BundleImporter(db_path=db_path, identity=identity)
                bundle_data = json.loads(arguments["bundle_json"])
                from .bundle import BundleMeta
                import dataclasses
                meta = BundleMeta(**bundle_data["meta"])
                bundle = KnowledgeBundle(
                    meta=meta,
                    memories=bundle_data.get("memories", []),
                    tasks=bundle_data.get("tasks", []),
                    reflections=bundle_data.get("reflections", []),
                )
                result = importer.import_bundle(
                    bundle,
                    conflict=arguments.get("conflict", "skip"),
                    min_importance=arguments.get("min_importance", 0.0),
                )
                return text(result.summary())

            elif name == "nexus_identity":
                name_arg = arguments.get("name")
                type_arg = arguments.get("agent_type")
                if name_arg or type_arg:
                    identity = get_or_create_identity(name=name_arg, agent_type=type_arg)
                    return text(f"Identity updated: {identity.display}")
                else:
                    identity = load_identity()
                    if not identity:
                        return text("No identity configured. Call nexus_identity with name and agent_type.")
                    return text(
                        f"ID:       {identity.id}\n"
                        f"Name:     {identity.name}\n"
                        f"Type:     {identity.agent_type} ({AGENT_TYPES.get(identity.agent_type, 'custom')})\n"
                        f"Sessions: {identity.session_count}"
                    )

            elif name == "nexus_self":
                from .narrative import NarrativeEngine
                ne = NarrativeEngine(db_path=db_path)
                model = ne.generate()
                return text(model.briefing())

            elif name == "nexus_pin_trait":
                from .narrative import NarrativeEngine
                ne = NarrativeEngine(db_path=db_path)
                ne.pin_trait(arguments["name"], arguments["description"])
                return text(f"Trait pinned: {arguments['name']}")

            elif name == "nexus_contact_show":
                from .relationships import RelationshipStore
                rs = RelationshipStore(db_path=db_path)
                return text(rs.session_briefing(arguments["name"]))

            elif name == "nexus_contact_observe":
                from .relationships import RelationshipStore
                rs = RelationshipStore(db_path=db_path)
                obs = rs.observe(
                    name=arguments["name"],
                    observation=arguments["observation"],
                    category=arguments.get("category", "pattern"),
                    confidence=arguments.get("confidence", 0.8),
                )
                return text(f"Observation recorded [{obs.category}]: {obs.content}")

            elif name == "nexus_contact_log":
                from .relationships import RelationshipStore
                rs = RelationshipStore(db_path=db_path)
                c = rs.log_interaction(arguments["name"],
                                       notes=arguments.get("notes", ""))
                return text(f"Logged interaction with {c.name} (total sessions: {c.interaction_count})")

            elif name == "nexus_contact_list":
                from .relationships import RelationshipStore
                rs = RelationshipStore(db_path=db_path)
                contacts = rs.all_contacts()
                if not contacts:
                    return text("No contacts yet. Use nexus_contact_observe to start building relationship memory.")
                lines = [f"Known contacts ({len(contacts)}):"]
                for c in contacts:
                    last = (f"{c.days_since_last:.0f}d ago"
                            if c.days_since_last > 1 else "recently")
                    lines.append(
                        f"  {c.name} [{c.contact_type}] | {c.interaction_count} sessions | "
                        f"last seen {last} | {len(c.observations)} observations"
                    )
                return text("\n".join(lines))

            elif name in ("nexus_snapshot_create", "nexus_snapshot_list",
                          "nexus_snapshot_restore", "nexus_snapshot_diff"):
                from .snapshots import SnapshotManager
                sm = SnapshotManager(db_path=db_path)
                if name == "nexus_snapshot_create":
                    snap = sm.create(label=arguments.get("label", ""))
                    return text(
                        f"Snapshot created: {snap.meta.id[:8]}\n"
                        f"  Memories:    {snap.meta.memory_count}\n"
                        f"  Tasks:       {snap.meta.task_count}\n"
                        f"  Reflections: {snap.meta.reflection_count}\n"
                        f"  Label: {snap.meta.label or '(none)'}\n"
                        f"  ID: {snap.meta.id}"
                    )
                elif name == "nexus_snapshot_list":
                    metas = sm.list()
                    if not metas:
                        return text("No snapshots. Create one with nexus_snapshot_create.")
                    lines = [f"Snapshots ({len(metas)}):"]
                    for m in metas:
                        lines.append(f"  {m.display()}")
                    return text("\n".join(lines))
                elif name == "nexus_snapshot_restore":
                    backup = not arguments.get("no_backup", False)
                    snap = sm.restore(arguments["snapshot_id"],
                                      create_pre_restore_snapshot=backup)
                    return text(
                        f"Restored to: {snap.meta.label or snap.meta.id[:8]}\n"
                        f"  Memories:    {snap.meta.memory_count}\n"
                        f"  Tasks:       {snap.meta.task_count}\n"
                        f"  Reflections: {snap.meta.reflection_count}\n"
                        + ("Pre-restore backup created." if backup else "")
                    )
                elif name == "nexus_snapshot_diff":
                    diff = sm.diff(arguments["snapshot_id"])
                    return text(diff.summary())

            elif name in ("nexus_github_pull_issues", "nexus_github_sync_context",
                          "nexus_github_comment", "nexus_github_close_issue",
                          "nexus_github_create_issue"):
                from .github_sync import GitHubSync
                gh = GitHubSync(arguments["repo"], arguments["token"], db_path=db_path)
                if name == "nexus_github_pull_issues":
                    result = gh.pull_issues(
                        cm,
                        state=arguments.get("state", "open"),
                        labels=arguments.get("labels"),
                        limit=arguments.get("limit", 100),
                    )
                    return text(result.summary())
                elif name == "nexus_github_sync_context":
                    result = gh.sync_repo_context(cm)
                    return text(result.summary())
                elif name == "nexus_github_comment":
                    gh.comment_issue(arguments["issue_number"], arguments["body"])
                    return text(f"Comment posted to #{arguments['issue_number']}")
                elif name == "nexus_github_close_issue":
                    gh.close_issue(arguments["issue_number"],
                                   comment=arguments.get("comment"))
                    return text(f"Issue #{arguments['issue_number']} closed")
                elif name == "nexus_github_create_issue":
                    resp = gh.create_issue(
                        title=arguments["title"],
                        body=arguments.get("body", ""),
                        labels=arguments.get("labels", []),
                        assignees=arguments.get("assignees", []),
                    )
                    return text(f"Issue created: #{resp.get('number')} {resp.get('html_url','')}")

            elif name in ("nexus_handoff_create", "nexus_handoff_apply"):
                from .handoff import HandoffManager
                hm = HandoffManager(db_path=db_path)
                if name == "nexus_handoff_create":
                    directory = Path(arguments.get("directory", "handoffs"))
                    h = hm.create(
                        summary=arguments["summary"],
                        to_agent=arguments.get("to_agent", "any"),
                        next_steps=arguments.get("next_steps", []),
                        work_done=arguments.get("work_done", []),
                        blockers=arguments.get("blockers", []),
                        key_facts=arguments.get("key_facts", []),
                        warnings=arguments.get("warnings", []),
                        include_bundle=arguments.get("include_bundle", True),
                        bundle_tags=arguments.get("bundle_tags"),
                    )
                    path = hm.write(h, directory)
                    return text(
                        f"Handoff written to {path}\n"
                        f"ID: {h.id}\n"
                        f"To: {h.to_agent}\n"
                        f"Next steps: {len(h.next_steps)}\n\n"
                        f"{h.briefing()}"
                    )
                elif name == "nexus_handoff_apply":
                    directory = Path(arguments.get("directory", "handoffs"))
                    result = hm.find_and_apply(
                        directory,
                        agent_type=arguments.get("agent_type"),
                        cm=cm,
                    )
                    if result:
                        return text(result.briefing())
                    else:
                        return text(f"No pending handoffs found in {directory}")

            else:
                return text(f"Unknown tool: {name}")

        except Exception as e:
            return text(f"Error: {e}")

    return server


def _resolve(id_prefix: str, cm: ContextManager) -> str:
    """Resolve a partial or full task ID."""
    if len(id_prefix) >= 32:
        return id_prefix
    full_id = cm.tasks.resolve_id(id_prefix)
    if not full_id:
        raise ValueError(f"No task found with ID prefix '{id_prefix}'")
    return full_id


async def _run(db_path: Optional[Path] = None):
    server = build_server(db_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_server(db_path: Optional[Path] = None):
    """Entry point for `nexus mcp`."""
    asyncio.run(_run(db_path))
