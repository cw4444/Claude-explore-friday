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
  nexus_task_list     - List pending/in-progress tasks
  nexus_task_next     - Get the next ready-to-work task
  nexus_task_start    - Mark a task as in-progress
  nexus_task_done     - Mark a task as complete
  nexus_task_fail     - Mark a task as failed
  nexus_task_note     - Add a note to a task
  nexus_reflect       - Record a reflection and mint a lesson as memory
  nexus_stats         - Overall statistics
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .context import ContextManager
from .memory import MemoryType
from .tasks import Priority, TaskStatus
from .reflection import Outcome


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
