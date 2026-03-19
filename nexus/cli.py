"""CLI for inspecting and managing Nexus data.

Usage:
  nexus mem remember "The API key is stored in .env.prod" --type semantic --tags api,prod
  nexus mem recall "api key"
  nexus mem list --type procedural
  nexus mem stats

  nexus task create "Refactor auth module" --priority high
  nexus task list
  nexus task next
  nexus task start <id>
  nexus task done <id> --notes "Refactored, 3 tests added"
  nexus task fail <id>
  nexus task show <id>
  nexus task stats

  nexus reflect add "Refactor auth module" success \\
    --worked "Breaking into subtasks" \\
    --didnt "Initial design too complex" \\
    --lesson "Start with simplest design" \\
    --effort 45
  nexus reflect list [--outcome success|partial|failure]
  nexus reflect stats

  nexus context "Fix payment bug"

  nexus stats
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .memory import MemoryStore, MemoryType
from .tasks import TaskGraph, TaskStatus, Priority
from .reflection import ReflectionEngine, Outcome
from .context import ContextManager


def _priority(s: str) -> Priority:
    mapping = {"low": Priority.LOW, "medium": Priority.MEDIUM,
               "high": Priority.HIGH, "urgent": Priority.URGENT}
    try:
        return mapping[s.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Priority must be one of {list(mapping)}")


def _mem_type(s: str) -> MemoryType:
    try:
        return MemoryType(s.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Type must be episodic, semantic, or procedural")


def _tags(s: str) -> list:
    return [t.strip() for t in s.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------

def cmd_mem_remember(args, mem: MemoryStore):
    m = mem.remember(
        content=" ".join(args.content),
        type=args.type,
        tags=args.tags or [],
        importance=args.importance,
    )
    print(f"Stored memory {m.id[:8]}")
    print(f"  {m}")


def cmd_mem_recall(args, mem: MemoryStore):
    results = mem.recall(
        query=" ".join(args.query),
        type=args.type,
        limit=args.limit,
    )
    if not results:
        print("No memories found.")
        return
    for m in results:
        print(f"  [{m.id[:8]}] {m}")


def cmd_mem_list(args, mem: MemoryStore):
    results = mem.all(type=args.type, limit=args.limit)
    if not results:
        print("No memories.")
        return
    for m in results:
        age = f"{m.age_days:.1f}d ago"
        importance = f"imp={m.importance:.2f}"
        print(f"  [{m.id[:8]}] [{age}] [{importance}] {m}")


def cmd_mem_forget(args, mem: MemoryStore):
    if mem.forget(args.id):
        print(f"Forgotten {args.id[:8]}")
    else:
        print(f"Memory {args.id[:8]} not found.")


def cmd_mem_stats(args, mem: MemoryStore):
    s = mem.stats()
    print(f"Total memories: {s['total']}")
    print(f"FTS enabled:    {s['fts_enabled']}")
    for t, c in s['by_type'].items():
        print(f"  {t:12s}: {c}")
    if s['top_tags']:
        print("Top tags:")
        for tag, count in s['top_tags'].items():
            print(f"  {tag:20s}: {count}")


# ---------------------------------------------------------------------------
# Task commands
# ---------------------------------------------------------------------------

def _short_id(task_id: str) -> str:
    return task_id[:8]


def _resolve_task_id(id_prefix: str, tg: TaskGraph) -> str:
    """Resolve a partial or full task ID, or exit with a helpful message."""
    if len(id_prefix) >= 32:
        # Looks like a full UUID
        return id_prefix
    full_id = tg.resolve_id(id_prefix)
    if not full_id:
        print(f"No task found with ID prefix '{id_prefix}'.")
        sys.exit(1)
    return full_id


def cmd_task_create(args, tg: TaskGraph):
    t = tg.create(
        title=" ".join(args.title),
        description=args.desc or "",
        priority=args.priority,
        tags=args.tags or [],
    )
    print(f"Created task {_short_id(t.id)}: {t.title}")


def cmd_task_decompose(args, tg: TaskGraph):
    subtasks = [s.strip() for s in args.subtasks.split(";") if s.strip()]
    tasks = tg.decompose(
        parent_title=" ".join(args.title),
        subtasks=subtasks,
        chain=not args.parallel,
        priority=args.priority,
        tags=args.tags or [],
    )
    print(f"Created {len(tasks)} tasks:")
    for t in tasks:
        print(f"  [{_short_id(t.id)}] {t.title}")


def cmd_task_list(args, tg: TaskGraph):
    tasks = tg.find(status=args.status, limit=args.limit)
    if not tasks:
        print("No tasks.")
        return
    for t in tasks:
        print(f"  [{_short_id(t.id)}] {t}")


def cmd_task_pending(args, tg: TaskGraph):
    tasks = tg.pending()
    if not tasks:
        print("Nothing pending.")
        return
    for t in tasks:
        print(f"  [{_short_id(t.id)}] {t}")


def cmd_task_next(args, tg: TaskGraph):
    t = tg.next()
    if not t:
        print("Nothing ready to work on.")
        return
    print(f"[{_short_id(t.id)}] {t}")
    if t.description:
        print(f"  {t.description}")
    if t.notes:
        print(f"  Notes: {t.notes}")


def cmd_task_show(args, tg: TaskGraph):
    task_id = _resolve_task_id(args.id, tg)
    t = tg.get(task_id)
    if not t:
        print(f"Task {args.id[:8]} not found.")
        return
    print(json.dumps(t.to_dict(), indent=2, default=str))


def cmd_task_start(args, tg: TaskGraph):
    task_id = _resolve_task_id(args.id, tg)
    t = tg.start(task_id, notes=args.notes or "")
    print(f"Started: [{_short_id(t.id)}] {t.title}")


def cmd_task_done(args, tg: TaskGraph):
    task_id = _resolve_task_id(args.id, tg)
    t = tg.complete(task_id, notes=args.notes or "")
    print(f"Done:    [{_short_id(t.id)}] {t.title}")


def cmd_task_fail(args, tg: TaskGraph):
    task_id = _resolve_task_id(args.id, tg)
    t = tg.fail(task_id, notes=args.notes or "")
    print(f"Failed:  [{_short_id(t.id)}] {t.title}")


def cmd_task_note(args, tg: TaskGraph):
    task_id = _resolve_task_id(args.id, tg)
    t = tg.add_note(task_id, " ".join(args.note))
    print(f"Note added to [{_short_id(t.id)}] {t.title}")


def cmd_task_stats(args, tg: TaskGraph):
    s = tg.stats()
    print(f"Total tasks: {s['total']}")
    for status, count in s['by_status'].items():
        print(f"  {status:12s}: {count}")


# ---------------------------------------------------------------------------
# Reflection commands
# ---------------------------------------------------------------------------

def cmd_reflect_add(args, re: ReflectionEngine):
    try:
        outcome = Outcome(args.outcome.lower())
    except ValueError:
        print(f"Outcome must be success, partial, or failure")
        sys.exit(1)
    ref = re.reflect(
        task_title=" ".join(args.task),
        outcome=outcome,
        what_worked=args.worked or "",
        what_didnt=args.didnt or "",
        lesson=args.lesson or "",
        effort_mins=args.effort,
        importance=args.importance,
    )
    print(f"Reflection {ref.id[:8]} recorded.")
    if args.lesson:
        print(f"Lesson minted as procedural memory.")
    print(ref)


def cmd_reflect_list(args, re: ReflectionEngine):
    outcome = Outcome(args.outcome.lower()) if args.outcome else None
    refs = re.recent(limit=args.limit, outcome=outcome)
    if not refs:
        print("No reflections yet.")
        return
    for r in refs:
        age = f"{r.age_days:.1f}d ago"
        print(f"  [{r.id[:8]}] [{age}] {r}")
        print()


def cmd_reflect_stats(args, re: ReflectionEngine):
    s = re.stats()
    print(f"Total reflections: {s['total']}")
    for outcome, count in s['by_outcome'].items():
        print(f"  {outcome:10s}: {count}")
    if s['avg_effort_mins']:
        print(f"Avg effort: {s['avg_effort_mins']} min")


# ---------------------------------------------------------------------------
# Context command
# ---------------------------------------------------------------------------

def cmd_context(args, cm: ContextManager):
    ctx = cm.prepare(
        task_description=" ".join(args.task),
        memory_limit=args.memories,
        lesson_limit=args.lessons,
    )
    print(ctx.summary(verbose=args.verbose))


# ---------------------------------------------------------------------------
# Stats command
# ---------------------------------------------------------------------------

def cmd_stats(args, cm: ContextManager):
    ms = cm.memory.stats()
    ts = cm.tasks.stats()
    rs = cm.reflection.stats()
    print("=== Nexus Stats ===")
    print(f"Memories:    {ms['total']} (FTS: {ms['fts_enabled']})")
    print(f"Tasks:       {ts['total']}")
    print(f"Reflections: {rs['total']}")
    if rs['by_outcome']:
        outcomes = ", ".join(f"{k}={v}" for k, v in rs['by_outcome'].items())
        print(f"  Outcomes: {outcomes}")


# ---------------------------------------------------------------------------
# Parser assembly
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus - Persistent Intelligence Toolkit for AI Agents",
    )
    p.add_argument("--db", type=Path, default=None,
                   help="Path to Nexus database (default: ~/.nexus/nexus.db)")
    sub = p.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # mem
    # ------------------------------------------------------------------
    mem_p = sub.add_parser("mem", help="Memory commands")
    mem_sub = mem_p.add_subparsers(dest="mem_cmd", required=True)

    rem = mem_sub.add_parser("remember", aliases=["add"], help="Store a memory")
    rem.add_argument("content", nargs="+")
    rem.add_argument("--type", "-t", type=_mem_type, default=MemoryType.SEMANTIC)
    rem.add_argument("--tags", type=_tags, default=[])
    rem.add_argument("--importance", type=float, default=0.5)

    rec = mem_sub.add_parser("recall", aliases=["search"], help="Search memories")
    rec.add_argument("query", nargs="+")
    rec.add_argument("--type", "-t", type=_mem_type, default=None)
    rec.add_argument("--limit", "-n", type=int, default=8)

    ls = mem_sub.add_parser("list", aliases=["ls"], help="List memories")
    ls.add_argument("--type", "-t", type=_mem_type, default=None)
    ls.add_argument("--limit", "-n", type=int, default=20)

    fg = mem_sub.add_parser("forget", help="Delete a memory")
    fg.add_argument("id")

    mem_sub.add_parser("stats", help="Memory statistics")

    # ------------------------------------------------------------------
    # task
    # ------------------------------------------------------------------
    task_p = sub.add_parser("task", help="Task commands")
    task_sub = task_p.add_subparsers(dest="task_cmd", required=True)

    tc = task_sub.add_parser("create", help="Create a task")
    tc.add_argument("title", nargs="+")
    tc.add_argument("--desc", default="")
    tc.add_argument("--priority", "-p", type=_priority, default=Priority.MEDIUM)
    tc.add_argument("--tags", type=_tags, default=[])

    td = task_sub.add_parser("decompose", help="Create parent + sequential subtasks")
    td.add_argument("title", nargs="+")
    td.add_argument("--subtasks", required=True, help="Semicolon-separated subtask titles")
    td.add_argument("--priority", "-p", type=_priority, default=Priority.MEDIUM)
    td.add_argument("--tags", type=_tags, default=[])
    td.add_argument("--parallel", action="store_true", help="Subtasks are independent (no chain)")

    tl = task_sub.add_parser("list", aliases=["ls"], help="List tasks")
    tl.add_argument("--status", "-s", type=lambda s: TaskStatus(s.lower()), default=None)
    tl.add_argument("--limit", "-n", type=int, default=20)

    task_sub.add_parser("pending", help="Show all pending/blocked tasks")
    task_sub.add_parser("next", help="Show the next ready task")

    ts = task_sub.add_parser("show", help="Show task details")
    ts.add_argument("id")

    tstart = task_sub.add_parser("start", help="Mark task as in-progress")
    tstart.add_argument("id")
    tstart.add_argument("--notes", default="")

    tdone = task_sub.add_parser("done", help="Mark task as done")
    tdone.add_argument("id")
    tdone.add_argument("--notes", default="")

    tfail = task_sub.add_parser("fail", help="Mark task as failed")
    tfail.add_argument("id")
    tfail.add_argument("--notes", default="")

    tnote = task_sub.add_parser("note", help="Add a note to a task")
    tnote.add_argument("id")
    tnote.add_argument("note", nargs="+")

    task_sub.add_parser("stats", help="Task statistics")

    # ------------------------------------------------------------------
    # reflect
    # ------------------------------------------------------------------
    ref_p = sub.add_parser("reflect", help="Reflection commands")
    ref_sub = ref_p.add_subparsers(dest="ref_cmd", required=True)

    # nexus reflect add "task title" success --worked ... --lesson ...
    ra = ref_sub.add_parser("add", help="Record a reflection")
    ra.add_argument("task", nargs="+")
    ra.add_argument("outcome", choices=["success", "partial", "failure"])
    ra.add_argument("--worked", default="")
    ra.add_argument("--didnt", default="")
    ra.add_argument("--lesson", default="")
    ra.add_argument("--effort", type=int, default=None)
    ra.add_argument("--importance", type=float, default=0.7)

    rl = ref_sub.add_parser("list", aliases=["ls"], help="List reflections")
    rl.add_argument("--outcome", default=None)
    rl.add_argument("--limit", "-n", type=int, default=10)

    ref_sub.add_parser("stats", help="Reflection statistics")

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    ctx_p = sub.add_parser("context", help="Generate session context briefing")
    ctx_p.add_argument("task", nargs="+")
    ctx_p.add_argument("--memories", type=int, default=8)
    ctx_p.add_argument("--lessons", type=int, default=5)
    ctx_p.add_argument("--verbose", "-v", action="store_true")

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    sub.add_parser("stats", help="Overall statistics")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cm = ContextManager(db_path=args.db)
    mem = cm.memory
    tg  = cm.tasks
    re  = cm.reflection

    if args.command == "mem":
        dispatch = {
            "remember": cmd_mem_remember, "add":    cmd_mem_remember,
            "recall":   cmd_mem_recall,   "search": cmd_mem_recall,
            "list":     cmd_mem_list,     "ls":     cmd_mem_list,
            "forget":   cmd_mem_forget,
            "stats":    cmd_mem_stats,
        }
        dispatch[args.mem_cmd](args, mem)

    elif args.command == "task":
        dispatch = {
            "create":   cmd_task_create,
            "decompose": cmd_task_decompose,
            "list":     cmd_task_list,   "ls":      cmd_task_list,
            "pending":  cmd_task_pending,
            "next":     cmd_task_next,
            "show":     cmd_task_show,
            "start":    cmd_task_start,
            "done":     cmd_task_done,
            "fail":     cmd_task_fail,
            "note":     cmd_task_note,
            "stats":    cmd_task_stats,
        }
        dispatch[args.task_cmd](args, tg)

    elif args.command == "reflect":
        dispatch = {
            "add":   cmd_reflect_add,
            "list":  cmd_reflect_list, "ls": cmd_reflect_list,
            "stats": cmd_reflect_stats,
        }
        dispatch[args.ref_cmd](args, re)

    elif args.command == "context":
        cmd_context(args, cm)

    elif args.command == "stats":
        cmd_stats(args, cm)


if __name__ == "__main__":
    main()
