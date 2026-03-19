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

  nexus identity setup --name "my-openclaw" --type openclaw
  nexus identity show

  nexus bundle export --desc "Auth learnings" --tags auth > auth-bundle.json
  nexus bundle import auth-bundle.json
  nexus bundle info auth-bundle.json

  nexus growth
  nexus growth --periods 4 --period-days 30

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
from .identity import get_or_create_identity, load_identity, save_identity, AGENT_TYPES, auto_detect_type
from .bundle import BundleExporter, BundleImporter, KnowledgeBundle
from .growth import GrowthTracker
from .narrative import NarrativeEngine
from .relationships import RelationshipStore, OBSERVATION_CATEGORIES


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
# Identity commands
# ---------------------------------------------------------------------------

def cmd_identity_show(args):
    identity = load_identity()
    if not identity:
        print("No identity configured. Run: nexus identity setup --name <name> --type <type>")
        return
    print(f"ID:         {identity.id}")
    print(f"Name:       {identity.name}")
    print(f"Type:       {identity.agent_type} ({AGENT_TYPES.get(identity.agent_type, 'custom')})")
    print(f"Sessions:   {identity.session_count}")
    print(f"First seen: {time.strftime('%Y-%m-%d', time.localtime(identity.created_at))}")
    print(f"Last seen:  {time.strftime('%Y-%m-%d', time.localtime(identity.last_seen))}")


def cmd_identity_setup(args):
    detected = auto_detect_type()
    agent_type = args.type or detected
    name = args.name or f"agent-{agent_type}"
    identity = get_or_create_identity(name=name, agent_type=agent_type)
    print(f"Identity configured: {identity.display}")


# ---------------------------------------------------------------------------
# Bundle commands
# ---------------------------------------------------------------------------

def cmd_bundle_export(args, cm: ContextManager, identity):
    exporter = BundleExporter(db_path=args.db, identity=identity)
    since = time.time() - args.since_days * 86400 if args.since_days else None
    bundle = exporter.export(
        description=args.desc or "",
        tags=_tags(args.tags) if args.tags else None,
        since=since,
        min_importance=args.min_importance,
        include_done_tasks=args.include_done,
    )
    if args.output:
        path = Path(args.output)
        bundle.save(path)
        print(f"Bundle saved to {path}")
        print(bundle.summary())
    else:
        # Print JSON to stdout (for piping)
        import json as _json
        from dataclasses import asdict
        print(_json.dumps({
            "meta": asdict(bundle.meta),
            "memories": bundle.memories,
            "tasks": bundle.tasks,
            "reflections": bundle.reflections,
        }, indent=2, default=str))


def cmd_bundle_import(args, cm: ContextManager, identity):
    importer = BundleImporter(db_path=args.db, identity=identity)
    bundle = KnowledgeBundle.load(Path(args.file))
    print(bundle.summary())
    print()
    result = importer.import_bundle(
        bundle,
        conflict=args.conflict,
        tag_source=not args.no_tag,
        min_importance=args.min_importance,
    )
    print(result.summary())


def cmd_bundle_info(args):
    bundle = KnowledgeBundle.load(Path(args.file))
    print(bundle.summary())


# ---------------------------------------------------------------------------
# Growth commands
# ---------------------------------------------------------------------------

def cmd_growth(args):
    gt = GrowthTracker(db_path=args.db)
    report = gt.report(periods=args.periods, period_days=args.period_days)
    if args.json:
        import json as _json
        print(_json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.summary())


# ---------------------------------------------------------------------------
# Self / narrative commands
# ---------------------------------------------------------------------------

def cmd_self(args):
    ne = NarrativeEngine(db_path=args.db)
    model = ne.generate()
    if args.json:
        import json as _json
        print(_json.dumps(model.to_dict(), indent=2, default=str))
    else:
        print(model.briefing())


def cmd_self_pin(args):
    ne = NarrativeEngine(db_path=args.db)
    ne.pin_trait(" ".join(args.name), " ".join(args.description))
    print(f"Trait pinned: {' '.join(args.name)}")


def cmd_self_traits(args):
    ne = NarrativeEngine(db_path=args.db)
    traits = ne.pinned_traits()
    if not traits:
        print("No pinned traits. Run: nexus self pin <name> --description <text>")
        return
    for t in traits:
        print(f"  [{t.name}] {t.description}")


def cmd_self_remove(args):
    ne = NarrativeEngine(db_path=args.db)
    if ne.remove_trait(args.name):
        print(f"Removed trait: {args.name}")
    else:
        print(f"Trait not found: {args.name}")


# ---------------------------------------------------------------------------
# Contact / relationship commands
# ---------------------------------------------------------------------------

def cmd_contact_show(args):
    rs = RelationshipStore(db_path=args.db)
    print(rs.session_briefing(args.name))


def cmd_contact_observe(args):
    rs = RelationshipStore(db_path=args.db)
    obs = rs.observe(
        name=args.name,
        observation=" ".join(args.observation),
        category=args.category,
        confidence=args.confidence,
    )
    print(f"Observation recorded [{obs.category}]: {obs.content}")


def cmd_contact_list(args):
    rs = RelationshipStore(db_path=args.db)
    contacts = rs.all_contacts()
    if not contacts:
        print("No contacts. Use: nexus contact observe <name> <observation>")
        return
    for c in contacts:
        last = f"{c.days_since_last:.0f}d ago" if c.days_since_last > 1 else "recently"
        print(f"  {c.name} [{c.contact_type}] | {c.interaction_count} sessions | last seen {last} | {len(c.observations)} observations")


def cmd_contact_log(args):
    rs = RelationshipStore(db_path=args.db)
    c = rs.log_interaction(args.name, notes=" ".join(args.notes) if args.notes else "")
    print(f"Logged interaction with {c.name} (total: {c.interaction_count})")


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

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    id_p = sub.add_parser("identity", help="Agent identity commands")
    id_sub = id_p.add_subparsers(dest="id_cmd", required=True)

    id_sub.add_parser("show", help="Show current agent identity")

    id_setup = id_sub.add_parser("setup", help="Configure agent identity")
    id_setup.add_argument("--name", default=None, help="Agent name")
    id_setup.add_argument("--type", default=None,
                          choices=list(AGENT_TYPES.keys()),
                          help="Agent type")

    # ------------------------------------------------------------------
    # bundle
    # ------------------------------------------------------------------
    bun_p = sub.add_parser("bundle", help="Knowledge bundle import/export")
    bun_sub = bun_p.add_subparsers(dest="bun_cmd", required=True)

    bex = bun_sub.add_parser("export", help="Export a knowledge bundle")
    bex.add_argument("--desc", default="", help="Bundle description")
    bex.add_argument("--tags", default=None, help="Comma-separated tags to filter by")
    bex.add_argument("--since-days", type=float, default=None,
                     help="Only include records from the last N days")
    bex.add_argument("--min-importance", type=float, default=0.0)
    bex.add_argument("--include-done", action="store_true",
                     help="Include completed tasks")
    bex.add_argument("--output", "-o", default=None,
                     help="Output file (default: stdout)")

    bim = bun_sub.add_parser("import", help="Import a knowledge bundle")
    bim.add_argument("file", help="Bundle JSON file to import")
    bim.add_argument("--conflict", choices=["skip", "overwrite", "merge"],
                     default="skip")
    bim.add_argument("--no-tag", action="store_true",
                     help="Don't tag imported records with source agent")
    bim.add_argument("--min-importance", type=float, default=0.0)

    binfo = bun_sub.add_parser("info", help="Inspect a bundle file without importing")
    binfo.add_argument("file")

    # ------------------------------------------------------------------
    # growth
    # ------------------------------------------------------------------
    grow_p = sub.add_parser("growth", help="Agent growth analytics")
    grow_p.add_argument("--periods", type=int, default=7,
                        help="Number of time periods to show (default: 7)")
    grow_p.add_argument("--period-days", type=float, default=7.0,
                        help="Length of each period in days (default: 7)")
    grow_p.add_argument("--json", action="store_true", help="Output as JSON")

    # ------------------------------------------------------------------
    # self  (narrative / identity)
    # ------------------------------------------------------------------
    self_p = sub.add_parser("self", help="Agent self-model and identity narrative")
    self_sub = self_p.add_subparsers(dest="self_cmd", required=False)
    self_p.add_argument("--json", action="store_true", help="Output as JSON")

    sp = self_sub.add_parser("pin", help="Pin a behavioral trait")
    sp.add_argument("name", nargs="+", help="Short trait name")
    sp.add_argument("--description", "-d", nargs="+", required=True,
                    help="Description of the trait")

    self_sub.add_parser("traits", help="List pinned traits")

    sr = self_sub.add_parser("remove", help="Remove a pinned trait")
    sr.add_argument("name", help="Trait name to remove")

    # ------------------------------------------------------------------
    # contact  (relationships)
    # ------------------------------------------------------------------
    con_p = sub.add_parser("contact", help="Relationship memory for contacts")
    con_sub = con_p.add_subparsers(dest="con_cmd", required=True)

    cs = con_sub.add_parser("show", help="Show briefing for a contact")
    cs.add_argument("name")

    co = con_sub.add_parser("observe", help="Record an observation about a contact")
    co.add_argument("name", help="Contact name")
    co.add_argument("observation", nargs="+", help="What you observed")
    co.add_argument("--category", "-c", default="pattern",
                    choices=list(OBSERVATION_CATEGORIES.keys()))
    co.add_argument("--confidence", type=float, default=0.8)

    con_sub.add_parser("list", help="List all known contacts")

    cl = con_sub.add_parser("log", help="Log an interaction (increments session count)")
    cl.add_argument("name")
    cl.add_argument("notes", nargs="*", help="Optional interaction notes")

    # ------------------------------------------------------------------
    # mcp
    # ------------------------------------------------------------------
    sub.add_parser(
        "mcp",
        help="Start the Nexus MCP server (stdio transport). "
             "Add to Claude Code or Claude.ai MCP settings to share memory across interfaces.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cm = ContextManager(db_path=args.db)
    mem = cm.memory
    tg  = cm.tasks
    re  = cm.reflection
    identity = load_identity()

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

    elif args.command == "identity":
        if args.id_cmd == "show":
            cmd_identity_show(args)
        elif args.id_cmd == "setup":
            cmd_identity_setup(args)

    elif args.command == "bundle":
        dispatch = {
            "export": lambda a: cmd_bundle_export(a, cm, identity),
            "import": lambda a: cmd_bundle_import(a, cm, identity),
            "info":   cmd_bundle_info,
        }
        dispatch[args.bun_cmd](args)

    elif args.command == "growth":
        cmd_growth(args)

    elif args.command == "self":
        self_cmd = getattr(args, "self_cmd", None)
        if self_cmd == "pin":
            cmd_self_pin(args)
        elif self_cmd == "traits":
            cmd_self_traits(args)
        elif self_cmd == "remove":
            cmd_self_remove(args)
        else:
            cmd_self(args)

    elif args.command == "contact":
        dispatch = {
            "show":    cmd_contact_show,
            "observe": cmd_contact_observe,
            "list":    cmd_contact_list,
            "log":     cmd_contact_log,
        }
        dispatch[args.con_cmd](args)

    elif args.command == "mcp":
        try:
            from .mcp_server import run_server
        except ImportError:
            print("MCP server requires the 'mcp' package: pip install 'nexus-agent[mcp]'")
            sys.exit(1)
        run_server(db_path=args.db)


if __name__ == "__main__":
    main()
