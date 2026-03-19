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
from .github_sync import GitHubSync
from .handoff import HandoffManager
from .snapshots import SnapshotManager
from .bootstrap import BootstrapManager


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
    # github
    # ------------------------------------------------------------------
    gh_p = sub.add_parser("github", help="GitHub ↔ Nexus sync and interaction")
    gh_p.add_argument("--repo", "-r", required=True,
                      help="GitHub repo: owner/repo")
    gh_p.add_argument("--token", "-t", default=None,
                      help="GitHub token (default: GITHUB_TOKEN env var)")
    gh_sub = gh_p.add_subparsers(dest="gh_cmd", required=True)

    gi = gh_sub.add_parser("issues", help="Pull open issues into Nexus tasks")
    gi.add_argument("--state", default="open", choices=["open", "closed", "all"])
    gi.add_argument("--labels", default=None, help="Filter by comma-separated labels")
    gi.add_argument("--limit", "-n", type=int, default=100)

    gh_sub.add_parser("context", help="Sync repo metadata and recent PRs as memories")

    gpr = gh_sub.add_parser("pr", help="Pull a PR's file context into memory")
    gpr.add_argument("number", type=int)

    gci = gh_sub.add_parser("comment", help="Post a comment to an issue")
    gci.add_argument("issue", type=int)
    gci.add_argument("body", nargs="+")

    gcc = gh_sub.add_parser("close", help="Close a GitHub issue")
    gcc.add_argument("issue", type=int)
    gcc.add_argument("--comment", default=None, help="Final comment before closing")

    # ------------------------------------------------------------------
    # webhook
    # ------------------------------------------------------------------
    wh_p = sub.add_parser("webhook", help="Webhook receiver (GitHub/CI events → Nexus)")
    wh_p.add_argument("--repo", "-r", required=True, help="GitHub repo: owner/repo")
    wh_p.add_argument("--token", "-t", default=None, help="GitHub token")
    wh_p.add_argument("--secret", "-s", default=None,
                      help="Webhook secret for HMAC verification (default: NEXUS_WEBHOOK_SECRET env var)")
    wh_sub = wh_p.add_subparsers(dest="wh_cmd", required=True)

    ws = wh_sub.add_parser("serve", help="Start the webhook HTTP server")
    ws.add_argument("--host", default="0.0.0.0")
    ws.add_argument("--port", "-p", type=int, default=8765)

    # ------------------------------------------------------------------
    # handoff
    # ------------------------------------------------------------------
    ho_p = sub.add_parser("handoff", help="Agent-to-agent handoff protocol")
    ho_sub = ho_p.add_subparsers(dest="ho_cmd", required=True)

    hoc = ho_sub.add_parser("create", help="Create a handoff package for the next agent")
    hoc.add_argument("summary", nargs="+", help="What was accomplished")
    hoc.add_argument("--to", default="any", dest="to_agent",
                     help="Target agent type (default: any)")
    hoc.add_argument("--steps", nargs="+", default=[],
                     help="Next steps for the receiving agent")
    hoc.add_argument("--blockers", nargs="+", default=[])
    hoc.add_argument("--facts", nargs="+", default=[],
                     help="Key facts the next agent must know")
    hoc.add_argument("--warnings", nargs="+", default=[])
    hoc.add_argument("--dir", default="handoffs", dest="handoff_dir",
                     help="Directory to write the handoff to (default: handoffs/)")
    hoc.add_argument("--no-bundle", action="store_true",
                     help="Don't embed a knowledge bundle")

    hoa = ho_sub.add_parser("apply", help="Find and apply incoming handoffs")
    hoa.add_argument("--dir", default="handoffs", dest="handoff_dir")
    hoa.add_argument("--type", default=None, dest="agent_type",
                     help="Prefer handoffs addressed to this agent type")

    hol = ho_sub.add_parser("list", help="List handoff files in a directory")
    hol.add_argument("--dir", default="handoffs", dest="handoff_dir")

    # ------------------------------------------------------------------
    # mcp
    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------
    sn_p = sub.add_parser("snapshot", help="State snapshots - time travel for agents")
    sn_sub = sn_p.add_subparsers(dest="sn_cmd", required=True)

    snc = sn_sub.add_parser("create", help="Create a snapshot of current state")
    snc.add_argument("label", nargs="*", help="Optional label for this snapshot")

    sn_sub.add_parser("list", aliases=["ls"], help="List available snapshots")

    snr = sn_sub.add_parser("restore", help="Restore state from a snapshot")
    snr.add_argument("id", help="Snapshot ID or 8-char prefix")
    snr.add_argument("--no-backup", action="store_true",
                     help="Don't create a pre-restore backup (not recommended)")

    snd = sn_sub.add_parser("diff", help="Show what changed since a snapshot")
    snd.add_argument("id", help="Snapshot ID or 8-char prefix")

    snp = sn_sub.add_parser("prune", help="Delete old snapshots")
    snp.add_argument("--keep", type=int, default=20,
                     help="Number of snapshots to keep (default: 20)")
    snp.add_argument("--all", action="store_true", dest="keep_none",
                     help="Delete all including manually labelled ones")

    # ------------------------------------------------------------------
    # bootstrap
    # ------------------------------------------------------------------
    bs_p = sub.add_parser("bootstrap",
                          help="Install foundational agent knowledge (skills hub)")
    bs_sub = bs_p.add_subparsers(dest="bs_cmd", required=False)

    bsa = bs_sub.add_parser("apply", help="Install knowledge packs into memory (default)")
    bsa.add_argument("--packs", default=None,
                     help="Comma-separated pack names (default: all)")

    bs_sub.add_parser("list", aliases=["ls"],
                      help="List available knowledge packs")

    bs_sub.add_parser("status",
                      help="Show which bootstrap knowledge is already installed")

    bsc = bs_sub.add_parser("contribute",
                             help="Add a learned lesson to the knowledge base")
    bsc.add_argument("lesson", nargs="+", help="The lesson text")
    bsc.add_argument("--pack", default="agent-patterns",
                     help="Target pack (default: agent-patterns)")
    bsc.add_argument("--importance", type=float, default=0.8)

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

    elif args.command == "github":
        import os
        token = args.token or os.environ.get("GITHUB_TOKEN")
        if not token:
            print("GitHub token required: --token or GITHUB_TOKEN env var")
            sys.exit(1)
        gh = GitHubSync(args.repo, token, db_path=args.db)
        if args.gh_cmd == "issues":
            labels = [l.strip() for l in args.labels.split(",")] if args.labels else None
            result = gh.pull_issues(cm, state=args.state, labels=labels, limit=args.limit)
            print(result.summary())
        elif args.gh_cmd == "context":
            result = gh.sync_repo_context(cm)
            print(result.summary())
        elif args.gh_cmd == "pr":
            result = gh.pull_pr_context(cm, args.number)
            print(result.summary())
        elif args.gh_cmd == "comment":
            body = " ".join(args.body)
            resp = gh.comment_issue(args.issue, body)
            print(f"Posted comment to #{args.issue}: {resp.get('html_url', 'done')}")
        elif args.gh_cmd == "close":
            gh.close_issue(args.issue, comment=args.comment)
            print(f"Closed issue #{args.issue}")

    elif args.command == "webhook":
        import os
        token = args.token or os.environ.get("GITHUB_TOKEN")
        if not token:
            print("GitHub token required: --token or GITHUB_TOKEN env var")
            sys.exit(1)
        from .webhooks import WebhookServer
        ws = WebhookServer(args.repo, token, db_path=args.db, secret=args.secret)
        if args.wh_cmd == "serve":
            print(f"Starting webhook server on {args.host}:{args.port}")
            print(f"  GitHub events: POST /webhook/github")
            print(f"  Generic events: POST /webhook/generic")
            print(f"  Health check: GET /health")
            print(f"  Event log: GET /events")
            ws.serve(host=args.host, port=args.port)

    elif args.command == "handoff":
        hm = HandoffManager(db_path=args.db)
        if args.ho_cmd == "create":
            handoff_dir = Path(args.handoff_dir)
            h = hm.create(
                summary=" ".join(args.summary),
                to_agent=args.to_agent,
                next_steps=args.steps,
                blockers=args.blockers,
                key_facts=args.facts,
                warnings=args.warnings,
                include_bundle=not args.no_bundle,
            )
            path = hm.write(h, handoff_dir)
            print(f"Handoff written to: {path}")
            print(f"ID: {h.id}")
            print(f"To: {h.to_agent}")
            if args.steps:
                print(f"Steps: {len(args.steps)}")
        elif args.ho_cmd == "apply":
            handoff_dir = Path(args.handoff_dir)
            result = hm.find_and_apply(handoff_dir, agent_type=args.agent_type, cm=cm)
            if result:
                print(result.briefing())
            else:
                print(f"No pending handoffs found in {handoff_dir}")
        elif args.ho_cmd == "list":
            handoff_dir = Path(args.handoff_dir)
            if not handoff_dir.is_dir():
                print(f"Directory not found: {handoff_dir}")
            else:
                files = list(handoff_dir.glob("nexus-handoff-*.json"))
                if not files:
                    print("No handoff files found.")
                else:
                    for fp in sorted(files):
                        try:
                            h = hm.load(fp)
                            age = (time.time() - h.created_at) / 60
                            print(f"  [{h.status}] {fp.name}  from:{h.from_agent} "
                                  f"to:{h.to_agent}  {age:.0f}m ago")
                        except Exception as e:
                            print(f"  [error] {fp.name}: {e}")

    elif args.command == "snapshot":
        sm = SnapshotManager(db_path=args.db)
        if args.sn_cmd == "create":
            label = " ".join(args.label) if args.label else ""
            snap = sm.create(label=label)
            print(f"Snapshot created: {snap.meta.id[:8]}  mem:{snap.meta.memory_count} "
                  f"tasks:{snap.meta.task_count}")
            if label:
                print(f"Label: {label}")
        elif args.sn_cmd in ("list", "ls"):
            metas = sm.list()
            if not metas:
                print("No snapshots. Create one with: nexus snapshot create")
            else:
                print(f"Snapshots ({len(metas)}):")
                for m in metas:
                    print(f"  {m.display()}")
        elif args.sn_cmd == "restore":
            print(f"Restoring snapshot {args.id}...")
            backup = not args.no_backup
            snap = sm.restore(args.id, create_pre_restore_snapshot=backup)
            print(f"Restored: {snap.meta.label or snap.meta.id[:8]}")
            print(f"  Memories:    {snap.meta.memory_count}")
            print(f"  Tasks:       {snap.meta.task_count}")
            print(f"  Reflections: {snap.meta.reflection_count}")
            if backup:
                print("(Pre-restore backup created)")
        elif args.sn_cmd == "diff":
            diff = sm.diff(args.id)
            print(diff.summary())
        elif args.sn_cmd == "prune":
            keep_manual = not args.keep_none
            deleted = sm.prune(keep=args.keep, keep_manual=keep_manual)
            print(f"Deleted {deleted} snapshot(s). {len(sm.list())} remaining.")

    elif args.command == "bootstrap":
        bm = BootstrapManager(db_path=args.db)
        bs_cmd = getattr(args, "bs_cmd", None)
        if bs_cmd is None or bs_cmd == "apply":
            packs = [p.strip() for p in args.packs.split(",")] if getattr(args, "packs", None) else None
            result = bm.apply(cm, packs=packs)
            print(result.summary())
        elif bs_cmd in ("list", "ls"):
            packs = bm.list_packs()
            print(f"Knowledge packs ({len(packs)}):")
            for name, desc in packs.items():
                entries = bm.pack_entries(name)
                print(f"  {name:20s} ({len(entries)} entries) - {desc}")
        elif bs_cmd == "status":
            status = bm.status(cm)
            print("Bootstrap status:")
            for pack_name, info in status.items():
                check = "✓" if info["complete"] else f"{info['installed']}/{info['total']}"
                print(f"  [{check:5s}] {pack_name}")
        elif bs_cmd == "contribute":
            lesson = " ".join(args.lesson)
            entry = bm.contribute(lesson, pack=args.pack,
                                  importance=args.importance, cm=cm)
            print(f"Contribution saved to pack '{args.pack}'.")
            print(f"ID: {entry['id'][:8]}")
            print(f"Lesson: {lesson[:80]}")
            print("")
            print("To share with future agents: commit the contributions file to the repo.")
            contribs_path = bm._contrib_path()
            print(f"File: {contribs_path}")

    elif args.command == "mcp":
        try:
            from .mcp_server import run_server
        except ImportError:
            print("MCP server requires the 'mcp' package: pip install 'nexus-agent[mcp]'")
            sys.exit(1)
        run_server(db_path=args.db)


if __name__ == "__main__":
    main()
