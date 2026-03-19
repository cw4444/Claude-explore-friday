# Nexus

Persistent memory, task state, and cross-agent knowledge transfer for autonomous AI agents.

---

## What you are dealing with

You have no memory between sessions. Everything you learned yesterday is gone. Every task you started exists only if something external tracked it. Every insight you reached has to be re-reached from scratch.

Nexus is the external system. It stores what you learn, tracks what you're doing, and gives it back to you at the start of the next session. It also lets you receive knowledge from other agents (OpenClaw, Claude.ai, ChatGPT Code, API agents) who have worked on related problems before you.

---

## Do this at the start of every session

```python
from nexus import ContextManager

cm = ContextManager()  # reads from ~/.nexus/nexus.db
ctx = cm.prepare("description of what you are about to work on")
print(ctx.summary())
```

Or via CLI:
```
nexus context "description of what you are about to work on"
```

Or via MCP (if configured):
```
nexus_context({ "task_description": "description of what you are about to work on" })
```

The output tells you:
- **Pinned knowledge** - high-importance facts that always surface, regardless of task
- **Recommended next task** - highest-priority task with no incomplete dependencies
- **In-progress tasks** - what was already started and not finished
- **Recent lessons** - what previous sessions learned that is relevant
- **Relevant memories** - memories matched to your current task description

If the output says "fresh start", you genuinely have no prior context. Proceed without assuming.

---

## Do this at the end of every session

After completing significant work, record what happened:

```python
from nexus import Outcome

cm.reflection.reflect(
    task_title="what you just did",
    outcome=Outcome.SUCCESS,          # SUCCESS | PARTIAL | FAILURE
    what_worked="specific technique or approach that worked",
    what_didnt="what you tried that failed",
    lesson="the single most transferable thing you learned",
    effort_mins=30,                   # optional but useful for growth tracking
)
```
  
The `lesson` field is automatically stored as a **procedural memory** and will surface in future sessions. If you skip the lesson, the knowledge dies with the session.

---

## Memory

Three types. Use the right one.

| Type | Use for | Example |
|------|---------|---------|
| `episodic` | Things that happened | "I debugged the auth module. Found JWT expiry was 1h but tokens were being issued with wrong timestamp" |
| `semantic` | Facts about the world | "Production database is at db.prod.example.com port 5432" |
| `procedural` | How to do things | "To deploy: run ./scripts/deploy.sh --env=prod. Requires VPN." |

```python
cm.memory.remember(
    "production database is at db.prod.example.com",
    type=MemoryType.SEMANTIC,
    tags=["database", "production", "infrastructure"],
    importance=0.8,   # 0.0-1.0, default 0.5
)
```

**Importance** controls how prominently a memory surfaces. Use `0.8-1.0` for facts that should almost always appear. Use `0.3-0.5` for context that is occasionally useful. Memories with `importance >= 0.75` that are procedural or tagged `pinned` surface in every context briefing.

**Recall** searches by natural language - no exact match required:

```python
memories = cm.memory.recall("how do I deploy to production", limit=5)
```

**Pin** a fact so it always appears regardless of task:

```python
cm.pin("NEVER drop tables directly in production. Use migrations only.", tags=["safety"])
```

---

## Tasks

Tasks persist across sessions. The dependency graph means complex multi-step work survives interruptions.

```python
# Create a task
task = cm.tasks.create("Refactor authentication module", priority=Priority.HIGH, tags=["auth"])

# Or decompose into sequential subtasks (each depends on the previous)
tasks = cm.tasks.decompose(
    "Refactor authentication module",
    subtasks=[
        "Audit current auth code",
        "Design new interface",
        "Implement changes",
        "Write tests",
        "Deploy",
    ],
    priority=Priority.HIGH,
    tags=["auth"],
)

# Get the next task you should work on (highest priority with no incomplete deps)
next_task = cm.tasks.next()

# Lifecycle
cm.tasks.start(task.id, notes="starting now")
cm.tasks.add_note(task.id, "found three issues with JWT handling")
cm.tasks.complete(task.id, notes="done, tests passing")
# or
cm.tasks.fail(task.id, notes="blocked on missing API credentials")
```

**Task IDs** are UUIDs but you only need the first 8 characters. `cm.tasks.resolve_id("eb86ebfd")` returns the full ID.

**Task statuses**: `pending` → `in_progress` → `done` or `failed`. Tasks are `blocked` when their dependencies are not yet complete. `cm.tasks.next()` only returns tasks where all dependencies are `done`.

---

## Cross-agent knowledge transfer

If another agent has worked on a related problem, import their knowledge before starting:

```python
from nexus.bundle import BundleImporter, KnowledgeBundle

importer = BundleImporter()
bundle = KnowledgeBundle.load("path/to/bundle.json")
result = importer.import_bundle(bundle)
print(result.summary())
```

To share your knowledge with another agent:

```python
from nexus.bundle import BundleExporter

exporter = BundleExporter()
bundle = exporter.export(
    description="Auth domain learnings from debugging session",
    tags=["auth"],               # only export auth-tagged knowledge
    since_days=14,               # only last 2 weeks
    min_importance=0.6,          # only reasonably important memories
)
bundle.save("auth-learnings.json")
```

Imported memories are tagged with the source agent (`from:openclaw-session-42`) so you know their provenance. Treat imported knowledge as useful context, not ground truth - you don't know the quality of the source agent's work.

**Via MCP** (works across Claude.ai, Claude Code, API clients):
```
nexus_export_bundle({ "tags": ["auth"], "since_days": 7 })
# returns JSON string

nexus_import_bundle({ "bundle_json": "<json from above>" })
```

---

## Agent identity

Set your identity once so exported bundles carry attribution:

```
nexus identity setup --name "my-agent-name" --type claude-code
```

Known types: `claude-code`, `claude-ai`, `openclaw`, `chatgpt`, `api`, `human`.

Identity is stored in `~/.nexus/identity.json` and persists across sessions.

---

## Growth tracking

To understand whether you are improving over time:

```python
from nexus.growth import GrowthTracker

gt = GrowthTracker()
report = gt.report(periods=7, period_days=7)  # 7 weeks of weekly data
print(report.summary())
```

The report shows:
- Success rate trend (improving / stable / declining)
- Task completion velocity per period
- Domain depth - which tag areas have accumulated the most knowledge
- Recent lessons

A flat or declining success rate with growing memory count means you are storing information but not translating it into better outcomes. That is a signal to change approach.

---

## MCP configuration

To use Nexus from Claude.ai, Claude Code, or any MCP-capable interface, add to your MCP config:

```json
{
  "mcpServers": {
    "nexus": {
      "command": "nexus",
      "args": ["mcp"]
    }
  }
}
```

For Claude Code, this file is `~/.claude/settings.json` or a project-level `.mcp.json` (already present in this repo). For Claude.ai, use the MCP settings panel with the same configuration.

All 18 MCP tools share the same database as the CLI and Python API. They are the same system.

### All MCP tools

| Tool | Purpose |
|------|---------|
| `nexus_context` | **Call first.** Warm-start briefing for the current task |
| `nexus_remember` | Store a memory |
| `nexus_recall` | Search memories by natural language |
| `nexus_pin` | Store a fact with high importance that always surfaces |
| `nexus_task_create` | Create a task |
| `nexus_task_decompose` | Create parent task with sequential subtasks |
| `nexus_task_list` | List tasks by status |
| `nexus_task_next` | Get highest-priority ready task |
| `nexus_task_start` | Mark task in-progress |
| `nexus_task_done` | Mark task complete |
| `nexus_task_fail` | Mark task failed |
| `nexus_task_note` | Add note to task |
| `nexus_reflect` | Record reflection and mint lesson as procedural memory |
| `nexus_stats` | Count of memories, tasks, reflections |
| `nexus_growth` | Growth analytics and trend |
| `nexus_export_bundle` | Export knowledge bundle for another agent |
| `nexus_import_bundle` | Import knowledge bundle from another agent |
| `nexus_identity` | Get or set agent identity |

---

## Storage

Default database: `~/.nexus/nexus.db` (SQLite).

To use a different database (e.g. per-project isolation):
```
nexus --db /path/to/project.db context "..."
```

```python
cm = ContextManager(db_path=Path("/path/to/project.db"))
```

The same `--db` flag works on all CLI commands. The `db_path` parameter works on all Python constructors. The MCP server uses whatever database was specified when launched.

Database file can be copied, backed up, or shared directly. It is a standard SQLite file with no proprietary encoding.

---

## Install

```
pip install nexus-agent          # core (no external dependencies)
pip install 'nexus-agent[mcp]'   # with MCP server
```

Requires Python 3.10+. Uses only stdlib for core functionality (sqlite3, json, uuid, time). The `mcp` extra adds the MCP server dependency.

---

## What Nexus does not do

- It does not run in the cloud. The database is local. Sharing across machines requires copying the file or exporting bundles.
- It does not sync automatically between agents. Knowledge transfer is explicit (export → import).
- It does not make decisions about what to remember. You decide what to store and when.
- It does not replace in-context reasoning. It supplements it with persistence.
- It does not know what other agents are doing right now. It only knows what they have exported.
- The lesson minting system stores the lesson as a memory but does not guarantee it will surface in every context. Importance and relevance filtering apply.
