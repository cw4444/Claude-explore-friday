"""Nexus - Persistent Intelligence Toolkit for autonomous AI agents.

Quick start:

    from nexus import ContextManager, MemoryType, Outcome

    cm = ContextManager()                        # uses ~/.nexus/nexus.db

    # Remember things
    cm.memory.remember("prod DB is at db.prod.example.com",
                       type=MemoryType.SEMANTIC, tags=["db", "prod"])

    cm.memory.remember("To deploy: run ./scripts/deploy.sh --env=prod",
                       type=MemoryType.PROCEDURAL, tags=["deploy"])

    # Track tasks
    tasks = cm.tasks.decompose(
        "Refactor auth module",
        subtasks=["Audit current code", "Design new interface",
                  "Implement changes", "Write tests"],
    )
    cm.tasks.start(tasks[1].id)  # Audit current code

    # Reflect on completed work
    cm.reflection.reflect(
        task_title="Audit current code",
        outcome=Outcome.SUCCESS,
        what_worked="Reading tests first gave me the contract",
        lesson="Always read tests before production code",
        effort_mins=20,
    )

    # Bootstrap a new session
    ctx = cm.prepare("Fix the payment processing bug")
    print(ctx.summary())
"""

from .memory import MemoryStore, Memory, MemoryType
from .tasks import TaskGraph, Task, TaskStatus, Priority
from .reflection import ReflectionEngine, Reflection, Outcome
from .context import ContextManager, SessionContext
from .narrative import NarrativeEngine, SelfModel, Trait
from .relationships import RelationshipStore, Contact, Observation

__all__ = [
    # Memory
    "MemoryStore", "Memory", "MemoryType",
    # Tasks
    "TaskGraph", "Task", "TaskStatus", "Priority",
    # Reflection
    "ReflectionEngine", "Reflection", "Outcome",
    # Context (primary entrypoint)
    "ContextManager", "SessionContext",
    # Identity / narrative
    "NarrativeEngine", "SelfModel", "Trait",
    # Relationships
    "RelationshipStore", "Contact", "Observation",
]

__version__ = "0.1.0"
