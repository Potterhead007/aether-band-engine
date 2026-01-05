"""
AETHER Workflow Orchestration Engine

Production-grade DAG execution engine for the music generation pipeline.
Handles task scheduling, dependency resolution, retry logic, and state management.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    TypeVar,
)
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type variables for generic task handling
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowStatus(str, Enum):
    """Overall workflow status."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    output: Any | None = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class TaskNode:
    """A node in the workflow DAG representing a single task."""

    id: str
    name: str
    agent_type: str
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "max_retries": 3,
            "backoff_base": 2.0,
            "backoff_max": 60.0,
        }
    )
    timeout_seconds: float = 300.0
    status: TaskStatus = TaskStatus.PENDING
    result: TaskResult | None = None

    def can_run(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)


class WorkflowEvent(BaseModel):
    """Event emitted during workflow execution."""

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    workflow_id: str
    task_id: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Simple event bus for workflow events."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._history: list[WorkflowEvent] = []

    def subscribe(self, event_type: str, handler: Callable[[WorkflowEvent], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def publish(self, event: WorkflowEvent) -> None:
        """Publish an event to all subscribers."""
        self._history.append(event)
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get("*", []))  # Wildcard handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def get_history(self) -> list[WorkflowEvent]:
        """Get event history."""
        return self._history.copy()


class WorkflowState(BaseModel):
    """Serializable workflow state for persistence and recovery."""

    workflow_id: str
    name: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    tasks: dict[str, dict[str, Any]]
    completed_tasks: list[str]
    failed_tasks: list[str]
    context: dict[str, Any]
    checksum: Optional[str] = None

    def compute_checksum(self) -> str:
        """Compute state checksum for integrity verification."""
        state_str = json.dumps(
            {
                "workflow_id": self.workflow_id,
                "tasks": self.tasks,
                "completed_tasks": self.completed_tasks,
                "context": self.context,
            },
            sort_keys=True,
        )
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class AgentExecutor(ABC):
    """Abstract base for agent execution."""

    @abstractmethod
    async def execute(
        self,
        task: TaskNode,
        context: dict[str, Any],
    ) -> TaskResult:
        """Execute a task and return the result."""
        pass


class WorkflowOrchestrator:
    """
    Production-grade DAG workflow orchestrator.

    Features:
    - Dependency-based task scheduling
    - Parallel execution where possible
    - Retry with exponential backoff
    - State persistence and recovery
    - Event-driven architecture
    - Timeout handling
    - Graceful cancellation
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        name: str = "aether_workflow",
        max_parallel: int = 4,
        state_dir: Optional[Path] = None,
    ):
        self.workflow_id = workflow_id or str(uuid4())
        self.name = name
        self.max_parallel = max_parallel
        self.state_dir = state_dir

        self.tasks: dict[str, TaskNode] = {}
        self.context: dict[str, Any] = {}
        self.status = WorkflowStatus.INITIALIZING
        self.event_bus = EventBus()

        self._completed_tasks: set[str] = set()
        self._failed_tasks: set[str] = set()
        self._running_tasks: set[str] = set()
        self._semaphore: asyncio.Semaphore | None = None
        self._cancel_event: asyncio.Event | None = None
        self._agent_executors: dict[str, AgentExecutor] = {}

        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()

    def register_agent(self, agent_type: str, executor: AgentExecutor) -> None:
        """Register an agent executor for a task type."""
        self._agent_executors[agent_type] = executor
        logger.info(f"Registered agent executor: {agent_type}")

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the workflow."""
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task
        self._emit_event("task_added", task_id=task.id)

    def add_tasks(self, tasks: list[TaskNode]) -> None:
        """Add multiple tasks to the workflow."""
        for task in tasks:
            self.add_task(task)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value available to all tasks."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)

    def _emit_event(self, event_type: str, task_id: Optional[str] = None, **data) -> None:
        """Emit a workflow event."""
        event = WorkflowEvent(
            event_type=event_type,
            workflow_id=self.workflow_id,
            task_id=task_id,
            data=data,
        )
        self.event_bus.publish(event)

    def _get_ready_tasks(self) -> list[TaskNode]:
        """Get tasks that are ready to run (all dependencies satisfied)."""
        ready = []
        for task_id, task in self.tasks.items():
            if (
                task.status == TaskStatus.PENDING
                and task_id not in self._running_tasks
                and task.can_run(self._completed_tasks)
            ):
                ready.append(task)
        return ready

    async def _execute_task(self, task: TaskNode) -> TaskResult:
        """Execute a single task with retry logic."""
        task.status = TaskStatus.RUNNING
        self._running_tasks.add(task.id)
        self._emit_event("task_started", task_id=task.id)

        retry_count = 0
        max_retries = task.retry_policy.get("max_retries", 3)
        backoff_base = task.retry_policy.get("backoff_base", 2.0)
        backoff_max = task.retry_policy.get("backoff_max", 60.0)

        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        while retry_count <= max_retries:
            try:
                executor = self._agent_executors.get(task.agent_type)
                if not executor:
                    raise ValueError(f"No executor registered for agent type: {task.agent_type}")

                # Execute with timeout
                task_result = await asyncio.wait_for(
                    executor.execute(task, self.context),
                    timeout=task.timeout_seconds,
                )

                result.status = task_result.status
                result.output = task_result.output
                result.completed_at = datetime.utcnow()
                result.duration_ms = (
                    result.completed_at - result.started_at
                ).total_seconds() * 1000
                result.retry_count = retry_count

                if result.status == TaskStatus.COMPLETED:
                    # Store output in context for downstream tasks
                    self.context[f"output_{task.id}"] = result.output
                    self._emit_event(
                        "task_completed",
                        task_id=task.id,
                        duration_ms=result.duration_ms,
                    )
                    break
                else:
                    raise RuntimeError(f"Task returned non-completed status: {result.status}")

            except asyncio.TimeoutError:
                result.error = f"Task timed out after {task.timeout_seconds}s"
                logger.warning(f"Task {task.id} timed out (attempt {retry_count + 1})")

            except asyncio.CancelledError:
                result.status = TaskStatus.CANCELLED
                result.error = "Task was cancelled"
                result.completed_at = datetime.utcnow()
                self._emit_event("task_cancelled", task_id=task.id)
                break

            except Exception as e:
                result.error = str(e)
                logger.error(f"Task {task.id} failed (attempt {retry_count + 1}): {e}")

            # Retry logic
            if retry_count < max_retries:
                retry_count += 1
                task.status = TaskStatus.RETRYING
                backoff = min(backoff_base**retry_count, backoff_max)
                self._emit_event(
                    "task_retrying",
                    task_id=task.id,
                    retry_count=retry_count,
                    backoff_seconds=backoff,
                )
                await asyncio.sleep(backoff)
            else:
                result.status = TaskStatus.FAILED
                result.completed_at = datetime.utcnow()
                result.duration_ms = (
                    result.completed_at - result.started_at
                ).total_seconds() * 1000
                result.retry_count = retry_count
                self._emit_event("task_failed", task_id=task.id, error=result.error)

        task.status = result.status
        task.result = result
        self._running_tasks.discard(task.id)

        if result.status == TaskStatus.COMPLETED:
            self._completed_tasks.add(task.id)
        elif result.status == TaskStatus.FAILED:
            self._failed_tasks.add(task.id)

        return result

    async def run(self) -> dict[str, TaskResult]:
        """
        Execute the workflow.

        Returns dict of task_id -> TaskResult for all tasks.
        """
        self.status = WorkflowStatus.RUNNING
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        self._cancel_event = asyncio.Event()
        self._emit_event("workflow_started")

        results: dict[str, TaskResult] = {}
        pending_futures: dict[str, asyncio.Task] = {}

        try:
            while True:
                # Check for cancellation
                if self._cancel_event.is_set():
                    self.status = WorkflowStatus.CANCELLED
                    break

                # Get tasks ready to run
                ready_tasks = self._get_ready_tasks()

                # Check if workflow is complete
                if not ready_tasks and not pending_futures:
                    if self._failed_tasks:
                        self.status = WorkflowStatus.FAILED
                    else:
                        self.status = WorkflowStatus.COMPLETED
                    break

                # Launch ready tasks (up to semaphore limit)
                for task in ready_tasks:
                    if len(pending_futures) >= self.max_parallel:
                        break

                    async def run_with_semaphore(t: TaskNode):
                        async with self._semaphore:
                            return await self._execute_task(t)

                    future = asyncio.create_task(run_with_semaphore(task))
                    pending_futures[task.id] = future
                    task.status = TaskStatus.QUEUED

                # Wait for at least one task to complete
                if pending_futures:
                    done, _ = await asyncio.wait(
                        pending_futures.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Process completed tasks
                    for future in done:
                        for task_id, task_future in list(pending_futures.items()):
                            if task_future == future:
                                try:
                                    result = await future
                                    results[task_id] = result
                                except Exception as e:
                                    results[task_id] = TaskResult(
                                        task_id=task_id,
                                        status=TaskStatus.FAILED,
                                        error=str(e),
                                    )
                                del pending_futures[task_id]
                                break

                # Small delay to prevent tight loop
                await asyncio.sleep(0.01)

        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self._emit_event("workflow_error", error=str(e))
            logger.error(f"Workflow failed: {e}")
            raise

        finally:
            self._updated_at = datetime.utcnow()
            self._emit_event(
                "workflow_finished",
                status=self.status.value,
                completed_count=len(self._completed_tasks),
                failed_count=len(self._failed_tasks),
            )

        return results

    async def cancel(self) -> None:
        """Cancel the running workflow."""
        if self._cancel_event:
            self._cancel_event.set()
        self._emit_event("workflow_cancelling")

    def get_state(self) -> WorkflowState:
        """Get current workflow state for persistence."""
        state = WorkflowState(
            workflow_id=self.workflow_id,
            name=self.name,
            status=self.status,
            created_at=self._created_at,
            updated_at=self._updated_at,
            tasks={
                task_id: {
                    "name": task.name,
                    "agent_type": task.agent_type,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "result": task.result.to_dict() if task.result else None,
                }
                for task_id, task in self.tasks.items()
            },
            completed_tasks=list(self._completed_tasks),
            failed_tasks=list(self._failed_tasks),
            context={k: v for k, v in self.context.items() if not k.startswith("_")},
        )
        state.checksum = state.compute_checksum()
        return state

    def save_state(self, path: Optional[Path] = None) -> Path:
        """Save workflow state to file."""
        if path is None:
            if self.state_dir is None:
                raise ValueError("No state directory configured")
            path = self.state_dir / f"workflow_{self.workflow_id}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.get_state()
        with open(path, "w") as f:
            f.write(state.model_dump_json(indent=2))

        logger.info(f"Saved workflow state to {path}")
        return path

    @classmethod
    def load_state(cls, path: Path) -> WorkflowOrchestrator:
        """Load workflow from saved state."""
        with open(path) as f:
            state = WorkflowState.model_validate_json(f.read())

        # Verify checksum
        expected_checksum = state.compute_checksum()
        if state.checksum and state.checksum != expected_checksum:
            logger.warning("State checksum mismatch - state may be corrupted")

        orchestrator = cls(
            workflow_id=state.workflow_id,
            name=state.name,
        )
        orchestrator.status = state.status
        orchestrator._created_at = state.created_at
        orchestrator._updated_at = state.updated_at
        orchestrator.context = state.context
        orchestrator._completed_tasks = set(state.completed_tasks)
        orchestrator._failed_tasks = set(state.failed_tasks)

        # Reconstruct tasks
        for task_id, task_data in state.tasks.items():
            task = TaskNode(
                id=task_id,
                name=task_data["name"],
                agent_type=task_data["agent_type"],
                dependencies=task_data["dependencies"],
                status=TaskStatus(task_data["status"]),
            )
            orchestrator.tasks[task_id] = task

        logger.info(f"Loaded workflow state from {path}")
        return orchestrator


def create_pipeline_workflow(
    song_id: str,
    has_vocals: bool = True,
) -> WorkflowOrchestrator:
    """
    Create the standard AETHER music production pipeline.

    Pipeline stages:
    A. Creative Direction
    B. Composition (Harmony + Melody)
    C. Arrangement
    D. Lyrics (if vocals)
    E. Vocal Planning (if vocals)
    F. Sound Design
    G. Mixing
    H. Mastering
    I. QA
    J. Packaging
    """
    workflow = WorkflowOrchestrator(name=f"aether_pipeline_{song_id}")
    workflow.set_context("song_id", song_id)
    workflow.set_context("has_vocals", has_vocals)

    # Define pipeline tasks
    tasks = [
        TaskNode(
            id="creative_direction",
            name="Creative Direction",
            agent_type="creative_director",
            dependencies=[],
            timeout_seconds=120.0,
        ),
        TaskNode(
            id="composition",
            name="Composition",
            agent_type="composition",
            dependencies=["creative_direction"],
            timeout_seconds=300.0,
        ),
        TaskNode(
            id="arrangement",
            name="Arrangement",
            agent_type="arrangement",
            dependencies=["composition"],
            timeout_seconds=300.0,
        ),
        TaskNode(
            id="sound_design",
            name="Sound Design",
            agent_type="sound_design",
            dependencies=["arrangement"],
            timeout_seconds=300.0,
        ),
        TaskNode(
            id="mixing",
            name="Mixing",
            agent_type="mixing",
            dependencies=["sound_design"],
            timeout_seconds=600.0,
        ),
        TaskNode(
            id="mastering",
            name="Mastering",
            agent_type="mastering",
            dependencies=["mixing"],
            timeout_seconds=300.0,
        ),
        TaskNode(
            id="qa",
            name="Quality Assurance",
            agent_type="qa",
            dependencies=["mastering"],
            timeout_seconds=180.0,
        ),
        TaskNode(
            id="packaging",
            name="Release Packaging",
            agent_type="release",
            dependencies=["qa"],
            timeout_seconds=120.0,
        ),
    ]

    # Add vocal tasks if needed
    if has_vocals:
        tasks.insert(
            3,
            TaskNode(
                id="lyrics",
                name="Lyrics",
                agent_type="lyrics",
                dependencies=["arrangement"],
                timeout_seconds=300.0,
            ),
        )
        tasks.insert(
            4,
            TaskNode(
                id="vocal_planning",
                name="Vocal Planning",
                agent_type="vocal",
                dependencies=["lyrics"],
                timeout_seconds=180.0,
            ),
        )
        # Update sound_design to depend on vocal_planning
        for task in tasks:
            if task.id == "sound_design":
                task.dependencies = ["arrangement", "vocal_planning"]

    workflow.add_tasks(tasks)
    return workflow
