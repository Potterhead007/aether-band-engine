"""
AETHER Orchestration Layer

Workflow DAG engine and music production pipeline.
"""

from aether.orchestration.pipeline import (
    MusicPipeline,
    PipelineAgentExecutor,
    generate_track,
)
from aether.orchestration.workflow import (
    AgentExecutor,
    EventBus,
    TaskNode,
    TaskResult,
    TaskStatus,
    WorkflowEvent,
    WorkflowOrchestrator,
    WorkflowStatus,
    create_pipeline_workflow,
)

__all__ = [
    # Workflow
    "WorkflowOrchestrator",
    "WorkflowStatus",
    "TaskNode",
    "TaskStatus",
    "TaskResult",
    "EventBus",
    "WorkflowEvent",
    "AgentExecutor",
    "create_pipeline_workflow",
    # Pipeline
    "MusicPipeline",
    "PipelineAgentExecutor",
    "generate_track",
]
