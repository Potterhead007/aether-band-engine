"""
AETHER Orchestration Layer

Workflow DAG engine and music production pipeline.
"""

from aether.orchestration.workflow import (
    WorkflowOrchestrator,
    WorkflowStatus,
    TaskNode,
    TaskStatus,
    TaskResult,
    EventBus,
    WorkflowEvent,
    AgentExecutor,
    create_pipeline_workflow,
)

from aether.orchestration.pipeline import (
    MusicPipeline,
    PipelineAgentExecutor,
    generate_track,
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
