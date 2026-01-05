"""
AETHER Agent Base Class

Foundation for all pipeline agents with common functionality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from aether.orchestration import TaskNode, TaskResult, TaskStatus
from aether.providers import get_provider_registry
from aether.storage import ArtifactStore, ArtifactType

logger = logging.getLogger(__name__)

# Type variables for input/output schemas
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


@dataclass
class AgentDecision:
    """Record of a decision made by an agent."""

    decision_id: str
    agent_id: str
    timestamp: datetime
    decision_type: str
    input_summary: str
    output_summary: str
    reasoning: str
    alternatives_considered: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """
    Base class for all AETHER pipeline agents.

    Features:
    - Typed input/output via Pydantic schemas
    - Decision logging for auditability
    - Integration with artifact store
    - Provider access
    - Error handling with context
    """

    # Override in subclasses
    agent_type: str = "base"
    agent_name: str = "Base Agent"
    input_schema: type = BaseModel
    output_schema: type = BaseModel

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.agent_id = f"{self.agent_type}_{uuid4().hex[:8]}"
        self.artifact_store = artifact_store
        self.config = config or {}
        self.decisions: list[AgentDecision] = []
        self._providers = get_provider_registry()

    @property
    def llm(self):
        """Get the LLM provider."""
        return self._providers.get("llm")

    @property
    def midi(self):
        """Get the MIDI provider."""
        return self._providers.get("midi")

    @property
    def audio(self):
        """Get the audio provider."""
        return self._providers.get("audio")

    @property
    def embedding(self):
        """Get the embedding provider."""
        return self._providers.get("embedding")

    def log_decision(
        self,
        decision_type: str,
        input_summary: str,
        output_summary: str,
        reasoning: str,
        alternatives: list[str] | None = None,
        confidence: float = 0.8,
        **metadata,
    ) -> AgentDecision:
        """Log a decision for auditability."""
        decision = AgentDecision(
            decision_id=str(uuid4()),
            agent_id=self.agent_id,
            timestamp=datetime.utcnow(),
            decision_type=decision_type,
            input_summary=input_summary,
            output_summary=output_summary,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence=confidence,
            metadata=metadata,
        )
        self.decisions.append(decision)
        logger.info(f"[{self.agent_name}] Decision: {decision_type} (confidence: {confidence:.2f})")
        return decision

    def store_artifact(
        self,
        data: Any,
        artifact_type: ArtifactType,
        name: str,
        song_id: str,
        **kwargs,
    ):
        """Store an artifact if artifact store is available."""
        if self.artifact_store:
            return self.artifact_store.store(
                data=data,
                artifact_type=artifact_type,
                name=name,
                song_id=song_id,
                created_by=self.agent_id,
                **kwargs,
            )
        return None

    @abstractmethod
    async def process(
        self,
        input_data: TInput,
        context: dict[str, Any],
    ) -> TOutput:
        """
        Main processing method - implement in subclasses.

        Args:
            input_data: Validated input matching input_schema
            context: Shared workflow context

        Returns:
            Output matching output_schema
        """
        pass

    async def execute(
        self,
        task: TaskNode,
        context: dict[str, Any],
    ) -> TaskResult:
        """
        Execute the agent as part of a workflow task.

        This is called by the WorkflowOrchestrator.
        """
        started_at = datetime.utcnow()

        try:
            # Get input from context
            input_key = f"input_{task.id}"
            raw_input = context.get(input_key, {})

            # Validate input
            if self.input_schema != BaseModel:
                input_data = self.input_schema.model_validate(raw_input)
            else:
                input_data = raw_input

            logger.info(f"[{self.agent_name}] Starting execution for task {task.id}")

            # Run the agent's main processing
            output = await self.process(input_data, context)

            # Validate output
            if self.output_schema != BaseModel and not isinstance(output, self.output_schema):
                output = self.output_schema.model_validate(output)

            completed_at = datetime.utcnow()
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            logger.info(f"[{self.agent_name}] Completed in {duration_ms:.0f}ms")

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=output.model_dump() if hasattr(output, "model_dump") else output,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                metadata={
                    "agent_id": self.agent_id,
                    "decisions_count": len(self.decisions),
                },
            )

        except Exception as e:
            completed_at = datetime.utcnow()
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            logger.error(f"[{self.agent_name}] Failed: {e}")

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                metadata={
                    "agent_id": self.agent_id,
                    "error_type": type(e).__name__,
                },
            )

    def get_decisions_summary(self) -> list[dict[str, Any]]:
        """Get summary of all decisions made."""
        return [
            {
                "decision_id": d.decision_id,
                "type": d.decision_type,
                "timestamp": d.timestamp.isoformat(),
                "confidence": d.confidence,
                "reasoning": d.reasoning[:100] + "..." if len(d.reasoning) > 100 else d.reasoning,
            }
            for d in self.decisions
        ]


class AgentRegistry:
    """Registry for agent types."""

    _agents: dict[str, type] = {}

    @classmethod
    def register(cls, agent_type: str):
        """Decorator to register an agent class."""

        def decorator(agent_cls: type):
            cls._agents[agent_type] = agent_cls
            return agent_cls

        return decorator

    @classmethod
    def get(cls, agent_type: str) -> type | None:
        """Get agent class by type."""
        return cls._agents.get(agent_type)

    @classmethod
    def create(
        cls,
        agent_type: str,
        artifact_store: ArtifactStore | None = None,
        config: dict[str, Any] | None = None,
    ) -> BaseAgent | None:
        """Create an agent instance by type."""
        agent_cls = cls.get(agent_type)
        if agent_cls:
            return agent_cls(artifact_store=artifact_store, config=config)
        return None

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered agent types."""
        return list(cls._agents.keys())
