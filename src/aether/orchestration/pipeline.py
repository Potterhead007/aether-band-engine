"""
AETHER Music Production Pipeline

Complete end-to-end orchestration of the music generation process.
Ties agents together with the workflow engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from uuid import uuid4

from aether.orchestration.workflow import (
    WorkflowOrchestrator,
    TaskNode,
    TaskResult,
    TaskStatus,
    AgentExecutor,
    create_pipeline_workflow,
)
from aether.storage import ArtifactStore, create_artifact_store
from aether.providers import ProviderManager, ProviderConfig
from aether.rendering import RenderingEngine, RenderingConfig, RenderingResult

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from aether.agents import AgentRegistry

logger = logging.getLogger(__name__)


class PipelineAgentExecutor(AgentExecutor):
    """
    Agent executor that routes tasks to the appropriate AETHER agent.

    Handles input/output marshaling between the workflow context
    and individual agent input/output schemas.
    """

    def __init__(self, artifact_store: Optional[ArtifactStore] = None):
        self.artifact_store = artifact_store or create_artifact_store()

    async def execute(
        self,
        task: TaskNode,
        context: Dict[str, Any],
    ) -> TaskResult:
        """Execute a pipeline task by calling the appropriate agent."""
        started_at = __import__("datetime").datetime.utcnow()

        try:
            # Import agents at runtime to avoid circular import
            from aether.agents import AgentRegistry

            # Get agent
            agent = AgentRegistry.create(task.agent_type)
            agent.artifact_store = self.artifact_store

            # Build agent input from context
            agent_input = self._build_agent_input(task.agent_type, context)

            # Execute agent
            logger.info(f"Executing agent: {agent.agent_name}")
            result = await agent.process(agent_input, context)

            # Store output in context
            output_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)

            # Update context with outputs
            self._update_context(task.agent_type, output_dict, context)

            completed_at = __import__("datetime").datetime.utcnow()

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=output_dict,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=(completed_at - started_at).total_seconds() * 1000,
                metadata={"agent": agent.agent_name, "decisions": len(agent.decisions)},
            )

        except Exception as e:
            logger.error(f"Agent execution failed for {task.agent_type}: {e}")
            completed_at = __import__("datetime").datetime.utcnow()
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=(completed_at - started_at).total_seconds() * 1000,
            )

    def _build_agent_input(self, agent_type: str, context: Dict[str, Any]) -> Any:
        """Build the appropriate input for an agent from context."""
        # Import at runtime to avoid circular import
        from aether.agents import (
            CreativeDirectorInput,
            CompositionInput,
            ArrangementInput,
            LyricsInput,
            VocalInput,
            SoundDesignInput,
            MixingInput,
            MasteringInput,
            QAInput,
            ReleaseInput,
        )

        genre_profile_id = context.get("genre_profile_id", "boom-bap")

        if agent_type == "creative_director":
            return CreativeDirectorInput(
                title=context.get("title", "Untitled"),
                genre_id=context.get("genre_id", genre_profile_id),
                creative_brief=context.get("creative_brief", "An energetic track"),
                bpm=context.get("bpm"),
                key=context.get("key"),
                mood=context.get("mood"),
                duration_seconds=context.get("duration_seconds"),
                has_vocals=context.get("has_vocals", True),
                random_seed=context.get("random_seed"),
            )

        elif agent_type == "composition":
            song_spec = context.get("song_spec", {})
            return CompositionInput(
                song_spec=song_spec,
                genre_profile_id=song_spec.get("genre_id", genre_profile_id),
            )

        elif agent_type == "arrangement":
            return ArrangementInput(
                song_spec=context.get("song_spec", {}),
                harmony_spec=context.get("harmony_spec", {}),
                melody_spec=context.get("melody_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "lyrics":
            return LyricsInput(
                song_spec=context.get("song_spec", {}),
                arrangement_spec=context.get("arrangement_spec", {}),
                melody_spec=context.get("melody_spec", {}),
            )

        elif agent_type == "vocal":
            return VocalInput(
                song_spec=context.get("song_spec", {}),
                lyric_spec=context.get("lyric_spec", {}),
                melody_spec=context.get("melody_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "sound_design":
            return SoundDesignInput(
                song_spec=context.get("song_spec", {}),
                arrangement_spec=context.get("arrangement_spec", {}),
                rhythm_spec=context.get("rhythm_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "mixing":
            return MixingInput(
                song_spec=context.get("song_spec", {}),
                arrangement_spec=context.get("arrangement_spec", {}),
                sound_design_spec=context.get("sound_design_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "mastering":
            return MasteringInput(
                song_spec=context.get("song_spec", {}),
                mix_spec=context.get("mix_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "qa":
            return QAInput(
                song_spec=context.get("song_spec", {}),
                harmony_spec=context.get("harmony_spec", {}),
                melody_spec=context.get("melody_spec", {}),
                lyric_spec=context.get("lyric_spec", {}),
                master_spec=context.get("master_spec", {}),
                genre_profile_id=genre_profile_id,
            )

        elif agent_type == "release":
            return ReleaseInput(
                song_spec=context.get("song_spec", {}),
                master_spec=context.get("master_spec", {}),
                lyric_spec=context.get("lyric_spec", {}),
                qa_report=context.get("qa_report", {}),
                genre_profile_id=genre_profile_id,
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _update_context(
        self, agent_type: str, output: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Update context with agent output."""
        if agent_type == "creative_director":
            context["song_spec"] = output.get("song_spec", {})
            context["genre_profile_id"] = output.get("genre_profile_id", context.get("genre_profile_id"))

        elif agent_type == "composition":
            context["harmony_spec"] = output.get("harmony_spec", {})
            context["melody_spec"] = output.get("melody_spec", {})

        elif agent_type == "arrangement":
            context["arrangement_spec"] = output.get("arrangement_spec", {})
            context["rhythm_spec"] = output.get("rhythm_spec", {})

        elif agent_type == "lyrics":
            context["lyric_spec"] = output.get("lyric_spec", {})

        elif agent_type == "vocal":
            context["vocal_spec"] = output.get("vocal_spec", {})

        elif agent_type == "sound_design":
            context["sound_design_spec"] = output.get("sound_design_spec", {})

        elif agent_type == "mixing":
            context["mix_spec"] = output.get("mix_spec", {})

        elif agent_type == "mastering":
            context["master_spec"] = output.get("master_spec", {})

        elif agent_type == "qa":
            context["qa_report"] = output.get("qa_report", {})
            context["qa_passed"] = output.get("passed", False)

        elif agent_type == "release":
            context["release_package"] = output.get("release_package", {})
            context["ready_for_distribution"] = output.get("ready_for_distribution", False)


class MusicPipeline:
    """
    High-level interface for running the complete music production pipeline.

    Usage:
        pipeline = MusicPipeline()
        result = await pipeline.generate(
            title="My Song",
            genre="boom-bap",
            creative_brief="An introspective hip-hop track about perseverance",
        )
    """

    def __init__(
        self,
        artifact_store: Optional[ArtifactStore] = None,
        state_dir: Optional[Path] = None,
        provider_config: Optional[ProviderConfig] = None,
        rendering_config: Optional[RenderingConfig] = None,
    ):
        self.artifact_store = artifact_store or create_artifact_store()
        self.state_dir = state_dir or Path.home() / ".aether" / "workflows"
        self.executor = PipelineAgentExecutor(self.artifact_store)
        self.provider_config = provider_config or ProviderConfig()
        self.rendering_config = rendering_config
        self._provider_manager: Optional[ProviderManager] = None
        self._rendering_engine: Optional[RenderingEngine] = None

    async def generate(
        self,
        title: str,
        genre: str,
        creative_brief: str,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        mood: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        has_vocals: bool = True,
        random_seed: Optional[int] = None,
        save_state: bool = True,
        render_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a complete track through the pipeline.

        Args:
            title: Song title
            genre: Genre profile ID (e.g., "boom-bap", "synthwave", "lo-fi")
            creative_brief: Description of the desired track
            bpm: Optional BPM override
            key: Optional key override (e.g., "Am", "C major")
            mood: Optional mood override
            duration_seconds: Optional duration override
            has_vocals: Whether to include vocal planning
            random_seed: Optional seed for reproducibility
            save_state: Whether to save workflow state

        Returns:
            Dict containing all pipeline outputs and metadata
        """
        song_id = str(uuid4())
        logger.info(f"Starting pipeline for: {title} (ID: {song_id})")

        # Initialize providers
        self._provider_manager = ProviderManager(self.provider_config)
        provider_results = await self._provider_manager.initialize()
        logger.info(f"Providers initialized: {provider_results}")

        # Create workflow
        workflow = create_pipeline_workflow(song_id, has_vocals)
        workflow.state_dir = self.state_dir

        # Set initial context
        workflow.set_context("title", title)
        workflow.set_context("genre_id", genre)
        workflow.set_context("genre_profile_id", genre)
        workflow.set_context("creative_brief", creative_brief)
        workflow.set_context("has_vocals", has_vocals)

        if bpm:
            workflow.set_context("bpm", bpm)
        if key:
            workflow.set_context("key", key)
        if mood:
            workflow.set_context("mood", mood)
        if duration_seconds:
            workflow.set_context("duration_seconds", duration_seconds)
        if random_seed:
            workflow.set_context("random_seed", random_seed)

        # Register agent executor for all agent types
        from aether.agents import get_pipeline_agents
        for agent_type in get_pipeline_agents():
            workflow.register_agent(agent_type, self.executor)

        # Subscribe to events for logging
        workflow.event_bus.subscribe("*", self._log_event)

        # Run pipeline
        try:
            results = await workflow.run()

            if save_state:
                workflow.save_state()

            pipeline_result = {
                "song_id": song_id,
                "title": title,
                "genre": genre,
                "status": workflow.status.value,
                "song_spec": workflow.context.get("song_spec"),
                "harmony_spec": workflow.context.get("harmony_spec"),
                "melody_spec": workflow.context.get("melody_spec"),
                "arrangement_spec": workflow.context.get("arrangement_spec"),
                "lyric_spec": workflow.context.get("lyric_spec"),
                "vocal_spec": workflow.context.get("vocal_spec"),
                "sound_design_spec": workflow.context.get("sound_design_spec"),
                "mix_spec": workflow.context.get("mix_spec"),
                "master_spec": workflow.context.get("master_spec"),
                "qa_report": workflow.context.get("qa_report"),
                "release_package": workflow.context.get("release_package"),
                "ready_for_distribution": workflow.context.get("ready_for_distribution", False),
                "task_results": {k: v.to_dict() for k, v in results.items()},
            }

            # Render audio if requested
            if render_audio:
                logger.info("Rendering audio from specifications...")
                self._rendering_engine = RenderingEngine(self.rendering_config)
                render_result = await self._rendering_engine.render(pipeline_result)

                pipeline_result["render_result"] = {
                    "success": render_result.success,
                    "duration_seconds": render_result.duration_seconds,
                    "output_paths": {k: str(v) for k, v in render_result.output_paths.items()},
                    "errors": render_result.errors,
                }

                if render_result.success:
                    logger.info(f"Audio rendered: {render_result.duration_seconds:.1f}s")
                else:
                    logger.warning(f"Audio rendering failed: {render_result.errors}")

            return pipeline_result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if save_state:
                workflow.save_state()
            raise

        finally:
            # Shutdown providers
            if self._provider_manager:
                await self._provider_manager.shutdown()
                logger.info("Providers shutdown complete")

    def _log_event(self, event) -> None:
        """Log workflow events."""
        if event.event_type == "task_started":
            logger.info(f"▶ Started: {event.task_id}")
        elif event.event_type == "task_completed":
            duration = event.data.get("duration_ms", 0)
            logger.info(f"✓ Completed: {event.task_id} ({duration:.0f}ms)")
        elif event.event_type == "task_failed":
            error = event.data.get("error", "Unknown error")
            logger.error(f"✗ Failed: {event.task_id} - {error}")
        elif event.event_type == "workflow_finished":
            status = event.data.get("status", "unknown")
            completed = event.data.get("completed_count", 0)
            failed = event.data.get("failed_count", 0)
            logger.info(f"Pipeline {status}: {completed} completed, {failed} failed")


async def generate_track(
    title: str,
    genre: str,
    creative_brief: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to generate a track.

    Example:
        result = await generate_track(
            title="Night Drive",
            genre="synthwave",
            creative_brief="A nostalgic synthwave track with pulsing arpeggios and warm pads",
        )
    """
    pipeline = MusicPipeline()
    return await pipeline.generate(title, genre, creative_brief, **kwargs)
