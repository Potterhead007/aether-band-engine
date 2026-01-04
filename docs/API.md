# AETHER Band Engine API Reference

## Overview

This document provides comprehensive API documentation for programmatic usage of AETHER Band Engine.

---

## Table of Contents

- [Core Modules](#core-modules)
- [Pipeline API](#pipeline-api)
- [Agents API](#agents-api)
- [Providers API](#providers-api)
- [Schemas](#schemas)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

---

## Core Modules

### Configuration

```python
from aether.config import (
    AetherConfig,
    get_config,
    init_config,
    set_config,
)
```

#### `init_config(config_path: Optional[Path] = None) -> AetherConfig`

Initialize configuration from file or defaults.

```python
from pathlib import Path
from aether.config import init_config

# Load from default location (~/.aether/config.yaml)
config = init_config()

# Load from custom path
config = init_config(Path("/path/to/config.yaml"))
```

#### `get_config() -> AetherConfig`

Get the global configuration singleton.

```python
from aether.config import get_config

config = get_config()
print(config.audio.default_lufs)  # -14.0
```

#### `AetherConfig`

Main configuration class with nested settings.

```python
from aether.config import AetherConfig

config = AetherConfig(
    debug=True,
    log_level="DEBUG",
)

# Access nested settings
config.paths.base_dir
config.providers.llm_provider
config.audio.working_sample_rate
config.qa.melody_similarity_threshold
```

---

## Pipeline API

### MusicPipeline

The main entry point for track generation.

```python
from aether.orchestration import MusicPipeline
from aether.config import init_config
```

#### Constructor

```python
pipeline = MusicPipeline(
    config: AetherConfig,
    artifact_store: Optional[ArtifactStore] = None,
)
```

#### `generate()` - Main Generation Method

```python
async def generate(
    self,
    title: str,
    genre: str,
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    mood: Optional[str] = None,
    duration: Optional[int] = None,
    creative_brief: Optional[Dict[str, Any]] = None,
    render_audio: bool = True,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `title` | str | Yes | Track title |
| `genre` | str | Yes | Genre ID (e.g., "synthwave", "boom-bap") |
| `bpm` | int | No | Tempo in BPM (auto-selected from genre if not specified) |
| `key` | str | No | Musical key (e.g., "C major", "A minor") |
| `mood` | str | No | Primary mood (e.g., "energetic", "melancholic") |
| `duration` | int | No | Target duration in seconds |
| `creative_brief` | dict | No | Additional creative direction |
| `render_audio` | bool | No | Whether to render audio (default: True) |

**Returns:**

```python
{
    "song_id": str,
    "title": str,
    "success": bool,
    "duration_seconds": float,
    "output_path": Optional[Path],
    "specs": {
        "song_spec": dict,
        "harmony_spec": dict,
        "melody_spec": dict,
        "arrangement_spec": dict,
        # ... all generated specs
    },
    "audio": {
        "sample_rate": int,
        "channels": int,
        "duration": float,
    },
    "qa_report": Optional[dict],
}
```

**Example:**

```python
import asyncio
from aether.orchestration import MusicPipeline
from aether.config import init_config

async def main():
    config = init_config()
    pipeline = MusicPipeline(config)

    result = await pipeline.generate(
        title="Neon Dreams",
        genre="synthwave",
        bpm=118,
        key="A minor",
        mood="nostalgic",
        duration=240,
        creative_brief={
            "theme": "night driving",
            "era": "1980s",
            "instruments": ["analog synths", "drum machine"],
        },
    )

    if result["success"]:
        print(f"Generated: {result['output_path']}")
    else:
        print(f"Failed: {result.get('error')}")

asyncio.run(main())
```

---

## Agents API

All agents follow the same interface pattern.

### Base Agent Interface

```python
from aether.agents.base import BaseAgent

class BaseAgent(ABC, Generic[TInput, TOutput]):
    agent_type: str          # Unique identifier
    agent_name: str          # Human-readable name
    input_schema: type       # Pydantic input model
    output_schema: type      # Pydantic output model

    async def process(
        self,
        input_data: TInput,
        context: Dict[str, Any],
    ) -> TOutput:
        """Main processing method."""
        ...

    def log_decision(
        self,
        decision_type: str,
        input_summary: str,
        output_summary: str,
        reasoning: str,
        confidence: float = 0.8,
    ) -> AgentDecision:
        """Log a decision for auditability."""
        ...
```

### Available Agents

#### CreativeDirectorAgent

Interprets creative briefs and generates song specifications.

```python
from aether.agents import CreativeDirectorAgent

agent = CreativeDirectorAgent()

# Input schema
input_data = agent.input_schema(
    prompt="upbeat summer anthem about freedom",
    title="Breaking Free",
    genre_id="pop",
    creative_brief={
        "theme": "liberation",
        "energy": "high",
        "target_audience": "young adults",
    },
)

# Process
result = await agent.process(input_data, context={})

# Output
result.song_spec  # Dict with full song specification
```

#### CompositionAgent

Generates harmony, melody, and rhythm specifications.

```python
from aether.agents import CompositionAgent

agent = CompositionAgent()

input_data = agent.input_schema(
    song_spec=song_spec_dict,
    genre_profile_id="synthwave",
)

result = await agent.process(input_data, context={})

result.harmony_spec  # Chord progressions, key, mode
result.melody_spec   # Melodic patterns, hooks
result.rhythm_spec   # Tempo, time signature, groove
```

#### ArrangementAgent

Structures the song and assigns instruments.

```python
from aether.agents import ArrangementAgent

agent = ArrangementAgent()

input_data = agent.input_schema(
    song_spec=song_spec,
    rhythm_spec=rhythm_spec,
    harmony_spec=harmony_spec,
    melody_spec=melody_spec,
    genre_profile_id="boom-bap",
)

result = await agent.process(input_data, context={})

result.arrangement_spec  # Sections, instruments, structure
```

#### LyricsAgent

Generates original lyrics.

```python
from aether.agents import LyricsAgent

agent = LyricsAgent()

input_data = agent.input_schema(
    song_spec=song_spec,
    arrangement_spec=arrangement_spec,
    melody_spec=melody_spec,
)

result = await agent.process(input_data, context={})

result.lyric_spec  # Verses, choruses, hooks with syllable counts
```

#### MixingAgent

Creates mix specifications.

```python
from aether.agents import MixingAgent

agent = MixingAgent()

input_data = agent.input_schema(
    song_spec=song_spec,
    arrangement_spec=arrangement_spec,
    sound_design_spec=sound_design_spec,
    genre_profile_id="lo-fi",
)

result = await agent.process(input_data, context={})

result.mix_spec  # Buses, tracks, levels, panning, effects
```

#### MasteringAgent

Generates mastering specifications.

```python
from aether.agents import MasteringAgent

agent = MasteringAgent()

input_data = agent.input_schema(
    song_spec=song_spec,
    mix_spec=mix_spec,
    genre_profile_id="synthwave",
)

result = await agent.process(input_data, context={})

result.master_spec  # Loudness, EQ, compression, limiting
```

### Agent Registry

```python
from aether.agents.base import AgentRegistry

# List all registered agents
agent_types = AgentRegistry.list_types()
# ['creative_director', 'composition', 'arrangement', ...]

# Get agent class by type
AgentClass = AgentRegistry.get("creative_director")

# Create agent instance
agent = AgentRegistry.create("mixing", config={"some": "config"})
```

---

## Providers API

### Provider Registry

```python
from aether.providers import get_provider_registry

registry = get_provider_registry()

# Get provider by type
llm = registry.get("llm")
midi = registry.get("midi")
audio = registry.get("audio")
embedding = registry.get("embedding")
```

### LLM Provider

```python
from aether.providers.llm import (
    ClaudeLLMProvider,
    OpenAILLMProvider,
    MockLLMProvider,
    LLMMessage,
)

# Initialize
provider = ClaudeLLMProvider(
    api_key="sk-ant-...",  # Or set ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-20250514",
)
await provider.initialize()

# Generate completion
response = await provider.complete(
    messages=[
        LLMMessage(role="system", content="You are a songwriter."),
        LLMMessage(role="user", content="Write a verse about rain."),
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response.content)
print(response.usage)  # {"input_tokens": 20, "output_tokens": 150}

# Structured output (JSON mode)
response = await provider.complete(
    messages=[...],
    json_mode=True,
)

# Shutdown
await provider.shutdown()
```

### MIDI Provider

```python
from aether.providers.midi import AlgorithmicMIDIProvider

provider = AlgorithmicMIDIProvider()
await provider.initialize()

# Generate MIDI from specs
midi_file = await provider.generate_from_spec(
    harmony_spec={
        "progression": ["Am", "F", "C", "G"],
        "key": "A",
        "mode": "minor",
    },
    melody_spec={
        "contour": "arch",
        "range_octaves": 1.5,
    },
    rhythm_spec={
        "bpm": 120,
        "time_signature": (4, 4),
    },
    arrangement_spec={
        "sections": ["verse", "chorus"],
    },
)

# Access tracks
for track in midi_file.tracks:
    print(f"{track.name}: {len(track.notes)} notes")

# Save to file
await provider.save_midi(midi_file, Path("output.mid"))
```

### Audio Provider

```python
from aether.providers.audio import SynthAudioProvider

provider = SynthAudioProvider(
    sample_rate=48000,
    soundfont_path=Path("/path/to/soundfont.sf2"),  # Optional
)
await provider.initialize()

# Render MIDI to audio
audio_buffer = await provider.render_midi(midi_file)

print(f"Shape: {audio_buffer.data.shape}")  # (2, samples)
print(f"Duration: {audio_buffer.data.shape[1] / audio_buffer.sample_rate}s")

# Mix stems
from aether.providers import AudioStem

stems = [
    AudioStem(name="drums", buffer=drums_buffer, category="drums"),
    AudioStem(name="bass", buffer=bass_buffer, category="bass"),
]

mixed = await provider.mix_stems(
    stems,
    levels={"drums": 0.0, "bass": -3.0},  # dB
    pans={"drums": 0.0, "bass": 0.0},     # -1 to 1
)
```

### Embedding Provider

```python
from aether.providers.embedding import (
    SentenceTransformerEmbeddingProvider,
    MockEmbeddingProvider,
)

# Local embeddings (requires [ml] extra)
provider = SentenceTransformerEmbeddingProvider(
    model_name="all-MiniLM-L6-v2",
)
await provider.initialize()

# Generate embeddings
result = await provider.embed_text("Walking in the rain")
print(f"Embedding shape: {result.embedding.shape}")  # (384,)

# Batch embeddings
results = await provider.embed_batch([
    "First text",
    "Second text",
    "Third text",
])

# Similarity
similarity = await provider.similarity(
    result1.embedding,
    result2.embedding,
)
print(f"Similarity: {similarity:.3f}")  # 0.0 to 1.0
```

---

## Schemas

All data structures are Pydantic models for validation and serialization.

### Core Schemas

```python
from aether.schemas import (
    SongSpec,
    HarmonySpec,
    MelodySpec,
    RhythmSpec,
    ArrangementSpec,
    LyricSpec,
    VocalSpec,
    SoundDesignSpec,
    MixSpec,
    MasterSpec,
    QAReport,
)
```

### Example: SongSpec

```python
from aether.schemas import SongSpec

song = SongSpec(
    id="song-123",
    title="Midnight Drive",
    primary_mood="nostalgic",
    secondary_moods=["melancholic", "hopeful"],
    tempo_bpm=118,
    key_signature="Am",
    time_signature="4/4",
    target_duration_seconds=240,
    creative_brief={
        "theme": "night driving",
        "era": "1980s",
    },
)

# Serialize to dict
data = song.model_dump()

# Serialize to JSON
json_str = song.model_dump_json()
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `AETHER_DEBUG` | Enable debug mode | `false` |
| `AETHER_LOG_LEVEL` | Log level | `INFO` |
| `AETHER_PATHS__BASE_DIR` | Base directory | `~/.aether` |
| `AETHER_AUDIO__DEFAULT_LUFS` | Default loudness | `-14.0` |

### Programmatic Configuration

```python
from aether.config import AetherConfig, set_config

config = AetherConfig(
    debug=True,
    log_level="DEBUG",
    audio=AudioConfig(
        default_lufs=-16.0,  # For YouTube
    ),
)

set_config(config)
```

---

## Error Handling

### Exception Hierarchy

```python
from aether.core.exceptions import (
    AetherError,              # Base exception
    ConfigurationError,       # Configuration issues
    MissingConfigError,       # Missing required config
    ProviderError,            # Provider failures
    ProviderInitializationError,
    ProviderUnavailableError,
    PipelineError,            # Pipeline failures
    AgentError,               # Agent failures
    ValidationError,          # Data validation failures
    AudioError,               # Audio processing errors
    MIDIError,                # MIDI processing errors
    RetryExhaustedError,      # Retry limit reached
    CircuitBreakerOpenError,  # Circuit breaker tripped
)
```

### Error Handling Pattern

```python
from aether.core.exceptions import (
    AetherError,
    ProviderError,
    PipelineError,
)

try:
    result = await pipeline.generate(...)
except ProviderError as e:
    print(f"Provider failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Recovery hints: {e.recovery_hints}")
except PipelineError as e:
    print(f"Pipeline failed at stage: {e.stage}")
except AetherError as e:
    print(f"AETHER error: {e}")
```

### Resilience Patterns

```python
from aether.core.resilience import (
    retry,
    circuit_breaker,
    timeout,
    fallback,
    BackoffStrategy,
)

@retry(max_attempts=3, backoff=BackoffStrategy.EXPONENTIAL)
async def unreliable_operation():
    ...

@circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
async def external_api_call():
    ...

@timeout(seconds=10.0)
async def time_sensitive_operation():
    ...

@fallback(default_value="default")
async def operation_with_fallback():
    ...
```

---

## Health Checks

```python
from aether.core.health import (
    HealthCheck,
    health_check,
    get_probe_manager,
)

# Register custom health check
health = HealthCheck.get_instance()

@health.register("my_service", timeout=5.0, critical=True)
async def check_my_service():
    # Return True/False or (bool, message) or (bool, message, details)
    return True, "Service is healthy", {"latency_ms": 10}

# Run all checks
system_health = await health.check_all()
print(f"Status: {system_health.status}")
print(f"Uptime: {system_health.uptime_seconds}s")

# Kubernetes-style probes
probes = get_probe_manager()
probes.set_ready(True)

liveness = await probes.check_liveness()
readiness = await probes.check_readiness()
```

---

## Metrics

```python
from aether.core.metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
)

metrics = MetricsCollector.get_instance()

# Counter
tracks_generated = Counter("aether_tracks_generated", "Total tracks generated")
tracks_generated.inc()
tracks_generated.inc(labels={"genre": "synthwave"})

# Gauge
active_pipelines = Gauge("aether_active_pipelines", "Currently running pipelines")
active_pipelines.set(5)

# Histogram
render_time = Histogram(
    "aether_render_duration_seconds",
    "Time to render audio",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
render_time.observe(1.5)

# Timer context manager
with Timer("aether_pipeline_duration_seconds", "Pipeline duration"):
    await pipeline.generate(...)

# Export Prometheus format
prometheus_output = metrics.export_prometheus()
```

---

## Logging

```python
from aether.core.logging import (
    get_logger,
    configure_logging,
    LogContext,
    log_operation,
)

# Get logger
logger = get_logger(__name__)

# Configure logging
configure_logging(
    level="DEBUG",
    format="json",  # or "human"
    log_file=Path("/var/log/aether.log"),
)

# Context propagation
with LogContext(trace_id="abc123", user_id="user1"):
    logger.info("Processing request")
    # All logs within this context include trace_id and user_id

# Decorator for operation logging
@log_operation(name="generate_track")
async def generate_track():
    ...
```
