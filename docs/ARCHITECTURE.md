# AETHER Band Engine Architecture

## Overview

AETHER Band Engine is a production-grade AI music generation system built on a layered architecture with clear separation of concerns. This document describes the system's design principles, component interactions, and data flow.

---

## Design Principles

### 1. Layered Architecture
Each layer has a single responsibility and communicates only with adjacent layers.

### 2. Provider Abstraction
All external dependencies (LLMs, audio synthesis, embeddings) are accessed through provider interfaces, enabling easy swapping and testing.

### 3. Specification-Driven Pipeline
The pipeline generates increasingly detailed specifications at each stage, with clear contracts between agents.

### 4. Production-Grade Infrastructure
Built-in resilience patterns, structured logging, health checks, and metrics from day one.

### 5. Reproducibility
Every generated track has a complete recipe that enables exact regeneration.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Layer 1: Interface                              │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │    CLI      │    │  REST API   │    │   Web UI    │                     │
│   │  (Click)    │    │  (Future)   │    │  (Future)   │                     │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                     │
│          │                  │                  │                             │
└──────────┴──────────────────┴──────────────────┴─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Layer 2: Orchestration                              │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        MusicPipeline                                 │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│   │  │   DAG       │  │  Workflow   │  │   Error     │                  │   │
│   │  │   Engine    │  │   State     │  │  Recovery   │                  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Layer 3: Agents                                   │
│                                                                              │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                   │
│   │Creative│→│Compos- │→│Arrange-│→│ Lyrics │→│ Vocal  │                   │
│   │Director│ │ition   │ │ment    │ │        │ │        │                   │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘                   │
│        │          │          │          │          │                        │
│        ▼          ▼          ▼          ▼          ▼                        │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                   │
│   │ Sound  │→│ Mixing │→│Master- │→│   QA   │→│Release │                   │
│   │ Design │ │        │ │ing     │ │        │ │        │                   │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Layer 4: Providers                                 │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │     LLM     │  │    MIDI     │  │    Audio    │  │  Embedding  │       │
│   │  Provider   │  │  Provider   │  │  Provider   │  │  Provider   │       │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤       │
│   │ - Claude    │  │ - Algorith- │  │ - Synth     │  │ - Sentence  │       │
│   │ - OpenAI    │  │   mic       │  │ - FluidSynth│  │   Transform │       │
│   │ - Mock      │  │             │  │             │  │ - OpenAI    │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Layer 5: Core Infrastructure                           │
│                                                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│   │Exception│  │ Logging │  │Resilience│ │ Health  │  │ Metrics │          │
│   │Hierarchy│  │ System  │  │ Patterns │ │ Checks  │  │Collector│          │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Layer 6: Knowledge                                  │
│                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │  Genre Profiles │  │  Music Theory   │  │  Instruments    │            │
│   │  (YAML)         │  │  Utilities      │  │  Definitions    │            │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Layer 7: Storage                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Artifact Store                                │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│   │  │   SQLite    │  │    Blob     │  │  Checksums  │                  │   │
│   │  │  Metadata   │  │   Storage   │  │ (SHA-256)   │                  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Layer 1: Interface

#### CLI (`aether/cli.py`)
- Built with Click framework
- Rich terminal output
- Async command execution
- Project and pipeline management

```
aether
├── init              # Initialize AETHER
├── new-project       # Create project
├── build-track       # Generate track
├── list-genres       # Show genres
├── pipeline
│   ├── list          # List pipelines
│   ├── status        # Show status
│   └── resume        # Resume failed
└── config            # Show configuration
```

### Layer 2: Orchestration

#### MusicPipeline (`aether/orchestration/pipeline.py`)

The central orchestrator that:
1. Initializes providers
2. Creates agent DAG
3. Executes agents in order
4. Handles failures and recovery
5. Manages state persistence

**Pipeline Flow:**
```
                    ┌──────────────┐
                    │   generate() │
                    └──────┬───────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Initialize Providers        │
            │   (LLM, MIDI, Audio, Embed)   │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Execute Agent Pipeline      │
            │   (10 agents in sequence)     │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Render Audio               │
            │   (MIDI → Audio → Master)    │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Shutdown Providers         │
            │   (Cleanup resources)        │
            └──────────────────────────────┘
```

### Layer 3: Agents

Each agent follows the same pattern:

```python
class Agent(BaseAgent[InputSchema, OutputSchema]):
    agent_type = "agent_name"

    async def process(self, input_data, context) -> output:
        # 1. Access providers (self.llm, self.midi, etc.)
        # 2. Generate specification
        # 3. Log decisions
        # 4. Return output
```

**Agent Data Flow:**

```
┌─────────────────┐
│Creative Director│ prompt, genre
└────────┬────────┘
         │ SongSpec
         ▼
┌─────────────────┐
│  Composition    │ SongSpec, genre_profile
└────────┬────────┘
         │ HarmonySpec, MelodySpec, RhythmSpec
         ▼
┌─────────────────┐
│  Arrangement    │ All specs, genre_profile
└────────┬────────┘
         │ ArrangementSpec
         ├─────────────────────────┐
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│    Lyrics       │       │  Sound Design   │
└────────┬────────┘       └────────┬────────┘
         │ LyricSpec              │ SoundDesignSpec
         ▼                         │
┌─────────────────┐               │
│     Vocal       │               │
└────────┬────────┘               │
         │ VocalSpec              │
         └─────────────┬──────────┘
                       ▼
              ┌─────────────────┐
              │     Mixing      │
              └────────┬────────┘
                       │ MixSpec
                       ▼
              ┌─────────────────┐
              │    Mastering    │
              └────────┬────────┘
                       │ MasterSpec
                       ▼
              ┌─────────────────┐
              │       QA        │
              └────────┬────────┘
                       │ QAReport
                       ▼
              ┌─────────────────┐
              │    Release      │
              └─────────────────┘
```

### Layer 4: Providers

#### Provider Interface

```python
class BaseProvider(ABC):
    async def initialize(self) -> bool
    async def shutdown(self) -> None
    async def health_check(self) -> bool
    def get_info(self) -> ProviderInfo
```

#### Provider Registry

Singleton that manages provider instances:

```python
registry = get_provider_registry()
registry.register("llm", ClaudeLLMProvider())
llm = registry.get("llm")
```

#### Provider Types

| Type | Interface | Implementations |
|------|-----------|-----------------|
| LLM | `LLMProvider` | Claude, OpenAI, Mock |
| MIDI | `MIDIProvider` | Algorithmic |
| Audio | `AudioProvider` | Synth, FluidSynth |
| Embedding | `EmbeddingProvider` | SentenceTransformer, OpenAI, Mock |

### Layer 5: Core Infrastructure

#### Exception Hierarchy

```
AetherError
├── ConfigurationError
│   ├── MissingConfigError
│   └── InvalidConfigError
├── ProviderError
│   ├── ProviderInitializationError
│   ├── ProviderUnavailableError
│   └── RateLimitError
├── PipelineError
│   └── PipelineStageError
├── AgentError
│   └── AgentNotFoundError
├── ValidationError
├── AudioError
├── MIDIError
└── ResilienceError
    ├── RetryExhaustedError
    └── CircuitBreakerOpenError
```

#### Resilience Patterns

```
┌─────────────────────────────────────────────────────┐
│                  @resilient                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │  @retry   │  │ @timeout  │  │ @fallback │       │
│  │           │  │           │  │           │       │
│  │ Exponential│  │ Async    │  │ Default   │       │
│  │ Backoff   │  │ Cancel   │  │ Values    │       │
│  └───────────┘  └───────────┘  └───────────┘       │
│                                                      │
│  ┌───────────────────┐  ┌───────────────────┐      │
│  │  CircuitBreaker   │  │     Bulkhead      │      │
│  │                   │  │                   │      │
│  │ CLOSED → OPEN →   │  │ Concurrency      │      │
│  │ HALF_OPEN         │  │ Isolation        │      │
│  └───────────────────┘  └───────────────────┘      │
└─────────────────────────────────────────────────────┘
```

#### Health Check System

```
┌─────────────────────────────────────────────────────┐
│                   HealthCheck                        │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Component   │  │ Component   │  │ Component   │ │
│  │ Check 1     │  │ Check 2     │  │ Check 3     │ │
│  │ (critical)  │  │             │  │             │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │        │
│         └────────────────┼────────────────┘        │
│                          │                         │
│                          ▼                         │
│              ┌───────────────────┐                 │
│              │   SystemHealth    │                 │
│              │                   │                 │
│              │ status: HEALTHY   │                 │
│              │ uptime: 3600s     │                 │
│              │ components: [...]  │                 │
│              └───────────────────┘                 │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │              ProbeManager                    │   │
│  │  ┌─────────────┐     ┌─────────────┐       │   │
│  │  │  Liveness   │     │  Readiness  │       │   │
│  │  │  /healthz   │     │  /ready     │       │   │
│  │  └─────────────┘     └─────────────┘       │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Layer 6: Knowledge

#### Genre Profile System

```yaml
# data/genres/synthwave.yaml
id: synthwave
name: Synthwave
lineage:
  primary_parent: electronic
  secondary_influences: [new-wave, italo-disco]

rhythm:
  tempo_range: [80, 118]
  time_signatures: ["4/4"]
  swing_amount: 0.0

harmony:
  common_keys: [Am, Dm, Em, Fm]
  common_modes: [minor, dorian]
  typical_progressions:
    - [i, VI, III, VII]
    - [i, iv, VI, V]

melody:
  typical_range_octaves: 1.5
  step_vs_leap_ratio: 0.7

instrumentation:
  core_instruments:
    - analog_synth_lead
    - analog_synth_pad
    - analog_bass
    - drum_machine

production:
  target_lufs: -14.0
  dynamic_range_lu: 8
```

### Layer 7: Storage

#### Artifact Store

```
~/.aether/artifacts/
├── artifacts.db          # SQLite metadata
└── blobs/                # Content-addressable storage
    ├── a1/
    │   └── a1b2c3...     # SHA-256 hash
    └── b2/
        └── b2c3d4...
```

**Database Schema:**

```sql
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    song_id TEXT,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    is_deleted INTEGER DEFAULT 0
);

CREATE TABLE artifact_tags (
    artifact_id TEXT,
    key TEXT,
    value TEXT,
    PRIMARY KEY (artifact_id, key)
);

CREATE TABLE artifact_parents (
    artifact_id TEXT,
    parent_id TEXT,
    PRIMARY KEY (artifact_id, parent_id)
);
```

---

## Audio Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rendering Engine                              │
│                                                                  │
│   Specs ──► MIDI Generation ──► Audio Synthesis ──► Mixing     │
│                                                                  │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│   │   MIDI     │    │   Synth    │    │   Stems    │           │
│   │ Provider   │───►│  Provider  │───►│            │           │
│   └────────────┘    └────────────┘    └─────┬──────┘           │
│                                              │                   │
│                                              ▼                   │
│                                     ┌────────────────┐          │
│                                     │ Mixing Engine  │          │
│                                     │ - Bus routing  │          │
│                                     │ - EQ/Comp     │          │
│                                     │ - Send/Return │          │
│                                     └───────┬────────┘          │
│                                              │                   │
│                                              ▼                   │
│                                     ┌────────────────┐          │
│                                     │Mastering Chain │          │
│                                     │ - Multiband    │          │
│                                     │ - Limiter     │          │
│                                     │ - Loudness    │          │
│                                     └───────┬────────┘          │
│                                              │                   │
│                                              ▼                   │
│                                     ┌────────────────┐          │
│                                     │  Audio I/O    │          │
│                                     │ WAV/FLAC/MP3  │          │
│                                     └────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Schemas

All data flows through typed Pydantic schemas:

```
SongSpec
├── id: UUID
├── title: str
├── primary_mood: str
├── tempo_bpm: int
├── key_signature: str
├── time_signature: str
├── target_duration_seconds: int
└── creative_brief: Dict

HarmonySpec
├── key: str
├── mode: str
├── progressions: List[Progression]
└── chord_voicings: Dict

MelodySpec
├── primary_hook: Hook
├── section_melodies: List[SectionMelody]
└── typical_range_octaves: float

ArrangementSpec
├── sections: List[Section]
├── instruments: List[Instrument]
└── transitions: List[Transition]

MixSpec
├── buses: List[Bus]
├── tracks: List[Track]
├── sends: List[Send]
└── automation: List[Automation]

MasterSpec
├── loudness: LoudnessSettings
├── multiband_compression: MultibandSettings
├── limiter: LimiterSettings
└── stereo_enhancement: float
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Deployment                        │
│                                                                  │
│   ┌─────────────┐                                               │
│   │   Docker    │                                               │
│   │  Container  │                                               │
│   │             │                                               │
│   │  ┌───────┐  │    ┌─────────────┐    ┌─────────────┐        │
│   │  │AETHER │  │───►│  Volumes    │    │ Environment │        │
│   │  │ App   │  │    │ - /data     │    │ - API Keys  │        │
│   │  └───────┘  │    │ - /output   │    │ - Config    │        │
│   │             │    └─────────────┘    └─────────────┘        │
│   │  Ports:     │                                               │
│   │  - 8080 API │                                               │
│   │  - 9090 Met │                                               │
│   └─────────────┘                                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Health Endpoints                      │   │
│   │   /health/live   /health/ready   /metrics               │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### API Key Management
- Keys loaded from environment variables only
- Never serialized to disk
- Fail-fast validation on startup

### Input Validation
- All user input sanitized (project names, file paths)
- Pydantic validation on all data structures
- Path traversal prevention

### Dependency Security
- Minimal dependencies
- Optional heavy ML packages
- No native code compilation required

---

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Agent Pipeline | 5-30s | Depends on LLM response time |
| MIDI Generation | <1s | Algorithmic, no network |
| Audio Rendering | 2-10s | Depends on duration |
| Mastering | 1-5s | CPU-bound DSP |
| Full Track | 30-60s | End-to-end |

### Memory Usage
- Base: ~100MB
- With ML: ~500MB (sentence-transformers model)
- Per track render: ~50-200MB (audio buffers)

---

## Extension Points

### Adding a New Agent

1. Create agent class in `aether/agents/`
2. Define input/output schemas
3. Register with `@AgentRegistry.register("name")`
4. Add to pipeline DAG

### Adding a New Provider

1. Implement provider interface
2. Add to provider registry
3. Update configuration options

### Adding a New Genre

1. Create YAML file in `data/genres/`
2. Define all musicological parameters
3. Genre is automatically available

---

## Testing Strategy

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_core_*.py       # Core infrastructure
│   ├── test_agents.py       # Agent logic
│   └── test_schemas.py      # Data validation
├── integration/             # Cross-component tests
│   ├── test_providers.py    # Provider functionality
│   └── test_pipeline.py     # End-to-end pipeline
└── conftest.py              # Shared fixtures
```

**Coverage Targets:**
- Core: 90%+
- Providers: 85%+
- Agents: 80%+
- Overall: 85%+
