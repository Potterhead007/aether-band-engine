# AETHER Band Engine

**Autonomous Ensemble for Thoughtful Harmonic Expression and Rendering**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade AI music generation engine capable of creating commercially viable, fully original music across any genre. From prompt to master-quality audio in a single command.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Pipeline Agents](#pipeline-agents)
- [Quality Standards](#quality-standards)
- [API Usage](#api-usage)
- [Development](#development)
- [License](#license)

---

## Features

### Core Capabilities

- **100% Original Content** - Every melody, harmony, lyric, and arrangement is algorithmically generated
- **Multi-Genre Support** - Deep musicological profiles for boom-bap, synthwave, lo-fi, and extensible to any genre
- **Commercial Quality** - Meets Spotify, Apple Music, and YouTube streaming specifications
- **Full Reproducibility** - Complete recipe files enable exact track regeneration
- **CLI-First Design** - No DAW required; entire workflow via command line

### Technical Highlights

- **10-Agent Pipeline** - Specialized agents for each production stage
- **Production-Grade Infrastructure** - Structured logging, circuit breakers, health checks, metrics
- **Flexible Providers** - Pluggable LLM (Claude, OpenAI), MIDI, audio, and embedding providers
- **Broadcast-Ready Mastering** - ITU-R BS.1770-4 loudness, true peak limiting, multiband compression
- **Originality Verification** - Melody fingerprinting, lyric n-gram analysis, audio embedding similarity

---

## Installation

### Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended for ML features)
- macOS, Linux, or Windows

### Base Installation (Minimal)

```bash
pip install aether-band-engine
```

This includes core functionality with algorithmic MIDI generation and audio synthesis (~150MB).

### With Machine Learning Features

```bash
pip install 'aether-band-engine[ml]'
```

Adds sentence-transformers for semantic embeddings and librosa for advanced audio analysis (~700MB).

### With LLM Providers

```bash
pip install 'aether-band-engine[llm]'
```

Adds Anthropic Claude and OpenAI SDK support for AI-powered creative generation.

### Full Installation

```bash
pip install 'aether-band-engine[full]'
```

All features enabled.

### Development Installation

```bash
git clone https://github.com/aether-band-engine/aether-band-engine.git
cd aether-band-engine
pip install -e '.[dev]'
```

---

## Quick Start

### 1. Initialize AETHER

```bash
aether init
```

Creates configuration directory at `~/.aether/` with default settings.

### 2. Create a Project

```bash
aether new-project "Summer Nights" --genre synthwave
```

### 3. Generate a Track

```bash
aether build-track "Midnight Drive" \
    --genre synthwave \
    --bpm 118 \
    --key "A minor" \
    --mood nostalgic \
    --duration 240
```

### 4. Check Output

```bash
ls ~/.aether/output/
# Midnight_Drive_master.wav
# Midnight_Drive_stems/
# Midnight_Drive_recipe.yaml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AETHER Band Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Interface                                              │
│  └── CLI (Click + Rich) | Future: REST API, Web UI              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Orchestration                                          │
│  └── Pipeline DAG Engine | Workflow State | Error Recovery       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Agents (10 Pipeline Agents)                            │
│  └── Creative Director → Composition → Arrangement → Lyrics →   │
│      Vocal → Sound Design → Mixing → Mastering → QA → Release   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Providers                                              │
│  └── LLM (Claude/OpenAI/Mock) | MIDI | Audio | Embedding        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Core Infrastructure                                    │
│  └── Exceptions | Logging | Resilience | Health | Metrics       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Knowledge                                              │
│  └── Genre Profiles | Music Theory | Instrument Definitions     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: Storage                                                │
│  └── Artifact Store (SQLite) | Content-Addressable Blobs        │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
aether-band-engine/
├── src/aether/
│   ├── agents/          # 10 pipeline agents
│   ├── audio/           # DSP, mixing, mastering, I/O
│   ├── core/            # Exceptions, logging, resilience, health, metrics
│   ├── knowledge/       # Genre profiles, music theory
│   ├── orchestration/   # Pipeline DAG, workflow engine
│   ├── providers/       # LLM, MIDI, audio, embedding providers
│   ├── qa/              # Quality assurance, originality checking
│   ├── rendering/       # Spec-to-audio conversion
│   ├── schemas/         # Pydantic models for all specs
│   └── storage/         # Artifact store, metadata
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
└── data/
    └── genres/          # Genre profile YAML files
```

---

## Configuration

### Configuration File

Located at `~/.aether/config.yaml`:

```yaml
paths:
  base_dir: ~/.aether
  output_dir: output
  projects_dir: projects

providers:
  llm_provider: anthropic          # anthropic, openai, mock
  llm_model: claude-sonnet-4-20250514
  midi_provider: internal
  audio_provider: soundfont
  embedding_provider: sentence-transformers

audio:
  working_sample_rate: 48000
  output_sample_rate: 44100
  output_bit_depth: 24
  default_lufs: -14.0
  default_true_peak: -1.0

qa:
  melody_similarity_threshold: 0.85
  lyric_ngram_threshold: 0.03
  genre_authenticity_threshold: 0.80
```

### Environment Variables

```bash
# LLM API Keys (required for LLM providers)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Configuration overrides
export AETHER_DEBUG=true
export AETHER_LOG_LEVEL=DEBUG
export AETHER_AUDIO__DEFAULT_LUFS=-16.0
```

---

## CLI Reference

### Project Management

```bash
aether new-project <name> [--genre GENRE]    # Create new project
aether list-projects                          # List all projects
aether project-status <name>                  # Show project status
```

### Track Generation

```bash
aether build-track <title> \
    --genre <genre> \
    [--bpm BPM] \
    [--key KEY] \
    [--mood MOOD] \
    [--duration SECONDS] \
    [--output-dir PATH]
```

### Genre Management

```bash
aether list-genres                            # List available genres
aether genre-info <genre>                     # Show genre details
```

### Pipeline Control

```bash
aether pipeline list                          # List running pipelines
aether pipeline status <id>                   # Show pipeline status
aether pipeline resume <id>                   # Resume failed pipeline
```

### System

```bash
aether init                                   # Initialize AETHER
aether config                                 # Show current configuration
aether health                                 # System health check
aether version                                # Show version info
```

---

## Pipeline Agents

| Agent | Input | Output | Description |
|-------|-------|--------|-------------|
| **Creative Director** | Prompt, Genre | SongSpec | Interprets creative brief, sets direction |
| **Composition** | SongSpec | Harmony, Melody, Rhythm | Generates musical foundations |
| **Arrangement** | Specs | ArrangementSpec | Structures song, assigns instruments |
| **Lyrics** | Song, Melody | LyricSpec | Writes original lyrics |
| **Vocal** | Lyrics, Melody | VocalSpec | Plans vocal performance |
| **Sound Design** | Arrangement | SoundDesignSpec | Designs patches and samples |
| **Mixing** | All Specs | MixSpec | Creates mix blueprint |
| **Mastering** | Mix | MasterSpec | Finalizes for distribution |
| **QA** | All | QAReport | Validates quality and originality |
| **Release** | All | ReleasePackage | Packages for distribution |

---

## Quality Standards

### Audio Technical Specifications

| Metric | Target | Standard |
|--------|--------|----------|
| Integrated Loudness | -14.0 LUFS | ITU-R BS.1770-4 |
| True Peak | ≤ -1.0 dBTP | AES-17 |
| Sample Rate | 44.1/48 kHz | CD/Streaming |
| Bit Depth | 24-bit | Professional |
| Dynamic Range | ≥ 6 LU | EBU R128 |

### Originality Thresholds

| Check | Threshold | Method |
|-------|-----------|--------|
| Melody Similarity | < 0.85 | Interval sequence fingerprinting |
| Lyric N-gram Overlap | < 3% | N-gram frequency analysis |
| Audio Embedding | < 0.15 | Cosine similarity |
| Harmonic Progression | < 0.90 | Chord sequence matching |

### Genre Authenticity

Minimum score of 0.80 across:
- Tempo/Groove adherence
- Harmonic vocabulary
- Melodic characteristics
- Production aesthetics
- Arrangement conventions

---

## API Usage

### Programmatic Pipeline

```python
import asyncio
from aether.orchestration import MusicPipeline
from aether.config import init_config

async def generate_track():
    config = init_config()
    pipeline = MusicPipeline(config)

    result = await pipeline.generate(
        title="Ocean Dreams",
        genre="lo-fi",
        bpm=85,
        key="F major",
        mood="relaxed",
        duration=180,
    )

    print(f"Generated: {result['output_path']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    print(f"Loudness: {result['loudness_lufs']:.1f} LUFS")

asyncio.run(generate_track())
```

### Using Individual Agents

```python
from aether.agents import CreativeDirectorAgent

agent = CreativeDirectorAgent()
input_data = agent.input_schema(
    prompt="upbeat summer anthem",
    title="Sunshine State",
    genre_id="pop",
    creative_brief={"theme": "summer", "energy": "high"},
)

result = await agent.process(input_data, context={})
print(result.song_spec)
```

### Provider Access

```python
from aether.providers import get_provider_registry

registry = get_provider_registry()

# MIDI generation
midi = registry.get("midi")
await midi.initialize()
midi_file = await midi.generate_from_spec(harmony_spec, melody_spec, rhythm_spec)

# Audio rendering
audio = registry.get("audio")
await audio.initialize()
buffer = await audio.render_midi(midi_file)
```

---

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=aether --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Adding a Genre Profile

Create `data/genres/<genre-name>.yaml`:

```yaml
id: my-genre
name: My Genre
description: Description of the genre

lineage:
  primary_parent: electronic
  secondary_influences: [ambient, techno]

rhythm:
  tempo_range: [110, 130]
  time_signatures: ["4/4"]
  swing_amount: 0.0

harmony:
  common_keys: [C, G, Am, Em]
  common_modes: [major, minor]
  typical_progressions:
    - [I, V, vi, IV]
    - [vi, IV, I, V]

# ... see existing profiles for full schema
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- ITU-R BS.1770-4 loudness metering implementation
- Audio EQ Cookbook by Robert Bristow-Johnson
- Linkwitz-Riley crossover design

---

**Built with precision. Powered by creativity.**
