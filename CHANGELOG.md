# Changelog

All notable changes to AETHER Band Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-04

### Added

#### Core Infrastructure (Phase 10)
- Comprehensive exception hierarchy with 40+ typed exceptions
- Structured JSON logging with context propagation (trace_id, span_id)
- Resilience patterns: retry, circuit breaker, timeout, fallback, bulkhead
- Health check system with Kubernetes-style liveness/readiness probes
- Metrics collection with Prometheus-compatible export
- 120+ test cases with 94% pass rate

#### Pipeline Agents (Phases 4-5)
- 10 specialized pipeline agents: Creative Director, Composition, Arrangement, Lyrics, Vocal, Sound Design, Mixing, Mastering, QA, Release
- Agent registry with type-based lookup and creation
- Decision logging for auditability
- Typed input/output schemas with Pydantic validation

#### Audio Processing (Phase 6)
- Professional DSP utilities (850+ lines)
  - Biquad filters (Audio EQ Cookbook implementation)
  - Parametric EQ (multi-band)
  - Compressor with soft knee and program-dependent release
  - True Peak Limiter (ITU-R BS.1770 compliant)
  - Loudness Meter (ITU-R BS.1770-4, EBU R128)
- Mixing engine with bus architecture and automation
- Broadcast-grade mastering chain with multiband compression
- Platform presets (Spotify, Apple Music, YouTube, Broadcast)

#### QA System (Phase 7)
- Technical validator (loudness, true peak, phase correlation)
- Originality checker (melody fingerprinting, lyric n-gram analysis)
- Genre authenticity evaluator with multi-dimensional scoring
- QA report generator with multiple output formats

#### Providers (Phase 8-9)
- LLM providers (Claude, OpenAI, Mock) with rate limiting and retry
- Algorithmic MIDI generation with music theory utilities
- Audio synthesis with FluidSynth support
- Embedding providers (SentenceTransformer, OpenAI, Mock)
- Provider registry with centralized lifecycle management
- Rendering engine for spec-to-audio conversion

#### Knowledge System (Phases 1-3)
- Genre profile system with YAML definitions
- Music theory utilities (chords, scales, progressions)
- 3 complete genre profiles (boom-bap, synthwave, lo-fi)

#### CLI (Phase 5)
- Full command-line interface with Click and Rich
- Project management commands
- Track generation with progress display
- Pipeline status and resume functionality

### Security
- API key validation with fail-fast on missing configuration
- Config serialization excludes sensitive fields
- CLI path sanitization prevents traversal attacks

### Changed
- Dependencies optimized: removed unused packages (sqlalchemy, chromadb, audioread, pretty_midi)
- Heavy dependencies made optional: sentence-transformers, librosa, anthropic, openai
- Base install size reduced from ~800MB to ~150MB

### Documentation
- Comprehensive README with installation, usage, and API examples
- Full API documentation
- Architecture documentation with system diagrams
- Docker configuration (Dockerfile, docker-compose.yml)
- GitHub Actions CI/CD pipelines

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-04 | Initial release with full pipeline |

[Unreleased]: https://github.com/aether-band-engine/aether-band-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aether-band-engine/aether-band-engine/releases/tag/v0.1.0
