# Contributing to AETHER Band Engine

Thank you for your interest in contributing to AETHER Band Engine! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Architecture Overview](#architecture-overview)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Make (optional, for convenience commands)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/aether-band-engine.git
   cd aether-band-engine
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/aether-band-engine/aether-band-engine.git
   ```

---

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Development Dependencies

```bash
pip install -e ".[dev,full]"
```

### Verify Installation

```bash
# Run tests
pytest

# Check CLI
aether --version
```

---

## Making Changes

### Branch Naming

Create a branch for your changes:

```bash
git checkout -b type/description
```

Branch types:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

Examples:
- `feature/add-jazz-genre`
- `fix/midi-note-overflow`
- `docs/improve-api-reference`

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructure
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(agents): add jazz improvisation agent

Implements a new agent for generating jazz-style melodic variations
using modal interchange and chord extensions.

Closes #123
```

```
fix(audio): prevent clipping in mastering chain

The limiter was allowing transients to exceed the ceiling.
Added 4-sample lookahead for true peak limiting.
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific test file
pytest tests/unit/test_core_exceptions.py

# With coverage
pytest --cov=aether --cov-report=html
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`

Example test:

```python
import pytest
from aether.agents import CompositionAgent

class TestCompositionAgent:
    @pytest.fixture
    def agent(self):
        return CompositionAgent()

    @pytest.mark.asyncio
    async def test_generates_harmony_spec(self, agent, sample_song_spec):
        input_data = agent.input_schema(
            song_spec=sample_song_spec,
            genre_profile_id="synthwave",
        )

        result = await agent.process(input_data, context={})

        assert result.harmony_spec is not None
        assert "key" in result.harmony_spec
        assert "progressions" in result.harmony_spec
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   black src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/

   # Tests
   pytest
   ```

3. **Update documentation** if needed

### Submitting

1. Push your branch:
   ```bash
   git push origin your-branch-name
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if UI changes)

### Review Process

- PRs require at least one approval
- All CI checks must pass
- Address reviewer feedback
- Keep PRs focused and reasonably sized

---

## Style Guide

### Python Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Use type hints

```python
from typing import Dict, List, Optional

async def process_track(
    title: str,
    genre: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a track through the pipeline.

    Args:
        title: Track title
        genre: Genre identifier
        options: Optional processing options

    Returns:
        Processing result with specs and audio
    """
    ...
```

### Documentation Style

- Use Google-style docstrings
- Include type hints in docstrings for complex types
- Add examples for public APIs

### Import Order

```python
# Standard library
import os
from pathlib import Path
from typing import Any, Dict

# Third-party
import numpy as np
from pydantic import BaseModel

# Local
from aether.core.exceptions import AetherError
from aether.agents.base import BaseAgent
```

---

## Architecture Overview

### Adding a New Agent

1. Create file in `src/aether/agents/`:

```python
from aether.agents.base import BaseAgent, AgentRegistry

@AgentRegistry.register("my_agent")
class MyAgent(BaseAgent[MyInput, MyOutput]):
    agent_type = "my_agent"
    agent_name = "My Agent"
    input_schema = MyInput
    output_schema = MyOutput

    async def process(self, input_data, context):
        # Implementation
        return MyOutput(...)
```

2. Add input/output schemas in `src/aether/schemas/`
3. Add tests in `tests/unit/test_agents.py`
4. Update pipeline if needed

### Adding a New Provider

1. Create file in `src/aether/providers/`:

```python
from aether.providers.base import BaseProvider

class MyProvider(BaseProvider):
    async def initialize(self) -> bool:
        ...

    async def shutdown(self) -> None:
        ...

    async def health_check(self) -> bool:
        ...
```

2. Register in provider registry
3. Add tests in `tests/integration/test_providers.py`

### Adding a New Genre Profile

1. Create YAML in `data/genres/`:

```yaml
id: my-genre
name: My Genre
lineage:
  primary_parent: electronic
rhythm:
  tempo_range: [100, 120]
  # ... full schema
```

2. Validate against genre schema
3. Test with pipeline

---

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues before creating new ones

Thank you for contributing!
