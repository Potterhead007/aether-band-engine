"""
AETHER LLM Providers

Production-grade LLM integrations for creative text generation.

Supported Providers:
- Anthropic Claude (claude-3-opus, claude-3-sonnet, claude-3-haiku)
- OpenAI GPT (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
- Mock (for testing)

Features:
- Structured output generation (JSON mode)
- Retry with exponential backoff
- Token usage tracking
- Rate limiting
- Streaming support
- Prompt templating

Example:
    provider = ClaudeLLMProvider(api_key="...")
    await provider.initialize()

    response = await provider.complete([
        LLMMessage(role="user", content="Write a verse about rain")
    ])
    print(response.content)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from aether.core.exceptions import MissingConfigError
from aether.core.resilience import BackoffStrategy, RetryPolicy
from aether.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderInfo,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Centralized Retry Helper (uses core resilience patterns)
# ============================================================================


async def _execute_with_retry(
    operation: Callable[[], Any],
    policy: RetryPolicy,
    operation_name: str = "operation",
) -> Any:
    """
    Execute an async operation with retry logic using centralized policy.

    Uses the RetryPolicy from core.resilience for consistent backoff
    calculation across all providers.

    Args:
        operation: Async callable to execute
        policy: RetryPolicy for backoff configuration
        operation_name: Name for logging

    Returns:
        Result from operation

    Raises:
        RuntimeError: If all retry attempts exhausted
    """
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except Exception as e:
            last_error = e

            if attempt >= policy.max_attempts:
                break

            if not policy.should_retry(e):
                raise

            delay = policy.calculate_delay(attempt)
            logger.warning(
                f"{operation_name} attempt {attempt}/{policy.max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"All {policy.max_attempts} retry attempts for {operation_name} failed: {last_error}"
    )


# ============================================================================
# Rate Limiter
# ============================================================================


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0

    def to_retry_policy(self) -> RetryPolicy:
        """Convert to core RetryPolicy for consistent retry behavior."""
        return RetryPolicy(
            max_attempts=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            backoff=BackoffStrategy.EXPONENTIAL,
            jitter=True,
        )


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._request_times: list[float] = []
        self._token_usage: list[tuple] = []  # (time, tokens)
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Wait until rate limit allows request."""
        async with self._lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old entries
            self._request_times = [t for t in self._request_times if t > minute_ago]
            self._token_usage = [(t, n) for t, n in self._token_usage if t > minute_ago]

            # Check request limit
            if len(self._request_times) >= self.config.requests_per_minute:
                wait_time = self._request_times[0] - minute_ago
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s for request slot")
                await asyncio.sleep(wait_time)

            # Check token limit
            current_tokens = sum(n for _, n in self._token_usage)
            if current_tokens + estimated_tokens > self.config.tokens_per_minute:
                wait_time = self._token_usage[0][0] - minute_ago
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s for token budget")
                await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    def record_usage(self, tokens: int) -> None:
        """Record token usage."""
        self._token_usage.append((time.time(), tokens))


# ============================================================================
# Claude Provider
# ============================================================================


class ClaudeLLMProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.

    Models:
    - claude-3-opus-20240229 (most capable)
    - claude-3-sonnet-20240229 (balanced)
    - claude-3-haiku-20240307 (fastest)
    - claude-3-5-sonnet-20241022 (latest)

    Example:
        provider = ClaudeLLMProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-20241022",
        )
        await provider.initialize()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        rate_limit: RateLimitConfig | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingConfigError(
                "ANTHROPIC_API_KEY",
                recovery_hints=[
                    "Set ANTHROPIC_API_KEY environment variable",
                    "Pass api_key parameter to ClaudeLLMProvider()",
                    "Use MockLLMProvider for testing without API key",
                ],
            )
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit or RateLimitConfig())
        self._client = None
        self._total_tokens = 0

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Claude LLM Provider",
            version="1.0.0",
            provider_type="llm",
            status=self._status,
            capabilities=["completion", "structured_output", "streaming"],
            config={"model": self.model},
        )

    async def initialize(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"Claude provider initialized with model: {self.model}")
            return True
        except ImportError:
            logger.error(
                "anthropic package not installed. "
                "Run: pip install 'aether-band-engine[llm]' or pip install anthropic"
            )
            self._status = ProviderStatus.UNAVAILABLE
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Claude provider: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._client = None
        self._status = ProviderStatus.UNAVAILABLE
        logger.info("Claude provider shut down")

    async def health_check(self) -> bool:
        """Check if the API is accessible."""
        if not self._client:
            return False
        try:
            # Simple test completion
            await self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion from messages using centralized retry logic."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        await self.rate_limiter.acquire(max_tokens)

        # Convert messages to Anthropic format
        system_message = None
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        # Build API kwargs
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        # Define the API call operation
        async def _api_call() -> LLMResponse:
            response = await self._client.messages.create(**kwargs)

            # Track usage
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            self._total_tokens += usage["input_tokens"] + usage["output_tokens"]
            self.rate_limiter.record_usage(usage["input_tokens"] + usage["output_tokens"])

            content = response.content[0].text if response.content else ""

            # If JSON mode requested, try to extract JSON
            if json_mode:
                content = self._extract_json(content)

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                finish_reason=response.stop_reason or "stop",
            )

        # Execute with centralized retry logic
        return await _execute_with_retry(
            operation=_api_call,
            policy=self.rate_limiter.config.to_retry_policy(),
            operation_name="Claude API call",
        )

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate structured output matching schema."""
        # Add schema instruction to system message
        schema_instruction = f"""
You must respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text.
"""
        enhanced_messages = [LLMMessage(role="system", content=schema_instruction)]
        enhanced_messages.extend(messages)

        response = await self.complete(
            messages=enhanced_messages,
            temperature=temperature,
            json_mode=True,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            extracted = self._extract_json(response.content)
            return json.loads(extracted)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may have additional content."""
        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used across all requests."""
        return self._total_tokens


# ============================================================================
# OpenAI Provider
# ============================================================================


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI GPT LLM provider.

    Models:
    - gpt-4-turbo-preview (most capable)
    - gpt-4 (strong reasoning)
    - gpt-3.5-turbo (fast, economical)

    Example:
        provider = OpenAILLMProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4-turbo-preview",
        )
        await provider.initialize()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        rate_limit: RateLimitConfig | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingConfigError(
                "OPENAI_API_KEY",
                recovery_hints=[
                    "Set OPENAI_API_KEY environment variable",
                    "Pass api_key parameter to OpenAILLMProvider()",
                    "Use MockLLMProvider for testing without API key",
                ],
            )
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit or RateLimitConfig())
        self._client = None
        self._total_tokens = 0

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="OpenAI LLM Provider",
            version="1.0.0",
            provider_type="llm",
            status=self._status,
            capabilities=["completion", "structured_output", "json_mode"],
            config={"model": self.model},
        )

    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            import openai

            self._client = openai.AsyncOpenAI(api_key=self.api_key)
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"OpenAI provider initialized with model: {self.model}")
            return True
        except ImportError:
            logger.error(
                "openai package not installed. "
                "Run: pip install 'aether-band-engine[llm]' or pip install openai"
            )
            self._status = ProviderStatus.UNAVAILABLE
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._client = None
        self._status = ProviderStatus.UNAVAILABLE
        logger.info("OpenAI provider shut down")

    async def health_check(self) -> bool:
        """Check if the API is accessible."""
        if not self._client:
            return False
        try:
            await self._client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion from messages."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        await self.rate_limiter.acquire(max_tokens)

        # Convert messages to OpenAI format
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.rate_limiter.config.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": api_messages,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = await self._client.chat.completions.create(**kwargs)

                # Track usage
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
                self._total_tokens += usage["input_tokens"] + usage["output_tokens"]
                self.rate_limiter.record_usage(usage["input_tokens"] + usage["output_tokens"])

                content = response.choices[0].message.content or ""

                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    finish_reason=response.choices[0].finish_reason,
                )

            except Exception as e:
                last_error = e
                delay = min(
                    self.rate_limiter.config.base_delay * (2**attempt),
                    self.rate_limiter.config.max_delay,
                )
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)

        raise RuntimeError(f"All retry attempts failed: {last_error}")

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate structured output matching schema."""
        schema_instruction = f"""
Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""
        enhanced_messages = [LLMMessage(role="system", content=schema_instruction)]
        enhanced_messages.extend(messages)

        response = await self.complete(
            messages=enhanced_messages,
            temperature=temperature,
            json_mode=True,
        )

        return json.loads(response.content)

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used across all requests."""
        return self._total_tokens


# ============================================================================
# Mock Provider (for testing)
# ============================================================================


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses or generates simple placeholder content.

    Example:
        provider = MockLLMProvider(responses={
            "write lyrics": "Here are some lyrics...",
        })
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.responses = responses or {}
        self._call_count = 0

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Mock LLM Provider",
            version="1.0.0",
            provider_type="llm",
            status=self._status,
            capabilities=["completion", "structured_output"],
            config={"mock": True},
        )

    async def initialize(self) -> bool:
        self._status = ProviderStatus.AVAILABLE
        return True

    async def shutdown(self) -> None:
        self._status = ProviderStatus.UNAVAILABLE

    async def health_check(self) -> bool:
        return self._status == ProviderStatus.AVAILABLE

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return mock response."""
        self._call_count += 1

        # Check for matching response
        last_message = messages[-1].content.lower() if messages else ""
        for key, value in self.responses.items():
            if key.lower() in last_message:
                return LLMResponse(
                    content=value,
                    model="mock",
                    usage={"input_tokens": 100, "output_tokens": 200},
                    finish_reason="stop",
                )

        # Default response
        if json_mode:
            content = json.dumps({"result": "mock_response", "success": True})
        else:
            content = f"Mock response #{self._call_count} to: {last_message[:50]}..."

        return LLMResponse(
            content=content,
            model="mock",
            usage={"input_tokens": 50, "output_tokens": 100},
            finish_reason="stop",
        )

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Return mock structured output."""
        # Generate default values based on schema
        result = {}
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                result[key] = f"mock_{key}"
            elif prop_type == "number" or prop_type == "integer":
                result[key] = 42
            elif prop_type == "boolean":
                result[key] = True
            elif prop_type == "array":
                result[key] = ["item1", "item2"]
            elif prop_type == "object":
                result[key] = {}
        return result

    @property
    def call_count(self) -> int:
        return self._call_count


# ============================================================================
# Creative Generation Utilities
# ============================================================================


class CreativePrompts:
    """Pre-built prompts for music generation tasks."""

    @staticmethod
    def lyrics_generation(
        genre: str,
        mood: str,
        theme: str,
        structure: list[str],
    ) -> list[LLMMessage]:
        """Generate prompt for lyrics writing."""
        return [
            LLMMessage(
                role="system",
                content=f"""You are a professional songwriter specializing in {genre} music.
Your lyrics are original, emotionally resonant, and authentic to the genre.
Never copy existing lyrics. Create entirely new content.
Use vivid imagery, metaphors, and genre-appropriate vocabulary.""",
            ),
            LLMMessage(
                role="user",
                content=f"""Write original lyrics for a {genre} song.

Mood: {mood}
Theme: {theme}
Structure: {', '.join(structure)}

Requirements:
- Each section should be clearly labeled
- Verses should tell a story or develop the theme
- Chorus should be memorable and hook-oriented
- Use rhyme schemes appropriate for {genre}
- Include vivid imagery and emotional depth

Write the complete lyrics:""",
            ),
        ]

    @staticmethod
    def creative_brief(
        genre: str,
        mood: str,
        influences: list[str],
    ) -> list[LLMMessage]:
        """Generate prompt for creative brief generation."""
        return [
            LLMMessage(
                role="system",
                content="""You are a music producer and creative director.
Generate detailed creative briefs for music production.
Be specific about sonic elements, production techniques, and artistic direction.""",
            ),
            LLMMessage(
                role="user",
                content=f"""Create a detailed creative brief for a {genre} track.

Target Mood: {mood}
Style Influences: {', '.join(influences)}

Include:
1. Overall vision and concept
2. Sonic palette (instruments, sounds, textures)
3. Production approach (mixing style, effects)
4. Arrangement suggestions (dynamics, structure)
5. Emotional journey (how the track should feel)

Generate the creative brief:""",
            ),
        ]

    @staticmethod
    def concept_album(
        genre: str,
        concept: str,
        track_count: int,
    ) -> list[LLMMessage]:
        """Generate prompt for concept album planning."""
        return [
            LLMMessage(
                role="system",
                content="""You are a visionary music artist planning a concept album.
Create cohesive album concepts with interconnected tracks.
Each track should contribute to the overall narrative.""",
            ),
            LLMMessage(
                role="user",
                content=f"""Plan a concept album in the {genre} genre.

Central Concept: {concept}
Number of Tracks: {track_count}

For each track, provide:
1. Track title
2. Theme/subject
3. Mood/energy level
4. Key sonic elements
5. Role in the album narrative

Create the album concept:""",
            ),
        ]


# ============================================================================
# Provider Factory
# ============================================================================


def create_llm_provider(
    provider_type: str = "claude",
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: "claude", "openai", or "mock"
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLM provider instance
    """
    providers = {
        "claude": ClaudeLLMProvider,
        "anthropic": ClaudeLLMProvider,
        "openai": OpenAILLMProvider,
        "gpt": OpenAILLMProvider,
        "mock": MockLLMProvider,
    }

    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider type: {provider_type}. " f"Available: {list(providers.keys())}"
        )

    return provider_class(**kwargs)


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Providers
    "ClaudeLLMProvider",
    "OpenAILLMProvider",
    "MockLLMProvider",
    # Utilities
    "RateLimitConfig",
    "RateLimiter",
    "CreativePrompts",
    # Factory
    "create_llm_provider",
]
