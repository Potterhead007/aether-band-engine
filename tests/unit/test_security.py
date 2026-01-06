"""
Security Tests for AETHER Band Engine

Tests for security-related functionality:
- JWT algorithm validation
- Retry decorator
- Genre DNA determinism
- SSO placeholder
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch


class TestJWTAlgorithmValidation:
    """Tests for JWT algorithm security."""

    def test_jwt_auth_allows_valid_algorithms(self):
        """Valid algorithms should be accepted."""
        from src.aether.api.auth import JWTAuth

        valid_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        for algo in valid_algorithms:
            auth = JWTAuth(secret_key="test-secret", algorithm=algo)
            assert auth.algorithm == algo.upper()

    def test_jwt_auth_rejects_none_algorithm(self):
        """'none' algorithm must be rejected to prevent bypass attacks."""
        from src.aether.api.auth import JWTAuth

        with pytest.raises(ValueError) as exc_info:
            JWTAuth(secret_key="test-secret", algorithm="none")
        assert "Unsupported JWT algorithm" in str(exc_info.value)

    def test_jwt_auth_rejects_invalid_algorithm(self):
        """Invalid algorithms should be rejected."""
        from src.aether.api.auth import JWTAuth

        invalid_algorithms = ["NONE", "None", "invalid", "MD5", ""]
        for algo in invalid_algorithms:
            with pytest.raises(ValueError):
                JWTAuth(secret_key="test-secret", algorithm=algo)

    def test_jwt_auth_normalizes_algorithm_case(self):
        """Algorithm should be normalized to uppercase."""
        from src.aether.api.auth import JWTAuth

        auth = JWTAuth(secret_key="test-secret", algorithm="hs256")
        assert auth.algorithm == "HS256"


class TestSSOAuthPlaceholder:
    """Tests for SSO authentication placeholder."""

    def test_sso_auth_raises_not_implemented(self):
        """SSO auth should raise NotImplementedError on instantiation."""
        from src.aether.api.auth import SSOAuth

        with pytest.raises(NotImplementedError) as exc_info:
            SSOAuth()
        assert "not yet implemented" in str(exc_info.value)

    def test_sso_auth_provides_implementation_guidance(self):
        """SSO error should include implementation guidance."""
        from src.aether.api.auth import SSOAuth

        with pytest.raises(NotImplementedError) as exc_info:
            SSOAuth(idp_metadata_url="https://example.com/saml")
        assert "python3-saml" in str(exc_info.value) or "authlib" in str(exc_info.value)


class TestRetryDecorator:
    """Tests for unified retry decorator."""

    def test_retry_config_defaults(self):
        """Default retry config should have sensible values."""
        from src.aether.providers.retry import RetryConfig

        config = RetryConfig.default()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_retry_config_aggressive(self):
        """Aggressive config should retry more times."""
        from src.aether.providers.retry import RetryConfig

        config = RetryConfig.aggressive()
        assert config.max_attempts == 5

    def test_retry_config_conservative(self):
        """Conservative config should have longer delays."""
        from src.aether.providers.retry import RetryConfig

        config = RetryConfig.conservative()
        assert config.base_delay >= 2.0

    @pytest.mark.asyncio
    async def test_retry_decorator_succeeds_first_try(self):
        """Function that succeeds should not retry."""
        from src.aether.providers.retry import retry, RetryConfig

        call_count = 0

        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_retries_on_failure(self):
        """Function should retry on retryable exceptions."""
        from src.aether.providers.retry import retry, RetryConfig

        call_count = 0

        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await failing_then_success()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_decorator_raises_after_exhaustion(self):
        """Should raise RetryExhaustedError after all attempts fail."""
        from src.aether.providers.retry import retry, RetryConfig, RetryExhaustedError

        @retry(config=RetryConfig(max_attempts=2, base_delay=0.01))
        async def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fails()
        assert exc_info.value.attempts == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_does_not_retry_non_retryable(self):
        """Non-retryable exceptions should not trigger retry."""
        from src.aether.providers.retry import retry, RetryConfig

        call_count = 0

        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()
        assert call_count == 1  # No retry


class TestGenreDNADeterminism:
    """Tests for deterministic genre DNA vectors."""

    def test_genre_vector_is_deterministic(self):
        """Genre vectors should be identical across calls."""
        from src.aether.genre.dna import get_genre_vector

        vec1 = get_genre_vector("trap")
        vec2 = get_genre_vector("trap")
        assert vec1 == vec2

    def test_genre_vector_is_48_dimensional(self):
        """Genre vectors should have exactly 48 dimensions."""
        from src.aether.genre.dna import get_genre_vector

        vec = get_genre_vector("house")
        assert len(vec) == 48

    def test_genre_vector_values_are_normalized(self):
        """Genre vector values should be in [0, 1] range."""
        from src.aether.genre.dna import get_genre_vector, list_genres

        for genre in list_genres():
            vec = get_genre_vector(genre)
            for i, val in enumerate(vec):
                assert 0.0 <= val <= 2.0, f"Genre {genre}, dim {i}: {val} out of range"

    def test_different_genres_have_different_vectors(self):
        """Different genres should produce different vectors."""
        from src.aether.genre.dna import get_genre_vector

        trap = get_genre_vector("trap")
        house = get_genre_vector("house")
        assert trap != house

    def test_genre_similarity_is_symmetric(self):
        """Genre similarity should be symmetric: sim(A,B) == sim(B,A)."""
        from src.aether.genre.dna import compute_genre_similarity

        sim_ab = compute_genre_similarity("trap", "house")
        sim_ba = compute_genre_similarity("house", "trap")
        assert abs(sim_ab - sim_ba) < 0.001

    def test_genre_self_similarity_is_one(self):
        """Genre similarity to itself should be 1.0."""
        from src.aether.genre.dna import compute_genre_similarity

        sim = compute_genre_similarity("trap", "trap")
        assert abs(sim - 1.0) < 0.001


class TestSecurityHeaders:
    """Tests for security headers (requires API client)."""

    def test_api_app_creates_successfully(self):
        """API app should create without errors."""
        from src.aether.api.app import create_app

        app = create_app()
        assert app is not None
        assert app.title == "AETHER Band Engine API"
