"""Unified LLM error types.

Pipeline §13.3: provider exceptions must be mapped to stable error types
so that services can decide retry/repair/degrade without knowing which
provider was called.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-related errors."""

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMTimeoutError(LLMError):
    """The LLM call exceeded its timeout."""


class LLMRateLimitError(LLMError):
    """The provider returned a rate-limit error."""

    def __init__(
        self, message: str, *, retry_after_s: float | None = None, **kwargs: str
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_s = retry_after_s


class LLMAuthenticationError(LLMError):
    """API key or credentials are invalid/missing."""


class LLMInvalidResponseError(LLMError):
    """The response could not be parsed or did not match the expected schema."""


class LLMProviderUnavailableError(LLMError):
    """The provider is temporarily unavailable (5xx or network error)."""


__all__ = [
    "LLMAuthenticationError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMProviderUnavailableError",
    "LLMRateLimitError",
    "LLMTimeoutError",
]
