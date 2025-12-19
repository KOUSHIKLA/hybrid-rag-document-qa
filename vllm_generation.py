
"""
LLM Pipeline: High-performance async client wrapper for chat and completion APIs.

Key design goals:
- Correct async/await usage with strict awaiting semantics (no "fire-and-forget").
- Efficient batching with bounded concurrency and robust retries + exponential backoff.
- First-class prompt templating (strict placeholders validation, explicit variable bindings).
- Clean API for system/user prompt handling and custom template injection.
- Strong typing, PEP 8 compliant code style, comprehensive inline documentation.

Requirements:
- Python 3.8+
- pip install openai (>=1.0.0) which provides 'from openai import AsyncOpenAI'

Security:
- Do not hardcode real API keys. Prefer secure environment or secret management.

Note:
- Replace placeholders like API_KEY, BASE_URL, and MODEL_NAME with real values suitable for your stack.
- This wrapper sticks to the vLLM-compatible routes as exemplified by chat.completions and completions.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import string
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from openai import AsyncOpenAI

# Configure logging level for this module. Adjust as needed (DEBUG/INFO for deeper introspection).
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("llm_pipeline")


# ------------------------------
# Exceptions
# ------------------------------

class PromptTemplateError(ValueError):
    """Raised when a prompt template is invalid or cannot be rendered due to missing variables."""


class LLMClientError(RuntimeError):
    """Generic wrapper for recoverable/unrecoverable errors originating from LLM calls or client logic."""


# ------------------------------
# Configuration & Models
# ------------------------------

@dataclasses.dataclass(frozen=True)
class LLMConfig:
    """
    Immutable configuration for the LLM client. Duplicate/override with dataclasses.replace(...)
    to provide per-call overrides without mutating the client-wide baseline.
    """
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None  # None -> let server default; provide explicit value for stricter control
    request_timeout_s: float = 60.0   # overall timeout per request (transport + server) applied via asyncio.wait_for
    max_retries: int = 3              # number of attempts per request (total tries = max_retries + 1)
    backoff_initial_s: float = 0.5    # initial backoff base
    backoff_max_s: float = 8.0        # cap for exponential backoff
    jitter_s: float = 0.25            # random jitter to de-synchronize colliding clients
    max_concurrency: int = 8          # upper bound for concurrent in-flight requests (per client instance)


@dataclasses.dataclass(frozen=True)
class ChatResult:
    """
    Normalized chat result surface. Includes raw response for diagnostics.
    """
    content: str
    raw: Any  # raw OpenAI SDK response object
    # Optionally, we could add tokens/usage when the backend returns them.


@dataclasses.dataclass(frozen=True)
class CompletionResult:
    """
    Normalized completion result surface. Includes raw response for diagnostics.
    """
    text: str
    raw: Any  # raw OpenAI SDK response object


# ------------------------------
# Prompt Template
# ------------------------------

class PromptTemplate:
    """
    Strict prompt templating based on Python's format string syntax.

    - Ensures all placeholders are supplied (no silent fallbacks).
    - Prevents accidental leaking of unresolved fields in production.
    - Keeps interface minimal, explicit, and type-friendly.

    Example:
        template = PromptTemplate("Write a haiku about {topic} in the style of {style}.")
        result = template.render({"topic": "rain", "style": "Bashō"})
    """

    __slots__ = ("_template", "_fields")

    def __init__(self, template: str) -> None:
        if not isinstance(template, str) or not template:
            raise PromptTemplateError("Template must be a non-empty string.")
        self._template = template
        self._fields = self._extract_fields(template)

    @staticmethod
    def _extract_fields(template: str) -> List[str]:
        """
        Parses the format string and extracts field names like {foo}, {bar.baz}, etc.
        Only top-level names are enforced; dotted names must still be supplied as a single key.
        """
        formatter = string.Formatter()
        fields: List[str] = []
        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            if field_name is not None and field_name != "":
                fields.append(field_name)
        return fields

    def required_fields(self) -> List[str]:
        """Returns the ordered list of required field names."""
        return list(self._fields)

    def render(self, data: Dict[str, Any]) -> str:
        """
        Renders the template with provided data. Raises PromptTemplateError on missing variables.
        """
        if not isinstance(data, dict):
            raise PromptTemplateError("Template variables must be provided as a dict.")
        missing = [f for f in self._fields if f not in data]
        if missing:
            raise PromptTemplateError(
                f"Missing variables for template rendering: {', '.join(missing)}"
            )
        try:
            return self._template.format(**data)
        except Exception as ex:
            # Provide a clean error message while preserving template details.
            raise PromptTemplateError(f"Template rendering failed: {ex}") from ex

    def __repr__(self) -> str:  # helpful in logs/debugging
        return f"PromptTemplate({self._template!r})"


# ------------------------------
# LLM Client
# ------------------------------

T = TypeVar("T")

class LLMClient:
    """
    High-performance async wrapper around AsyncOpenAI with:
    - Robust retry policy and timeouts
    - Bounded concurrency for batch workloads
    - Prompt templating and standardized chat/completion methods
    - Explicit system/user prompt support and custom template injection

    Life-cycle:
    - Instantiate once per process/service or context where shared pooling is desired.
    - Methods are safe to call concurrently; concurrency is bounded internally by a semaphore.

    Note:
    - The AsyncOpenAI client is thread-compatible for asyncio contexts. Avoid cross-event-loop use.
    """

    __slots__ = ("_cfg", "_client", "_semaphore")

    def __init__(self, cfg: LLMConfig) -> None:
        if not isinstance(cfg, LLMConfig):
            raise LLMClientError("LLMClient requires a valid LLMConfig instance.")
        self._cfg = cfg
        # Instantiate a single async client. The underlying HTTP resources are reused across requests.
        self._client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        # Bound concurrency to avoid overloading the model server.
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)

    @property
    def config(self) -> LLMConfig:
        """Returns the current immutable configuration."""
        return self._cfg

    async def chat(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        # Optional template injection for system and/or user prompts.
        system_template: Optional[PromptTemplate] = None,
        user_template: Optional[PromptTemplate] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        # Optional additional messages, already precomposed in OpenAI format.
        extra_messages: Optional[List[Dict[str, str]]] = None,
        # Optional per-call overrides.
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout_s: Optional[float] = None,
        model: Optional[str] = None,
    ) -> ChatResult:
        """
        Single chat completion request.

        Parameters:
        - system_prompt: Raw system message. If system_template is provided, it will be rendered and override this.
        - user_prompt: Raw user message. If user_template is provided, it will be rendered and override this.
        - system_template/user_template: Use to inject variables via template_vars for structured prompts.
        - template_vars: Dict of variables applied to provided template(s).
        - extra_messages: Optional additional messages appended after system and user; each must be {'role': str, 'content': str}.
        - temperature, max_tokens, request_timeout_s, model: Optional per-call overrides.

        Returns:
        - ChatResult containing normalized content and raw response.
        """
        messages: List[Dict[str, str]] = []
        tv = template_vars or {}

        # Strict template rendering if templates are provided.
        if system_template is not None:
            system_prompt = system_template.render(tv)
        if user_template is not None:
            user_prompt = user_template.render(tv)

        # Compose messages in valid order.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        if extra_messages:
            # Validate minimal structure to avoid subtle role/content mistakes.
            for m in extra_messages:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise LLMClientError("Each extra message must be a dict with 'role' and 'content' keys.")
            messages.extend(extra_messages)

        if not messages:
            raise LLMClientError("At least one message is required to perform a chat completion.")

        # Effective parameters apply overrides when provided; otherwise fallback to client defaults.
        eff_temperature = self._cfg.temperature if temperature is None else float(temperature)
        eff_max_tokens = self._cfg.max_tokens if max_tokens is None else int(max_tokens)
        eff_timeout_s = self._cfg.request_timeout_s if request_timeout_s is None else float(request_timeout_s)
        eff_model = self._cfg.model if model is None else str(model)

        async def _invoke() -> Any:
            return await self._client.chat.completions.create(
                model=eff_model,
                messages=messages,
                temperature=eff_temperature,
                **({"max_tokens": eff_max_tokens} if eff_max_tokens is not None else {}),
            )

        raw = await self._bounded_retry_wait(_invoke, timeout_s=eff_timeout_s)
        # Normalize result shape. Defensive checks ensure graceful failure if the backend schema changes.
        try:
            content = raw.choices[0].message["content"] if isinstance(raw.choices[0].message, dict) else raw.choices[0].message.content
        except Exception as ex:
            raise LLMClientError(f"Unexpected response schema for chat completion: {ex}") from ex

        return ChatResult(content=content or "", raw=raw)

    async def complete(
        self,
        *,
        prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout_s: Optional[float] = None,
        model: Optional[str] = None,
    ) -> CompletionResult:
        """
        Single text completion request (non-chat).

        Parameters:
        - prompt: Raw text prompt. If prompt_template is provided, it will be rendered and override this.
        - prompt_template: Use to inject variables via template_vars for structured prompts.
        - template_vars: Dict of variables applied to provided prompt_template.
        - temperature, max_tokens, request_timeout_s, model: Optional per-call overrides.

        Returns:
        - CompletionResult containing normalized text and raw response.
        """
        tv = template_vars or {}
        if prompt_template is not None:
            prompt = prompt_template.render(tv)
        if not prompt:
            raise LLMClientError("A prompt string is required to perform a text completion.")

        eff_temperature = self._cfg.temperature if temperature is None else float(temperature)
        eff_max_tokens = self._cfg.max_tokens if max_tokens is None else int(max_tokens)
        eff_timeout_s = self._cfg.request_timeout_s if request_timeout_s is None else float(request_timeout_s)
        eff_model = self._cfg.model if model is None else str(model)

        async def _invoke() -> Any:
            return await self._client.completions.create(
                model=eff_model,
                prompt=prompt,
                temperature=eff_temperature,
                **({"max_tokens": eff_max_tokens} if eff_max_tokens is not None else {}),
            )

        raw = await self._bounded_retry_wait(_invoke, timeout_s=eff_timeout_s)
        try:
            text = raw.choices[0].text
        except Exception as ex:
            raise LLMClientError(f"Unexpected response schema for completion: {ex}") from ex

        return CompletionResult(text=text or "", raw=raw)

    async def run_batch_complete(
        self,
        prompts: Sequence[Union[str, Tuple[PromptTemplate, Dict[str, Any]]]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout_s: Optional[float] = None,
        model: Optional[str] = None,
        return_exceptions: bool = False,
    ) -> List[Union[CompletionResult, BaseException]]:
        """
        Batch execution for non-chat completions.

        Parameters:
        - prompts: Sequence of either raw strings or (PromptTemplate, variables) tuples.
        - temperature, max_tokens, request_timeout_s, model: Optional per-call overrides applied to each item.
        - return_exceptions: If True, failed tasks return the exception object instead of raising at the end.

        Returns:
        - List of CompletionResult in the same order as input. If return_exceptions=True, exceptions are included.
        """
        async def _task(item: Union[str, Tuple[PromptTemplate, Dict[str, Any]]]) -> CompletionResult:
            if isinstance(item, tuple):
                tmpl, vars_ = item
                return await self.complete(
                    prompt_template=tmpl,
                    template_vars=vars_,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    model=model,
                )
            else:
                return await self.complete(
                    prompt=item,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    model=model,
                )

        tasks = [asyncio.create_task(self._with_semaphore(lambda it=item: _task(it))) for item in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        if not return_exceptions:
            # If any exception occurred, gather(..., return_exceptions=False) would raise.
            # Thus, at this point, results is List[CompletionResult].
            return results  # type: ignore[return-value]
        return results

    async def run_batch_chat(
        self,
        user_prompts: Sequence[Union[str, Tuple[PromptTemplate, Dict[str, Any]]]],
        *,
        system_prompt: Optional[str] = None,
        system_template: Optional[PromptTemplate] = None,
        shared_template_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout_s: Optional[float] = None,
        model: Optional[str] = None,
        return_exceptions: bool = False,
    ) -> List[Union[ChatResult, BaseException]]:
        """
        Batch execution for chat completions.

        Parameters:
        - user_prompts: Sequence of either raw user strings or (PromptTemplate, variables) tuples for user messages.
        - system_prompt: Raw system prompt applied to every item unless system_template is provided.
        - system_template: If provided, this overrides system_prompt and is rendered with shared_template_vars.
        - shared_template_vars: Variables applied to system_template. Ignored if system_template is None.
        - temperature, max_tokens, request_timeout_s, model: Optional per-call overrides applied to each item.
        - return_exceptions: If True, failed tasks return the exception object instead of raising at the end.

        Returns:
        - List of ChatResult in the same order as input. If return_exceptions=True, exceptions are included.
        """
        effective_system_prompt = None
        if system_template is not None:
            effective_system_prompt = system_template.render(shared_template_vars or {})
        else:
            effective_system_prompt = system_prompt

        async def _task(item: Union[str, Tuple[PromptTemplate, Dict[str, Any]]]) -> ChatResult:
            if isinstance(item, tuple):
                user_tmpl, vars_ = item
                return await self.chat(
                    system_prompt=effective_system_prompt,
                    user_template=user_tmpl,
                    template_vars=vars_,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    model=model,
                )
            else:
                return await self.chat(
                    system_prompt=effective_system_prompt,
                    user_prompt=item,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout_s=request_timeout_s,
                    model=model,
                )

        tasks = [asyncio.create_task(self._with_semaphore(lambda it=item: _task(it))) for item in user_prompts]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        if not return_exceptions:
            return results  # type: ignore[return-value]
        return results

    # --------------------------
    # Internal helpers
    # --------------------------

    async def _with_semaphore(self, fn: Callable[[], Awaitable[T]]) -> T:
        """
        Enforces the concurrency limit via a semaphore around a single coroutine-producing call.
        """
        async with self._semaphore:
            return await fn()

    async def _bounded_retry_wait(self, fn: Callable[[], Awaitable[T]], *, timeout_s: float) -> T:
        """
        Applies:
        - asyncio.wait_for for total call timeout.
        - Bounded retry policy with exponential backoff + jitter on transient errors.

        Retries:
        - On generic Exceptions except PromptTemplateError and ValueError (likely programmer errors).
        - Backoff increases exponentially and is capped at config.backoff_max_s.

        Returns:
        - The awaited result if successful within retries.

        Raises:
        - The last encountered exception after exhausting retries.
        """
        attempts = self._cfg.max_retries + 1  # total tries
        last_ex: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(fn(), timeout=timeout_s)
            except (PromptTemplateError, ValueError) as ex:
                # Non-retryable by design (bad inputs/templates).
                logger.debug("Non-retryable error (template/validation): %s", ex)
                raise
            except asyncio.TimeoutError as ex:
                last_ex = ex
                logger.warning("Request timed out at attempt %d/%d (timeout=%.2fs)", attempt, attempts, timeout_s)
            except Exception as ex:
                last_ex = ex
                logger.warning("Request failed at attempt %d/%d: %s", attempt, attempts, ex)

            if attempt < attempts:
                # Compute exponential backoff with jitter.
                sleep_s = min(
                    self._cfg.backoff_initial_s * (2 ** (attempt - 1)),
                    self._cfg.backoff_max_s,
                ) + random.uniform(0.0, self._cfg.jitter_s)
                await asyncio.sleep(sleep_s)

        # Exhausted retries; surface the most recent error.
        assert last_ex is not None  # last_ex must be set if we reached here
        raise LLMClientError(f"Operation failed after {attempts} attempts") from last_ex

