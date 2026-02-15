"""
GCAW Lock Engine — Grammar-Constrained Extraction Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Takes raw image bytes from the Scout layer and forces them into
strict Pydantic schemas using Groq (Llama 3.2 Vision) + ``instructor``.

Design Decisions
~~~~~~~~~~~~~~~~

1. **Generic Extraction**
   The ``extract()`` method accepts ``Type[T]`` where ``T`` is any
   ``BaseModel`` subclass.  This makes the Lock entirely schema-
   agnostic — extract cement prices today, turbine specs tomorrow,
   zero code changes required.

2. **Grammar Lock via instructor**
   ``instructor.from_groq`` with ``Mode.JSON`` constrains the LLM's
   output to valid JSON matching the Pydantic schema.  Combined with
   Pydantic V2 validation on the returned object, we get a double-lock:

     LLM JSON output → JSON decode → Pydantic validation

   If the JSON is structurally valid but semantically wrong (e.g. a
   string where a float is expected), instructor re-prompts the model
   with the validation error — up to ``max_retries`` times internally.

3. **Two-Layer Retry Strategy**
   - **Layer 1 — instructor** (validation retries): If the LLM
     returns JSON that fails Pydantic validation, instructor
     automatically re-prompts with the error message.
   - **Layer 2 — tenacity** (API retries): If the Groq API
     itself fails (rate limits, transient 500s, network errors),
     tenacity retries with exponential backoff + jitter.

   These layers are orthogonal: tenacity wraps the entire API call
   (including instructor's internal retries), so a single tenacity
   attempt may contain multiple instructor validation rounds.

4. **Image Handling (Groq Vision)**
   Raw PNG bytes from the Scout are base64-encoded and sent as a
   ``data:image/png;base64,...`` URL in the OpenAI-compatible
   messages format that Groq's vision models expect.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import instructor
from groq import Groq
from loguru import logger
from pydantic import BaseModel
from src.config import settings
from src.domain.events import LockResult
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ────────────────────────────────────────────────────────────────────
# Type Variable
# ────────────────────────────────────────────────────────────────────

T = TypeVar("T", bound=BaseModel)

# ────────────────────────────────────────────────────────────────────
# System Prompt
# ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = (
    "You are a precise data extraction engine. "
    "Extract data from this image strictly according to the "
    "provided JSON schema. "
    "If a value is missing or ambiguous, output null. "
    "Do not hallucinate. "
    "NO PREAMBLE. JSON ONLY."
)

# ────────────────────────────────────────────────────────────────────
# Retry Callback (loguru-compatible)
# ────────────────────────────────────────────────────────────────────


def _log_retry(retry_state: RetryCallState) -> None:
    """Emit a loguru warning before each tenacity sleep.

    ``tenacity.before_sleep_log`` expects a stdlib logger, which
    doesn't play nicely with loguru.  This thin callback bridges
    the gap so all GCAW logs go through a single sink.
    """
    exc = (
        retry_state.outcome.exception()
        if retry_state.outcome
        else None
    )
    logger.warning(
        "Lock retry {}/{} — sleeping {:.1f}s (error: {}).",
        retry_state.attempt_number,
        3,  # mirrors stop_after_attempt(3)
        retry_state.idle_for if hasattr(retry_state, "idle_for") else 0,
        exc,
    )


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LockConfig:
    """Immutable configuration for the GrammarLock.

    Sensible defaults target Groq's Llama 4 Scout with
    deterministic output (``temperature=0``).
    """

    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.0
    max_output_tokens: int = 8_192
    max_api_retries: int = 3
    max_validation_retries: int = 2


# ────────────────────────────────────────────────────────────────────
# Domain Exception
# ────────────────────────────────────────────────────────────────────


class LockError(Exception):
    """Raised for any unrecoverable failure in the Lock layer.

    Wraps underlying Groq / instructor / validation errors with a
    domain-meaningful message so callers never need to catch infra
    exceptions directly.
    """


# ────────────────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────────────────


class GrammarLock:
    """Grammar-constrained extraction engine.

    Given raw image bytes (typically a Scout screenshot), forces the
    LLM's output into a caller-supplied Pydantic schema via the
    ``instructor`` library and Groq (Llama 3.2 Vision).

    Lifecycle::

        lock = GrammarLock(api_key="YOUR_KEY")
        result = lock.extract(
            image_data=screenshot_bytes,
            schema=IndustrialItem,
            context_text="Cement price table from Q3 2024",
        )
        print(result.items)  # list[ItemRow]

    The engine is **stateless** between ``extract()`` calls — no
    session, no conversation history.  Each call is an independent
    single-shot extraction.
    """

    # ── Construction ────────────────────────────────────────────────

    def __init__(
        self,
        api_key: str,
        *,
        config: LockConfig | None = None,
    ) -> None:
        self._config: LockConfig = config or LockConfig()

        # ── Initialise Groq client ──────────────────────────────────
        self._groq = Groq(api_key=api_key)

        # ── Patch with instructor ───────────────────────────────────
        #    JSON mode constrains output to valid JSON matching the
        #    response_model's schema.
        self._client = instructor.from_groq(
            client=self._groq,
            mode=instructor.Mode.JSON,
        )

        logger.info(
            "GrammarLock initialised (provider=Groq, model={}, "
            "temp={}, max_tokens={}).",
            self._config.model_name,
            self._config.temperature,
            self._config.max_output_tokens,
        )

    # ── Public API ──────────────────────────────────────────────────

    def extract(
        self,
        image_data: bytes,
        schema: Type[T],
        context_text: str = "",
    ) -> LockResult:
        """Extract structured data from an image into a LockResult envelope.

        Parameters
        ----------
        image_data : bytes
            Raw screenshot PNG bytes.
        schema : Type[T]
            Target Pydantic V2 model class.
        context_text : str, optional
            Free-text hint for the LLM.

        Returns
        -------
        LockResult
            Typed envelope with ``data`` (validated ``T``), ``tokens_used``,
            ``validation_retries``, ``model_name``, ``latency_ms``.

        Raises
        ------
        LockError
            If extraction fails after all retry attempts (both
            instructor validation retries and tenacity API retries).
        """
        logger.info("── Lock: extract(schema={}) ──", schema.__name__)
        import time
        t_start: float = time.perf_counter()

        try:
            data: T = self._extract_with_retry(
                image_data, schema, context_text
            )
            latency_ms = (time.perf_counter() - t_start) * 1000

            # Attempt to extract token usage from instructor's raw response
            tokens_used: int = 0
            try:
                raw = getattr(data, "_raw_response", None)
                if raw and hasattr(raw, "usage"):
                    tokens_used = getattr(raw.usage, "total_tokens", 0)
            except Exception:
                pass

            return LockResult(
                data=data,
                tokens_used=tokens_used,
                validation_retries=self._config.max_validation_retries,
                model_name=self._config.model_name,
                latency_ms=round(latency_ms, 1),
            )
        except LockError:
            raise
        except Exception as exc:
            logger.exception("Lock extraction failed after retries.")
            raise LockError(
                f"Failed to extract {schema.__name__} from image: {exc}"
            ) from exc

    # ── Retry-Wrapped Extraction ────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _extract_with_retry(
        self,
        image_data: bytes,
        schema: Type[T],
        context_text: str,
    ) -> T:
        """Inner extraction loop with tenacity retry decoration.

        Each attempt:
          1. Encodes raw bytes → base64 data URL.
          2. Builds an OpenAI-compatible multimodal message.
          3. Calls Groq via the instructor-patched client.
          4. Returns a validated ``schema`` instance.

        instructor may internally re-prompt up to
        ``max_validation_retries`` times if the JSON is structurally
        valid but fails Pydantic validation.  tenacity handles the
        outer API-level failures (rate limits, 500s, network).
        """
        # ── Image → base64 data URL ─────────────────────────────────
        b64: str = base64.b64encode(image_data).decode("ascii")
        data_url: str = f"data:image/png;base64,{b64}"
        logger.debug(
            "Image encoded: {} bytes → {} base64 chars.",
            len(image_data),
            len(b64),
        )

        # ── Build prompt ────────────────────────────────────────────
        prompt: str = _SYSTEM_PROMPT
        if context_text:
            prompt += f"\n\nAdditional context: {context_text}"

        # ── Call Groq via instructor ────────────────────────────────
        logger.debug(
            "Sending to Groq {} (temp={}).",
            self._config.model_name,
            self._config.temperature,
        )

        result: T = self._client.chat.completions.create(
            model=self._config.model_name,
            response_model=schema,
            max_retries=self._config.max_validation_retries,
            max_tokens=self._config.max_output_tokens,
            temperature=self._config.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                },
            ],
        )

        # ── Log success ─────────────────────────────────────────────
        dump: dict[str, Any] = result.model_dump()
        populated = sum(1 for v in dump.values() if v is not None)
        total = len(schema.model_fields)
        logger.success(
            "Extracted {} — {}/{} top-level fields populated.",
            schema.__name__,
            populated,
            total,
        )
        return result
