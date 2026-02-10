"""
GCAW Orchestrator — Pipeline Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wires **Scout → Lock → Memory** into a single extraction pipeline
with an integrated self-healing correction loop.

Design Decisions
~~~~~~~~~~~~~~~~

1. **Composition — not orchestration-by-inheritance**
   The Pipeline holds references to Scout-config, Lock, and Memory.
   Each component is independently testable; the Pipeline just
   sequences them and threads data through.

2. **Async Scout, sync Lock & Memory**
   Scout requires ``async`` (Playwright).  Lock (Gemini API via
   ``instructor``) and Memory (ChromaDB) are synchronous.  The
   Pipeline's public ``run()`` is ``async``; a ``run_sync()``
   convenience wrapper calls ``asyncio.run()`` for sync contexts
   such as Streamlit.

3. **Self-Healing Step (Stage 3)**
   After extraction, the Pipeline walks every string field in the
   returned Pydantic model — including nested ``BaseModel`` lists —
   and queries Memory for a stored correction.  This is recursive:
   ``IndustrialItem.items[*].name`` is checked just like
   ``IndustrialItem.source_title``.

4. **Domain exceptions only**
   Infrastructure exceptions (Playwright timeouts, Gemini 500s,
   ChromaDB I/O errors) are caught and re-raised as
   ``PipelineError`` so callers never import infra packages.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from src.domain.events import (
    CorrectionDetail,
    FlightRecord,
    LockResult,
    MemoryResult,
    ScoutResult,
)
from src.lock.engine import GrammarLock, LockError
from src.memory.store import MemoryStore, MemoryStoreError
from src.scout.engine import AsyncScout, ScoutConfig, ScoutError

# ────────────────────────────────────────────────────────────────────
# Type Variable
# ────────────────────────────────────────────────────────────────────

T = TypeVar("T", bound=BaseModel)

# ────────────────────────────────────────────────────────────────────
# Domain Exception
# ────────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Raised for any unrecoverable failure in the Pipeline.

    Wraps Scout / Lock / Memory domain errors into a single
    top-level exception that the UI layer can catch cleanly.
    """


# ────────────────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────────────────


class Pipeline:
    """Orchestrates the full GCAW extraction pipeline.

    Sequences three stages per extraction:

    1. **Scout** — navigate, lazy-load, capture ROI screenshot.
    2. **Lock** — extract structured data via grammar-constrained LLM.
    3. **Memory** — apply stored corrections to every string field.

    Lifecycle::

        pipeline = Pipeline(
            lock=GrammarLock(api_key="..."),
            memory=MemoryStore(),
        )

        # Sync (for Streamlit)
        image_bytes, data = pipeline.run_sync(
            url="https://example.com/prices",
            schema=IndustrialItem,
        )

        # Or async
        image_bytes, data = await pipeline.run(url, IndustrialItem)

    The ``teach()`` method feeds human corrections back into Memory
    for future auto-application.
    """

    # ── Construction ────────────────────────────────────────────────

    def __init__(
        self,
        lock: GrammarLock,
        memory: MemoryStore,
        *,
        headless: bool = True,
        groq_api_key: str = "",
    ) -> None:
        self._lock: GrammarLock = lock
        self._memory: MemoryStore = memory
        self._scout_config: ScoutConfig = ScoutConfig(
            headless=headless,
            groq_api_key=groq_api_key,
        )

        logger.info("Pipeline initialised.")

    # ── Sync API (Streamlit-friendly) ───────────────────────────────

    def run_sync(
        self,
        url: str,
        schema: Type[T],
        context_text: str = "",
    ) -> tuple[bytes, T]:
        """Synchronous wrapper around the async pipeline.

        Creates a fresh event loop via ``asyncio.run()``.  On Windows,
        switches to ``ProactorEventLoop`` which supports subprocess
        creation (required by Playwright).
        """
        # Windows SelectorEventLoop cannot spawn subprocesses —
        # Playwright needs ProactorEventLoop.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy()
            )
        return asyncio.run(self.run(url, schema, context_text))

    # ── Async API ───────────────────────────────────────────────────

    async def run(
        self,
        url: str,
        schema: Type[T],
        context_text: str = "",
    ) -> tuple[FlightRecord, T]:
        """Execute the full extraction pipeline.

        Parameters
        ----------
        url : str
            Target URL containing an industrial data table.
        schema : Type[T]
            Pydantic V2 model defining the expected structure.
        context_text : str, optional
            Free-text hint for the Lock (forwarded to Gemini).

        Returns
        -------
        tuple[FlightRecord, T]
            ``(flight_record, corrected_data_object)``

        Raises
        ------
        PipelineError
            If any stage fails after its internal retries.
        """
        logger.info("── Pipeline: run({}) ──", url)

        try:
            # ── Stage 1: Scout ──────────────────────────────────────
            logger.info("Stage 1/3: Scout — capturing ROI.")
            async with AsyncScout(config=self._scout_config) as scout:
                scout_result: ScoutResult = await scout.capture(url)

            logger.info(
                "Scout complete — {} coords, method={}, {:.0f}ms.",
                scout_result.roi_coords,
                scout_result.detection_method,
                scout_result.latency_ms,
            )

            # ── Debug: save screenshot to disk ──────────────────────
            debug_path = Path("debug_roi.png")
            debug_path.write_bytes(scout_result.image_bytes)
            logger.info("Debug screenshot saved → {}", debug_path.resolve())

            # ── Stage 2: Lock ───────────────────────────────────────
            logger.info("Stage 2/3: Lock — extracting {}.", schema.__name__)
            lock_result: LockResult = self._lock.extract(
                image_data=scout_result.image_bytes,
                schema=schema,
                context_text=context_text,
            )
            logger.info(
                "Lock complete — {} extracted ({:.0f}ms, {} tokens).",
                schema.__name__,
                lock_result.latency_ms,
                lock_result.tokens_used,
            )

            # ── Stage 3: Memory (Self-Healing) ──────────────────────
            logger.info("Stage 3/3: Memory — applying corrections.")
            t_mem_start: float = time.perf_counter()
            corrected, corrections = self._apply_corrections(
                lock_result.data
            )
            mem_latency_ms = (time.perf_counter() - t_mem_start) * 1000

            memory_result = MemoryResult(
                final_data=corrected,
                corrections=corrections,
                latency_ms=round(mem_latency_ms, 1),
            )

            # ── Assemble FlightRecord ─────────────────────────────
            flight_record = FlightRecord(
                scout=scout_result,
                lock=lock_result,
                memory=memory_result,
            )

            logger.success(
                "Pipeline complete for {} — total {:.0f}ms, "
                "{} corrections applied.",
                url,
                flight_record.total_latency_ms,
                len(corrections),
            )
            return flight_record, corrected

        except (ScoutError, LockError, MemoryStoreError) as exc:
            logger.error("Pipeline domain error: {}", exc)
            raise PipelineError(str(exc)) from exc
        except PipelineError:
            raise
        except Exception as exc:
            logger.exception("Pipeline unexpected error.")
            raise PipelineError(
                f"Unexpected pipeline failure: {exc}"
            ) from exc

    # ── Self-Healing ────────────────────────────────────────────────

    def _apply_corrections(
        self, obj: T
    ) -> tuple[T, list[CorrectionDetail]]:
        """Recursively walk a Pydantic model and fix string fields.

        Returns the (potentially new) model plus a list of every
        correction applied, for audit in the ``MemoryResult``.
        """
        updates: dict[str, Any] = {}
        corrections: list[CorrectionDetail] = []

        for field_name in obj.model_fields:
            value = getattr(obj, field_name)

            if value is None:
                continue

            # ── String field → query Memory ─────────────────────────
            if isinstance(value, str):
                fix = self._memory.recall_correction(value, field_name)
                if fix is not None:
                    updates[field_name] = fix
                    corrections.append(CorrectionDetail(
                        field_name=field_name,
                        original_value=value,
                        corrected_value=fix,
                    ))
                    logger.debug(
                        "Corrected {}: '{}' → '{}'.",
                        field_name,
                        value,
                        fix,
                    )

            # ── List of BaseModels → recurse each item ──────────────
            elif isinstance(value, list):
                new_items: list[Any] = []
                list_modified = False
                for item in value:
                    if isinstance(item, BaseModel):
                        corrected, sub_corrections = (
                            self._apply_corrections(item)
                        )
                        new_items.append(corrected)
                        corrections.extend(sub_corrections)
                        if corrected is not item:
                            list_modified = True
                    else:
                        new_items.append(item)
                if list_modified:
                    updates[field_name] = new_items

            # ── Nested BaseModel → recurse ──────────────────────────
            elif isinstance(value, BaseModel):
                corrected, sub_corrections = (
                    self._apply_corrections(value)
                )
                corrections.extend(sub_corrections)
                if corrected is not value:
                    updates[field_name] = corrected

        if updates:
            logger.info(
                "Applied {} correction(s) to {}.",
                len(updates),
                type(obj).__name__,
            )
            return obj.model_copy(update=updates), corrections
        return obj, corrections

    # ── Teaching ────────────────────────────────────────────────────

    def teach(
        self,
        mistake: str,
        fix: str,
        field: str,
    ) -> None:
        """Feed a human correction into Memory for future recall.

        Convenience wrapper around ``MemoryStore.learn_correction``.

        Parameters
        ----------
        mistake : str
            The incorrect value as extracted by the Lock.
        fix : str
            The human-verified correct value.
        field : str
            The schema field name this correction applies to.
        """
        self._memory.learn_correction(
            mistake=mistake,
            correction=fix,
            field_name=field,
        )
