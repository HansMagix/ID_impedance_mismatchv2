"""
GCAW Domain Events — Typed Result Envelopes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Immutable result objects returned by each pipeline stage.  Instead of
prop-drilling a mutable ``Trace`` through every layer, each stage returns
a rich envelope containing **both data and metadata**.

The ``FlightRecord`` aggregates all three stage results into a single
auditable snapshot of the entire extraction run.

Design Philosophy (Hexagonal / Envelope Pattern):
  - Each layer is self-contained and knows nothing about global state.
  - Metadata originates where it's measured (latency in Scout, tokens in Lock).
  - The Orchestrator simply collects envelopes — no coupling to internals.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Scout Envelope
# ────────────────────────────────────────────────────────────────────


class ScoutResult(BaseModel):
    """Result envelope from the Scout (Perception) stage.

    Contains the raw ROI screenshot plus Vision detection metadata.
    ``image_bytes`` is stored as ``bytes`` — not serialisable to JSON,
    but ``FlightRecord`` is an in-process object, not a wire format.
    """

    image_bytes: bytes
    roi_coords: list[int] = Field(
        default_factory=list,
        description="[ymin, xmin, ymax, xmax] as detected by Vision AI.",
    )
    detection_method: str = Field(
        default="unknown",
        description="'vision' or 'fallback'.",
    )
    vision_confidence: float | None = Field(
        default=None,
        description="Model confidence, if reported.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock time for the full Scout stage.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ────────────────────────────────────────────────────────────────────
# Lock Envelope
# ────────────────────────────────────────────────────────────────────


class LockResult(BaseModel):
    """Result envelope from the Lock (Extraction) stage.

    ``data`` holds the extracted Pydantic model instance typed as
    ``Any`` to avoid forcing ``FlightRecord`` to be generic.  The
    Orchestrator returns the typed ``T`` alongside the record.
    """

    data: Any = Field(
        default=None,
        description="The extracted Pydantic model instance.",
    )
    tokens_used: int = Field(
        default=0,
        description="Total tokens consumed by the LLM call.",
    )
    validation_retries: int = Field(
        default=0,
        description="Number of instructor validation retries.",
    )
    model_name: str = Field(
        default="",
        description="LLM model name used for extraction.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock time for the Lock stage.",
    )
    inferred_schema: dict | None = Field(
        default=None,
        description="SchemaDefinition dict when dynamic mode is used.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ────────────────────────────────────────────────────────────────────
# Correction Detail
# ────────────────────────────────────────────────────────────────────


class CorrectionDetail(BaseModel):
    """Record of a single auto-correction applied by Memory."""

    field_name: str
    original_value: str
    corrected_value: str


# ────────────────────────────────────────────────────────────────────
# Memory Envelope
# ────────────────────────────────────────────────────────────────────


class MemoryResult(BaseModel):
    """Result envelope from the Memory (Self-Healing) stage.

    ``final_data`` holds the corrected Pydantic model.
    ``corrections`` is the audit trail of every auto-fix applied.
    """

    final_data: Any = Field(
        default=None,
        description="Corrected Pydantic model instance.",
    )
    corrections: list[CorrectionDetail] = Field(
        default_factory=list,
        description="All auto-corrections applied to this extraction.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Wall-clock time for the Memory stage.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ────────────────────────────────────────────────────────────────────
# Flight Record — Aggregated Pipeline Trace
# ────────────────────────────────────────────────────────────────────


class FlightRecord(BaseModel):
    """Aggregated trace of a single extraction run.

    Combines the three stage envelopes into one auditable snapshot.
    Enables:
      - Process-timeline visualisation in the dashboard.
      - Per-stage latency breakdown.
      - Confidence-based alerting.
      - Full correction audit trail.
    """

    scout: ScoutResult
    lock: LockResult
    memory: MemoryResult

    @property
    def total_latency_ms(self) -> float:
        """Sum of all stage latencies."""
        return (
            self.scout.latency_ms
            + self.lock.latency_ms
            + self.memory.latency_ms
        )

    model_config = {"arbitrary_types_allowed": True}
