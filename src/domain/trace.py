"""
GCAW Trace — Pipeline Observability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lightweight dataclass that flows through the pipeline, accumulating
timing and confidence metrics at each stage.  Enables:
  - Latency dashboards (per-stage breakdown)
  - Confidence-based threshold alerting
  - Audit trail for extracted data provenance

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trace:
    """Per-extraction trace record for pipeline observability.

    Attributes
    ----------
    scout_latency_ms : float
        Wall-clock time for the full Scout stage (navigation + scroll +
        Vision ROI detection + screenshot).
    detection_method : str
        How the ROI was detected: ``"vision"`` (Groq Vision API) or
        ``"fallback"`` (full-viewport crop).
    vision_confidence : float | None
        Model confidence for the Vision bbox, if available.  ``None``
        when fallback was used or model doesn't report confidence.
    lock_prompt_tokens : int | None
        Number of prompt tokens consumed by the Lock stage.  Populated
        by the Orchestrator after Lock completes.
    extraction_confidence : float | None
        Semantic confidence of the extracted data.  Populated by
        downstream validation or correction stages.
    """

    scout_latency_ms: float = 0.0
    detection_method: str = "unknown"
    vision_confidence: float | None = None
    lock_prompt_tokens: int | None = None
    extraction_confidence: float | None = None
