"""GCAW Scout Module — Perception Layer.

Provides async browser automation for navigating industrial data sources,
handling lazy-loaded content, and capturing targeted screenshots of the
primary data region (ROI).
"""

from src.scout.engine import AsyncScout, ScoutConfig, ScoutError

__all__ = ["AsyncScout", "ScoutConfig", "ScoutError"]
