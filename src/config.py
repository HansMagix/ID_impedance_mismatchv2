"""
GCAW Configuration — Single Source of Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loads environment variables from ``.env``, validates required keys,
and exposes a typed ``Settings`` singleton consumed by all layers.

Design Decisions
~~~~~~~~~~~~~~~~

1. **Dynamic ``.env`` discovery**
   ``pathlib`` resolves the ``.env`` path relative to this file's
   parent directory (``src/``), then walks up to the project root.
   This works regardless of the working directory — whether you
   run ``streamlit run src/app.py`` from the project root or
   invoke the module from a test harness.

2. **Typed properties with sensible defaults**
   Every setting has a safe default.  ``GEMINI_API_KEY`` defaults
   to ``""`` (empty), which downstream code handles gracefully.
   ``HEADLESS`` is a bool parsed from the env string.

3. **Singleton pattern**
   ``settings = Settings()`` is instantiated at module level.
   Import it anywhere: ``from src.config import settings``.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ────────────────────────────────────────────────────────────────────
# .env Discovery
# ────────────────────────────────────────────────────────────────────

# Walk from this file (src/config.py) up to the project root
_THIS_DIR: Path = Path(__file__).resolve().parent          # src/
_PROJECT_ROOT: Path = _THIS_DIR.parent                     # project root
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

# Load .env if it exists — does NOT override already-set env vars
if _ENV_FILE.is_file():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
    logger.debug("Loaded .env from {}.", _ENV_FILE)
else:
    logger.warning(
        "No .env file found at {}.  "
        "Falling back to system environment variables.",
        _ENV_FILE,
    )


# ────────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────────


class Settings:
    """Typed, validated access to GCAW configuration.

    All values are read from environment variables (populated by
    ``.env`` or the system environment).  Properties provide typed
    access with sensible defaults.

    Usage::

        from src.config import settings

        lock = GrammarLock(api_key=settings.GEMINI_API_KEY)
        store = MemoryStore(persistence_path=settings.CHROMA_PATH)
        scout = AsyncScout(headless=settings.HEADLESS)
    """

    # ── Core Properties ─────────────────────────────────────────────

    @property
    def GROQ_API_KEY(self) -> str:
        """Groq API key for the Lock module."""
        return os.environ.get("GROQ_API_KEY", "")

    @property
    def CHROMA_PATH(self) -> str:
        """Filesystem path for ChromaDB persistence.

        Resolved relative to the project root so it works
        regardless of working directory.
        """
        raw = os.environ.get("CHROMA_PATH", "./chroma_db")
        # If relative, anchor to project root
        path = Path(raw)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return str(path)

    @property
    def HEADLESS(self) -> bool:
        """Whether Playwright runs headless (no visible browser).

        Accepts ``"true"``/``"1"``/``"yes"`` (case-insensitive).
        Defaults to ``True``.
        """
        raw = os.environ.get("HEADLESS", "true").lower()
        return raw in ("true", "1", "yes")

    @property
    def GROQ_MODEL(self) -> str:
        """Groq model name for the Lock module."""
        return os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    @property
    def LOG_LEVEL(self) -> str:
        """Loguru log level."""
        return os.environ.get("LOG_LEVEL", "INFO")

    # ── Validation ──────────────────────────────────────────────────

    def validate(self) -> bool:
        """Check all required settings and log warnings for missing ones.

        Returns
        -------
        bool
            ``True`` if all critical settings are present.
        """
        ok = True

        if not self.GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set. "
                "The Lock module will not function. "
                "Add it to .env or export it as an environment variable."
            )
            ok = False
        else:
            # Show only first/last 4 chars for confirmation
            key = self.GROQ_API_KEY
            masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            logger.info("GROQ_API_KEY loaded ({}).", masked)

        logger.info("CHROMA_PATH  = {}", self.CHROMA_PATH)
        logger.info("HEADLESS     = {}", self.HEADLESS)
        logger.info("GROQ_MODEL   = {}", self.GROQ_MODEL)
        logger.info("LOG_LEVEL    = {}", self.LOG_LEVEL)

        return ok

    def __repr__(self) -> str:
        return (
            f"Settings(GROQ_API_KEY={'SET' if self.GROQ_API_KEY else 'MISSING'}, "
            f"CHROMA_PATH={self.CHROMA_PATH!r}, "
            f"HEADLESS={self.HEADLESS}, "
            f"GROQ_MODEL={self.GROQ_MODEL!r})"
        )


# ────────────────────────────────────────────────────────────────────
# Singleton
# ────────────────────────────────────────────────────────────────────

settings = Settings()
