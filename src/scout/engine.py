"""
GCAW Scout Engine — Async Perception Layer (Visual AI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Playwright-based browser automation that:
  1. Navigates to a target URL with anti-bot stealth measures.
  2. Forces lazy-loaded content into the DOM via progressive scrolling.
  3. Detects the primary data region (ROI) using **Groq Vision AI**
     (Llama 4 Scout) instead of brittle CSS-heuristics.
  4. Captures a pixel-perfect screenshot of *only* the ROI region.

Design Decisions:
  - **Visual ROI Detection**: A full-page screenshot is sent to Groq's
    Vision model, which returns a ``[ymin, xmin, ymax, xmax]`` bounding
    box for the main data region.  This replaces the old density-scoring
    heuristic, making detection layout-agnostic (works on ``<table>``,
    CSS Grid, product cards, dashboards — anything visual).
  - **Graceful Fallback**: If the Vision call fails (API error, invalid
    JSON, out-of-bounds bbox), the Scout falls back to a full-viewport
    crop so the pipeline never halts.
  - **No ``networkidle``**: Uses ``domcontentloaded`` + explicit waits.
  - **Anti-Bot**: UA rotation, Chromium stealth args, navigator masking.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from src.domain.trace import Trace

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

_USER_AGENTS: list[str] = [
    # Chrome 120 — Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    # Chrome 121 — Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    # Chrome 120 — macOS
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    # Chrome 120 — Linux
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    # Firefox 121 — Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0"
    ),
    # Edge 120 — Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    ),
]

_STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
    "--disable-extensions",
    "--disable-popup-blocking",
    "--disable-background-timer-throttling",
]

_VISION_PROMPT: str = (
    "You are a bounding-box detector for data tables. "
    "Analyse this webpage screenshot and find the MAIN data table, "
    "data grid, or product listing — the primary content region. "
    "Ignore navigation bars, headers, footers, sidebars, and ads. "
    "Return ONLY a JSON array of 4 integers: [ymin, xmin, ymax, xmax] "
    "representing the pixel coordinates of the bounding box. "
    "Example: [120, 50, 800, 1400]. "
    "NO explanation, NO markdown, ONLY the JSON array."
)


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoutConfig:
    """Immutable configuration for the AsyncScout.

    All timing values are chosen for industrial dashboards that typically
    render within 2-5 seconds.  Tune ``scroll_pause_sec`` upward for
    exceptionally slow-loading sites.
    """

    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080

    # Scroll behaviour
    scroll_step_px: int = 800
    scroll_pause_sec: float = 0.4
    max_scroll_attempts: int = 50
    stale_height_threshold: int = 3

    # Timeouts
    page_load_timeout_ms: int = 30_000

    # Groq Vision (for ROI detection)
    groq_api_key: str = ""
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    @property
    def viewport(self) -> dict[str, int]:
        return {"width": self.viewport_width, "height": self.viewport_height}


# ────────────────────────────────────────────────────────────────────
# Domain Exception
# ────────────────────────────────────────────────────────────────────


class ScoutError(Exception):
    """Raised for any unrecoverable failure in the Scout layer.

    Wraps underlying Playwright / Vision / timeout errors with a
    domain-meaningful message so callers never need to catch infra
    exceptions directly.
    """


# ────────────────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────────────────


class AsyncScout:
    """Async perception engine for the GCAW pipeline.

    Manages a long-lived Chromium instance (reused across multiple
    ``capture()`` calls) with per-call isolated ``BrowserContext``s for
    cookie / state separation.

    Lifecycle::

        scout = AsyncScout(headless=True)
        try:
            result = await scout.capture("https://example.com/data")
            png_bytes = result["screenshot_bytes"]
            metadata  = result["roi_metadata"]
        finally:
            await scout.shutdown()

    Or use the async-context-manager shorthand::

        async with AsyncScout(config=cfg) as scout:
            result = await scout.capture(url)
    """

    # ── Construction ────────────────────────────────────────────────

    def __init__(
        self,
        headless: bool = True,
        *,
        config: ScoutConfig | None = None,
    ) -> None:
        self._config: ScoutConfig = config or ScoutConfig(headless=headless)
        self._pw: Playwright | None = None
        self._browser: Browser | None = None

        # Initialise Groq client for Vision ROI detection
        if self._config.groq_api_key:
            self._groq: Groq | None = Groq(
                api_key=self._config.groq_api_key
            )
        else:
            self._groq = None
            logger.warning(
                "Scout: No GROQ_API_KEY — Vision ROI disabled, "
                "will use full-viewport fallback."
            )

    @classmethod
    async def create(cls, **kwargs: Any) -> "AsyncScout":
        """Factory that eagerly initialises the browser.

        Returns the instance — also usable as
        ``async with AsyncScout.create() as s:``.
        """
        instance = cls(**kwargs)
        await instance._ensure_browser()
        return instance

    async def __aenter__(self) -> "AsyncScout":
        await self._ensure_browser()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    # ── Browser Lifecycle ───────────────────────────────────────────

    async def _ensure_browser(self) -> Browser:
        """Lazily start Playwright + Chromium.  Reuses a live instance."""
        if self._browser is not None and self._browser.is_connected():
            return self._browser

        logger.info(
            "Launching Chromium (headless={}).", self._config.headless
        )
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._config.headless,
            args=_STEALTH_ARGS,
        )
        return self._browser

    async def _new_stealth_context(self) -> BrowserContext:
        """Spin up an isolated context with randomised fingerprint."""
        browser = await self._ensure_browser()
        user_agent = random.choice(_USER_AGENTS)
        logger.debug("UA  → {}", user_agent[:70])

        ctx = await browser.new_context(
            viewport=self._config.viewport,
            user_agent=user_agent,
            java_script_enabled=True,
            locale="en-US",
            timezone_id="America/New_York",
        )

        # Mask automation tells
        await ctx.add_init_script(
            """() => {
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                // Fake plugin array (real browsers never have length 0)
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                // Chrome-specific: window.chrome stub
                window.chrome = { runtime: {} };
            }"""
        )
        return ctx

    async def shutdown(self) -> None:
        """Gracefully tear down browser and Playwright server."""
        if self._browser is not None:
            logger.info("Closing Chromium.")
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    # ── Public API ──────────────────────────────────────────────────

    async def capture(self, url: str) -> dict[str, Any]:
        """Scout a URL and return a targeted screenshot of the data region.

        Parameters
        ----------
        url : str
            Fully-qualified URL to navigate to.

        Returns
        -------
        dict
            ``screenshot_bytes`` — PNG bytes of the ROI crop.
            ``roi_metadata``     — dict with ``bounding_box``,
            ``detection_method``, ``vision_confidence``.
            ``trace``            — ``Trace`` dataclass with timing info.

        Raises
        ------
        ScoutError
            If navigation fails or an unexpected runtime error occurs.
        """
        logger.info("── Scout: capture({}) ──", url)
        ctx: BrowserContext | None = None
        t_start: float = time.perf_counter()

        try:
            ctx = await self._new_stealth_context()
            page: Page = await ctx.new_page()

            # ── Navigate (domcontentloaded — NOT networkidle) ───────
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self._config.page_load_timeout_ms,
            )
            if response is None or response.status >= 400:
                status = response.status if response else "no response"
                raise ScoutError(
                    f"Navigation failed → HTTP {status} for {url}"
                )
            logger.info("Loaded (HTTP {}).", response.status)

            # Brief pause for client-side JS to hydrate
            await page.wait_for_timeout(1_500)

            # ── Lazy-load ──────────────────────────────────────────
            await self._force_lazy_load(page)

            # ── Full-page screenshot for Vision ────────────────────
            full_screenshot: bytes = await page.screenshot(
                type="png", full_page=True
            )
            logger.debug(
                "Full-page screenshot: {} bytes.", len(full_screenshot)
            )

            # ── Vision ROI detection ───────────────────────────────
            bbox, detection_method, vision_confidence = (
                self._get_visual_bbox(full_screenshot)
            )

            # ── Crop to ROI ────────────────────────────────────────
            if detection_method == "vision":
                roi_screenshot: bytes = await page.screenshot(
                    type="png",
                    clip={
                        "x": bbox["x"],
                        "y": bbox["y"],
                        "width": bbox["width"],
                        "height": bbox["height"],
                    },
                )
            else:
                # Fallback: use the full-page screenshot as-is
                roi_screenshot = full_screenshot

            # ── Build metadata ─────────────────────────────────────
            scout_latency_ms = (time.perf_counter() - t_start) * 1000
            roi_meta: dict[str, Any] = {
                "bounding_box": bbox,
                "detection_method": detection_method,
                "vision_confidence": vision_confidence,
            }
            trace = Trace(
                scout_latency_ms=round(scout_latency_ms, 1),
                detection_method=detection_method,
                vision_confidence=vision_confidence,
            )

            logger.success(
                "Captured ROI — {}×{} px, method={}, "
                "confidence={}, latency={:.0f}ms.",
                bbox["width"],
                bbox["height"],
                detection_method,
                vision_confidence,
                scout_latency_ms,
            )
            return {
                "screenshot_bytes": roi_screenshot,
                "roi_metadata": roi_meta,
                "trace": trace,
            }

        except ScoutError:
            raise
        except Exception as exc:
            logger.exception("Unhandled Scout error.")
            raise ScoutError(
                f"Unexpected failure during capture: {exc}"
            ) from exc
        finally:
            if ctx is not None:
                await ctx.close()

    # ── Lazy-Load Handler ───────────────────────────────────────────

    async def _force_lazy_load(self, page: Page) -> None:
        """Progressively scroll the page to trigger deferred content.

        **Strategy:**

        1. Read ``document.body.scrollHeight`` as baseline.
        2. Smooth-scroll down by ``scroll_step_px`` pixels.
        3. Pause ``scroll_pause_sec`` for JS to render new DOM nodes.
        4. Re-read height.  If unchanged for ``stale_height_threshold``
           consecutive steps → page is fully loaded.
        5. Hard-cap at ``max_scroll_attempts`` to escape infinite-scroll
           traps (social feeds, analytics dashboards with server paging).
        6. Scroll back to top so that ROI bounding-box coordinates are
           measured relative to a deterministic scroll origin.
        """
        cfg = self._config
        logger.debug("Lazy-load: scrolling (step={}px).", cfg.scroll_step_px)

        prev_height: int = await page.evaluate("document.body.scrollHeight")
        stale_count: int = 0

        for step in range(cfg.max_scroll_attempts):
            await page.evaluate(
                f"window.scrollBy({{ top: {cfg.scroll_step_px}, "
                f"behavior: 'smooth' }})"
            )
            await asyncio.sleep(cfg.scroll_pause_sec)

            cur_height: int = await page.evaluate(
                "document.body.scrollHeight"
            )

            if cur_height == prev_height:
                stale_count += 1
                if stale_count >= cfg.stale_height_threshold:
                    logger.debug(
                        "Scroll stabilised after {} steps (h={}).",
                        step + 1,
                        cur_height,
                    )
                    break
            else:
                stale_count = 0
                prev_height = cur_height
        else:
            logger.warning(
                "Hit scroll cap ({}).  Page may have infinite scroll.",
                cfg.max_scroll_attempts,
            )

        # Reset viewport to top-of-page for consistent ROI coordinates
        await page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
        await asyncio.sleep(0.3)

    # ── Vision ROI Detection ────────────────────────────────────────

    def _get_visual_bbox(
        self, screenshot_bytes: bytes
    ) -> tuple[dict[str, int], str, float | None]:
        """Use Groq Vision to detect the data region bounding box.

        Parameters
        ----------
        screenshot_bytes : bytes
            Full-page PNG screenshot.

        Returns
        -------
        tuple[dict, str, float | None]
            - Bounding box dict: ``{"x", "y", "width", "height"}``
            - Detection method: ``"vision"`` or ``"fallback"``
            - Vision confidence (placeholder, currently ``None``
              for fallback)
        """
        if self._groq is None:
            logger.info("Vision ROI: no Groq client, using fallback.")
            return self._fallback_bbox(screenshot_bytes), "fallback", None

        try:
            # ── Encode image ────────────────────────────────────────
            b64: str = base64.b64encode(screenshot_bytes).decode("ascii")
            data_url: str = f"data:image/png;base64,{b64}"

            logger.debug(
                "Vision ROI: sending {}KB to {}.",
                len(screenshot_bytes) // 1024,
                self._config.vision_model,
            )

            # ── Call Groq Vision ────────────────────────────────────
            response = self._groq.chat.completions.create(
                model=self._config.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=64,
            )

            raw_text: str = response.choices[0].message.content.strip()
            logger.debug("Vision ROI raw response: {}", raw_text)

            # ── Parse [ymin, xmin, ymax, xmax] ──────────────────────
            # Strip markdown fences if the model wraps in ```json
            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            coords: list[int] = json.loads(cleaned)
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError(
                    f"Expected [ymin, xmin, ymax, xmax], got: {coords}"
                )

            ymin, xmin, ymax, xmax = [int(c) for c in coords]

            # Sanity checks
            if ymin >= ymax or xmin >= xmax:
                raise ValueError(
                    f"Invalid bbox: ymin={ymin} >= ymax={ymax} "
                    f"or xmin={xmin} >= xmax={xmax}"
                )
            if ymax - ymin < 50 or xmax - xmin < 50:
                raise ValueError(
                    f"Bbox too small: {xmax - xmin}×{ymax - ymin} px"
                )

            bbox: dict[str, int] = {
                "x": max(0, xmin),
                "y": max(0, ymin),
                "width": xmax - xmin,
                "height": ymax - ymin,
            }

            logger.info(
                "Vision ROI detected: x={}, y={}, {}×{} px.",
                bbox["x"], bbox["y"], bbox["width"], bbox["height"],
            )
            return bbox, "vision", None

        except Exception as exc:
            logger.warning(
                "Vision ROI failed ({}), falling back to full-page crop.",
                exc,
            )
            return self._fallback_bbox(screenshot_bytes), "fallback", None

    @staticmethod
    def _fallback_bbox(screenshot_bytes: bytes) -> dict[str, int]:
        """Generate a full-viewport fallback bbox.

        Uses the standard viewport size as the crop region.
        """
        return {
            "x": 0,
            "y": 0,
            "width": 1920,
            "height": 1080,
        }
