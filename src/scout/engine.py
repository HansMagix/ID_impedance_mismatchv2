"""
GCAW Scout Engine — Async Perception Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Playwright-based browser automation that:
  1. Navigates to a target URL with anti-bot stealth measures.
  2. Forces lazy-loaded content into the DOM via progressive scrolling.
  3. Detects the primary data region (ROI) using a scored heuristic.
  4. Captures a pixel-perfect screenshot of *only* the ROI element.

Design Decisions:
  - **No `networkidle`**: Playwright's `networkidle` is fundamentally flaky
    on dynamic industrial dashboards. We use `domcontentloaded` + explicit
    selector waits, which is deterministic and fast.
  - **Lazy-Load Strategy**: Incremental smooth-scroll in viewport-sized steps.
    We track `document.body.scrollHeight`; if it stabilizes for N consecutive
    scrolls, we declare the page fully loaded. A hard cap prevents infinite
    scroll traps (e.g., social media feeds).
  - **ROI Heuristic**: We score every `<table>`, `[role="grid"]`, and grid-
    classed `<div>` by `visible_area × log(data_density + 1)`. This naturally
    ranks large, data-dense tables above tiny nav-bars or footer links.
    Candidates below 10,000 px² or with fewer than 2×2 cells are excluded.
  - **Anti-Bot**: User-Agent rotation from a curated pool, Chromium stealth
    launch args (disable automation flags), and navigator property masking
    via `addInitScript`.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Page,
    Playwright,
    async_playwright,
)

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

_TABLE_SELECTORS: str = (
    "table, "
    "[role='grid'], [role='table'], "
    "[class*='data-table'], [class*='datatable'], "
    "[class*='DataTable'], [class*='grid-table'], "
    "[class*='ag-body'], [class*='el-table'], "
    "[class*='ant-table'], [class*='MuiTable']"
)

_MIN_ROI_AREA_PX: int = 10_000
_MIN_ROI_ROWS: int = 2
_MIN_ROI_COLS: int = 2


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
    roi_wait_timeout_ms: int = 5_000

    @property
    def viewport(self) -> dict[str, int]:
        return {"width": self.viewport_width, "height": self.viewport_height}


# ────────────────────────────────────────────────────────────────────
# Domain Exception
# ────────────────────────────────────────────────────────────────────


class ScoutError(Exception):
    """Raised for any unrecoverable failure in the Scout layer.

    Wraps underlying Playwright / selector / timeout errors with a
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

        async with AsyncScout.create(headless=True) as scout:
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

    @classmethod
    async def create(cls, **kwargs: Any) -> "AsyncScout":
        """Factory that eagerly initialises the browser.

        Returns the instance — also usable as ``async with AsyncScout.create() as s:``.
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
        """Scout a URL and return a targeted screenshot of the primary table.

        Parameters
        ----------
        url : str
            Fully-qualified URL to navigate to.

        Returns
        -------
        dict
            ``screenshot_bytes``  — PNG bytes of the ROI element.
            ``roi_metadata``      — dict with ``tag``, ``bounding_box``,
            ``row_count``, ``col_count``, ``data_density``, ``score``,
            ``selector``.

        Raises
        ------
        ScoutError
            If navigation fails, no data table is found, or an
            unexpected runtime error occurs.
        """
        logger.info("── Scout: capture({}) ──", url)
        ctx: BrowserContext | None = None

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

            # ── ROI detection ──────────────────────────────────────
            roi_handle, roi_meta = await self._detect_roi(page)

            # ── Screenshot of ROI only ─────────────────────────────
            screenshot_bytes: bytes = await roi_handle.screenshot(type="png")
            logger.success(
                "Captured ROI <{tag}> — {w}×{h} px, "
                "{rows}r × {cols}c, score={score:.0f}.",
                tag=roi_meta["tag"],
                w=roi_meta["bounding_box"]["width"],
                h=roi_meta["bounding_box"]["height"],
                rows=roi_meta["row_count"],
                cols=roi_meta["col_count"],
                score=roi_meta["score"],
            )
            return {
                "screenshot_bytes": screenshot_bytes,
                "roi_metadata": roi_meta,
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

    # ── ROI Detection ───────────────────────────────────────────────

    async def _detect_roi(
        self, page: Page
    ) -> tuple[ElementHandle, dict[str, Any]]:
        """Identify the primary data table on the page.

        **Heuristic — Scored Ranking:**

        For every candidate element matching ``_TABLE_SELECTORS``:

        * Compute **visible area** = ``width × height`` from bounding box.
        * Compute **data density** = ``row_count × col_count``.
        * **Score** = ``area × log(density + 1)``.

        The logarithm prevents a huge-but-empty ``<div>`` from winning
        over a moderately-sized table packed with cells.

        Candidates are filtered out if:
        * Bounding box area < 10 000 px²  (nav bars, icon grids).
        * Fewer than 2 rows or 2 columns  (single-cell wrappers).

        Returns
        -------
        tuple[ElementHandle, dict]
            The winning element handle and its metadata dict.

        Raises
        ------
        ScoutError
            If no qualifying element is found.
        """
        logger.debug("ROI: scanning for candidates …")

        # Wait briefly for at least one table-like element
        try:
            await page.wait_for_selector(
                "table, [role='grid'], [role='table']",
                timeout=self._config.roi_wait_timeout_ms,
            )
        except Exception:
            logger.warning("No table/grid appeared within timeout.")

        # Gather all candidate elements
        candidates: list[ElementHandle] = await page.query_selector_all(
            _TABLE_SELECTORS
        )
        if not candidates:
            raise ScoutError(
                "No table or grid-like elements found on the page."
            )

        logger.debug("ROI: {} candidates found.", len(candidates))

        best_score: float = -1.0
        best_handle: ElementHandle | None = None
        best_meta: dict[str, Any] = {}

        for handle in candidates:
            try:
                meta = await self._score_candidate(handle)
            except Exception as exc:
                logger.trace("Skipping candidate: {}", exc)
                continue
            if meta is None:
                continue
            if meta["score"] > best_score:
                best_score = meta["score"]
                best_handle = handle
                best_meta = meta

        if best_handle is None:
            raise ScoutError(
                "All candidate elements were too small or data-sparse "
                "to qualify as the primary data table."
            )

        logger.info(
            "ROI winner: <{tag}> score={score:.0f}, "
            "area={area}px², density={density}.",
            tag=best_meta["tag"],
            score=best_score,
            area=(
                best_meta["bounding_box"]["width"]
                * best_meta["bounding_box"]["height"]
            ),
            density=best_meta["data_density"],
        )
        return best_handle, best_meta

    # ── Candidate Scoring ───────────────────────────────────────────

    @staticmethod
    async def _score_candidate(
        handle: ElementHandle,
    ) -> dict[str, Any] | None:
        """Evaluate a single DOM element as a potential ROI.

        Returns ``None`` (silently skips) if the element fails any gate:
        area too small, too few rows, or too few columns.
        """
        bbox = await handle.bounding_box()
        if bbox is None:
            return None

        width = int(bbox["width"])
        height = int(bbox["height"])
        area = width * height

        if area < _MIN_ROI_AREA_PX:
            return None

        tag: str = await handle.evaluate("el => el.tagName.toLowerCase()")

        # ── Count data rows & columns ──────────────────────────────
        if tag == "table":
            row_count: int = await handle.evaluate(
                "el => el.querySelectorAll('tr').length"
            )
            col_count: int = await handle.evaluate(
                """el => {
                    const row = el.querySelector('tr');
                    return row
                        ? row.querySelectorAll('td, th').length
                        : 0;
                }"""
            )
        else:
            # Div-based grids (AG Grid, MUI DataGrid, Ant Design, etc.)
            row_count = await handle.evaluate(
                """el => {
                    const explicit = el.querySelectorAll(
                        '[role="row"], .row, tr'
                    ).length;
                    return explicit || el.children.length;
                }"""
            )
            col_count = await handle.evaluate(
                """el => {
                    const firstRow = el.querySelector(
                        '[role="row"], .row, tr'
                    );
                    if (firstRow) return firstRow.children.length;
                    const cols = getComputedStyle(el)
                        .gridTemplateColumns;
                    return cols
                        ? cols.split(' ').length
                        : 1;
                }"""
            )

        if row_count < _MIN_ROI_ROWS or col_count < _MIN_ROI_COLS:
            return None

        density = row_count * col_count
        score = area * math.log(density + 1)

        # Build a best-effort CSS selector for downstream reference
        selector: str = await handle.evaluate(
            """el => {
                if (el.id) return '#' + el.id;
                let sel = el.tagName.toLowerCase();
                if (el.className && typeof el.className === 'string') {
                    sel += '.' + el.className.trim()
                        .split(/\\s+/).join('.');
                }
                return sel;
            }"""
        )

        return {
            "tag": tag,
            "bounding_box": {
                "x": int(bbox["x"]),
                "y": int(bbox["y"]),
                "width": width,
                "height": height,
            },
            "row_count": row_count,
            "col_count": col_count,
            "data_density": density,
            "score": score,
            "selector": selector,
        }
