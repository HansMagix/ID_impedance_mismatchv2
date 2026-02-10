"""
GCAW Dashboard — Streamlit Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with::

    streamlit run src/app.py

Design Decisions
~~~~~~~~~~~~~~~~

1. **Session-state diffing for the learning loop**
   On extraction, the raw ``IndustrialItem.items`` are stored as
   ``original_items`` in ``st.session_state``.  ``st.data_editor``
   returns the edited DataFrame live.  "Brain Transplant" diffs the
   two cell-by-cell and feeds every changed string value into
   ``Pipeline.teach()``.

2. **Component caching**
   ``GrammarLock`` and ``MemoryStore`` are cached via
   ``@st.cache_resource`` so re-runs don't re-initialise them.
   ``Pipeline`` is cheap (just holds references) so it's created
   inline.  ``AsyncScout`` is spun up per extraction inside the
   Pipeline.

3. **No async leakage into Streamlit**
   The UI calls ``Pipeline.run_sync()`` which internally calls
   ``asyncio.run()``.  The Streamlit layer is 100 % synchronous.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import settings
from src.lock.engine import GrammarLock
from src.lock.schemas import IndustrialItem
from src.memory.store import MemoryStore
from src.orchestrator import Pipeline

# ────────────────────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GCAW — Data Workstation",
    page_icon="\U0001f52c",  # 🔬
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .stDataFrame { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────
# Cached Component Factories
# ────────────────────────────────────────────────────────────────────


@st.cache_resource
def _init_memory() -> MemoryStore:
    """One-time Memory initialisation (persisted across re-runs)."""
    return MemoryStore(persistence_path=settings.CHROMA_PATH)


@st.cache_resource
def _init_lock(api_key: str) -> GrammarLock | None:
    """One-time Lock initialisation, keyed by API key."""
    if not api_key:
        return None
    return GrammarLock(api_key=api_key)


# ────────────────────────────────────────────────────────────────────
# Sidebar — System Status & Config
# ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("\u2699\ufe0f GCAW Control")  # ⚙️
    st.markdown("---")

    # ── API Key ─────────────────────────────────────────────────────
    api_key: str = st.text_input(
        "Gemini API Key",
        type="password",
        value=settings.GEMINI_API_KEY,
        help="Required for the Lock module (Gemini 1.5 Flash).",
    )

    st.markdown("---")
    st.subheader("System Status")

    # ── Scout ───────────────────────────────────────────────────────
    scout_ok = False
    try:
        from playwright.async_api import async_playwright  # noqa: F401

        st.markdown("\U0001f7e2 **Scout** — Playwright ready")  # 🟢
        scout_ok = True
    except ImportError:
        st.markdown("\U0001f534 **Scout** — Playwright not installed")  # 🔴

    # ── Lock ────────────────────────────────────────────────────────
    lock: GrammarLock | None = _init_lock(api_key)
    lock_ok = lock is not None
    if lock_ok:
        st.markdown("\U0001f7e2 **Lock** — Gemini configured")
    else:
        st.markdown("\U0001f534 **Lock** — No API key")

    # ── Memory ──────────────────────────────────────────────────────
    memory: MemoryStore = _init_memory()
    memory_ok = True
    st.markdown(
        f"\U0001f7e2 **Memory** — {memory.correction_count} "
        f"corrections stored"
    )

    st.markdown("---")
    all_systems_go: bool = scout_ok and lock_ok and memory_ok
    if all_systems_go:
        st.success("All systems operational.")
    else:
        st.warning("Some systems need configuration.")

# ────────────────────────────────────────────────────────────────────
# Main Area — Header
# ────────────────────────────────────────────────────────────────────

st.title("\U0001f52c GCAW")  # 🔬
st.caption(
    "Grammar-Constrained Agentic Workstation — "
    "Industrial Data Extraction with Self-Healing Memory"
)

# ────────────────────────────────────────────────────────────────────
# Input Section
# ────────────────────────────────────────────────────────────────────

url: str = st.text_input(
    "\U0001f310 Target URL",  # 🌐
    placeholder="https://example.com/prices",
    help="Full URL of the page containing an industrial data table.",
)

context: str = st.text_input(
    "\U0001f4dd Context (optional)",  # 📝
    placeholder="e.g., Cement price list from Kenya, Q3 2024",
    help="Extra context passed to Gemini for better accuracy.",
)

extract_disabled: bool = not (all_systems_go and url.strip())
extract_clicked: bool = st.button(
    "\U0001f680 Extract",  # 🚀
    disabled=extract_disabled,
    use_container_width=False,
)

# ────────────────────────────────────────────────────────────────────
# Extraction Execution
# ────────────────────────────────────────────────────────────────────

if extract_clicked and all_systems_go and lock is not None:
    pipeline = Pipeline(lock=lock, memory=memory)

    progress = st.progress(0, text="Initialising Scout...")

    try:
        progress.progress(15, text="\U0001f575\ufe0f Scout: Navigating...")
        image_bytes, data = pipeline.run_sync(
            url=url.strip(),
            schema=IndustrialItem,
            context_text=context.strip(),
        )
        progress.progress(100, text="\u2705 Extraction complete!")

        # Persist to session state
        st.session_state["image_bytes"] = image_bytes
        st.session_state["data"] = data
        st.session_state["original_items"] = [
            item.model_dump() for item in data.items
        ]
        st.session_state["source_title"] = data.source_title
        st.session_state["extraction_notes"] = data.extraction_notes

    except Exception as exc:
        progress.empty()
        st.error(f"\u274c Extraction failed: {exc}")

# ────────────────────────────────────────────────────────────────────
# Results Section
# ────────────────────────────────────────────────────────────────────

if st.session_state.get("original_items"):
    st.markdown("---")
    st.subheader("\U0001f4cb Results")  # 📋

    # ── Metadata ────────────────────────────────────────────────────
    title = st.session_state.get("source_title")
    notes = st.session_state.get("extraction_notes")
    if title:
        st.caption(f"**Source:** {title}")
    if notes:
        st.info(f"\U0001f4dd {notes}")  # 📝

    # ── Two-column layout ──────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    # ── Column 1: Visual Verification ──────────────────────────────
    with col1:
        st.markdown("#### \U0001f4f8 Visual Verification")  # 📸
        if st.session_state.get("image_bytes"):
            st.image(
                st.session_state["image_bytes"],
                caption="ROI captured by Scout",
                use_container_width=True,
            )

    # ── Column 2: Data Editor ──────────────────────────────────────
    with col2:
        st.markdown("#### \U0001f4ca Extracted Data")  # 📊

        original_df = pd.DataFrame(st.session_state["original_items"])

        edited_df: pd.DataFrame = st.data_editor(
            original_df,
            key="data_editor",
            use_container_width=True,
            num_rows="dynamic",
        )

    # ────────────────────────────────────────────────────────────────
    # Learning Loop — Brain Transplant
    # ────────────────────────────────────────────────────────────────

    st.markdown("---")

    brain_col1, brain_col2 = st.columns([1, 3])

    with brain_col1:
        teach_clicked: bool = st.button(
            "\U0001f9e0 Brain Transplant",  # 🧠
            help=(
                "Commit your edits as corrections. "
                "GCAW will auto-apply them in future extractions."
            ),
            use_container_width=True,
        )

    if teach_clicked and lock is not None:
        pipeline = Pipeline(lock=lock, memory=memory)
        changes: list[tuple[str, str, str]] = []

        min_rows = min(len(original_df), len(edited_df))

        for idx in range(min_rows):
            for col_name in original_df.columns:
                old_val: Any = original_df.iloc[idx][col_name]
                new_val: Any = edited_df.iloc[idx][col_name]

                # Only learn meaningful string corrections
                if (
                    str(old_val) != str(new_val)
                    and pd.notna(old_val)
                    and pd.notna(new_val)
                    and str(old_val).strip()
                    and str(new_val).strip()
                ):
                    old_str = str(old_val).strip()
                    new_str = str(new_val).strip()
                    pipeline.teach(
                        mistake=old_str,
                        fix=new_str,
                        field=col_name,
                    )
                    changes.append((old_str, new_str, col_name))

        if changes:
            for old, new, field in changes:
                st.toast(
                    f"Learned: '{old}' = '{new}' [{field}]"
                )

            # Update the baseline so future diffs are clean
            st.session_state["original_items"] = (
                edited_df.to_dict("records")
            )

            st.success(
                f"\U0001f9e0 Committed {len(changes)} correction(s)! "
                f"Memory now has {memory.correction_count} total."
            )
        else:
            st.info(
                "No changes detected. "
                "Edit cells in the table above, then click again."
            )
