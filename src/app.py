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
from src.domain.events import FlightRecord
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
        "Groq API Key",
        type="password",
        value=settings.GROQ_API_KEY,
        help="Required for the Lock module (Llama 3.2 Vision via Groq).",
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
        st.markdown("\U0001f7e2 **Lock** — Groq configured")
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

    # ── Dynamic Schema Toggle ────────────────────────────────────────
    st.markdown("---")
    dynamic_mode: bool = st.toggle(
        "\U0001f9ec Dynamic Schema Mode",  # 🧬
        value=False,
        help=(
            "When ON, the Lock infers the schema autonomously "
            "(Two-Pass). When OFF, uses the static IndustrialItem "
            "schema."
        ),
    )
    if dynamic_mode:
        st.caption("\U0001f50d Agent will discover fields autonomously.")
    else:
        st.caption("\U0001f512 Using static IndustrialItem schema.")

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
    pipeline = Pipeline(lock=lock, memory=memory, groq_api_key=api_key)

    progress = st.progress(0, text="Initialising Scout...")

    try:
        progress.progress(15, text="\U0001f575\ufe0f Scout: Navigating...")

        chosen_schema = None if dynamic_mode else IndustrialItem
        flight_record, data = pipeline.run_sync(
            url=url.strip(),
            schema=chosen_schema,
            context_text=context.strip(),
        )
        progress.progress(100, text="\u2705 Extraction complete!")

        # Persist to session state
        st.session_state["flight_record"] = flight_record
        st.session_state["image_bytes"] = flight_record.scout.image_bytes
        st.session_state["data"] = data
        st.session_state["dynamic_mode"] = dynamic_mode

        # Static mode: data has .items / .source_title
        # Dynamic mode: data is flat Pydantic model
        if not dynamic_mode and hasattr(data, "items"):
            st.session_state["original_items"] = [
                item.model_dump() for item in data.items
            ]
            st.session_state["source_title"] = data.source_title
            st.session_state["extraction_notes"] = data.extraction_notes
        else:
            st.session_state["original_items"] = [data.model_dump()]
            st.session_state["source_title"] = None
            st.session_state["extraction_notes"] = None

    except Exception as exc:
        progress.empty()
        st.error(f"\u274c Extraction failed: {exc}")

# ────────────────────────────────────────────────────────────────────
# Results Section
# ────────────────────────────────────────────────────────────────────

if st.session_state.get("original_items"):
    st.markdown("---")
    st.subheader("\U0001f4cb Results")  # 📋

    # ── Metadata ──────────────────────────────────────────────────
    title = st.session_state.get("source_title")
    notes = st.session_state.get("extraction_notes")
    if title:
        st.caption(f"**Source:** {title}")
    if notes:
        st.info(f"\U0001f4dd {notes}")  # 📝

    # ── Explainable AI Tabs (Process Timeline) ────────────────────
    fr: FlightRecord | None = st.session_state.get("flight_record")

    tab_vision, tab_logic, tab_healing, tab_data = st.tabs([
        "\U0001f50d Vision",  # 🔍
        "\U0001f9ea Logic",   # 🧪
        "\U0001f9e0 Self-Healing",  # 🧠
        "\U0001f4ca Data",   # 📊
    ])

    # ── Tab 1: Vision ─────────────────────────────────────────
    with tab_vision:
        st.markdown("#### \U0001f4f8 ROI Detection")  # 📸
        if st.session_state.get("image_bytes"):
            st.image(
                st.session_state["image_bytes"],
                caption="ROI captured by Scout",
                use_container_width=True,
            )
        if fr:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Detection Method", fr.scout.detection_method)
            col_b.metric("Latency", f"{fr.scout.latency_ms:.0f} ms")
            col_c.metric(
                "Confidence",
                f"{fr.scout.vision_confidence:.2f}"
                if fr.scout.vision_confidence is not None
                else "N/A",
            )
            st.caption(
                f"ROI Coords: `{fr.scout.roi_coords}`"
            )

    # ── Tab 2: Logic ──────────────────────────────────────────
    with tab_logic:
        st.markdown("#### \U0001f916 LLM Extraction")  # 🤖
        if fr:
            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Model", fr.lock.model_name.split("/")[-1])
            col_e.metric("Tokens Used", fr.lock.tokens_used or "~")
            col_f.metric("Latency", f"{fr.lock.latency_ms:.0f} ms")

            extraction_mode = (
                "Two-Pass (Dynamic)" if fr.lock.inferred_schema
                else "Static Schema"
            )
            st.caption(
                f"Mode: **{extraction_mode}** · "
                f"Max validation retries: {fr.lock.validation_retries}"
            )

            # Inferred Schema Expander (dynamic mode only)
            if fr.lock.inferred_schema:
                with st.expander(
                    "\U0001f9ec Inferred Schema", expanded=True  # 🧬
                ):
                    schema_info = fr.lock.inferred_schema
                    st.markdown(
                        f"**Entity:** `{schema_info.get('entity_name', '?')}`"
                    )
                    for field in schema_info.get("fields", []):
                        st.markdown(
                            f"- `{field['name']}` : "
                            f"**{field['type']}** — "
                            f"{field.get('description', '')}"
                        )

    # ── Tab 3: Self-Healing ───────────────────────────────────
    with tab_healing:
        st.markdown("#### \U0001f9ec Correction Audit Trail")  # 🧬
        if fr and fr.memory.corrections:
            for c in fr.memory.corrections:
                st.markdown(
                    f"- **{c.field_name}**: "
                    f"`{c.original_value}` → `{c.corrected_value}`"
                )
            st.metric(
                "Memory Latency",
                f"{fr.memory.latency_ms:.0f} ms",
            )
        else:
            st.caption("No auto-corrections were applied.")

        if fr:
            st.metric(
                "Total Pipeline Latency",
                f"{fr.total_latency_ms:.0f} ms",
            )

    # ── Tab 4: Data Editor ────────────────────────────────────
    with tab_data:
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
        pipeline = Pipeline(lock=lock, memory=memory, groq_api_key=api_key)
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
