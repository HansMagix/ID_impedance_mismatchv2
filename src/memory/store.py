"""
GCAW Memory Store — Semantic Self-Healing Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ChromaDB-backed correction engine that learns from human fixes
and applies them automatically to future extractions.

Design Decisions
~~~~~~~~~~~~~~~~

1. **Correction-as-Vector-Search**
   We embed the *mistake* text and store the *fix* as metadata.
   When new text arrives, we search for semantically similar past
   mistakes.  This catches typos, abbreviations, and OCR artefacts
   that exact-string matching would miss:

     "Rhno Cem" → (semantic search) → "Rhino Cem" → fix: "Rhino Cement"

2. **Field-Scoped Corrections**
   Corrections are scoped by ``field_name`` (e.g. ``"name"``,
   ``"supplier"``).  A fix for "Rhino Cem" in the ``name`` field
   won't accidentally apply to the ``supplier`` field.  This is
   enforced via ChromaDB's metadata-level ``where`` filter.

3. **Strict Cosine Distance Threshold (0.3)**
   We use cosine distance (``hnsw:space = cosine``).  Range:
   ``0.0`` = identical, ``2.0`` = opposite.  A threshold of ``0.3``
   corresponds to cosine similarity ≥ 0.85 — deliberately strict.
   Better to miss a fix than to apply the wrong one.  Tune
   ``MemoryConfig.distance_threshold`` per-deployment if needed.

4. **Deterministic IDs via SHA-256**
   ``id = sha256(field_name + ":" + normalised_mistake)[:16]``.
   Consequences:
     - Re-learning the same mistake is an idempotent **upsert**.
     - Different fields can store independent corrections for
       identical raw text.
     - No UUID proliferation, no external ID generators.

5. **Default Embeddings**
   ChromaDB ships ``all-MiniLM-L6-v2`` built-in.  This is
   adequate for short industrial text (product names, specs,
   supplier codes).  Swap the embedding function at the collection
   level if extraction quality demands a heavier model.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from loguru import logger

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Immutable configuration for the MemoryStore.

    ``distance_threshold`` controls the strictness of the recall
    gate.  Lower = stricter (fewer auto-corrections, fewer false
    positives).  The default ``0.3`` is tuned for short industrial
    text with ``all-MiniLM-L6-v2`` under cosine distance.
    """

    persistence_path: str = "./chroma_db"
    collection_name: str = "corrections"
    distance_threshold: float = 0.3
    distance_space: str = "cosine"


# ────────────────────────────────────────────────────────────────────
# Domain Exception
# ────────────────────────────────────────────────────────────────────


class MemoryStoreError(Exception):
    """Raised for any unrecoverable failure in the Memory layer.

    Wraps underlying ChromaDB / embedding errors with a domain-
    meaningful message so callers never need to catch infra
    exceptions directly.
    """


# ────────────────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────────────────


class MemoryStore:
    """Semantic correction engine backed by ChromaDB.

    Implements a two-phase learning loop:

    * **Learn** — A human corrects an extraction error.  We store
      the raw mistake as the vector embedding key and the verified
      fix as metadata.
    * **Recall** — New text arrives from the Lock.  We search for
      semantically similar past mistakes.  If the closest match
      is below the distance threshold, we return the stored fix.

    Lifecycle::

        store = MemoryStore(persistence_path="./chroma_db")

        # Phase 1: Human corrects an error
        store.learn_correction(
            mistake="Rhino Cem",
            correction="Rhino Cement",
            field_name="name",
        )

        # Phase 2: Auto-correct future extractions
        fix = store.recall_correction(
            text="Rhno Cement",
            field_name="name",
        )
        # fix → "Rhino Cement"  (if distance < 0.3)
        # fix → None            (if match is too weak)
    """

    # ── Construction ────────────────────────────────────────────────

    def __init__(
        self,
        persistence_path: str = "./chroma_db",
        *,
        config: MemoryConfig | None = None,
    ) -> None:
        self._config: MemoryConfig = config or MemoryConfig(
            persistence_path=persistence_path,
        )

        # ── ChromaDB client (file-backed persistence) ──────────────
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=self._config.persistence_path,
        )

        # ── Corrections collection ─────────────────────────────────
        #    Uses cosine distance for semantic similarity.
        #    The default embedding function (all-MiniLM-L6-v2) is
        #    loaded automatically by ChromaDB.
        self._collection: Collection = (
            self._client.get_or_create_collection(
                name=self._config.collection_name,
                metadata={"hnsw:space": self._config.distance_space},
            )
        )

        logger.info(
            "MemoryStore initialised (path={}, collection={}, "
            "entries={}).",
            self._config.persistence_path,
            self._config.collection_name,
            self._collection.count(),
        )

    # ── ID Generation ───────────────────────────────────────────────

    @staticmethod
    def _make_id(field_name: str, mistake: str) -> str:
        """Deterministic document ID from field + normalised mistake.

        Ensures:
          - Same mistake + field → same ID → idempotent upsert.
          - Different fields → different IDs → independent corrections.
        """
        raw = f"{field_name}:{mistake.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ── Learning Phase ──────────────────────────────────────────────

    def learn_correction(
        self,
        mistake: str,
        correction: str,
        field_name: str,
    ) -> None:
        """Store a human-verified correction for future recall.

        Parameters
        ----------
        mistake : str
            The raw/incorrect value as extracted by the Lock.
        correction : str
            The human-verified correct value.
        field_name : str
            Schema field name (e.g. ``"name"``, ``"supplier"``).
            Corrections are scoped per field to prevent cross-field
            contamination.

        Raises
        ------
        MemoryStoreError
            If the ChromaDB upsert fails.

        Notes
        -----
        Uses **upsert**: if the same mistake + field was previously
        learned, the correction is silently updated.  No duplicates
        will accumulate.
        """
        doc_id = self._make_id(field_name, mistake)

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[mistake.strip()],
                metadatas=[
                    {
                        "fix": correction,
                        "field": field_name,
                        "original_mistake": mistake.strip(),
                    },
                ],
            )
            logger.info(
                "Learned: '{}' → '{}' (field={}, id={}).",
                mistake.strip(),
                correction,
                field_name,
                doc_id,
            )
        except Exception as exc:
            logger.exception("Failed to store correction.")
            raise MemoryStoreError(
                f"Failed to learn correction '{mistake}' → "
                f"'{correction}': {exc}"
            ) from exc

    # ── Recall Phase ────────────────────────────────────────────────

    def recall_correction(
        self,
        text: str,
        field_name: str,
    ) -> str | None:
        """Check if a semantically similar mistake was previously corrected.

        Parameters
        ----------
        text : str
            The raw extracted value to look up.
        field_name : str
            Schema field to scope the search.

        Returns
        -------
        str or None
            The stored correction if a confident match is found
            (distance < ``distance_threshold``), otherwise ``None``.
        """
        # ── Empty collection guard ─────────────────────────────────
        if self._collection.count() == 0:
            logger.trace("Collection empty — skipping recall.")
            return None

        # ── Query ──────────────────────────────────────────────────
        try:
            results: dict[str, Any] = self._collection.query(
                query_texts=[text.strip()],
                n_results=1,
                where={"field": field_name},
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            # Can happen if no documents match the where-filter
            # (ChromaDB behaviour varies by version)
            logger.debug(
                "Recall query failed for '{}' (field={}): {}.",
                text,
                field_name,
                exc,
            )
            return None

        # ── No results ─────────────────────────────────────────────
        if (
            not results
            or not results.get("ids")
            or not results["ids"][0]
        ):
            logger.trace(
                "No matches for '{}' in field={}.", text, field_name
            )
            return None

        # ── Distance gate ──────────────────────────────────────────
        distance: float = results["distances"][0][0]
        metadata: dict[str, Any] = results["metadatas"][0][0]
        matched_doc: str = results["documents"][0][0]
        fix: str | None = metadata.get("fix")

        logger.debug(
            "Nearest match: '{}' → '{}' (distance={:.4f}, "
            "threshold={}).",
            matched_doc,
            fix,
            distance,
            self._config.distance_threshold,
        )

        if distance > self._config.distance_threshold:
            logger.debug(
                "Distance {:.4f} > threshold {:.2f} — no correction.",
                distance,
                self._config.distance_threshold,
            )
            return None

        # ── Confident match — apply correction ─────────────────────
        logger.info(
            "Auto-correcting: '{}' → '{}' (distance={:.4f}, "
            "field={}).",
            text.strip(),
            fix,
            distance,
            field_name,
        )
        return fix

    # ── Diagnostics ─────────────────────────────────────────────────

    @property
    def correction_count(self) -> int:
        """Number of corrections currently stored."""
        return self._collection.count()
