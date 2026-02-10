"""
GCAW Lock Schemas — Domain Models for Structured Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample Pydantic V2 schemas that demonstrate the Lock's generic
extraction capability.  ``IndustrialItem`` is the reference schema;
create additional schemas for new data domains by following the
same pattern:

  1. All fields nullable with ``None`` default — the model must
     output ``null`` rather than hallucinate a missing value.
  2. Every field has a ``description`` — instructor forwards these
     to Gemini as part of the JSON schema, guiding extraction.
  3. Nested models (``IndustrialItemRow``) for tabular structures.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Row-level schema
# ────────────────────────────────────────────────────────────────────


class IndustrialItemRow(BaseModel):
    """A single row extracted from an industrial data table.

    Every field is ``Optional`` — if the source image doesn't
    contain a value for a column, the model must emit ``null``
    instead of guessing.
    """

    name: str | None = Field(
        default=None,
        description="Product or component name / identifier.",
    )
    specification: str | None = Field(
        default=None,
        description="Technical specification, grade, or model number.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement (e.g., kg, ton, m³, piece).",
    )
    price: float | None = Field(
        default=None,
        description="Unit price in the local currency.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code (e.g., USD, KES, EUR).",
    )
    supplier: str | None = Field(
        default=None,
        description="Supplier or manufacturer name, if listed.",
    )


# ────────────────────────────────────────────────────────────────────
# Table-level schema
# ────────────────────────────────────────────────────────────────────


class IndustrialItem(BaseModel):
    """Structured extraction target for an industrial price/spec table.

    This is the reference schema used to prove the Lock works.
    Callers can define any ``BaseModel`` subclass and pass it to
    ``GrammarLock.extract()`` — the Lock is schema-agnostic.

    Example usage::

        result = lock.extract(
            image_data=screenshot_bytes,
            schema=IndustrialItem,
            context_text="Cement price list from Bamburi, Q3 2024",
        )
        for row in result.items:
            print(f"{row.name}: {row.price} {row.currency}/{row.unit}")
    """

    source_title: str | None = Field(
        default=None,
        description=(
            "Title or heading of the data table as visible in the image."
        ),
    )
    date_reference: str | None = Field(
        default=None,
        description=(
            "Date or period the data refers to "
            "(e.g., 'March 2024', 'Q3 2024')."
        ),
    )
    items: list[IndustrialItemRow] = Field(
        default_factory=list,
        description="Ordered list of rows extracted from the data table.",
    )
    extraction_notes: str | None = Field(
        default=None,
        description=(
            "Any caveats about data quality, missing columns, "
            "or partially obscured values."
        ),
    )
