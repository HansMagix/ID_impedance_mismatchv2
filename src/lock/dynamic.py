"""
GCAW Dynamic Schema Induction — Meta-Schema & Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the meta-schema that the LLM writes *about* data schemas,
and a compiler that turns that meta-schema into a real Pydantic model
at runtime.

Two-Pass Workflow:
  1. LLM examines a screenshot and outputs a ``SchemaDefinition``
     describing the fields it observes.
  2. ``compile_pydantic_model()`` converts that definition into a
     live ``BaseModel`` subclass that the Lock can extract against.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from typing import Any, Literal, Type

from pydantic import BaseModel, Field, create_model


# ────────────────────────────────────────────────────────────────────
# Meta-Schema — the LLM writes *this*
# ────────────────────────────────────────────────────────────────────


class FieldDefinition(BaseModel):
    """A single field discovered by the LLM in the source data."""

    name: str = Field(
        ...,
        description=(
            "Python-safe field name (snake_case, no spaces). "
            "Example: 'unit_price', 'supplier_name'."
        ),
    )
    type: Literal["str", "int", "float", "bool"] = Field(
        ...,
        description=(
            "Python type for this field. Use 'str' when unsure. "
            "Use 'float' for prices/quantities with decimals."
        ),
    )
    description: str = Field(
        default="",
        description="Brief description of what this field represents.",
    )


class SchemaDefinition(BaseModel):
    """LLM-inferred schema describing a data entity on the page.

    The Lock asks the LLM to emit one of these in Pass 1.
    Pass 2 compiles this into a real Pydantic model and extracts
    the data constrained to that structure.
    """

    entity_name: str = Field(
        ...,
        description=(
            "PascalCase name for the entity. "
            "Example: 'CementProduct', 'SteelPrice'."
        ),
    )
    fields: list[FieldDefinition] = Field(
        ...,
        min_length=1,
        description="All fields the LLM can identify in the data.",
    )


# ────────────────────────────────────────────────────────────────────
# Type Mapping
# ────────────────────────────────────────────────────────────────────

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


# ────────────────────────────────────────────────────────────────────
# Compiler — SchemaDefinition → live Pydantic model
# ────────────────────────────────────────────────────────────────────


def compile_pydantic_model(
    schema_def: SchemaDefinition,
) -> Type[BaseModel]:
    """Convert a ``SchemaDefinition`` into a real Pydantic ``BaseModel``.

    Parameters
    ----------
    schema_def : SchemaDefinition
        The meta-schema emitted by the LLM in Pass 1.

    Returns
    -------
    Type[BaseModel]
        A dynamically-created Pydantic model class whose fields
        match the LLM's inferred structure.

    Example
    -------
    >>> sd = SchemaDefinition(
    ...     entity_name="CementProduct",
    ...     fields=[
    ...         FieldDefinition(name="name", type="str"),
    ...         FieldDefinition(name="price", type="float"),
    ...     ],
    ... )
    >>> Model = compile_pydantic_model(sd)
    >>> Model.__name__
    'CementProduct'
    >>> list(Model.model_fields.keys())
    ['name', 'price']
    """
    field_definitions: dict[str, Any] = {}

    for f in schema_def.fields:
        python_type: type = _TYPE_MAP.get(f.type, str)
        # All fields are Optional with None default to handle
        # missing data gracefully — LLM may not populate every cell.
        field_definitions[f.name] = (
            python_type | None,
            Field(default=None, description=f.description),
        )

    return create_model(schema_def.entity_name, **field_definitions)
