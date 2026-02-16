"""
GCAW Dynamic Schema — Meta-Schema & Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the meta-schema that the LLM uses to describe a data table's
structure, and a compiler that turns that description into a live
Pydantic model at runtime.

Two-Pass Workflow:
  1. **Pass 1 — Schema Inference:** The LLM examines an image and
     outputs a ``SchemaDefinition`` (a list of ``FieldDefinition``
     objects describing the columns/fields it sees).
  2. **Pass 2 — Data Extraction:** ``compile_pydantic_model()``
     turns the ``SchemaDefinition`` into an actual ``BaseModel``
     subclass, which is then fed back to the LLM as a grammar
     constraint for extraction.

:copyright: 2026 GCAW Project
:license: MIT
"""

from __future__ import annotations

from typing import Any, Literal, Type

from pydantic import BaseModel, Field, create_model


# ────────────────────────────────────────────────────────────────────
# Meta-Schema: How the LLM describes data structures
# ────────────────────────────────────────────────────────────────────


class FieldDefinition(BaseModel):
    """Description of a single data field discovered by the LLM.

    The ``type`` field is a string literal that maps to Python types
    during compilation.  We keep the set small to avoid confusing
    the LLM — ``str`` is the safe fallback for ambiguous types.
    """

    name: str = Field(
        ...,
        description=(
            "The column/field name, snake_case. "
            "E.g. 'product_name', 'unit_price', 'quantity'."
        ),
    )
    type: Literal["str", "int", "float", "bool"] = Field(
        default="str",
        description=(
            "Python type for this field. Use 'str' for text, "
            "'int' for whole numbers, 'float' for decimals, "
            "'bool' for true/false flags."
        ),
    )
    description: str = Field(
        default="",
        description="Brief description of what this field contains.",
    )


class SchemaDefinition(BaseModel):
    """LLM-generated description of a data table's structure.

    The ``entity_name`` becomes the Pydantic model class name.
    ``fields`` lists every column/field the LLM found in the image.
    """

    entity_name: str = Field(
        ...,
        description=(
            "Name for the data entity, PascalCase. "
            "E.g. 'CementPrice', 'InventoryItem', 'SalesRecord'."
        ),
    )
    fields: list[FieldDefinition] = Field(
        ...,
        min_length=1,
        description="List of fields/columns discovered in the data.",
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
# Compiler: SchemaDefinition → live Pydantic BaseModel
# ────────────────────────────────────────────────────────────────────


def compile_pydantic_model(
    schema_def: SchemaDefinition,
) -> Type[BaseModel]:
    """Compile a ``SchemaDefinition`` into a live Pydantic model class.

    Parameters
    ----------
    schema_def : SchemaDefinition
        The LLM-generated schema description.

    Returns
    -------
    Type[BaseModel]
        A dynamically created Pydantic model class whose fields
        match the ``SchemaDefinition``.

    Example
    -------
    >>> sd = SchemaDefinition(
    ...     entity_name="CementPrice",
    ...     fields=[
    ...         FieldDefinition(name="brand", type="str"),
    ...         FieldDefinition(name="price", type="float"),
    ...     ],
    ... )
    >>> Model = compile_pydantic_model(sd)
    >>> Model.__name__
    'CementPrice'
    >>> Model.model_fields.keys()
    dict_keys(['brand', 'price'])
    """
    field_definitions: dict[str, Any] = {}

    for f in schema_def.fields:
        python_type = _TYPE_MAP.get(f.type, str)
        # All fields are Optional with None default to handle
        # missing/ambiguous values gracefully.
        field_definitions[f.name] = (
            python_type | None,
            Field(default=None, description=f.description),
        )

    return create_model(schema_def.entity_name, **field_definitions)
