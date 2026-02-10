"""GCAW Lock Module — Grammar-Constrained Extraction Layer.

Forces raw image data into strict Pydantic schemas via
Gemini 1.5 Flash + instructor.
"""

from src.lock.engine import GrammarLock, LockConfig, LockError
from src.lock.schemas import IndustrialItem, IndustrialItemRow

__all__ = [
    "GrammarLock",
    "LockConfig",
    "LockError",
    "IndustrialItem",
    "IndustrialItemRow",
]
