"""GCAW Memory Module — Semantic Self-Healing Layer.

Provides a ChromaDB-backed correction engine that learns from
human fixes and auto-applies them to future extractions.
"""

from src.memory.store import MemoryConfig, MemoryStore, MemoryStoreError

__all__ = ["MemoryConfig", "MemoryStore", "MemoryStoreError"]
