from __future__ import annotations

import logging
from typing import Dict, List, Protocol, runtime_checkable

from memex.memory.schema import (
    CategoryItem as CategoryItemRecord,
    MemoryCategory as MemoryCategoryRecord,
    MemoryItem as MemoryItemRecord,
    Resource as ResourceRecord,
)

from memex.storage.repositories import (
    CategoryItemRepo,
    MemoryCategoryRepo,
    MemoryItemRepo,
    ResourceRepo,
)


logger = logging.getLogger(__name__)


@runtime_checkable
class Database(Protocol):
    """
    Backend-agnostic database interface.

    This protocol defines the contract required by the Memex
    memory storage layer. Any database backend must implement
    this interface to be compatible with the Memex memory system.

    Supported implementations may include:

    - In-memory database
    - SQLite
    - PostgreSQL
    - Redis
    - Distributed storage
    """

    # ------------------------------------------------------------------
    # Repository Interfaces
    # ------------------------------------------------------------------

    resource_repo: ResourceRepo

    memory_category_repo: MemoryCategoryRepo

    memory_item_repo: MemoryItemRepo

    category_item_repo: CategoryItemRepo

    # ------------------------------------------------------------------
    # In-Memory State Mirrors
    # ------------------------------------------------------------------

    resources: Dict[str, ResourceRecord]

    items: Dict[str, MemoryItemRecord]

    categories: Dict[str, MemoryCategoryRecord]

    relations: List[CategoryItemRecord]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close database connections and release resources.
        """
        ...
        

__all__ = [
    "Database",
    "ResourceRecord",
    "MemoryItemRecord",
    "MemoryCategoryRecord",
    "CategoryItemRecord",
]
