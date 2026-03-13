from __future__ import annotations

from typing import Any, Type
from pydantic import BaseModel

from memex.storage.database import Database
from memex.storage.backends.inmemory.models import build_inmemory_models
from memex.storage.backends.inmemory.state import InMemoryState

from memex.storage.backends.inmemory.repositories import (
    InMemoryCategoryItemRepository,
    InMemoryMemoryCategoryRepository,
    InMemoryMemoryItemRepository,
    InMemoryResourceRepository,
)

from memex.memory.schema import (
    CategoryItem,
    MemoryCategory,
    MemoryItem,
    Resource,
)

from memex.storage.repositories import (
    ResourceRepo,
    MemoryCategoryRepo,
)


class InMemoryDatabase(Database):
    """
    In-memory database backend for Memex.

    This backend stores all data in Python memory structures and
    is primarily intended for:

    - Local development
    - Testing
    - Lightweight agents
    - Ephemeral memory

    No persistence is provided.
    """

    def __init__(
        self,
        *,
        scope_model: Type[BaseModel] | None = None,
        resource_model: Type[Any] | None = None,
        memory_item_model: Type[Any] | None = None,
        memory_category_model: Type[Any] | None = None,
        category_item_model: Type[Any] | None = None,
        state: InMemoryState | None = None,
    ) -> None:

        # ---------------------------------------------------------
        # Scope model
        # ---------------------------------------------------------

        self.scope_model = scope_model or BaseModel

        (
            default_resource_model,
            default_memory_category_model,
            default_memory_item_model,
            default_category_item_model,
        ) = build_inmemory_models(self.scope_model)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------

        self.state = state or InMemoryState()

        self.resources: dict[str, Resource] = self.state.resources
        self.items: dict[str, MemoryItem] = self.state.items
        self.categories: dict[str, MemoryCategory] = self.state.categories
        self.relations: list[CategoryItem] = self.state.relations

        # ---------------------------------------------------------
        # Model resolution
        # ---------------------------------------------------------

        resource_model = resource_model or default_resource_model or Resource

        memory_item_model = (
            memory_item_model
            or default_memory_item_model
            or MemoryItem
        )

        memory_category_model = (
            memory_category_model
            or default_memory_category_model
            or MemoryCategory
        )

        category_item_model = (
            category_item_model
            or default_category_item_model
            or CategoryItem
        )

        # ---------------------------------------------------------
        # Repositories
        # ---------------------------------------------------------

        self.resource_repo: ResourceRepo = InMemoryResourceRepository(
            state=self.state,
            resource_model=resource_model,
        )

        self.memory_category_repo: MemoryCategoryRepo = (
            InMemoryMemoryCategoryRepository(
                state=self.state,
                memory_category_model=memory_category_model,
            )
        )

        self.memory_item_repo = InMemoryMemoryItemRepository(
            state=self.state,
            memory_item_model=memory_item_model,
        )

        self.category_item_repo = InMemoryCategoryItemRepository(
            state=self.state,
            category_item_model=category_item_model,
        )

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def close(self) -> None:
        """
        Close database resources.

        For in-memory backend there is nothing to release.
        """
        return None


# ============================================================
# Factory
# ============================================================


def build_inmemory_database(
    *,
    config: Any,
    user_model: Type[BaseModel],
) -> InMemoryDatabase:
    """
    Factory method for creating an InMemoryDatabase.

    Args:
        config:
            Database configuration

        user_model:
            User scope model

    Returns:
        InMemoryDatabase instance
    """

    return InMemoryDatabase(
        scope_model=user_model,
    )


__all__ = [
    "InMemoryDatabase",
    "build_inmemory_database",
]
