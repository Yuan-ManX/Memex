from __future__ import annotations

from typing import Type, Tuple
from pydantic import BaseModel

from memex.memory.schema import (
    CategoryItem,
    MemoryCategory,
    MemoryItem,
    Resource,
    merge_scope_model,
)


# ============================================================
# InMemory Concrete Models
# ============================================================


class InMemoryResource(Resource):
    """
    Concrete resource model for the in-memory backend.
    """

    pass


class InMemoryMemoryItem(MemoryItem):
    """
    Concrete memory item model for the in-memory backend.
    """

    pass


class InMemoryMemoryCategory(MemoryCategory):
    """
    Concrete memory category model for the in-memory backend.
    """

    pass


class InMemoryCategoryItem(CategoryItem):
    """
    Concrete relation model between category and memory item.
    """

    pass


# ============================================================
# Scoped Model Builder
# ============================================================


def build_inmemory_models(
    scope_model: Type[BaseModel],
) -> Tuple[
    Type[InMemoryResource],
    Type[InMemoryMemoryCategory],
    Type[InMemoryMemoryItem],
    Type[InMemoryCategoryItem],
]:
    """
    Build scoped models for the in-memory backend.

    The scoped models dynamically merge:

        - User scope model
        - Base Memex schema model

    This allows the database models to inherit additional
    user-specific fields such as:

        user_id
        workspace_id
        agent_id
        tenant_id

    Example:

        class UserScope(BaseModel):
            user_id: str

    The resulting model becomes:

        class UserScopeMemoryItem(UserScope, MemoryItem):
            ...

    Args:
        scope_model:
            User-defined scope model

    Returns:
        Tuple of scoped models:
            (
                ResourceModel,
                MemoryCategoryModel,
                MemoryItemModel,
                CategoryItemModel
            )
    """

    resource_model = merge_scope_model(
        scope_model,
        InMemoryResource,
        name_suffix="Resource",
    )

    memory_category_model = merge_scope_model(
        scope_model,
        InMemoryMemoryCategory,
        name_suffix="MemoryCategory",
    )

    memory_item_model = merge_scope_model(
        scope_model,
        InMemoryMemoryItem,
        name_suffix="MemoryItem",
    )

    category_item_model = merge_scope_model(
        scope_model,
        InMemoryCategoryItem,
        name_suffix="CategoryItem",
    )

    return (
        resource_model,
        memory_category_model,
        memory_item_model,
        category_item_model,
    )


__all__ = [
    "InMemoryResource",
    "InMemoryMemoryItem",
    "InMemoryMemoryCategory",
    "InMemoryCategoryItem",
    "build_inmemory_models",
]
