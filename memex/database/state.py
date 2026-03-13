from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from memex.memory.schema import (
    CategoryItem,
    MemoryCategory,
    MemoryItem,
    Resource,
)


logger = logging.getLogger(__name__)


@dataclass
class DatabaseState:
    """
    In-memory database state for Memex.

    This structure stores all loaded records during runtime and acts as the
    core state container for the storage layer.

    The state includes:

    resources
        External resources such as images, videos, or documents.

    items
        Core memory items stored in the system.

    categories
        Memory categories used to organize memory items.

    relations
        Mapping between memory items and categories.
    """

    resources: Dict[str, Resource] = field(default_factory=dict)

    items: Dict[str, MemoryItem] = field(default_factory=dict)

    categories: Dict[str, MemoryCategory] = field(default_factory=dict)

    relations: List[CategoryItem] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Resource Operations
    # ------------------------------------------------------------------

    def add_resource(self, resource: Resource) -> None:
        """Add or update a resource."""
        self.resources[resource.id] = resource

    def get_resource(self, resource_id: str) -> Resource | None:
        """Retrieve resource by ID."""
        return self.resources.get(resource_id)

    # ------------------------------------------------------------------
    # Memory Item Operations
    # ------------------------------------------------------------------

    def add_item(self, item: MemoryItem) -> None:
        """Add or update a memory item."""
        self.items[item.id] = item

    def get_item(self, item_id: str) -> MemoryItem | None:
        """Retrieve memory item by ID."""
        return self.items.get(item_id)

    # ------------------------------------------------------------------
    # Category Operations
    # ------------------------------------------------------------------

    def add_category(self, category: MemoryCategory) -> None:
        """Add or update a memory category."""
        self.categories[category.id] = category

    def get_category(self, category_id: str) -> MemoryCategory | None:
        """Retrieve category by ID."""
        return self.categories.get(category_id)

    # ------------------------------------------------------------------
    # Relation Operations
    # ------------------------------------------------------------------

    def add_relation(self, relation: CategoryItem) -> None:
        """Create relationship between memory item and category."""
        self.relations.append(relation)

    def get_item_categories(self, item_id: str) -> List[str]:
        """
        Get category IDs associated with a memory item.
        """
        return [
            r.category_id
            for r in self.relations
            if r.item_id == item_id
        ]

    def get_category_items(self, category_id: str) -> List[str]:
        """
        Get memory item IDs belonging to a category.
        """
        return [
            r.item_id
            for r in self.relations
            if r.category_id == category_id
        ]


__all__ = [
    "DatabaseState",
]
