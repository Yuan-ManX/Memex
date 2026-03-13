from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Literal

import pendulum
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Memory Types
# ---------------------------------------------------------------------

MemoryType = Literal[
    "profile",
    "event",
    "knowledge",
    "behavior",
    "skill",
    "tool",
]


# ---------------------------------------------------------------------
# Hash Utilities
# ---------------------------------------------------------------------


def compute_content_hash(summary: str, memory_type: str) -> str:
    """
    Compute a stable hash for memory deduplication.

    The hash is computed from normalized summary content and memory type.

    Normalization rules
    -------------------
    - lowercase
    - trim whitespace
    - collapse repeated spaces

    Example
    -------
    "I love coffee"
    "I  love  coffee"

    → same hash
    """

    normalized = " ".join(summary.lower().split())
    content = f"{memory_type}:{normalized}"

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------
# Base Record
# ---------------------------------------------------------------------


class BaseRecord(BaseModel):
    """
    Base database record.

    All persistent Memex entities inherit from this model.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    created_at: datetime = Field(
        default_factory=lambda: pendulum.now("UTC"),
        description="Record creation timestamp",
    )

    updated_at: datetime = Field(
        default_factory=lambda: pendulum.now("UTC"),
        description="Last update timestamp",
    )


# ---------------------------------------------------------------------
# Tool Memory
# ---------------------------------------------------------------------


class ToolCallResult(BaseModel):
    """
    Represents the result of a tool invocation.

    Used by Tool Memory to track tool performance and history.
    """

    tool_name: str = Field(..., description="Tool name")

    input: Dict[str, Any] | str = Field(
        default="",
        description="Tool input parameters",
    )

    output: str = Field(
        default="",
        description="Tool output result",
    )

    success: bool = Field(
        default=True,
        description="Whether the tool call succeeded",
    )

    time_cost: float = Field(
        default=0.0,
        description="Execution time in seconds",
    )

    token_cost: int = Field(
        default=-1,
        description="Token consumption (-1 if unknown)",
    )

    score: float = Field(
        default=0.0,
        description="Quality score between 0.0 and 1.0",
    )

    call_hash: str = Field(
        default="",
        description="Hash for deduplication",
    )

    created_at: datetime = Field(
        default_factory=lambda: pendulum.now("UTC")
    )

    # -------------------------------------------------------------

    def generate_hash(self) -> str:
        """
        Generate MD5 hash from tool input + output.

        Used to deduplicate tool memory entries.
        """

        if isinstance(self.input, dict):
            input_str = json.dumps(self.input, sort_keys=True)
        else:
            input_str = str(self.input)

        combined = f"{self.tool_name}|{input_str}|{self.output}"

        return hashlib.md5(
            combined.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

    def ensure_hash(self) -> None:
        """
        Ensure call_hash exists.
        """

        if not self.call_hash:
            self.call_hash = self.generate_hash()


# ---------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------


class Resource(BaseRecord):
    """
    Represents an external resource linked to memories.

    Examples
    --------
    - images
    - videos
    - documents
    """

    url: str

    modality: str

    local_path: str

    caption: Optional[str] = None

    embedding: Optional[List[float]] = None


# ---------------------------------------------------------------------
# Memory Item
# ---------------------------------------------------------------------


class MemoryItem(BaseRecord):
    """
    Core memory unit stored in Memex.
    """

    resource_id: Optional[str] = None

    memory_type: MemoryType

    summary: str

    embedding: Optional[List[float]] = None

    happened_at: Optional[datetime] = None

    extra: Dict[str, Any] = Field(default_factory=dict)

    """
    extra may include:

    Reinforcement fields
    --------------------
    content_hash: str
    reinforcement_count: int
    last_reinforced_at: str

    Reference tracking
    ------------------
    ref_id: str

    Tool memory fields
    ------------------
    when_to_use: str
    metadata: dict
    tool_calls: list
    """


# ---------------------------------------------------------------------
# Memory Category
# ---------------------------------------------------------------------


class MemoryCategory(BaseRecord):
    """
    Category used to group related memories.
    """

    name: str

    description: str

    embedding: Optional[List[float]] = None

    summary: Optional[str] = None


# ---------------------------------------------------------------------
# Category Link
# ---------------------------------------------------------------------


class CategoryItem(BaseRecord):
    """
    Many-to-many relationship between memory items and categories.
    """

    item_id: str

    category_id: str


# ---------------------------------------------------------------------
# Scoped Model Builder
# ---------------------------------------------------------------------


def merge_scope_model(
    user_model: Type[BaseModel],
    core_model: Type[BaseRecord],
    *,
    name_suffix: str,
) -> Type[BaseRecord]:
    """
    Merge user scope model with core memory model.

    This allows multi-tenant memory separation
    (user_id / workspace_id / agent_id etc).
    """

    overlap = set(user_model.model_fields) & set(core_model.model_fields)

    if overlap:
        raise TypeError(
            f"Scope fields conflict with core model fields: {sorted(overlap)}"
        )

    model_name = f"{user_model.__name__}{core_model.__name__}{name_suffix}"

    return type(
        model_name,
        (user_model, core_model),
        {
            "model_config": ConfigDict(extra="allow")
        },
    )


# ---------------------------------------------------------------------
# Scoped Model Factory
# ---------------------------------------------------------------------


def build_scoped_models(
    user_model: Type[BaseModel],
) -> Tuple[
    Type[Resource],
    Type[MemoryCategory],
    Type[MemoryItem],
    Type[CategoryItem],
]:
    """
    Build scoped versions of memory models.

    Example
    -------
    UserScopeModel(user_id)

    → UserResource
    → UserMemoryItem
    → UserMemoryCategory
    """

    resource_model = merge_scope_model(
        user_model,
        Resource,
        name_suffix="Resource",
    )

    category_model = merge_scope_model(
        user_model,
        MemoryCategory,
        name_suffix="MemoryCategory",
    )

    memory_model = merge_scope_model(
        user_model,
        MemoryItem,
        name_suffix="MemoryItem",
    )

    category_item_model = merge_scope_model(
        user_model,
        CategoryItem,
        name_suffix="CategoryItem",
    )

    return (
        resource_model,
        category_model,
        memory_model,
        category_item_model,
    )


# ---------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------

__all__ = [
    "BaseRecord",
    "CategoryItem",
    "MemoryCategory",
    "MemoryItem",
    "MemoryType",
    "Resource",
    "ToolCallResult",
    "build_scoped_models",
    "compute_content_hash",
    "merge_scope_model",
]
