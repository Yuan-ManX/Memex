from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from memex.config.settings import DatabaseConfig
from memex.storage.backends.inmemory import build_inmemory_database
from memex.storage.database import Database


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def build_database(
    *,
    config: DatabaseConfig,
    user_model: Type[BaseModel],
) -> Database:
    """
    Build and initialize a database backend.

    This function acts as the central factory for all Memex
    database backends.

    Supported providers
    -------------------

    inmemory
        In-memory storage. No persistence. Suitable for testing
        or lightweight local agents.

    postgres
        PostgreSQL backend with optional pgvector support for
        semantic search.

    sqlite
        SQLite file-based storage. Lightweight and portable.
    """

    provider = config.metadata_store.provider

    logger.info("Initializing database backend: %s", provider)

    # --------------------------------------------------------------
    # InMemory Database
    # --------------------------------------------------------------

    if provider == "inmemory":
        return build_inmemory_database(
            config=config,
            user_model=user_model,
        )

    # --------------------------------------------------------------
    # PostgreSQL Database
    # --------------------------------------------------------------

    if provider == "postgres":
        # Lazy import to avoid requiring postgres dependencies
        from memex.storage.backends.postgres import build_postgres_database

        return build_postgres_database(
            config=config,
            user_model=user_model,
        )

    # --------------------------------------------------------------
    # SQLite Database
    # --------------------------------------------------------------

    if provider == "sqlite":
        # Lazy import to avoid loading sqlite dependencies
        from memex.storage.backends.sqlite import build_sqlite_database

        return build_sqlite_database(
            config=config,
            user_model=user_model,
        )

    # --------------------------------------------------------------
    # Unsupported Provider
    # --------------------------------------------------------------

    raise ValueError(
        f"Unsupported metadata_store provider '{provider}'. "
        "Supported providers: inmemory, postgres, sqlite"
    )
  
