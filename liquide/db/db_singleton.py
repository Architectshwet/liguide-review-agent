import asyncio
import logging
from typing import Optional

import asyncpg

from liquide.db.postgres_config import PostgresConfig

logger = logging.getLogger(__name__)


class DatabaseSingleton:
    """Singleton class to manage a single PostgreSQL connection pool"""

    _instance: Optional["DatabaseSingleton"] = None
    _pool: asyncpg.Pool | None = None
    _lock = asyncio.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """
        Get or create the connection pool.
        Thread-safe lazy initialization.
        """
        if cls._pool is None:
            async with cls._lock:
                if cls._pool is None:
                    config = PostgresConfig.from_env()
                    try:
                        logger.info(
                            f"Creating PostgreSQL connection pool to {config.host}:{config.port}/{config.database}"
                        )
                        cls._pool = await asyncpg.create_pool(
                            host=config.host,
                            port=config.port,
                            database=config.database,
                            user=config.user,
                            password=config.password,
                            min_size=config.min_connections,
                            max_size=config.max_connections,
                            command_timeout=config.command_timeout,
                        )
                        cls._initialized = True
                        logger.info("PostgreSQL connection pool created successfully")
                    except Exception as e:
                        logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                        raise
        return cls._pool

    @classmethod
    async def close(cls):
        """Close the connection pool"""
        if cls._pool is not None:
            async with cls._lock:
                if cls._pool is not None:
                    await cls._pool.close()
                    cls._pool = None
                    cls._initialized = False
                    logger.info("PostgreSQL connection pool closed")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the pool is initialized"""
        return cls._initialized