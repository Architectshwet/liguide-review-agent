import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from liquide.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_checkpointer: AsyncPostgresSaver | None = None
_pool: AsyncConnectionPool | None = None


def get_checkpointer() -> AsyncPostgresSaver | None:
    """Initialize and return the PostgreSQL checkpointer.

    Returns:
        AsyncPostgresSaver or None if initialization fails
    """
    global _checkpointer, _pool

    if _checkpointer is not None:
        return _checkpointer

    # Build connection string from env vars
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "liquide_db")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")

    db_uri = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    try:
        _pool = AsyncConnectionPool(
            conninfo=db_uri,
            min_size=1,
            max_size=10,
            open=False,  # Don't open immediately, open explicitly in lifespan
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )
        _checkpointer = AsyncPostgresSaver(_pool)
        logger.info(f"PostgreSQL checkpointer configured successfully to {postgres_host}:{postgres_port}/{postgres_db}")
        return _checkpointer
    except Exception as e:
        logger.error(f"Failed to configure checkpointer: {e}")
        logger.warning("Continuing without persistence. Conversation history will not be saved.")
        return None


def get_pool() -> AsyncConnectionPool | None:
    """Get the connection pool instance."""
    return _pool


async def cleanup():
    """Cleanup checkpointer resources."""
    global _pool, _checkpointer
    if _pool:
        await _pool.close()
        _pool = None
        _checkpointer = None
        logger.info("Checkpointer resources cleaned up")