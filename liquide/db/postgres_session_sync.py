import asyncio
import os
from typing import Any

from liquide.db import postgres_repository
from liquide.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresSessionSyncService:
    """Service for syncing InMemoryStore data to PostgreSQL for debugging"""

    def __init__(self):
        # Read config from environment
        self._enabled = os.getenv("ENABLE_SESSION_SYNC", "false").lower() == "true"

        if self._enabled:
            logger.info("PostgreSQL session sync enabled - mirroring to session_store table")
        else:
            logger.info("PostgreSQL session sync disabled - using InMemoryStore only")

    @property
    def is_sync_enabled(self) -> bool:
        """Check if session sync is enabled"""
        return self._enabled

    async def sync_session_field(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        thread_id: str,
        max_retries: int = 2,
    ) -> bool:
        """
        Sync session data to PostgreSQL (non-blocking, with retry)

        Args:
            namespace: Tuple representing the namespace (e.g., ("sessions", "thread-123"))
            key: The field key (e.g., "session_data")
            value: The field value (dict)
            thread_id: Thread identifier
            max_retries: Number of retries (default: 2)

        Returns:
            True if successful, False otherwise
        """
        # If sync is disabled, exit immediately
        if not self._enabled:
            return False

        # Retry logic
        for attempt in range(max_retries + 1):
            try:
                # Use existing repository function (reuses connection pool)
                await postgres_repository.upsert_session_data(thread_id=thread_id, key=key, value=value)

                # Log success
                if attempt == 0:
                    logger.debug(f"PostgreSQL sync: {thread_id}/{key}")
                else:
                    logger.info(f"PostgreSQL sync succeeded on attempt {attempt + 1}: {thread_id}/{key}")

                return True

            except Exception as e:
                if attempt < max_retries:
                    # Calculate backoff delay
                    delay = 0.5 * (attempt + 1)  # 0.5s, 1s
                    logger.warning(
                        f"PostgreSQL sync attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final failure after all retries
                    logger.warning(f"PostgreSQL sync failed after {max_retries + 1} attempts: {thread_id}/{key} - {e}")
                    return False

        return False


# Global sync service instance
postgres_session_sync_service = PostgresSessionSyncService()