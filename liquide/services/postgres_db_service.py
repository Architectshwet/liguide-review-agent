import logging
from typing import Any

from liquide.db import postgres_repository
from liquide.db.db_singleton import DatabaseSingleton

logger = logging.getLogger(__name__)


class PostgresDBService:
    """
    Service layer for PostgreSQL conversation history operations.
    Handles business logic and delegates to repository for DB operations.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """
        Initialize the database service.
        Creates connection pool and tables.
        Should be called on application startup.
        """
        if self._initialized:
            logger.info("PostgresDBService already initialized")
            return

        try:
            # Initialize connection pool (via singleton)
            await DatabaseSingleton.get_pool()

            # Create tables if they don't exist
            await postgres_repository.initialize_conversation_table()
            await postgres_repository.initialize_session_store_table()

            self._initialized = True
            logger.info("PostgresDBService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgresDBService: {e}")
            # Don't raise - allow app to continue without conversation history
            logger.warning("App will continue without conversation history persistence")

    async def append_conversation_message(
        self, conversation_id: str, role: str, content: str, agent_name: str = "LiquideAgent"
    ) -> dict[str, Any]:
        """
        Insert a new message row into the conversation history table.

        Each message is stored as a separate row with individual columns.

        Args:
            conversation_id: Unique conversation identifier
            role: "user", "assistant", or "tool"
            content: Message content
            agent_name: Name of the agent (default: "LiquideAgent")

        Returns:
            Dict with operation result

        Table Schema:
            conversation_history (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """
        if not self._initialized:
            logger.warning("PostgresDBService not initialized, skipping message save")
            return {"success": False, "error": "Service not initialized"}

        try:
            result = await postgres_repository.append_message_to_conversation(
                conversation_id=conversation_id, role=role, content=content, agent_name=agent_name
            )

            logger.debug(f"Saved {role} message (id: {result.get('id')}) to conversation {conversation_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to append message to conversation {conversation_id}: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close database connections"""
        if self._initialized:
            await DatabaseSingleton.close()
            self._initialized = False
            logger.info("PostgresDBService closed")