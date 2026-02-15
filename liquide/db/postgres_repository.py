import json
import logging
from typing import Any

from liquide.db.db_singleton import DatabaseSingleton

logger = logging.getLogger(__name__)


# SQL for conversation history table - normalized structure
_CONVERSATION_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_history (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    agent_name TEXT NOT NULL DEFAULT 'LiquideAgent',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_history_conversation_id ON conversation_history(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at ON conversation_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_history_agent ON conversation_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_conversation_history_role ON conversation_history(role);
"""

# SQL for session store table - for debugging session data
_SESSION_STORE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_store (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(thread_id, key)
);

CREATE INDEX IF NOT EXISTS idx_session_store_thread_id ON session_store(thread_id);
CREATE INDEX IF NOT EXISTS idx_session_store_created_at ON session_store(created_at DESC);
"""


async def initialize_conversation_table():
    """
    Create the conversation_history table if it doesn't exist.
    Should be called on application startup.
    """
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(_CONVERSATION_HISTORY_TABLE_SQL)
            logger.info("Conversation history table initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize conversation history table: {e}")
            raise


async def initialize_session_store_table():
    """
    Create the session_store table if it doesn't exist.
    Should be called on application startup.
    """
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(_SESSION_STORE_TABLE_SQL)
            logger.info("Session store table initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize session store table: {e}")
            raise


async def append_message_to_conversation(
    conversation_id: str, role: str, content: str, agent_name: str = "LiquideAgent"
) -> dict[str, Any]:
    """
    Insert a new message row into the conversation history table.

    Args:
        conversation_id: Unique conversation identifier
        role: "user", "assistant", or "tool"
        content: Message content
        agent_name: Name of the agent (default: "LiquideAgent")

    Returns:
        Dict with operation result info
    """
    pool = await DatabaseSingleton.get_pool()

    async with pool.acquire() as conn:
        try:
            # Insert new message row
            result = await conn.fetchrow(
                """
                INSERT INTO conversation_history (
                    conversation_id,
                    agent_name,
                    role,
                    content,
                    created_at
                )
                VALUES ($1, $2, $3, $4, NOW())
                RETURNING
                    id,
                    conversation_id,
                    role,
                    created_at
            """,
                conversation_id,
                agent_name,
                role,
                content,
            )

            return {
                "id": result["id"],
                "conversation_id": result["conversation_id"],
                "role": result["role"],
                "created_at": result["created_at"].isoformat(),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to append message to conversation {conversation_id}: {e}")
            raise


async def upsert_session_data(thread_id: str, key: str, value: dict[str, Any]) -> dict[str, Any]:
    """
    Insert or update session data for a thread.
    Uses PostgreSQL's ON CONFLICT for atomic upsert.

    Args:
        thread_id: Unique thread/conversation identifier
        key: Field key (e.g., "session_data")
        value: JSON data to store

    Returns:
        Dict with operation result
    """
    pool = await DatabaseSingleton.get_pool()

    async with pool.acquire() as conn:
        try:
            # Convert dict to JSON string for PostgreSQL JSONB
            value_json = json.dumps(value)

            # Upsert using ON CONFLICT
            result = await conn.fetchrow(
                """
                INSERT INTO session_store (thread_id, key, value, created_at, updated_at)
                VALUES ($1, $2, $3::jsonb, NOW(), NOW())
                ON CONFLICT (thread_id, key)
                DO UPDATE SET
                    value = $3::jsonb,
                    updated_at = NOW()
                RETURNING id, thread_id, key, created_at, updated_at
            """,
                thread_id,
                key,
                value_json,
            )

            return {
                "id": result["id"],
                "thread_id": result["thread_id"],
                "key": result["key"],
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to upsert session data for {thread_id}/{key}: {e}")
            raise