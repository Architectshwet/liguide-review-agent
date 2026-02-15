from typing import Any

from langgraph.store.memory import InMemoryStore

from liquide.utils.logger import get_logger

logger = get_logger(__name__)

_store: InMemoryStore | None = None


def get_store() -> InMemoryStore:
    """Get or create the global store instance.

    Returns:
        InMemoryStore: The singleton store instance
    """
    global _store
    if _store is None:
        _store = InMemoryStore()
        logger.info("InMemoryStore initialized for session data")
    return _store


async def get_session(store: InMemoryStore, thread_id: str) -> dict[str, Any]:
    """
    Get the entire session data for a thread.

    Args:
        store: The InMemoryStore instance
        thread_id: The thread identifier

    Returns:
        Dict containing all session data for this thread
    """
    namespace = ("session", thread_id)

    # Search for all items in this session namespace
    # asearch() expects namespace_prefix as a positional argument
    items = await store.asearch(namespace, limit=100)

    # Build session dict from all items
    session = {}
    for item in items:
        if item.key and item.value is not None:
            session[item.key] = item.value

    return session


async def set_session(store: InMemoryStore, thread_id: str, session: dict[str, Any]) -> None:
    """
    Set the entire session data for a thread.

    Args:
        store: The InMemoryStore instance
        thread_id: The thread identifier
        session: Dict containing all session data to save
    """
    namespace = ("session", thread_id)

    # Save each key-value pair in the session
    for key, value in session.items():
        await store.aput(namespace=namespace, key=key, value=value)

        # Sync to PostgreSQL if enabled
        try:
            from liquide.db.postgres_session_sync import postgres_session_sync_service

            if postgres_session_sync_service.is_sync_enabled:
                await postgres_session_sync_service.sync_session_field(
                    namespace=namespace, key=key, value=value, thread_id=thread_id
                )
        except Exception as e:
            # Don't fail the operation if sync fails
            logger.warning(f"Failed to sync session data to PostgreSQL: {e}")


def reset_store():
    """Reset the store (useful for testing)."""
    global _store
    _store = None
    logger.info("Store reset")


def cleanup_store():
    """Cleanup the store resources."""
    global _store
    _store = None
    logger.info("Store cleanup")