import os

from liquide.utils.logger import get_logger

logger = get_logger(__name__)


class LangfuseService:
    """
    Service for managing Langfuse observability integration.
    Provides singleton management of Langfuse handlers and clients.
    """

    _instance = None
    _handler: object | None = None
    _client: object | None = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize service (only runs once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._initialized = False
            self._enabled = False

    def initialize(self) -> bool:
        """
        Initialize Langfuse service.
        Should be called on application startup.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.info("LangfuseService already initialized")
            return self._enabled

        # Check if Langfuse is enabled
        langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        if not langfuse_enabled:
            logger.info("Langfuse observability is disabled (set LANGFUSE_ENABLED=true to enable)")
            self._initialized = True
            return False

        # Get required configuration
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning("Langfuse keys not configured - observability disabled")
            logger.info("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file")
            self._initialized = True
            return False

        try:
            # Import here to avoid dependency issues if langfuse not installed
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            # Initialize callback handler
            self._handler = CallbackHandler()

            # Get client for verification
            self._client = get_client()

            logger.info("Langfuse observability initialized successfully")
            logger.info(f"Host: {host}")
            logger.info(f"Public Key: {public_key[:8]}...{public_key[-4:]}")

            # Verify connection
            if self.verify_connection():
                self._enabled = True
                self._initialized = True
                return True
            else:
                logger.error("Langfuse connection verification failed")
                self._handler = None
                self._client = None
                self._initialized = True
                return False

        except ImportError:
            logger.warning("Langfuse package not installed - run: pip install langfuse")
            self._initialized = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self._initialized = True
            return False

    def verify_connection(self) -> bool:
        """
        Verify LangFuse connection and credentials.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self._client:
            return False

        try:
            if self._client.auth_check():
                logger.info("LangFuse connection verified successfully!")
                return True
            else:
                logger.error("LangFuse authentication failed!")
                return False
        except Exception as e:
            logger.error(f"LangFuse connection error: {e}")
            return False

    def get_handler(self):
        """
        Get the Langfuse callback handler for LangChain integration.

        Returns:
            CallbackHandler or None if not initialized/disabled
        """
        return self._handler

    def get_client(self):
        """
        Get the Langfuse client instance.

        Returns:
            Langfuse client or None if not initialized/disabled
        """
        return self._client

    def flush(self):
        """
        Flush any pending traces to Langfuse.
        Should be called before application shutdown.
        """
        if self._handler:
            try:
                if hasattr(self._handler, "flush"):
                    self._handler.flush()
                logger.debug("Langfuse handler traces flushed")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse handler: {e}")

        if self._client:
            try:
                if hasattr(self._client, "flush"):
                    self._client.flush()
                logger.debug("Langfuse client traces flushed")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse client: {e}")


# Singleton instance
langfuse_service = LangfuseService()