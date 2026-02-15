import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Define a mapping of log level strings to logging module constants
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


# Configure LiveKit logger separately
def configure_special_loggers():
    """Configure special loggers to appropriate levels."""
    # Set all loggers to INFO level
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("asyncio"):
            asyncio_logger = logging.getLogger(logger_name)
            asyncio_logger.setLevel(logging.WARNING)
        elif logger_name.startswith("pymongo"):
            mongo_logger = logging.getLogger(logger_name)
            mongo_logger.setLevel(logging.WARNING)
        elif logger_name.startswith("openai"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        elif logger_name.startswith("groq"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Explicitly set certain loggers to WARNING level
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("pymongo.connection").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
    logging.getLogger("groq._base_client").setLevel(logging.WARNING)

    # You can also set specific modules in a package to different levels if needed
    # For example:
    # logging.getLogger("langfuse.client").setLevel(logging.INFO)


def get_logger(logger_name, log_level=None):
    configure_special_loggers()
    # Get log level from environment variable if not provided
    if log_level is None:
        # Get the log level from environment variable, default to INFO if not set
        env_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        # Map the string to a logging level constant, default to INFO if invalid
        log_level = LOG_LEVEL_MAP.get(env_log_level, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent log propagation to avoid double logging

    # Only add handler if it doesn't already exist to prevent duplicate handlers
    if not logger.handlers:
        log_format = "%(asctime)s [%(process)d] %(levelname)s %(filename)s : %(funcName)s(%(lineno)d) - %(message)s"
        formatter = logging.Formatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Enable file logging if configured in .env
        if os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true":
            log_file = os.getenv("LOG_FILE_PATH", "app.log")
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
