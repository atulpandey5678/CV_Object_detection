"""
Structured logging setup for the Industrial Surface Defect Detection Platform.

Provides a centralized logging configuration with both console (colored) and
file (JSON-structured) handlers. Log level is configurable via the LOG_LEVEL
environment variable (default: INFO).

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started", extra={"epochs": 50})
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOGS_DIR

# =============================================================================
# Constants
# =============================================================================

LOG_FILE = LOGS_DIR / "app.log"
DEFAULT_LOG_LEVEL = "INFO"

# ANSI color codes for console output
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
}
_RESET = "\033[0m"

# Track whether logging has been initialized
_logging_initialized = False


# =============================================================================
# Custom Formatters
# =============================================================================


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log level and timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Build the base message
        msg = f"{color}{timestamp} | {record.levelname:<8}{_RESET} | {record.name} | {record.getMessage()}"

        # Append extra fields if present (skip standard LogRecord attributes)
        extra = _extract_extra(record)
        if extra:
            msg += f" | {extra}"

        # Append exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


class JSONFileFormatter(logging.Formatter):
    """Formatter that outputs structured JSON lines for file logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include extra fields
        extra = _extract_extra(record)
        if extra:
            log_entry["extra"] = extra

        # Include exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_extra(record: logging.LogRecord) -> dict:
    """Extract user-supplied extra fields from a LogRecord."""
    standard_attrs = {
        "name", "msg", "args", "created", "relativeCreated", "thread",
        "threadName", "msecs", "filename", "funcName", "levelno",
        "lineno", "module", "exc_info", "exc_text", "pathname",
        "process", "processName", "levelname", "message", "stack_info",
        "taskName",
    }
    extra = {}
    for key, value in record.__dict__.items():
        if key not in standard_attrs and not key.startswith("_"):
            extra[key] = value
    return extra


def _get_log_level() -> int:
    """Get log level from LOG_LEVEL environment variable."""
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


# =============================================================================
# Public API
# =============================================================================


def setup_logging() -> None:
    """
    Initialize the root logger with console and file handlers.

    This function is idempotent — calling it multiple times has no additional
    effect after the first invocation.

    Console handler: colored output with timestamp, level, module, and message.
    File handler: JSON-structured lines written to PROJECT_ROOT/logs/app.log.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = _get_log_level()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (StreamHandler → stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (JSON lines → logs/app.log)
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFileFormatter())
    root_logger.addHandler(file_handler)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Ensures logging is set up before returning the logger. Use this as the
    primary way to obtain loggers throughout the application.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing image", extra={"image_id": "img_001"})
    """
    setup_logging()
    return logging.getLogger(name)
