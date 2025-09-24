"""Structured logging configuration for ITR system."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": (
                    self.formatException(record.exc_info) if record.exc_info else None
                ),
            }

        # Add extra fields from the record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, ensure_ascii=False)


class ITRLogger:
    """Centralized logging configuration for ITR system."""

    _instance: Optional["ITRLogger"] = None
    _configured: bool = False

    def __new__(cls) -> "ITRLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def configure(
        self,
        level: str = "INFO",
        format_type: str = "structured",
        output: str = "console",
        log_file: Optional[Path] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure logging for ITR system.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Format type ('structured' for JSON, 'standard' for text)
            output: Output destination ('console', 'file', 'both')
            log_file: Path to log file (required if output includes 'file')
            extra_config: Additional logging configuration

        Example:
            >>> logger = ITRLogger()
            >>> logger.configure(
            ...     level="DEBUG",
            ...     format_type="structured",
            ...     output="both",
            ...     log_file=Path("itr.log")
            ... )
        """
        if self._configured:
            return

        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set level
        log_level = getattr(logging, level.upper())
        root_logger.setLevel(log_level)

        # Configure ITR logger specifically
        itr_logger = logging.getLogger("itr")
        itr_logger.setLevel(log_level)

        # Choose formatter
        formatter: logging.Formatter
        if format_type == "structured":
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Add handlers based on output preference
        if output in ("console", "both"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            itr_logger.addHandler(console_handler)

        if output in ("file", "both"):
            if log_file is None:
                raise ValueError(
                    "log_file must be provided when output includes 'file'"
                )

            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            itr_logger.addHandler(file_handler)

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the given name.

        Args:
            name: Logger name (typically module name)

        Returns:
            Logger instance configured with ITR settings

        Example:
            >>> itr_logger = ITRLogger()
            >>> logger = itr_logger.get_logger(__name__)
            >>> logger.info("This is a test message")
        """
        return logging.getLogger(f"itr.{name}")

    def log_performance(
        self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics.

        Args:
            operation: Name of the operation being measured
            duration: Duration in seconds
            details: Additional performance details

        Example:
            >>> itr_logger = ITRLogger()
            >>> itr_logger.log_performance(
            ...     "retrieval",
            ...     0.25,
            ...     {"query_length": 50, "candidates": 20}
            ... )
        """
        logger = self.get_logger("performance")
        logger.info(
            "Performance metric",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "performance_details": details or {},
            },
        )

    def log_retrieval(
        self,
        query: str,
        result_count: int,
        total_tokens: int,
        confidence: float,
        fallback_triggered: bool = False,
    ) -> None:
        """Log retrieval operation details.

        Args:
            query: The search query
            result_count: Number of results returned
            total_tokens: Total tokens in results
            confidence: Confidence score
            fallback_triggered: Whether fallback was used

        Example:
            >>> itr_logger = ITRLogger()
            >>> itr_logger.log_retrieval(
            ...     "How to calculate average?",
            ...     5,
            ...     150,
            ...     0.85
            ... )
        """
        logger = self.get_logger("retrieval")
        logger.info(
            "Retrieval completed",
            extra={
                "query": query,
                "query_length": len(query),
                "result_count": result_count,
                "total_tokens": total_tokens,
                "confidence_score": confidence,
                "fallback_triggered": fallback_triggered,
            },
        )

    def log_error(
        self, operation: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error with context.

        Args:
            operation: Operation where error occurred
            error: The exception that occurred
            context: Additional context about the error

        Example:
            >>> try:
            ...     # Some operation that might fail
            ...     pass
            ... except Exception as e:
            ...     itr_logger.log_error("corpus_loading", e, {"file": "data.txt"})
        """
        logger = self.get_logger("error")
        logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_context": context or {},
            },
            exc_info=True,
        )

    def is_configured(self) -> bool:
        """Check if logging has been configured.

        Returns:
            True if logging is configured, False otherwise
        """
        return self._configured


def setup_logging(
    level: str = "INFO",
    format_type: str = "standard",
    output: str = "console",
    log_file: Optional[Path] = None,
) -> ITRLogger:
    """Convenient function to set up ITR logging.

    Args:
        level: Logging level
        format_type: Format type ('structured' or 'standard')
        output: Output destination ('console', 'file', 'both')
        log_file: Path to log file

    Returns:
        Configured ITRLogger instance

    Example:
        >>> from itr.core.logging import setup_logging
        >>> setup_logging(level="DEBUG", output="both", log_file=Path("itr.log"))
    """
    logger = ITRLogger()
    logger.configure(
        level=level, format_type=format_type, output=output, log_file=log_file
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> from itr.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    itr_logger = ITRLogger()
    if not itr_logger.is_configured():
        itr_logger.configure()  # Use default configuration
    return itr_logger.get_logger(name)
