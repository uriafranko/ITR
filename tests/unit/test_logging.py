"""Test cases for itr.core.logging module."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from itr.core.logging import ITRLogger, StructuredFormatter, get_logger, setup_logging


class TestStructuredFormatter:
    """Test cases for StructuredFormatter."""

    def test_format_basic_record(self):
        """Test formatting of basic log record."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "file"
        record.funcName = "test_function"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "file"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
        assert "exception" not in log_data

    def test_format_record_with_exception(self):
        """Test formatting of log record with exception info."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.module = "file"
        record.funcName = "test_function"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "traceback" in log_data["exception"]

    def test_format_record_with_extra_fields(self):
        """Test formatting of log record with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "file"
        record.funcName = "test_function"
        record.custom_field = "custom_value"
        record.operation = "test_operation"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert "extra" in log_data
        assert log_data["extra"]["custom_field"] == "custom_value"
        assert log_data["extra"]["operation"] == "test_operation"


class TestITRLogger:
    """Test cases for ITRLogger."""

    def setup_method(self):
        """Set up test method."""
        # Reset singleton state
        ITRLogger._instance = None
        ITRLogger._configured = False

        # Clear all loggers to ensure clean state
        import logging

        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("itr"):
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.setLevel(logging.NOTSET)

        # Clear root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_singleton_behavior(self):
        """Test that ITRLogger is a singleton."""
        logger1 = ITRLogger()
        logger2 = ITRLogger()
        assert logger1 is logger2

    def test_configure_console_output(self):
        """Test configuring logger for console output."""
        logger = ITRLogger()
        logger.configure(level="DEBUG", format_type="standard", output="console")

        assert logger.is_configured()
        itr_logger = logging.getLogger("itr")
        assert itr_logger.level == logging.DEBUG
        assert len(itr_logger.handlers) == 1
        assert isinstance(itr_logger.handlers[0], logging.StreamHandler)

    def test_configure_file_output(self):
        """Test configuring logger for file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = ITRLogger()
            logger.configure(
                level="INFO", format_type="structured", output="file", log_file=log_file
            )

            assert logger.is_configured()
            itr_logger = logging.getLogger("itr")
            assert len(itr_logger.handlers) == 1
            assert isinstance(itr_logger.handlers[0], logging.FileHandler)
            assert log_file.exists()

    def test_configure_both_outputs(self):
        """Test configuring logger for both console and file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = ITRLogger()
            logger.configure(level="WARNING", output="both", log_file=log_file)

            assert logger.is_configured()
            itr_logger = logging.getLogger("itr")
            assert len(itr_logger.handlers) == 2

    def test_configure_file_output_without_log_file(self):
        """Test that file output without log_file raises ValueError."""
        logger = ITRLogger()
        with pytest.raises(ValueError, match="log_file must be provided"):
            logger.configure(output="file")

    def test_configure_creates_log_directory(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "nested" / "dir" / "test.log"
            logger = ITRLogger()
            logger.configure(output="file", log_file=log_file)

            assert log_file.parent.exists()

    def test_configure_only_once(self):
        """Test that configure only works once."""
        logger = ITRLogger()
        logger.configure(level="DEBUG")

        # Second call should be ignored
        logger.configure(level="ERROR")

        itr_logger = logging.getLogger("itr")
        assert itr_logger.level == logging.DEBUG

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = ITRLogger()
        logger.configure()

        test_logger = logger.get_logger("test_module")
        assert test_logger.name == "itr.test_module"

    def test_log_performance(self):
        """Test performance logging."""
        logger = ITRLogger()
        logger.configure(level="INFO", format_type="structured")

        with patch.object(logger.get_logger("performance"), "info") as mock_info:
            logger.log_performance("test_op", 0.25, {"items": 10})

            mock_info.assert_called_once_with(
                "Performance metric",
                extra={
                    "operation": "test_op",
                    "duration_seconds": 0.25,
                    "performance_details": {"items": 10},
                },
            )

    def test_log_performance_without_details(self):
        """Test performance logging without details."""
        logger = ITRLogger()
        logger.configure()

        with patch.object(logger.get_logger("performance"), "info") as mock_info:
            logger.log_performance("test_op", 0.15)

            mock_info.assert_called_once_with(
                "Performance metric",
                extra={
                    "operation": "test_op",
                    "duration_seconds": 0.15,
                    "performance_details": {},
                },
            )

    def test_log_retrieval(self):
        """Test retrieval logging."""
        logger = ITRLogger()
        logger.configure()

        with patch.object(logger.get_logger("retrieval"), "info") as mock_info:
            logger.log_retrieval("test query", 5, 150, 0.85, True)

            mock_info.assert_called_once_with(
                "Retrieval completed",
                extra={
                    "query": "test query",
                    "query_length": 10,
                    "result_count": 5,
                    "total_tokens": 150,
                    "confidence_score": 0.85,
                    "fallback_triggered": True,
                },
            )

    def test_log_retrieval_default_fallback(self):
        """Test retrieval logging with default fallback value."""
        logger = ITRLogger()
        logger.configure()

        with patch.object(logger.get_logger("retrieval"), "info") as mock_info:
            logger.log_retrieval("test", 3, 100, 0.7)

            args, kwargs = mock_info.call_args
            assert kwargs["extra"]["fallback_triggered"] is False

    def test_log_error(self):
        """Test error logging."""
        logger = ITRLogger()
        logger.configure()

        test_error = ValueError("Test error")

        with patch.object(logger.get_logger("error"), "error") as mock_error:
            logger.log_error("test_operation", test_error, {"context": "test"})

            mock_error.assert_called_once_with(
                "Error in test_operation: Test error",
                extra={
                    "operation": "test_operation",
                    "error_type": "ValueError",
                    "error_context": {"context": "test"},
                },
                exc_info=True,
            )

    def test_log_error_without_context(self):
        """Test error logging without context."""
        logger = ITRLogger()
        logger.configure()

        test_error = RuntimeError("Runtime error")

        with patch.object(logger.get_logger("error"), "error") as mock_error:
            logger.log_error("runtime_op", test_error)

            args, kwargs = mock_error.call_args
            assert kwargs["extra"]["error_context"] == {}


class TestModuleFunctions:
    """Test module-level functions."""

    def setup_method(self):
        """Set up test method."""
        # Reset singleton state
        ITRLogger._instance = None
        ITRLogger._configured = False

        # Clear all loggers to ensure clean state
        import logging

        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("itr"):
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.setLevel(logging.NOTSET)

        # Clear root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_logging(self):
        """Test setup_logging convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging(
                level="DEBUG",
                format_type="structured",
                output="both",
                log_file=log_file,
            )

            assert isinstance(logger, ITRLogger)
            assert logger.is_configured()

    def test_get_logger_configures_if_needed(self):
        """Test that get_logger configures logging if not already configured."""
        logger_instance = get_logger("test_module")

        assert logger_instance.name == "itr.test_module"
        itr_logger_singleton = ITRLogger()
        assert itr_logger_singleton.is_configured()

    def test_get_logger_uses_existing_config(self):
        """Test that get_logger uses existing configuration."""
        # Configure first
        itr_logger = ITRLogger()
        itr_logger.configure(level="ERROR")

        # Get logger
        logger_instance = get_logger("test_module")

        assert logger_instance.name == "itr.test_module"
        itr_logger_instance = logging.getLogger("itr")
        assert itr_logger_instance.level == logging.ERROR


class TestIntegration:
    """Integration tests for logging functionality."""

    def setup_method(self):
        """Set up test method."""
        # Reset singleton state
        ITRLogger._instance = None
        ITRLogger._configured = False

        # Clear all loggers to ensure clean state
        import logging

        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("itr"):
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.setLevel(logging.NOTSET)

        # Clear root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_end_to_end_structured_logging(self):
        """Test complete structured logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Setup logging
            logger = setup_logging(
                level="DEBUG",
                format_type="structured",
                output="file",
                log_file=log_file,
            )

            # Log various types of messages
            logger.log_performance("retrieval", 0.15, {"results": 5})
            logger.log_retrieval("test query", 3, 75, 0.9)
            logger.log_error("test_op", ValueError("test"), {"file": "test.py"})

            # Read and verify log file
            log_content = log_file.read_text()
            lines = log_content.strip().split("\n")

            assert len(lines) == 3

            # Verify each log entry is valid JSON
            for line in lines:
                log_data = json.loads(line)
                assert "timestamp" in log_data
                assert "level" in log_data
                assert "message" in log_data

    def test_end_to_end_standard_logging(self):
        """Test complete standard logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Setup standard logging
            logger = setup_logging(
                level="INFO", format_type="standard", output="file", log_file=log_file
            )

            # Get a regular logger and log messages
            module_logger = logger.get_logger("test_module")
            module_logger.info("Test info message")
            module_logger.warning("Test warning message")

            # Read and verify log file
            log_content = log_file.read_text()
            lines = log_content.strip().split("\n")

            assert len(lines) == 2
            assert "Test info message" in log_content
            assert "Test warning message" in log_content
