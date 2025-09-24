"""Unit tests for exceptions module."""

import pytest

from itr.core.exceptions import (
    BudgetExceededException,
    ConfigurationException,
    CorpusException,
    ITRException,
    RetrievalException,
)


class TestITRException:
    """Tests for base ITRException class."""

    def test_basic_exception(self):
        """Test creating basic ITR exception."""
        exc = ITRException("Test error")

        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.details == {}
        assert isinstance(exc, Exception)

    def test_exception_with_details(self):
        """Test ITR exception with details."""
        details = {"code": 500, "component": "retriever"}
        exc = ITRException("Error with details", details)

        assert exc.message == "Error with details"
        assert exc.details == details
        assert exc.details["code"] == 500
        assert exc.details["component"] == "retriever"

    def test_exception_inheritance(self):
        """Test that ITRException properly inherits from Exception."""
        exc = ITRException("Test")

        assert isinstance(exc, ITRException)
        assert isinstance(exc, Exception)

    def test_exception_details_default(self):
        """Test that details defaults to empty dict."""
        exc = ITRException("Test")
        assert exc.details == {}

        # Modifying details should work
        exc.details["new_key"] = "new_value"
        assert exc.details["new_key"] == "new_value"

    def test_exception_str_representation(self):
        """Test string representation of exception."""
        exc = ITRException("Test message")
        assert str(exc) == "Test message"

    def test_exception_with_none_details(self):
        """Test exception when details is explicitly None."""
        exc = ITRException("Test", None)
        assert exc.details == {}


class TestCorpusException:
    """Tests for CorpusException."""

    def test_corpus_exception_inheritance(self):
        """Test that CorpusException inherits from ITRException."""
        exc = CorpusException("Corpus error")

        assert isinstance(exc, CorpusException)
        assert isinstance(exc, ITRException)
        assert isinstance(exc, Exception)

    def test_corpus_exception_with_file_details(self):
        """Test corpus exception with file-related details."""
        details = {
            "file_path": "/path/to/instructions.txt",
            "operation": "load",
            "error_type": "FileNotFound",
        }
        exc = CorpusException("Failed to load instructions", details)

        assert exc.message == "Failed to load instructions"
        assert exc.details["file_path"] == "/path/to/instructions.txt"
        assert exc.details["operation"] == "load"

    def test_corpus_exception_basic(self):
        """Test basic corpus exception."""
        exc = CorpusException("Invalid corpus format")

        assert str(exc) == "Invalid corpus format"
        assert exc.message == "Invalid corpus format"


class TestRetrievalException:
    """Tests for RetrievalException."""

    def test_retrieval_exception_inheritance(self):
        """Test that RetrievalException inherits from ITRException."""
        exc = RetrievalException("Retrieval failed")

        assert isinstance(exc, RetrievalException)
        assert isinstance(exc, ITRException)
        assert isinstance(exc, Exception)

    def test_retrieval_exception_with_query_details(self):
        """Test retrieval exception with query details."""
        details = {
            "query": "How to calculate average?",
            "retrieval_type": "hybrid",
            "embedding_model": "all-MiniLM-L6-v2",
            "error": "Model not found",
        }
        exc = RetrievalException("Embedding generation failed", details)

        assert exc.message == "Embedding generation failed"
        assert exc.details["query"] == "How to calculate average?"
        assert exc.details["embedding_model"] == "all-MiniLM-L6-v2"

    def test_retrieval_exception_basic(self):
        """Test basic retrieval exception."""
        exc = RetrievalException("Search index corrupted")

        assert str(exc) == "Search index corrupted"
        assert exc.details == {}


class TestBudgetExceededException:
    """Tests for BudgetExceededException."""

    def test_budget_exception_inheritance(self):
        """Test that BudgetExceededException inherits from ITRException."""
        exc = BudgetExceededException("Budget exceeded")

        assert isinstance(exc, BudgetExceededException)
        assert isinstance(exc, ITRException)
        assert isinstance(exc, Exception)

    def test_budget_exception_with_budget_details(self):
        """Test budget exception with budget-related details."""
        details = {
            "requested_tokens": 2500,
            "available_budget": 2000,
            "safety_tokens": 200,
            "selected_items": 15,
        }
        exc = BudgetExceededException("Cannot fit items in budget", details)

        assert exc.message == "Cannot fit items in budget"
        assert exc.details["requested_tokens"] == 2500
        assert exc.details["available_budget"] == 2000
        assert exc.details["safety_tokens"] == 200

    def test_budget_exception_calculation(self):
        """Test budget exception with calculated values."""
        details = {"budget": 1000, "required": 1200, "overflow": 200}
        exc = BudgetExceededException("Budget overflow", details)

        overflow = exc.details["required"] - exc.details["budget"]
        assert overflow == exc.details["overflow"]


class TestConfigurationException:
    """Tests for ConfigurationException."""

    def test_configuration_exception_inheritance(self):
        """Test that ConfigurationException inherits from ITRException."""
        exc = ConfigurationException("Invalid configuration")

        assert isinstance(exc, ConfigurationException)
        assert isinstance(exc, ITRException)
        assert isinstance(exc, Exception)

    def test_configuration_exception_with_validation_details(self):
        """Test configuration exception with validation details."""
        details = {
            "parameter": "token_budget",
            "provided_value": -100,
            "expected": "positive integer",
            "minimum_value": 50,
        }
        exc = ConfigurationException("Token budget must be positive", details)

        assert exc.message == "Token budget must be positive"
        assert exc.details["parameter"] == "token_budget"
        assert exc.details["provided_value"] == -100

    def test_configuration_exception_multiple_errors(self):
        """Test configuration exception with multiple validation errors."""
        details = {
            "errors": [
                {"field": "k_a_instructions", "issue": "must be positive"},
                {"field": "confidence_threshold", "issue": "must be between 0 and 1"},
                {"field": "embedding_model", "issue": "cannot be empty"},
            ]
        }
        exc = ConfigurationException("Multiple configuration errors", details)

        assert len(exc.details["errors"]) == 3
        assert exc.details["errors"][0]["field"] == "k_a_instructions"


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""

    def test_catch_specific_exceptions(self):
        """Test catching specific exception types."""
        # Test that specific exceptions can be caught
        with pytest.raises(CorpusException):
            raise CorpusException("Corpus error")

        with pytest.raises(RetrievalException):
            raise RetrievalException("Retrieval error")

        with pytest.raises(BudgetExceededException):
            raise BudgetExceededException("Budget error")

        with pytest.raises(ConfigurationException):
            raise ConfigurationException("Config error")

    def test_catch_base_exception(self):
        """Test catching all ITR exceptions via base class."""
        exceptions_to_test = [
            CorpusException("Corpus error"),
            RetrievalException("Retrieval error"),
            BudgetExceededException("Budget error"),
            ConfigurationException("Config error"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(ITRException):
                raise exc

    def test_catch_as_general_exception(self):
        """Test that ITR exceptions can be caught as general exceptions."""
        with pytest.raises(Exception):
            raise ITRException("General ITR error")

        with pytest.raises(Exception):
            raise CorpusException("Corpus error")

    def test_exception_chaining(self):
        """Test exception chaining functionality."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            details = {"original_error": str(e)}
            new_exc = RetrievalException("Wrapped error", details)

            assert "Original error" in new_exc.details["original_error"]

            # Test raising chained exception
            with pytest.raises(RetrievalException) as exc_info:
                raise new_exc from e

            assert exc_info.value.details["original_error"] == "Original error"
