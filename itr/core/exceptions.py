from typing import Optional


class ITRException(Exception):
    """Base exception for ITR system.

    All ITR-specific exceptions inherit from this base class, allowing
    for easy exception handling at the system level.

    Attributes:
        message: Human-readable error message
        details: Optional additional error details
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize ITR exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CorpusException(ITRException):
    """Exception related to corpus operations.

    Raised when errors occur during corpus loading, saving, or management
    operations such as file I/O errors, format issues, or invalid data.

    Example:
        >>> raise CorpusException(
        ...     "Failed to load instructions",
        ...     {"file_path": "/path/to/file.txt", "error": "File not found"}
        ... )
    """


class RetrievalException(ITRException):
    """Exception during retrieval operations.

    Raised when errors occur during the retrieval process, including
    embedding generation failures, search index issues, or scoring problems.

    Example:
        >>> raise RetrievalException(
        ...     "Embedding model failed",
        ...     {"model": "all-MiniLM-L6-v2", "query": "search text"}
        ... )
    """


class BudgetExceededException(ITRException):
    """Token budget exceeded during selection.

    Raised when the selection process cannot fit required items within
    the specified token budget, even after optimization attempts.

    Example:
        >>> raise BudgetExceededException(
        ...     "Cannot fit required items in budget",
        ...     {"budget": 1000, "required": 1500, "safety_tokens": 200}
        ... )
    """


class ConfigurationException(ITRException):
    """Invalid configuration parameters.

    Raised when configuration parameters are invalid, missing, or
    inconsistent with system requirements.

    Example:
        >>> raise ConfigurationException(
        ...     "Token budget must be positive",
        ...     {"provided_budget": -100, "minimum_budget": 50}
        ... )
    """
