import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomllib  # Python 3.11+

    HAS_TOMLLIB = True
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python

        HAS_TOMLLIB = True
    except ImportError:
        HAS_TOMLLIB = False

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

from .exceptions import ConfigurationException


@dataclass
class ITRConfig:
    """Configuration for the ITR (Instruction-Tool Retrieval) system.

    Centralizes all configuration parameters for ITR operations including
    retrieval parameters, scoring weights, budget constraints, and model settings.
    All parameters have sensible defaults for typical use cases.

    Attributes:
        top_m_instructions: Number of instruction candidates to retrieve initially
        top_m_tools: Number of tool candidates to retrieve initially
        tool_exemplars: Number of exemplars to include per tool
        k_a_instructions: Maximum instructions to select for final result
        k_b_tools: Maximum tools to select for final result
        dense_weight: Weight for dense (embedding) retrieval scores
        sparse_weight: Weight for sparse (BM25) retrieval scores
        rerank_weight: Weight for reranking scores
        token_budget: Maximum total tokens for selected items
        safety_overlay_tokens: Tokens reserved for safety overlays
        confidence_threshold: Minimum confidence to avoid fallback
        discovery_expansion_factor: Multiplier for fallback tool expansion
        cache_ttl_seconds: Cache time-to-live in seconds
        max_cache_size_mb: Maximum cache size in megabytes
        embedding_model: Name of the sentence transformer model
        reranker_model: Name of the cross-encoder reranking model

    Example:
        >>> config = ITRConfig(
        ...     k_a_instructions=3,
        ...     k_b_tools=2,
        ...     token_budget=1500
        ... )
        >>> itr = ITR(config)
    """

    # Retrieval parameters
    top_m_instructions: int = 20
    top_m_tools: int = 15
    tool_exemplars: int = 2
    k_a_instructions: int = 4
    k_b_tools: int = 2

    # Scoring weights
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    rerank_weight: float = 0.3

    # Budget parameters
    token_budget: int = 2000
    safety_overlay_tokens: int = 200

    # Fallback thresholds
    confidence_threshold: float = 0.7
    discovery_expansion_factor: float = 2.0

    # Cache settings
    cache_ttl_seconds: int = 900  # 15 minutes
    max_cache_size_mb: int = 100

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Using a simpler model for now
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Environment variable mapping
    _env_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "ITR_TOP_M_INSTRUCTIONS": "top_m_instructions",
            "ITR_TOP_M_TOOLS": "top_m_tools",
            "ITR_K_A_INSTRUCTIONS": "k_a_instructions",
            "ITR_K_B_TOOLS": "k_b_tools",
            "ITR_TOKEN_BUDGET": "token_budget",
            "ITR_CONFIDENCE_THRESHOLD": "confidence_threshold",
            "ITR_EMBEDDING_MODEL": "embedding_model",
            "ITR_RERANKER_MODEL": "reranker_model",
            "ITR_CACHE_TTL_SECONDS": "cache_ttl_seconds",
            "ITR_MAX_CACHE_SIZE_MB": "max_cache_size_mb",
        },
        repr=False,
    )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ITRConfig":
        """Load configuration from a file.

        Supports JSON, YAML, and TOML formats based on file extension.

        Args:
            config_path: Path to configuration file

        Returns:
            ITRConfig instance loaded from file

        Raises:
            ConfigurationException: If file cannot be loaded or parsed

        Example:
            >>> config = ITRConfig.from_file("config.yaml")
            >>> config = ITRConfig.from_file("config.toml")
            >>> config = ITRConfig.from_file("config.json")
        """
        path = Path(config_path)

        if not path.exists():
            raise ConfigurationException(
                f"Configuration file not found: {path}", {"file_path": str(path)}
            )

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigurationException(
                f"Failed to read configuration file: {e}",
                {"file_path": str(path), "error": str(e)},
            ) from e

        # Parse based on file extension
        suffix = path.suffix.lower()

        try:
            if suffix == ".json":
                config_data = json.loads(content)
            elif suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    raise ConfigurationException(
                        "PyYAML is required for YAML configuration files",
                        {"install_command": "pip install PyYAML"},
                    )
                config_data = yaml.safe_load(content)
            elif suffix == ".toml":
                if not HAS_TOMLLIB:
                    raise ConfigurationException(
                        "tomli/tomllib is required for TOML configuration files",
                        {"install_command": "pip install tomli"},
                    )
                config_data = tomllib.loads(content)
            else:
                # Try to parse as JSON by default
                config_data = json.loads(content)

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationException(
                f"Failed to parse configuration file: {e}",
                {"file_path": str(path), "format": suffix, "error": str(e)},
            ) from e

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ITRConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ITRConfig instance

        Example:
            >>> config_data = {"k_a_instructions": 5, "token_budget": 1500}
            >>> config = ITRConfig.from_dict(config_data)
        """
        # Filter out unknown keys and validate types
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {
            k: v
            for k, v in config_dict.items()
            if k in valid_fields and not k.startswith("_")
        }

        try:
            return cls(**filtered_dict)
        except TypeError as e:
            raise ConfigurationException(
                f"Invalid configuration parameters: {e}",
                {"provided_config": config_dict, "valid_fields": list(valid_fields)},
            ) from e

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "ITRConfig":
        """Load configuration from environment variables.

        Args:
            load_dotenv_file: Whether to load from .env file if available

        Returns:
            ITRConfig instance with values from environment

        Example:
            >>> # With ITR_TOKEN_BUDGET=1500 in environment
            >>> config = ITRConfig.from_env()
            >>> assert config.token_budget == 1500
        """
        if load_dotenv_file and HAS_DOTENV:
            load_dotenv()

        config_dict = {}

        for env_var, field_name in cls._default_env_mapping().items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                field_type = cls.__dataclass_fields__[field_name].type
                try:
                    if field_type is int:
                        config_dict[field_name] = int(value)
                    elif field_type is float:
                        config_dict[field_name] = float(value)
                    elif field_type is bool:
                        config_dict[field_name] = value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    else:
                        config_dict[field_name] = value
                except ValueError as e:
                    logging.warning(f"Invalid value for {env_var}: {value} ({e})")

        return cls.from_dict(config_dict)

    @classmethod
    def from_multiple_sources(
        cls,
        config_file: Optional[Union[str, Path]] = None,
        env_override: bool = True,
        dotenv_file: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> "ITRConfig":
        """Load configuration from multiple sources with precedence.

        Precedence order (highest to lowest):
        1. extra_config dictionary
        2. Environment variables (if env_override=True)
        3. Configuration file (if provided)
        4. Default values

        Args:
            config_file: Path to configuration file
            env_override: Whether environment variables override file config
            dotenv_file: Whether to load .env file
            extra_config: Additional configuration dictionary

        Returns:
            ITRConfig instance with merged configuration

        Example:
            >>> config = ITRConfig.from_multiple_sources(
            ...     config_file="config.yaml",
            ...     env_override=True,
            ...     extra_config={"token_budget": 2000}
            ... )
        """
        # Start with default configuration
        config_dict = {}

        # Load from file if provided
        if config_file:
            file_config = cls.from_file(config_file)
            config_dict.update(file_config.to_dict())

        # Override with environment variables
        if env_override:
            env_config = cls.from_env(load_dotenv_file=dotenv_file)
            config_dict.update(
                {
                    k: v
                    for k, v in env_config.to_dict().items()
                    if v != getattr(cls(), k)  # Only override if different from default
                }
            )

        # Apply extra configuration
        if extra_config:
            config_dict.update(extra_config)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary

        Example:
            >>> config = ITRConfig(token_budget=1500)
            >>> config_dict = config.to_dict()
            >>> assert config_dict['token_budget'] == 1500
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k in self.__dataclass_fields__
        }

    def save_to_file(self, config_path: Union[str, Path], format: str = "auto") -> None:
        """Save configuration to file.

        Args:
            config_path: Output file path
            format: File format ("json", "yaml", "toml", or "auto" to detect from extension)

        Raises:
            ConfigurationException: If format is not supported or file cannot be written

        Example:
            >>> config = ITRConfig(token_budget=1500)
            >>> config.save_to_file("config.yaml")
        """
        path = Path(config_path)
        config_data = self.to_dict()

        # Determine format
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix == ".toml":
                format = "toml"
            else:
                format = "json"  # Default to JSON

        # Generate content
        try:
            if format == "json":
                content = json.dumps(config_data, indent=2, ensure_ascii=False)
            elif format == "yaml":
                if not HAS_YAML:
                    raise ConfigurationException(
                        "PyYAML is required for YAML output",
                        {"install_command": "pip install PyYAML"},
                    )
                content = yaml.dump(
                    config_data, default_flow_style=False, allow_unicode=True
                )
            elif format == "toml":
                raise ConfigurationException(
                    "TOML output is not supported (use JSON or YAML)",
                    {"suggested_formats": ["json", "yaml"]},
                )
            else:
                raise ConfigurationException(
                    f"Unsupported format: {format}",
                    {"supported_formats": ["json", "yaml"]},
                )

        except Exception as e:
            raise ConfigurationException(
                f"Failed to serialize configuration: {e}",
                {"format": format, "error": str(e)},
            ) from e

        # Write to file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise ConfigurationException(
                f"Failed to write configuration file: {e}",
                {"file_path": str(path), "error": str(e)},
            ) from e

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ConfigurationException: If configuration is invalid

        Example:
            >>> config = ITRConfig(token_budget=-100)
            >>> config.validate()  # Raises ConfigurationException
        """
        errors = []

        # Validate positive integers
        positive_int_fields = [
            "top_m_instructions",
            "top_m_tools",
            "k_a_instructions",
            "k_b_tools",
            "token_budget",
            "cache_ttl_seconds",
            "max_cache_size_mb",
        ]

        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{field_name} must be positive, got {value}")

        # Validate non-negative integers
        non_negative_fields = ["safety_overlay_tokens", "tool_exemplars"]
        for negative_field in non_negative_fields:
            value = getattr(self, negative_field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{negative_field} must be a non-negative integer, got {value}")

        # Validate weights (should be non-negative floats)
        weight_fields = ["dense_weight", "sparse_weight", "rerank_weight"]
        for negative_field in weight_fields:
            value = getattr(self, negative_field)
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{negative_field} must be a non-negative number, got {value}")

        # Validate confidence threshold (0-1 range)
        if not 0 <= self.confidence_threshold <= 1:
            errors.append(
                f"confidence_threshold must be between 0 and 1, got {self.confidence_threshold}"
            )

        # Validate discovery expansion factor
        if self.discovery_expansion_factor < 1.0:
            errors.append(
                f"discovery_expansion_factor must be >= 1.0, got {self.discovery_expansion_factor}"
            )

        # Validate model names are not empty
        if not self.embedding_model.strip():
            errors.append("embedding_model cannot be empty")
        if not self.reranker_model.strip():
            errors.append("reranker_model cannot be empty")

        # Validate selection counts don't exceed retrieval counts
        if self.k_a_instructions > self.top_m_instructions:
            errors.append(
                f"k_a_instructions ({self.k_a_instructions}) cannot exceed top_m_instructions ({self.top_m_instructions})"
            )
        if self.k_b_tools > self.top_m_tools:
            errors.append(
                f"k_b_tools ({self.k_b_tools}) cannot exceed top_m_tools ({self.top_m_tools})"
            )

        if errors:
            raise ConfigurationException(
                f"Configuration validation failed: {'; '.join(errors)}",
                {"validation_errors": errors},
            )

    @staticmethod
    def _default_env_mapping() -> Dict[str, str]:
        """Get default environment variable mapping."""
        return {
            "ITR_TOP_M_INSTRUCTIONS": "top_m_instructions",
            "ITR_TOP_M_TOOLS": "top_m_tools",
            "ITR_K_A_INSTRUCTIONS": "k_a_instructions",
            "ITR_K_B_TOOLS": "k_b_tools",
            "ITR_TOKEN_BUDGET": "token_budget",
            "ITR_CONFIDENCE_THRESHOLD": "confidence_threshold",
            "ITR_DENSE_WEIGHT": "dense_weight",
            "ITR_SPARSE_WEIGHT": "sparse_weight",
            "ITR_RERANK_WEIGHT": "rerank_weight",
            "ITR_EMBEDDING_MODEL": "embedding_model",
            "ITR_RERANKER_MODEL": "reranker_model",
            "ITR_CACHE_TTL_SECONDS": "cache_ttl_seconds",
            "ITR_MAX_CACHE_SIZE_MB": "max_cache_size_mb",
        }
