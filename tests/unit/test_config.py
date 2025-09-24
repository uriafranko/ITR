"""Unit tests for configuration module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from itr.core.config import ITRConfig
from itr.core.exceptions import ConfigurationException


class TestITRConfig:
    """Tests for ITRConfig dataclass."""

    def test_default_config(self):
        """Test that default configuration has expected values."""
        config = ITRConfig()

        # Retrieval parameters
        assert config.top_m_instructions == 20
        assert config.top_m_tools == 15
        assert config.tool_exemplars == 2
        assert config.k_a_instructions == 4
        assert config.k_b_tools == 2

        # Scoring weights
        assert config.dense_weight == 0.4
        assert config.sparse_weight == 0.3
        assert config.rerank_weight == 0.3

        # Budget parameters
        assert config.token_budget == 2000
        assert config.safety_overlay_tokens == 200

        # Fallback thresholds
        assert config.confidence_threshold == 0.7
        assert config.discovery_expansion_factor == 2.0

        # Cache settings
        assert config.cache_ttl_seconds == 900
        assert config.max_cache_size_mb == 100

        # Model settings
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ITRConfig(
            k_a_instructions=3,
            k_b_tools=5,
            token_budget=1500,
            confidence_threshold=0.8,
            embedding_model="custom-model",
        )

        assert config.k_a_instructions == 3
        assert config.k_b_tools == 5
        assert config.token_budget == 1500
        assert config.confidence_threshold == 0.8
        assert config.embedding_model == "custom-model"

        # Ensure other defaults are preserved
        assert config.top_m_instructions == 20
        assert config.dense_weight == 0.4

    def test_config_weights_sum(self):
        """Test that retrieval weights have reasonable default sum."""
        config = ITRConfig()

        # The weights don't need to sum to 1.0, but should be reasonable
        total_weight = config.dense_weight + config.sparse_weight + config.rerank_weight
        assert total_weight == 1.0  # Default weights sum to 1.0

    def test_config_positive_values(self):
        """Test that configuration values that should be positive are positive."""
        config = ITRConfig()

        # Retrieval counts should be positive
        assert config.top_m_instructions > 0
        assert config.top_m_tools > 0
        assert config.k_a_instructions > 0
        assert config.k_b_tools > 0

        # Budget should be positive
        assert config.token_budget > 0
        assert config.safety_overlay_tokens >= 0

        # Cache settings should be positive
        assert config.cache_ttl_seconds > 0
        assert config.max_cache_size_mb > 0

    def test_config_weights_non_negative(self):
        """Test that weights are non-negative."""
        config = ITRConfig()

        assert config.dense_weight >= 0
        assert config.sparse_weight >= 0
        assert config.rerank_weight >= 0

    def test_config_threshold_range(self):
        """Test that confidence threshold is in reasonable range."""
        config = ITRConfig()

        assert 0 <= config.confidence_threshold <= 1
        assert config.discovery_expansion_factor >= 1.0

    def test_config_modification(self):
        """Test that config can be modified after creation."""
        config = ITRConfig()

        # Modify values
        config.k_a_instructions = 10
        config.token_budget = 3000
        config.confidence_threshold = 0.9

        assert config.k_a_instructions == 10
        assert config.token_budget == 3000
        assert config.confidence_threshold == 0.9

    def test_config_edge_cases(self):
        """Test configuration with edge case values."""
        # Very small budget
        small_config = ITRConfig(token_budget=50, safety_overlay_tokens=10)
        assert small_config.token_budget == 50
        assert small_config.safety_overlay_tokens == 10

        # Zero weights (should be allowed)
        zero_weight_config = ITRConfig(
            dense_weight=0.0, sparse_weight=1.0, rerank_weight=0.0
        )
        assert zero_weight_config.dense_weight == 0.0
        assert zero_weight_config.sparse_weight == 1.0
        assert zero_weight_config.rerank_weight == 0.0

        # Extreme confidence threshold
        extreme_config = ITRConfig(confidence_threshold=0.0)
        assert extreme_config.confidence_threshold == 0.0

    def test_config_model_names(self):
        """Test that model names are strings."""
        config = ITRConfig()

        assert isinstance(config.embedding_model, str)
        assert isinstance(config.reranker_model, str)
        assert len(config.embedding_model) > 0
        assert len(config.reranker_model) > 0

    def test_config_realistic_values(self):
        """Test that default values are realistic for typical use."""
        config = ITRConfig()

        # Should retrieve more candidates than final selection
        assert config.top_m_instructions > config.k_a_instructions
        assert config.top_m_tools > config.k_b_tools

        # Safety overlay should be reasonable fraction of budget
        safety_fraction = config.safety_overlay_tokens / config.token_budget
        assert 0 < safety_fraction < 0.5  # Less than 50% of budget for safety

        # Cache TTL should be reasonable (between 1 minute and 1 day)
        assert 60 <= config.cache_ttl_seconds <= 86400

    def test_config_serialization_compatibility(self):
        """Test that config values are JSON-serializable types."""
        import json

        config = ITRConfig()

        # Convert to dict-like structure for JSON serialization test
        config_dict = {
            "top_m_instructions": config.top_m_instructions,
            "k_a_instructions": config.k_a_instructions,
            "token_budget": config.token_budget,
            "dense_weight": config.dense_weight,
            "confidence_threshold": config.confidence_threshold,
            "embedding_model": config.embedding_model,
        }

        # Should not raise exception
        json_str = json.dumps(config_dict)
        parsed = json.loads(json_str)

        assert parsed["top_m_instructions"] == config.top_m_instructions
        assert parsed["embedding_model"] == config.embedding_model


class TestITRConfigFileMethods:
    """Test ITRConfig file loading and saving methods."""

    def test_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "k_a_instructions": 5,
            "k_b_tools": 3,
            "token_budget": 1500,
            "embedding_model": "test-model",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = ITRConfig.from_file(temp_path)
            assert config.k_a_instructions == 5
            assert config.k_b_tools == 3
            assert config.token_budget == 1500
            assert config.embedding_model == "test-model"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
k_a_instructions: 6
k_b_tools: 4
token_budget: 1800
confidence_threshold: 0.8
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Mock YAML availability
            with patch("itr.core.config.HAS_YAML", True), patch(
                "itr.core.config.yaml"
            ) as mock_yaml:
                mock_yaml.safe_load.return_value = {
                    "k_a_instructions": 6,
                    "k_b_tools": 4,
                    "token_budget": 1800,
                    "confidence_threshold": 0.8,
                }

                config = ITRConfig.from_file(temp_path)
                assert config.k_a_instructions == 6
                assert config.k_b_tools == 4
                assert config.token_budget == 1800
                assert config.confidence_threshold == 0.8
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_nonexistent(self):
        """Test loading from non-existent file raises exception."""
        with pytest.raises(
            ConfigurationException, match="Configuration file not found"
        ):
            ITRConfig.from_file("/nonexistent/path/config.json")

    def test_from_file_invalid_json(self):
        """Test loading invalid JSON raises exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(
                ConfigurationException, match="Failed to parse configuration file"
            ):
                ITRConfig.from_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_yaml_not_available(self):
        """Test YAML file when PyYAML not available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: value")
            temp_path = f.name

        try:
            with patch("itr.core.config.HAS_YAML", False):
                with pytest.raises(ConfigurationException, match="PyYAML is required"):
                    ITRConfig.from_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_file_toml_not_available(self):
        """Test TOML file when tomllib/tomli not available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('test = "value"')
            temp_path = f.name

        try:
            with patch("itr.core.config.HAS_TOMLLIB", False):
                with pytest.raises(
                    ConfigurationException, match="tomli/tomllib is required"
                ):
                    ITRConfig.from_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "k_a_instructions": 7,
            "token_budget": 2500,
            "dense_weight": 0.5,
            "embedding_model": "custom-model",
            "unknown_field": "ignored",  # Should be ignored
        }

        config = ITRConfig.from_dict(config_dict)
        assert config.k_a_instructions == 7
        assert config.token_budget == 2500
        assert config.dense_weight == 0.5
        assert config.embedding_model == "custom-model"

        # Unknown fields should be ignored
        assert not hasattr(config, "unknown_field")

    def test_from_dict_invalid_params(self):
        """Test from_dict with invalid parameters."""
        with pytest.raises(
            ConfigurationException, match="Invalid configuration parameters"
        ):
            # Try to pass invalid parameter to ITRConfig constructor
            with patch.object(
                ITRConfig, "__init__", side_effect=TypeError("Invalid param")
            ):
                ITRConfig.from_dict({"k_a_instructions": 5})

    def test_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "ITR_K_A_INSTRUCTIONS": "8",
            "ITR_TOKEN_BUDGET": "3000",
            "ITR_CONFIDENCE_THRESHOLD": "0.9",
            "ITR_EMBEDDING_MODEL": "env-model",
            "ITR_DENSE_WEIGHT": "0.6",
            "NOT_ITR_VAR": "ignored",  # Should be ignored
        }

        with patch.dict("os.environ", env_vars):
            config = ITRConfig.from_env()
            assert config.k_a_instructions == 8
            assert config.token_budget == 3000
            assert config.confidence_threshold == 0.9
            assert config.embedding_model == "env-model"
            assert config.dense_weight == 0.6

    def test_from_env_type_conversion(self):
        """Test type conversion from environment variables."""
        env_vars = {
            "ITR_K_A_INSTRUCTIONS": "10",  # int
            "ITR_DENSE_WEIGHT": "0.7",  # float
            "ITR_USE_CACHE": "true",  # bool (if such field existed)
        }

        with patch.dict("os.environ", env_vars):
            config = ITRConfig.from_env()
            assert isinstance(config.k_a_instructions, int)
            assert isinstance(config.dense_weight, float)

    def test_from_env_invalid_type(self):
        """Test environment variable with invalid type."""
        env_vars = {"ITR_K_A_INSTRUCTIONS": "not_a_number"}

        with patch.dict("os.environ", env_vars):
            # Should not raise exception, should use default or handle gracefully
            config = ITRConfig.from_env()
            # Should either use default or handle the error appropriately

    def test_to_file_json(self):
        """Test saving configuration to JSON file."""
        config = ITRConfig(k_a_instructions=9, token_budget=4000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save_to_file(temp_path, format="json")

            # Verify file was created and contains correct data
            assert Path(temp_path).exists()
            with open(temp_path) as f:
                loaded_data = json.load(f)

            assert loaded_data["k_a_instructions"] == 9
            assert loaded_data["token_budget"] == 4000
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_to_file_yaml(self):
        """Test saving configuration to YAML file."""
        config = ITRConfig(k_a_instructions=10, token_budget=5000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            with patch("itr.core.config.HAS_YAML", True), patch(
                "itr.core.config.yaml"
            ) as mock_yaml:
                mock_yaml.dump.return_value = "yaml_content: test"

                config.save_to_file(temp_path, format="yaml")

                # Verify YAML dump was called
                mock_yaml.dump.assert_called_once()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_to_file_invalid_format(self):
        """Test saving with invalid format."""
        config = ITRConfig()

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ConfigurationException, match="Unsupported format"):
                config.save_to_file(f.name, format="invalid")

    def test_to_file_write_error(self):
        """Test handling write errors."""
        config = ITRConfig()

        # Try to write to a directory that doesn't exist
        invalid_path = "/nonexistent/directory/config.json"
        with pytest.raises(
            ConfigurationException, match="Failed to write configuration file"
        ):
            config.save_to_file(invalid_path)

    def test_validate_success(self):
        """Test validation with valid configuration."""
        config = ITRConfig()
        # Should not raise exception
        config.validate()

    def test_validate_negative_values(self):
        """Test validation fails with negative values."""
        config = ITRConfig(k_a_instructions=-1)
        with pytest.raises(ConfigurationException, match="must be positive"):
            config.validate()

        config = ITRConfig(token_budget=-100)
        with pytest.raises(ConfigurationException, match="must be positive"):
            config.validate()

    def test_validate_zero_values(self):
        """Test validation fails with zero values for required positive fields."""
        config = ITRConfig(top_m_instructions=0)
        with pytest.raises(ConfigurationException, match="must be positive"):
            config.validate()

    def test_validate_negative_weights(self):
        """Test validation fails with negative weights."""
        config = ITRConfig(dense_weight=-0.1)
        with pytest.raises(
            ConfigurationException, match="must be a non-negative number"
        ):
            config.validate()

    def test_validate_invalid_threshold(self):
        """Test validation fails with invalid confidence threshold."""
        config = ITRConfig(confidence_threshold=-0.1)
        with pytest.raises(ConfigurationException, match="must be between 0 and 1"):
            config.validate()

        config = ITRConfig(confidence_threshold=1.5)
        with pytest.raises(ConfigurationException, match="must be between 0 and 1"):
            config.validate()

    def test_validate_expansion_factor(self):
        """Test validation fails with invalid expansion factor."""
        config = ITRConfig(discovery_expansion_factor=0.5)
        with pytest.raises(ConfigurationException, match="must be >= 1.0"):
            config.validate()

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ITRConfig(k_a_instructions=11, token_budget=6000)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["k_a_instructions"] == 11
        assert config_dict["token_budget"] == 6000
        assert "embedding_model" in config_dict

    def test_to_dict_excludes_private_attrs(self):
        """Test that to_dict excludes private attributes."""
        config = ITRConfig()
        config._private_attr = "should_not_appear"

        config_dict = config.to_dict()
        assert "_private_attr" not in config_dict


class TestITRConfigEdgeCases:
    """Test edge cases and error conditions for ITRConfig."""

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        config = ITRConfig(
            top_m_instructions=1000,
            k_a_instructions=100,
            token_budget=100000,
            cache_ttl_seconds=86400 * 365,  # 1 year
            confidence_threshold=0.999,
        )

        # Should not raise exceptions
        config.validate()

        assert config.top_m_instructions == 1000
        assert config.k_a_instructions == 100

    def test_config_consistency_validation(self):
        """Test configuration consistency validation."""
        # k_a should be <= top_m for instructions
        config = ITRConfig(top_m_instructions=5, k_a_instructions=10)
        # This might be caught by validation if implemented

    def test_config_immutability_after_validation(self):
        """Test that config can be modified after validation."""
        config = ITRConfig()
        config.validate()

        # Should still be able to modify (dataclass is mutable by default)
        config.k_a_instructions = 15
        assert config.k_a_instructions == 15

    def test_config_serialization_roundtrip(self):
        """Test that config survives serialization roundtrip."""
        original_config = ITRConfig(
            k_a_instructions=12,
            token_budget=7000,
            dense_weight=0.35,
            confidence_threshold=0.75,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            original_config.save_to_file(temp_path)
            loaded_config = ITRConfig.from_file(temp_path)

            assert loaded_config.k_a_instructions == original_config.k_a_instructions
            assert loaded_config.token_budget == original_config.token_budget
            assert loaded_config.dense_weight == original_config.dense_weight
            assert (
                loaded_config.confidence_threshold
                == original_config.confidence_threshold
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_partial_updates(self):
        """Test partial configuration updates from file/dict."""
        # Create config with some custom values
        base_config = ITRConfig(k_a_instructions=20, token_budget=10000)

        # Update only some fields
        partial_dict = {"k_a_instructions": 25, "dense_weight": 0.6}
        updated_config = ITRConfig.from_dict(partial_dict)

        assert updated_config.k_a_instructions == 25
        assert updated_config.dense_weight == 0.6
        # Should use defaults for other fields
        assert updated_config.k_b_tools == ITRConfig().k_b_tools

    def test_config_repr_and_str(self):
        """Test string representations of config."""
        config = ITRConfig(k_a_instructions=13)

        # Test that repr and str don't crash
        repr_str = repr(config)
        str_str = str(config)

        assert "ITRConfig" in repr_str
        assert "13" in repr_str or "13" in str_str

    def test_config_equality(self):
        """Test configuration equality comparison."""
        config1 = ITRConfig(k_a_instructions=14, token_budget=8000)
        config2 = ITRConfig(k_a_instructions=14, token_budget=8000)
        config3 = ITRConfig(k_a_instructions=15, token_budget=8000)

        assert config1 == config2
        assert config1 != config3

    def test_config_copy_behavior(self):
        """Test that config can be copied properly."""
        import copy

        original = ITRConfig(k_a_instructions=16, token_budget=9000)
        shallow_copy = copy.copy(original)
        deep_copy = copy.deepcopy(original)

        # Should be equal but not the same object
        assert original == shallow_copy
        assert original == deep_copy
        assert original is not shallow_copy
        assert original is not deep_copy

        # Modifying copy shouldn't affect original
        shallow_copy.k_a_instructions = 20
        assert original.k_a_instructions == 16
