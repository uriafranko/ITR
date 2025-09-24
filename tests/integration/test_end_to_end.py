"""End-to-end integration tests for ITR system."""

import json
import tempfile
from pathlib import Path

import pytest

from itr import ITR, ITRConfig
from itr.core.types import FragmentType


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete ITR workflows from start to finish."""

    def test_complete_retrieval_workflow(self):
        """Test a complete retrieval workflow with instructions and tools."""
        # Setup
        config = ITRConfig(
            k_a_instructions=2, k_b_tools=2, token_budget=1000, confidence_threshold=0.5
        )
        itr = ITR(config)

        # Add some instructions
        instructions = [
            "You are a helpful AI assistant specialized in mathematics and data analysis.",
            "Always provide step-by-step explanations for complex calculations.",
            "When working with data, validate inputs before processing.",
            "For safety, never execute potentially harmful operations on user data.",
        ]

        for instruction in instructions:
            itr.add_instruction(instruction, metadata={"source": "test"})

        # Add some tools
        tools = [
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
            {
                "name": "data_analyzer",
                "description": "Analyze datasets and compute statistics",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["mean", "median", "std"],
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "chart_generator",
                "description": "Create charts and visualizations",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter"],
                        },
                    },
                },
            },
        ]

        for tool in tools:
            itr.add_tool(tool)

        # Test queries
        queries = [
            "How do I calculate the average of a list of numbers?",
            "What's the best way to visualize time series data?",
            "Help me analyze customer data safely",
        ]

        for query in queries:
            result = itr.step(query)

            # Verify result structure
            assert result.instructions is not None
            assert result.tools is not None
            assert isinstance(result.total_tokens, int)
            assert 0 <= result.confidence_score <= 1
            assert isinstance(result.fallback_triggered, bool)

            # Verify budget constraints
            assert result.total_tokens <= config.token_budget

            # Verify selection counts
            assert len(result.instructions) <= config.k_a_instructions
            assert len(result.tools) <= config.k_b_tools

            # Test prompt generation
            prompt = itr.get_prompt(query)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # The prompt contains instructions and tools, but not the query itself
            assert "## Instructions" in prompt or "## Available Tools" in prompt

    def test_fallback_mechanism(self):
        """Test that fallback mechanism works correctly."""
        # Use high confidence threshold to trigger fallback
        config = ITRConfig(
            confidence_threshold=0.95,  # Very high threshold
            discovery_expansion_factor=3.0,
            k_b_tools=1,
        )
        itr = ITR(config)

        # Add minimal content to trigger low confidence
        itr.add_instruction("Basic instruction.")

        # Add several tools
        for i in range(5):
            itr.add_tool(
                {
                    "name": f"tool_{i}",
                    "description": f"Test tool number {i}",
                    "schema": {"type": "object"},
                }
            )

        # Perform retrieval
        query = "Complex query that should trigger fallback"
        result = itr.step(query)

        # If confidence is low, trigger fallback
        if result.confidence_score < config.confidence_threshold:
            fallback_result = itr.handle_fallback(result, query)

            assert fallback_result.fallback_triggered is True
            assert len(fallback_result.tools) >= len(result.tools)
            assert fallback_result.confidence_score >= result.confidence_score

    def test_file_loading_workflow(self, temp_instruction_file, temp_tools_file):
        """Test loading instructions and tools from files."""
        itr = ITR()

        # Load from files
        itr.load_instructions(str(temp_instruction_file))
        itr.load_tools(str(temp_tools_file))

        # Verify content was loaded
        assert len(itr.instruction_corpus.get_all()) > 0
        assert len(itr.tool_corpus.get_all()) > 0

        # Test retrieval with loaded content
        result = itr.step("How do I process data safely?")

        assert len(result.instructions) > 0
        assert len(result.tools) > 0

    def test_empty_corpus_handling(self):
        """Test ITR behavior with empty corpus."""
        itr = ITR()

        # Test with no instructions or tools
        result = itr.step("Any query")

        assert result.instructions == []
        assert result.tools == []
        assert result.total_tokens >= 0  # Should include safety overlay
        assert result.confidence_score >= 0

    def test_large_content_handling(self):
        """Test ITR with large instructions and many tools."""
        config = ITRConfig(k_a_instructions=5, k_b_tools=5, token_budget=2000)
        itr = ITR(config)

        # Add a large instruction that will be chunked
        large_instruction = """This is a comprehensive guide for AI assistants working with data analysis and visualization.

When processing data:
1. Always validate input data formats and types
2. Handle missing values appropriately
3. Check for outliers and anomalies
4. Ensure data privacy and security

For mathematical calculations:
- Use appropriate numerical precision
- Handle edge cases like division by zero
- Validate mathematical expressions before evaluation
- Provide clear error messages for invalid operations

When creating visualizations:
- Choose appropriate chart types for the data
- Ensure proper scaling and labeling
- Use accessible color schemes
- Provide meaningful titles and legends

Safety considerations:
- Never execute arbitrary code from user input
- Validate all parameters before processing
- Log important operations for audit trails
- Respect user privacy and data confidentiality

Best practices for communication:
- Provide step-by-step explanations
- Use clear, non-technical language when appropriate
- Offer examples and demonstrations
- Ask for clarification when requirements are ambiguous"""

        itr.add_instruction(
            large_instruction, metadata={"source": "comprehensive_guide"}
        )

        # Add many tools
        tool_types = [
            "calculator",
            "data_processor",
            "chart_generator",
            "file_reader",
            "statistics_calculator",
            "data_validator",
            "format_converter",
            "data_cleaner",
            "visualization_helper",
            "report_generator",
        ]

        for i, tool_type in enumerate(tool_types):
            itr.add_tool(
                {
                    "name": tool_type,
                    "description": f"A {tool_type.replace('_', ' ')} tool for data operations",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "options": {"type": "object"},
                        },
                    },
                }
            )

        # Test retrieval
        result = itr.step(
            "I need to analyze sales data and create a comprehensive report"
        )

        # Should handle large content appropriately
        assert len(result.instructions) <= config.k_a_instructions
        assert len(result.tools) <= config.k_b_tools
        assert result.total_tokens <= config.token_budget

        # Should have selected relevant content
        assert len(result.instructions) > 0
        assert len(result.tools) > 0

    def test_different_query_types(self):
        """Test ITR with various types of queries."""
        config = ITRConfig(k_a_instructions=3, k_b_tools=3)
        itr = ITR(config)

        # Setup diverse content
        itr.add_instruction(
            "Handle mathematical queries with precision", metadata={"type": "math"}
        )
        itr.add_instruction(
            "For data analysis, always validate inputs", metadata={"type": "data"}
        )
        itr.add_instruction(
            "Provide step-by-step explanations", metadata={"type": "communication"}
        )
        itr.add_instruction(
            "Prioritize user safety and data privacy", metadata={"type": "safety"}
        )

        math_tool = {
            "name": "math_calculator",
            "description": "Perform complex mathematical operations",
            "schema": {"type": "object"},
        }
        data_tool = {
            "name": "data_analyzer",
            "description": "Analyze and process datasets",
            "schema": {"type": "object"},
        }
        viz_tool = {
            "name": "visualizer",
            "description": "Create charts and graphs",
            "schema": {"type": "object"},
        }

        for tool in [math_tool, data_tool, viz_tool]:
            itr.add_tool(tool)

        # Test different query types
        test_queries = [
            ("Calculate the derivative of x^2 + 3x + 1", "math"),
            ("Analyze customer purchase patterns", "data"),
            ("Create a bar chart of sales data", "visualization"),
            ("How do I safely process user uploaded files?", "safety"),
            ("What's the best approach for data validation?", "methodology"),
        ]

        for query, query_type in test_queries:
            result = itr.step(query)

            # Basic validations
            assert isinstance(result.instructions, list)
            assert isinstance(result.tools, list)
            assert result.total_tokens > 0

            # Should retrieve some relevant content
            if query_type in ["math", "data", "visualization"]:
                assert len(result.tools) > 0

    def test_configuration_impact(self):
        """Test how different configurations affect retrieval results."""
        # Setup base content
        instructions = [
            "You are a helpful assistant",
            "Always be accurate and precise",
            "Provide detailed explanations",
            "Consider safety in all operations",
        ]

        tools = [
            {"name": "tool1", "description": "First tool", "schema": {}},
            {"name": "tool2", "description": "Second tool", "schema": {}},
            {"name": "tool3", "description": "Third tool", "schema": {}},
        ]

        # Test different configurations
        configs = [
            ITRConfig(k_a_instructions=1, k_b_tools=1, token_budget=500),
            ITRConfig(k_a_instructions=3, k_b_tools=2, token_budget=1000),
            ITRConfig(k_a_instructions=5, k_b_tools=3, token_budget=1500),
        ]

        query = "Help me with data processing"
        results = []

        for config in configs:
            itr = ITR(config)

            for instruction in instructions:
                itr.add_instruction(instruction)
            for tool in tools:
                itr.add_tool(tool)

            result = itr.step(query)
            results.append((config, result))

        # Verify that different configurations produce different results
        for i, (config, result) in enumerate(results):
            assert len(result.instructions) <= config.k_a_instructions
            assert len(result.tools) <= config.k_b_tools
            assert result.total_tokens <= config.token_budget

        # More generous configs should generally retrieve more content
        assert results[2][1].total_tokens >= results[0][1].total_tokens


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration and functionality."""

    def test_cli_components_importable(self):
        """Test that CLI components can be imported and initialized."""
        from click.testing import CliRunner

        from itr.cli.main import cli

        runner = CliRunner()

        # Test that CLI can be invoked (even if it shows help)
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ITR" in result.output

    def test_config_generation(self):
        """Test configuration file generation."""
        from click.testing import CliRunner

        from itr.cli.main import cli

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            config_path = tmp.name

        try:
            result = runner.invoke(cli, ["init-config", "--output", config_path])
            assert result.exit_code == 0

            # Verify config file was created and is valid
            assert Path(config_path).exists()

            with open(config_path) as f:
                config_data = json.load(f)

            # Verify it has expected configuration fields
            expected_fields = [
                "k_a_instructions",
                "k_b_tools",
                "token_budget",
                "embedding_model",
            ]
            for field in expected_fields:
                assert field in config_data

        finally:
            # Cleanup
            if Path(config_path).exists():
                Path(config_path).unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    def test_large_corpus_performance(self):
        """Test ITR performance with large corpus."""
        itr = ITR()

        # Add many instructions
        for i in range(100):
            instruction = (
                f"Instruction {i}: This is a test instruction for performance testing. "
                * 10
            )
            itr.add_instruction(instruction, metadata={"id": i})

        # Add many tools
        for i in range(50):
            tool = {
                "name": f"performance_tool_{i}",
                "description": f"Performance test tool {i} with detailed description. "
                * 5,
                "schema": {
                    "type": "object",
                    "properties": {f"param_{j}": {"type": "string"} for j in range(5)},
                },
            }
            itr.add_tool(tool)

        # Test retrieval performance
        import time

        queries = [
            "How do I process large datasets efficiently?",
            "What tools are available for data analysis?",
            "Explain the best practices for performance optimization",
            "Help me choose the right tool for my task",
        ]

        for query in queries:
            start_time = time.time()
            result = itr.step(query)
            end_time = time.time()

            # Verify retrieval completed successfully
            assert result.instructions is not None
            assert result.tools is not None

            # Performance should be reasonable (less than 5 seconds for this size)
            retrieval_time = end_time - start_time
            assert retrieval_time < 5.0, (
                f"Retrieval took {retrieval_time:.2f}s, which is too slow"
            )

    def test_memory_efficiency(self):
        """Test that ITR doesn't consume excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        itr = ITR()

        # Add substantial content
        for i in range(200):
            long_instruction = "This is a long instruction for memory testing. " * 50
            itr.add_instruction(long_instruction, metadata={"test_id": i})

        for i in range(100):
            itr.add_tool(
                {
                    "name": f"memory_test_tool_{i}",
                    "description": "Memory test tool description. " * 20,
                    "schema": {
                        "type": "object",
                        "properties": {
                            f"prop_{j}": {"type": "string"} for j in range(10)
                        },
                    },
                }
            )

        # Perform several retrievals
        for _ in range(10):
            result = itr.step("Memory test query for performance evaluation")
            assert result is not None

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, (
            f"Memory increased by {memory_increase:.2f}MB, which seems excessive"
        )
