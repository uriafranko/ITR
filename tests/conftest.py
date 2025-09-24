"""Shared pytest fixtures and configuration for ITR tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from itr import ITR, ITRConfig
from itr.core.types import FragmentType, InstructionFragment, Tool


@pytest.fixture
def sample_config():
    """Provide a sample ITR configuration for testing."""
    return ITRConfig(
        k_a_instructions=2,
        k_b_tools=2,
        token_budget=500,
        top_m_instructions=5,
        top_m_tools=5,
        dense_weight=0.5,
        sparse_weight=0.5,
        confidence_threshold=0.7,
    )


@pytest.fixture
def sample_instruction_fragments():
    """Provide sample instruction fragments for testing."""
    return [
        InstructionFragment(
            id="test_role_1",
            content="You are a helpful AI assistant.",
            token_count=7,
            fragment_type=FragmentType.ROLE_GUIDANCE,
            metadata={"source": "base", "priority": 1},
        ),
        InstructionFragment(
            id="test_safety_1",
            content="Always prioritize safety and ethical considerations.",
            token_count=8,
            fragment_type=FragmentType.SAFETY_POLICY,
            metadata={"source": "safety", "priority": 2},
        ),
        InstructionFragment(
            id="test_style_1",
            content="Use clear and concise language in responses.",
            token_count=8,
            fragment_type=FragmentType.STYLE_RULE,
            metadata={"source": "style", "priority": 1},
        ),
        InstructionFragment(
            id="test_domain_1",
            content="When explaining technical concepts, provide step-by-step examples.",
            token_count=10,
            fragment_type=FragmentType.DOMAIN_SPECIFIC,
            metadata={"source": "technical", "priority": 1},
        ),
    ]


@pytest.fixture
def sample_tools():
    """Provide sample tools for testing."""
    return [
        Tool(
            id="calculator",
            name="calculator",
            description="Perform mathematical calculations",
            schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
            token_count=25,
        ),
        Tool(
            id="web_search",
            name="web_search",
            description="Search the web for information",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
            token_count=30,
        ),
        Tool(
            id="text_analyzer",
            name="text_analyzer",
            description="Analyze text for various properties",
            schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["sentiment", "keywords"],
                    },
                },
                "required": ["text"],
            },
            token_count=35,
        ),
    ]


@pytest.fixture
def sample_tool_specs():
    """Provide sample tool specifications as dictionaries."""
    return [
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
            "name": "file_reader",
            "description": "Read and analyze file contents",
            "schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "format": {"type": "string", "enum": ["text", "json", "csv"]},
                },
                "required": ["file_path"],
            },
        },
    ]


@pytest.fixture
def populated_itr(sample_config, sample_instruction_fragments, sample_tools):
    """Provide an ITR instance populated with test data."""
    itr = ITR(sample_config)

    # Add instruction fragments
    itr.add_instruction_fragments(sample_instruction_fragments)

    # Add tools
    for tool in sample_tools:
        itr.tool_corpus.tools[tool.id] = tool

    return itr


@pytest.fixture
def temp_instruction_file():
    """Create a temporary file with sample instructions."""
    content = """You are a helpful AI assistant specialized in data analysis.

Always be precise and accurate in your responses.

When working with data:
- Validate input data before processing
- Handle edge cases gracefully
- Provide clear explanations of your methodology

For safety considerations:
- Never execute potentially harmful operations
- Always ask for confirmation before making changes
- Respect user privacy and data confidentiality"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def temp_tools_file():
    """Create a temporary file with sample tool specifications."""
    tools = [
        {
            "name": "data_processor",
            "description": "Process and transform data",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "operation": {
                        "type": "string",
                        "enum": ["filter", "transform", "aggregate"],
                    },
                },
                "required": ["data", "operation"],
            },
            "exemplars": ["Process sales data", "Transform user records"],
        },
        {
            "name": "chart_generator",
            "description": "Generate charts and visualizations",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]},
                    "title": {"type": "string"},
                },
                "required": ["data", "chart_type"],
            },
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tools, f, indent=2)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def sample_queries():
    """Provide sample queries for testing retrieval."""
    return [
        "How do I calculate the average of a list of numbers?",
        "What is the best way to visualize time series data?",
        "Help me analyze customer sentiment from reviews",
        "How can I safely process user uploaded files?",
        "Explain the steps for data validation",
    ]


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


# Test configuration
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
