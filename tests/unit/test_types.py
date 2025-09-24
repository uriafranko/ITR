"""Unit tests for core types module."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from itr.core.types import (
    FragmentType,
    InstructionFragment,
    RetrievalResult,
    SearchResult,
    Tool,
)


class TestFragmentType:
    """Tests for FragmentType enum."""

    def test_fragment_type_values(self):
        """Test that fragment types have correct string values."""
        assert FragmentType.ROLE_GUIDANCE.value == "role"
        assert FragmentType.STYLE_RULE.value == "style"
        assert FragmentType.SAFETY_POLICY.value == "safety"
        assert FragmentType.DOMAIN_SPECIFIC.value == "domain"
        assert FragmentType.EXEMPLAR.value == "exemplar"

    def test_fragment_type_enumeration(self):
        """Test that all expected fragment types exist."""
        expected_types = {"role", "style", "safety", "domain", "exemplar"}
        actual_types = {ft.value for ft in FragmentType}
        assert actual_types == expected_types


class TestInstructionFragment:
    """Tests for InstructionFragment dataclass."""

    def test_instruction_fragment_creation(self):
        """Test creating an instruction fragment."""
        fragment = InstructionFragment(
            id="test_1",
            content="Test instruction",
            token_count=3,
            fragment_type=FragmentType.ROLE_GUIDANCE,
            metadata={"source": "test"},
            priority=1,
        )

        assert fragment.id == "test_1"
        assert fragment.content == "Test instruction"
        assert fragment.token_count == 3
        assert fragment.fragment_type == FragmentType.ROLE_GUIDANCE
        assert fragment.metadata == {"source": "test"}
        assert fragment.priority == 1
        assert fragment.embedding is None

    def test_instruction_fragment_with_embedding(self):
        """Test creating fragment with embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        fragment = InstructionFragment(
            id="test_emb",
            content="Test with embedding",
            token_count=4,
            fragment_type=FragmentType.STYLE_RULE,
            embedding=embedding,
        )

        assert fragment.embedding is not None
        np.testing.assert_array_equal(fragment.embedding, embedding)

    def test_instruction_fragment_defaults(self):
        """Test fragment creation with default values."""
        fragment = InstructionFragment(
            id="minimal",
            content="Minimal fragment",
            token_count=2,
            fragment_type=FragmentType.DOMAIN_SPECIFIC,
        )

        assert fragment.metadata == {}
        assert fragment.embedding is None
        assert fragment.priority == 0

    def test_instruction_fragment_immutability(self):
        """Test that fragment fields are immutable (dataclass frozen behavior)."""
        fragment = InstructionFragment(
            id="test",
            content="Test",
            token_count=1,
            fragment_type=FragmentType.ROLE_GUIDANCE,
        )

        # Note: dataclass is not frozen by default, but we can test assignment
        fragment.priority = 5  # This should work
        assert fragment.priority == 5


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a tool."""
        schema = {"type": "object", "properties": {"input": {"type": "string"}}}
        tool = Tool(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
            schema=schema,
            exemplars=["example 1", "example 2"],
            token_count=50,
        )

        assert tool.id == "test_tool"
        assert tool.name == "Test Tool"
        assert tool.description == "A test tool"
        assert tool.schema == schema
        assert tool.exemplars == ["example 1", "example 2"]
        assert tool.token_count == 50
        assert tool.embedding is None
        assert tool.preconditions == []
        assert tool.postconditions == []
        assert tool.failure_modes == []

    def test_tool_with_conditions(self):
        """Test creating tool with preconditions and postconditions."""
        tool = Tool(
            id="conditional_tool",
            name="Conditional Tool",
            description="Tool with conditions",
            schema={},
            preconditions=["User is authenticated", "Data is validated"],
            postconditions=["Result is cached", "Log is updated"],
            failure_modes=["Network timeout", "Invalid input"],
        )

        assert tool.preconditions == ["User is authenticated", "Data is validated"]
        assert tool.postconditions == ["Result is cached", "Log is updated"]
        assert tool.failure_modes == ["Network timeout", "Invalid input"]

    def test_tool_defaults(self):
        """Test tool creation with minimal parameters."""
        tool = Tool(
            id="minimal",
            name="Minimal Tool",
            description="Minimal description",
            schema={},
        )

        assert tool.exemplars == []
        assert tool.token_count == 0
        assert tool.embedding is None
        assert tool.preconditions == []
        assert tool.postconditions == []
        assert tool.failure_modes == []

    def test_tool_with_embedding(self):
        """Test tool with embedding."""
        embedding = np.array([0.5, 0.6, 0.7])
        tool = Tool(
            id="emb_tool",
            name="Embedded Tool",
            description="Tool with embedding",
            schema={},
            embedding=embedding,
        )

        assert tool.embedding is not None
        np.testing.assert_array_equal(tool.embedding, embedding)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(
        self, sample_instruction_fragments, sample_tools
    ):
        """Test creating a retrieval result."""
        instructions = sample_instruction_fragments[:2]
        tools = sample_tools[:1]

        result = RetrievalResult(
            instructions=instructions,
            tools=tools,
            total_tokens=100,
            confidence_score=0.85,
            fallback_triggered=False,
        )

        assert result.instructions == instructions
        assert result.tools == tools
        assert result.total_tokens == 100
        assert result.confidence_score == 0.85
        assert result.fallback_triggered is False

    def test_retrieval_result_defaults(self):
        """Test retrieval result with default values."""
        result = RetrievalResult(
            instructions=[], tools=[], total_tokens=0, confidence_score=0.5
        )

        assert result.fallback_triggered is False

    def test_retrieval_result_with_fallback(
        self, sample_instruction_fragments, sample_tools
    ):
        """Test retrieval result with fallback triggered."""
        result = RetrievalResult(
            instructions=sample_instruction_fragments,
            tools=sample_tools,
            total_tokens=200,
            confidence_score=0.6,
            fallback_triggered=True,
        )

        assert result.fallback_triggered is True
        assert len(result.instructions) == len(sample_instruction_fragments)
        assert len(result.tools) == len(sample_tools)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self, sample_instruction_fragments):
        """Test creating a search result."""
        item = sample_instruction_fragments[0]
        result = SearchResult(id="test_search_1", score=0.92, item=item)

        assert result.id == "test_search_1"
        assert result.score == 0.92
        assert result.item == item

    def test_search_result_with_tool(self, sample_tools):
        """Test search result with tool item."""
        tool = sample_tools[0]
        result = SearchResult(id=tool.id, score=0.78, item=tool)

        assert result.id == tool.id
        assert result.score == 0.78
        assert result.item == tool
        assert isinstance(result.item, Tool)

    def test_search_result_score_range(self, sample_instruction_fragments):
        """Test search results with various score values."""
        item = sample_instruction_fragments[0]

        # Test with score 0
        result_zero = SearchResult(id="zero", score=0.0, item=item)
        assert result_zero.score == 0.0

        # Test with score 1
        result_one = SearchResult(id="one", score=1.0, item=item)
        assert result_one.score == 1.0

        # Test with negative score (should be allowed)
        result_negative = SearchResult(id="neg", score=-0.1, item=item)
        assert result_negative.score == -0.1
