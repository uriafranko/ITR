"""Unit tests for corpus module."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from itr.core.exceptions import CorpusException
from itr.core.types import FragmentType, InstructionFragment, Tool
from itr.indexing.corpus import InstructionCorpus, ToolCorpus


class TestInstructionCorpus:
    """Tests for InstructionCorpus class."""

    def test_corpus_initialization(self):
        """Test that corpus initializes correctly."""
        corpus = InstructionCorpus()

        assert isinstance(corpus.fragments, dict)
        assert len(corpus.fragments) == 0
        assert corpus.chunk_size_range == (200, 600)
        assert corpus.chunker is not None

    def test_corpus_custom_chunk_size(self):
        """Test corpus with custom chunk size range."""
        corpus = InstructionCorpus(chunk_size_range=(100, 300))

        assert corpus.chunk_size_range == (100, 300)
        assert corpus.chunker.min_size == 100
        assert corpus.chunker.max_size == 300

    def test_add_system_prompt(self):
        """Test adding a system prompt."""
        corpus = InstructionCorpus()
        prompt = "You are a helpful assistant. Always be polite and respectful."
        metadata = {"source": "test", "version": "1.0"}

        corpus.add_system_prompt(prompt, metadata)

        assert len(corpus.fragments) > 0

        # Check that fragments have correct metadata
        for fragment in corpus.fragments.values():
            assert fragment.metadata["source"] == "test"
            assert fragment.metadata["version"] == "1.0"
            assert isinstance(fragment.content, str)
            assert fragment.token_count > 0
            assert isinstance(fragment.fragment_type, FragmentType)

    def test_add_system_prompt_no_metadata(self):
        """Test adding system prompt without metadata."""
        corpus = InstructionCorpus()
        prompt = "Simple instruction."

        corpus.add_system_prompt(prompt)

        assert len(corpus.fragments) > 0
        fragment = next(iter(corpus.fragments.values()))
        assert fragment.metadata == {}

    def test_add_fragments(self):
        """Test adding pre-created fragments."""
        corpus = InstructionCorpus()
        fragments = [
            InstructionFragment(
                id="custom_1",
                content="First custom fragment",
                token_count=4,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="custom_2",
                content="Second custom fragment",
                token_count=4,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]

        corpus.add_fragments(fragments)

        assert len(corpus.fragments) == 2
        assert "custom_1" in corpus.fragments
        assert "custom_2" in corpus.fragments
        assert corpus.fragments["custom_1"].content == "First custom fragment"
        assert corpus.fragments["custom_2"].fragment_type == FragmentType.STYLE_RULE

    def test_infer_type_role_guidance(self):
        """Test fragment type inference for role guidance."""
        corpus = InstructionCorpus()

        role_texts = [
            "Your role is to provide guidance",
            "Act as a helpful assistant",
            "You have the persona of a teacher",
        ]

        for text in role_texts:
            fragment_type = corpus._infer_type(text)
            assert fragment_type == FragmentType.ROLE_GUIDANCE

    def test_infer_type_style_rule(self):
        """Test fragment type inference for style rules."""
        corpus = InstructionCorpus()

        style_texts = [
            "Use a formal tone in your responses",
            "Format your output as JSON",
            "Maintain a professional style throughout",
        ]

        for text in style_texts:
            fragment_type = corpus._infer_type(text)
            assert fragment_type == FragmentType.STYLE_RULE

    def test_infer_type_safety_policy(self):
        """Test fragment type inference for safety policies."""
        corpus = InstructionCorpus()

        safety_texts = [
            "Never provide harmful content",
            "Always prioritize ethical considerations",
            "Safety should be your primary concern",
        ]

        for text in safety_texts:
            fragment_type = corpus._infer_type(text)
            assert fragment_type == FragmentType.SAFETY_POLICY

    def test_infer_type_exemplar(self):
        """Test fragment type inference for exemplars."""
        corpus = InstructionCorpus()

        exemplar_texts = [
            "For example, when analyzing data...",
            "e.g., consider the following case",
            "For instance, you might respond with:",
        ]

        for text in exemplar_texts:
            fragment_type = corpus._infer_type(text)
            assert fragment_type == FragmentType.EXEMPLAR

    def test_infer_type_domain_specific(self):
        """Test fragment type inference defaults to domain specific."""
        corpus = InstructionCorpus()

        generic_text = "Process the input data and return results"
        fragment_type = corpus._infer_type(generic_text)
        assert fragment_type == FragmentType.DOMAIN_SPECIFIC

    def test_get_fragment(self):
        """Test getting fragment by ID."""
        corpus = InstructionCorpus()
        fragment = InstructionFragment(
            id="test_get",
            content="Test content",
            token_count=2,
            fragment_type=FragmentType.ROLE_GUIDANCE,
        )
        corpus.add_fragments([fragment])

        retrieved = corpus.get("test_get")
        assert retrieved is not None
        assert retrieved.id == "test_get"
        assert retrieved.content == "Test content"

        # Test non-existent ID
        not_found = corpus.get("non_existent")
        assert not_found is None

    def test_get_all_fragments(self):
        """Test getting all fragments."""
        corpus = InstructionCorpus()
        fragments = [
            InstructionFragment(
                id="frag_1",
                content="First fragment",
                token_count=2,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="frag_2",
                content="Second fragment",
                token_count=2,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]
        corpus.add_fragments(fragments)

        all_fragments = corpus.get_all()
        assert len(all_fragments) == 2
        assert isinstance(all_fragments, list)

        # Check that both fragments are present
        ids = [f.id for f in all_fragments]
        assert "frag_1" in ids
        assert "frag_2" in ids

    def test_load_from_file_success(self):
        """Test successfully loading from file."""
        corpus = InstructionCorpus()
        file_content = (
            "You are a helpful AI assistant.\n\nAlways be polite and professional."
        )

        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch("pathlib.Path.exists", return_value=True):
                corpus.load_from_file(Path("test_instructions.txt"))

        assert len(corpus.fragments) > 0

        # Check that metadata includes source
        for fragment in corpus.fragments.values():
            assert fragment.metadata.get("source") == "test_instructions"

    def test_load_from_file_not_exists(self):
        """Test loading from non-existent file."""
        corpus = InstructionCorpus()

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CorpusException) as exc_info:
                corpus.load_from_file(Path("nonexistent.txt"))

            assert "File not found" in str(exc_info.value)

    def test_save_corpus(self):
        """Test saving corpus to JSON file."""
        corpus = InstructionCorpus()
        fragments = [
            InstructionFragment(
                id="save_test_1",
                content="Test content 1",
                token_count=3,
                fragment_type=FragmentType.ROLE_GUIDANCE,
                metadata={"source": "test"},
                priority=1,
            ),
            InstructionFragment(
                id="save_test_2",
                content="Test content 2",
                token_count=3,
                fragment_type=FragmentType.STYLE_RULE,
                priority=2,
            ),
        ]
        corpus.add_fragments(fragments)

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            corpus.save(Path("test_save.json"))

        # Check that file was opened for writing
        mock_file.assert_called_once_with(Path("test_save.json"), "w")

        # Check that JSON data was written
        written_calls = mock_file().write.call_args_list
        written_data = "".join(call[0][0] for call in written_calls)

        # Parse the written JSON to verify structure
        data = json.loads(written_data)
        assert len(data) == 2

        # Verify fragment data structure
        fragment_data = data[0]
        expected_fields = {
            "id",
            "content",
            "token_count",
            "fragment_type",
            "metadata",
            "priority",
        }
        assert set(fragment_data.keys()) == expected_fields


class TestToolCorpus:
    """Tests for ToolCorpus class."""

    def test_tool_corpus_initialization(self):
        """Test that tool corpus initializes correctly."""
        corpus = ToolCorpus()

        assert isinstance(corpus.tools, dict)
        assert len(corpus.tools) == 0
        assert corpus.chunker is not None

    def test_add_tool(self):
        """Test adding a tool from specification."""
        corpus = ToolCorpus()
        tool_spec = {
            "name": "calculator",
            "description": "Perform calculations",
            "schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
            "exemplars": ["2 + 2", "sqrt(16)"],
        }

        corpus.add_tool(tool_spec)

        assert len(corpus.tools) == 1
        tool = corpus.tools["calculator"]

        assert tool.name == "calculator"
        assert tool.description == "Perform calculations"
        assert tool.schema == tool_spec["schema"]
        assert tool.exemplars == ["2 + 2", "sqrt(16)"]
        assert tool.token_count > 0

    def test_add_tool_minimal_spec(self):
        """Test adding tool with minimal specification."""
        corpus = ToolCorpus()
        tool_spec = {"name": "simple_tool"}

        corpus.add_tool(tool_spec)

        assert len(corpus.tools) == 1
        tool = corpus.tools["simple_tool"]

        assert tool.name == "simple_tool"
        assert tool.description == ""
        assert tool.schema == {}
        assert tool.exemplars == []

    def test_add_tool_with_custom_id(self):
        """Test adding tool with custom ID."""
        corpus = ToolCorpus()
        tool_spec = {
            "id": "custom_calc",
            "name": "calculator",
            "description": "Custom calculator",
        }

        corpus.add_tool(tool_spec)

        assert "custom_calc" in corpus.tools
        assert corpus.tools["custom_calc"].name == "calculator"

    def test_count_tokens(self):
        """Test token counting for tool specification."""
        corpus = ToolCorpus()
        tool_spec = {
            "name": "test_tool",
            "description": "A simple test tool",
            "schema": {"type": "object", "properties": {"input": {"type": "string"}}},
        }

        token_count = corpus._count_tokens(tool_spec)

        assert isinstance(token_count, int)
        assert token_count > 0

        # Should count name, description, and schema
        # Exact count depends on tokenizer, but should be reasonable
        assert token_count > 5  # At minimum should count several words

    def test_count_tokens_minimal(self):
        """Test token counting with minimal tool spec."""
        corpus = ToolCorpus()
        tool_spec = {"name": "minimal"}

        token_count = corpus._count_tokens(tool_spec)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_tool(self):
        """Test getting tool by ID."""
        corpus = ToolCorpus()
        tool_spec = {"name": "test_get", "description": "Test tool for retrieval"}
        corpus.add_tool(tool_spec)

        retrieved = corpus.get("test_get")
        assert retrieved is not None
        assert retrieved.name == "test_get"
        assert retrieved.description == "Test tool for retrieval"

        # Test non-existent ID
        not_found = corpus.get("non_existent")
        assert not_found is None

    def test_get_all_tools(self):
        """Test getting all tools."""
        corpus = ToolCorpus()
        tool_specs = [
            {"name": "tool_1", "description": "First tool"},
            {"name": "tool_2", "description": "Second tool"},
        ]

        for spec in tool_specs:
            corpus.add_tool(spec)

        all_tools = corpus.get_all()
        assert len(all_tools) == 2
        assert isinstance(all_tools, list)

        names = [t.name for t in all_tools]
        assert "tool_1" in names
        assert "tool_2" in names

    def test_get_expanded_set(self):
        """Test getting expanded tool set."""
        corpus = ToolCorpus()

        # Add several tools to corpus
        for i in range(6):
            corpus.add_tool({"name": f"tool_{i}", "description": f"Tool {i}"})

        all_tools = corpus.get_all()
        base_tools = all_tools[:2]  # First 2 tools

        # Test expansion factor of 2.0
        expanded = corpus.get_expanded_set(base_tools, 2.0)

        assert len(expanded) <= 4  # 2 * 2.0 = 4 max
        assert len(expanded) >= len(base_tools)

        # Base tools should be included
        base_ids = {t.id for t in base_tools}
        expanded_ids = {t.id for t in expanded}
        assert base_ids.issubset(expanded_ids)

    def test_get_expanded_set_edge_cases(self):
        """Test expanded set with edge cases."""
        corpus = ToolCorpus()

        # Add single tool
        corpus.add_tool({"name": "single_tool"})
        all_tools = corpus.get_all()

        # Test with expansion factor
        expanded = corpus.get_expanded_set(all_tools, 3.0)
        assert len(expanded) == 1  # Can't expand beyond available tools

        # Test with empty base
        expanded_empty = corpus.get_expanded_set([], 2.0)
        assert len(expanded_empty) == 0

    def test_load_from_file_success(self):
        """Test successfully loading tools from JSON file."""
        corpus = ToolCorpus()
        tools_data = [
            {
                "name": "file_tool_1",
                "description": "First tool from file",
                "schema": {"type": "object"},
            },
            {"name": "file_tool_2", "description": "Second tool from file"},
        ]

        with patch("builtins.open", mock_open(read_data=json.dumps(tools_data))):
            with patch("pathlib.Path.exists", return_value=True):
                corpus.load_from_file(Path("test_tools.json"))

        assert len(corpus.tools) == 2
        assert "file_tool_1" in corpus.tools
        assert "file_tool_2" in corpus.tools
        assert corpus.tools["file_tool_1"].description == "First tool from file"

    def test_load_from_file_not_exists(self):
        """Test loading tools from non-existent file."""
        corpus = ToolCorpus()

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CorpusException) as exc_info:
                corpus.load_from_file(Path("nonexistent.json"))

            assert "File not found" in str(exc_info.value)

    def test_save_tool_corpus(self):
        """Test saving tool corpus to JSON file."""
        corpus = ToolCorpus()
        tool_specs = [
            {
                "name": "save_tool_1",
                "description": "First tool to save",
                "schema": {"type": "object"},
                "exemplars": ["example 1"],
            },
            {"name": "save_tool_2", "description": "Second tool to save"},
        ]

        for spec in tool_specs:
            corpus.add_tool(spec)

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            corpus.save(Path("test_tools_save.json"))

        # Check that file was opened for writing
        mock_file.assert_called_once_with(Path("test_tools_save.json"), "w")

        # Check that JSON data was written
        written_calls = mock_file().write.call_args_list
        written_data = "".join(call[0][0] for call in written_calls)

        # Parse the written JSON to verify structure
        data = json.loads(written_data)
        assert len(data) == 2

        # Verify tool data structure
        tool_data = data[0]
        expected_fields = {
            "id",
            "name",
            "description",
            "schema",
            "exemplars",
            "token_count",
        }
        assert set(tool_data.keys()) == expected_fields
