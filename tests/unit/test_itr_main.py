"""Test cases for itr.__init__ module (ITR main class)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from itr import ITR, ITRConfig
from itr.core.types import FragmentType, InstructionFragment, RetrievalResult, Tool


class TestITRInitialization:
    """Test cases for ITR class initialization."""

    def test_init_with_default_config(self):
        """Test ITR initialization with default config."""
        itr = ITR()

        assert isinstance(itr.config, ITRConfig)
        assert itr.instruction_corpus is not None
        assert itr.tool_corpus is not None
        assert itr.retriever is not None
        assert itr.selector is not None
        assert itr.prompt_builder is not None

    def test_init_with_custom_config(self):
        """Test ITR initialization with custom config."""
        custom_config = ITRConfig(
            k_a_instructions=5,
            k_b_tools=3,
            token_budget=2000,
        )
        itr = ITR(custom_config)

        assert itr.config is custom_config
        assert itr.config.k_a_instructions == 5
        assert itr.config.k_b_tools == 3
        assert itr.config.token_budget == 2000

    def test_components_use_same_config(self):
        """Test that all components receive the same config instance."""
        config = ITRConfig(token_budget=1500)
        itr = ITR(config)

        # All components should use the same config
        assert itr.retriever.config is config
        assert itr.selector.config is config
        assert itr.prompt_builder.config is config


class TestITRStep:
    """Test cases for the step method."""

    @pytest.fixture
    def itr_with_data(self):
        """Provide ITR instance with sample data."""
        config = ITRConfig(
            top_m_instructions=5,
            top_m_tools=3,
            token_budget=500,
            safety_overlay_tokens=50,
        )
        itr = ITR(config)

        # Add sample instructions
        instructions = [
            InstructionFragment(
                id="inst_1",
                content="You are helpful.",
                token_count=20,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="inst_2",
                content="Be precise.",
                token_count=15,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]
        itr.add_instruction_fragments(instructions)

        # Add sample tools
        tools = [
            {
                "id": "calc",
                "name": "calculator",
                "description": "Math calculations",
                "schema": {"type": "object"},
            },
            {
                "id": "search",
                "name": "web_search",
                "description": "Search the web",
                "schema": {"type": "object"},
            },
        ]
        for tool in tools:
            itr.add_tool(tool)

        return itr

    def test_step_basic_functionality(self, itr_with_data):
        """Test basic step functionality."""
        result = itr_with_data.step("Calculate 2+2")

        assert isinstance(result, RetrievalResult)
        assert isinstance(result.instructions, list)
        assert isinstance(result.tools, list)
        assert result.total_tokens > 0
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.fallback_triggered, bool)

    def test_step_with_empty_query(self, itr_with_data):
        """Test step with empty query."""
        result = itr_with_data.step("")

        assert isinstance(result, RetrievalResult)
        # Should still work with empty query

    def test_step_with_history(self, itr_with_data):
        """Test step with conversation history."""
        history = ["Previous question", "Previous answer"]
        result = itr_with_data.step("Follow-up question", history)

        assert isinstance(result, RetrievalResult)

    def test_step_with_none_history(self, itr_with_data):
        """Test step with None history (should default to empty list)."""
        result = itr_with_data.step("Test query", None)

        assert isinstance(result, RetrievalResult)

    def test_step_token_calculation(self, itr_with_data):
        """Test that total tokens are calculated correctly."""
        result = itr_with_data.step("Test query")

        # Calculate expected tokens
        instruction_tokens = sum(inst.token_count for inst in result.instructions)
        tool_tokens = sum(tool.token_count for tool in result.tools)
        expected_total = (
            instruction_tokens
            + tool_tokens
            + itr_with_data.config.safety_overlay_tokens
        )

        assert result.total_tokens == expected_total

    @patch("itr.retrieval.hybrid_retriever.HybridRetriever.retrieve")
    def test_step_retrieval_calls(self, mock_retrieve, itr_with_data):
        """Test that retrieval is called correctly for both corpora."""
        # Mock return values
        mock_retrieve.side_effect = [
            [InstructionFragment("test", "Test", 10, FragmentType.ROLE_GUIDANCE)],
            [Tool("test", "test", "Test", {})],
        ]

        itr_with_data.step("Test query")

        # Should be called twice - once for instructions, once for tools
        assert mock_retrieve.call_count == 2

        # Check the calls
        calls = mock_retrieve.call_args_list
        assert calls[0][0][0] == "Test query"  # query
        assert calls[0][0][1] == itr_with_data.instruction_corpus  # corpus
        assert calls[0][0][2] == itr_with_data.config.top_m_instructions  # top_m

        assert calls[1][0][0] == "Test query"  # query
        assert calls[1][0][1] == itr_with_data.tool_corpus  # corpus
        assert calls[1][0][2] == itr_with_data.config.top_m_tools  # top_m

    @patch("itr.selection.budget_selector.BudgetAwareSelector.select")
    def test_step_selection_call(self, mock_select, itr_with_data):
        """Test that selection is called correctly."""
        # Mock return values
        mock_instructions = [
            InstructionFragment("test", "Test", 10, FragmentType.ROLE_GUIDANCE)
        ]
        mock_tools = [Tool("test", "test", "Test", {})]
        mock_select.return_value = (mock_instructions, mock_tools)

        _result = itr_with_data.step("Test query")

        # Should be called once
        mock_select.assert_called_once()

        # Check arguments
        call_args = mock_select.call_args[0]
        assert len(call_args) == 3  # inst_candidates, tool_candidates, budget
        assert call_args[2] == itr_with_data.config.token_budget


class TestITRGetPrompt:
    """Test cases for the get_prompt method."""

    @pytest.fixture
    def itr_instance(self):
        """Provide basic ITR instance."""
        return ITR(ITRConfig())

    def test_get_prompt_basic(self, itr_instance):
        """Test basic get_prompt functionality."""
        prompt = itr_instance.get_prompt("Test query")

        assert isinstance(prompt, str)

    def test_get_prompt_with_history(self, itr_instance):
        """Test get_prompt with history."""
        history = ["Previous interaction"]
        prompt = itr_instance.get_prompt("Test query", history)

        assert isinstance(prompt, str)

    @patch("itr.ITR.step")
    @patch("itr.assembly.prompt_builder.PromptBuilder.assemble")
    def test_get_prompt_calls_step_and_assemble(
        self, mock_assemble, mock_step, itr_instance
    ):
        """Test that get_prompt calls step and assemble methods."""
        # Mock return values
        mock_result = RetrievalResult(
            instructions=[
                InstructionFragment("test", "Test", 10, FragmentType.ROLE_GUIDANCE)
            ],
            tools=[Tool("test", "test", "Test", {})],
            total_tokens=60,
            confidence_score=0.8,
            fallback_triggered=False,
        )
        mock_step.return_value = mock_result
        mock_assemble.return_value = "Assembled prompt"

        prompt = itr_instance.get_prompt("Test query", ["history"])

        # Check calls
        mock_step.assert_called_once_with("Test query", ["history"])
        mock_assemble.assert_called_once_with(
            mock_result.instructions, mock_result.tools
        )
        assert prompt == "Assembled prompt"


class TestITRHandleFallback:
    """Test cases for the handle_fallback method."""

    @pytest.fixture
    def itr_instance(self):
        """Provide ITR instance with config."""
        config = ITRConfig(
            discovery_expansion_factor=1.5,
            safety_overlay_tokens=50,
        )
        return ITR(config)

    @pytest.fixture
    def sample_result(self):
        """Provide sample retrieval result."""
        return RetrievalResult(
            instructions=[
                InstructionFragment(
                    "inst1", "Instruction 1", 20, FragmentType.ROLE_GUIDANCE
                ),
                InstructionFragment(
                    "inst2", "Instruction 2", 15, FragmentType.STYLE_RULE
                ),
            ],
            tools=[
                Tool("tool1", "tool1", "Tool 1", {}, token_count=25),
            ],
            total_tokens=110,
            confidence_score=0.7,
            fallback_triggered=False,
        )

    @patch("itr.indexing.corpus.ToolCorpus.get_expanded_set")
    def test_handle_fallback_basic(
        self, mock_get_expanded, itr_instance, sample_result
    ):
        """Test basic fallback handling."""
        # Mock expanded tools
        expanded_tools = [
            Tool("tool1", "tool1", "Tool 1", {}, token_count=25),
            Tool("tool2", "tool2", "Tool 2", {}, token_count=30),
        ]
        mock_get_expanded.return_value = expanded_tools

        new_result = itr_instance.handle_fallback(sample_result, "Test query")

        # Check the result
        assert isinstance(new_result, RetrievalResult)
        assert (
            new_result.instructions == sample_result.instructions
        )  # Same instructions
        assert new_result.tools == expanded_tools  # Expanded tools
        assert new_result.fallback_triggered is True
        assert new_result.confidence_score == 0.9  # Higher confidence

        # Check token calculation
        expected_tokens = 35 + 55 + 50  # instructions + tools + safety
        assert new_result.total_tokens == expected_tokens

    @patch("itr.indexing.corpus.ToolCorpus.get_expanded_set")
    def test_handle_fallback_expansion_call(
        self, mock_get_expanded, itr_instance, sample_result
    ):
        """Test that expansion is called with correct parameters."""
        mock_get_expanded.return_value = []

        itr_instance.handle_fallback(sample_result, "Test query")

        # Check the call
        mock_get_expanded.assert_called_once_with(
            sample_result.tools,
            expansion_factor=itr_instance.config.discovery_expansion_factor,
        )

    @patch("itr.indexing.corpus.ToolCorpus.get_expanded_set")
    def test_handle_fallback_empty_expanded_tools(
        self, mock_get_expanded, itr_instance, sample_result
    ):
        """Test fallback with empty expanded tools."""
        mock_get_expanded.return_value = []

        new_result = itr_instance.handle_fallback(sample_result, "Test query")

        assert new_result.tools == []
        assert new_result.fallback_triggered is True


class TestITRCorpusManagement:
    """Test cases for corpus management methods."""

    @pytest.fixture
    def itr_instance(self):
        """Provide basic ITR instance."""
        return ITR()

    def test_add_instruction(self, itr_instance):
        """Test adding a single instruction."""
        with patch.object(
            itr_instance.instruction_corpus, "add_system_prompt"
        ) as mock_add:
            itr_instance.add_instruction("Test instruction", {"source": "test"})
            mock_add.assert_called_once_with("Test instruction", {"source": "test"})

    def test_add_instruction_without_metadata(self, itr_instance):
        """Test adding instruction without metadata."""
        with patch.object(
            itr_instance.instruction_corpus, "add_system_prompt"
        ) as mock_add:
            itr_instance.add_instruction("Test instruction")
            mock_add.assert_called_once_with("Test instruction", None)

    def test_add_instruction_fragments(self, itr_instance):
        """Test adding pre-created instruction fragments."""
        fragments = [
            InstructionFragment("1", "Test 1", 10, FragmentType.ROLE_GUIDANCE),
            InstructionFragment("2", "Test 2", 15, FragmentType.STYLE_RULE),
        ]

        with patch.object(itr_instance.instruction_corpus, "add_fragments") as mock_add:
            itr_instance.add_instruction_fragments(fragments)
            mock_add.assert_called_once_with(fragments)

    def test_add_tool(self, itr_instance):
        """Test adding a tool."""
        tool_spec = {
            "id": "test_tool",
            "name": "test_tool",
            "description": "Test tool",
            "schema": {"type": "object"},
        }

        with patch.object(itr_instance.tool_corpus, "add_tool") as mock_add:
            itr_instance.add_tool(tool_spec)
            mock_add.assert_called_once_with(tool_spec)

    def test_load_instructions(self, itr_instance):
        """Test loading instructions from file."""
        with patch.object(
            itr_instance.instruction_corpus, "load_from_file"
        ) as mock_load:
            itr_instance.load_instructions("/path/to/instructions.txt")
            mock_load.assert_called_once_with(Path("/path/to/instructions.txt"))

    def test_load_tools(self, itr_instance):
        """Test loading tools from file."""
        with patch.object(itr_instance.tool_corpus, "load_from_file") as mock_load:
            itr_instance.load_tools("/path/to/tools.json")
            mock_load.assert_called_once_with(Path("/path/to/tools.json"))


class TestITRIntegration:
    """Integration tests for ITR functionality."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create ITR with specific config
        config = ITRConfig(
            k_a_instructions=2,
            k_b_tools=2,
            token_budget=1000,
            safety_overlay_tokens=50,
        )
        itr = ITR(config)

        # Add instructions
        instructions = [
            InstructionFragment(
                id="role",
                content="You are a helpful assistant.",
                token_count=25,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="safety",
                content="Always prioritize safety.",
                token_count=20,
                fragment_type=FragmentType.SAFETY_POLICY,
            ),
        ]
        itr.add_instruction_fragments(instructions)

        # Add tools
        tools = [
            {
                "id": "calc",
                "name": "calculator",
                "description": "Perform calculations",
                "schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                },
            },
            {
                "id": "search",
                "name": "web_search",
                "description": "Search the web",
                "schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        ]
        for tool in tools:
            itr.add_tool(tool)

        # Test step functionality
        result = itr.step("Calculate the square root of 16")

        assert isinstance(result, RetrievalResult)
        assert len(result.instructions) <= config.k_a_instructions
        assert len(result.tools) <= config.k_b_tools
        assert result.total_tokens <= config.token_budget + config.safety_overlay_tokens

        # Test prompt generation
        prompt = itr.get_prompt("What is 2+2?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_fallback_workflow(self):
        """Test fallback handling workflow."""
        itr = ITR(ITRConfig(discovery_expansion_factor=2.0))

        # Create a sample result
        original_result = RetrievalResult(
            instructions=[
                InstructionFragment(
                    "inst", "Test instruction", 20, FragmentType.ROLE_GUIDANCE
                )
            ],
            tools=[Tool("tool", "test_tool", "Test tool", {}, token_count=30)],
            total_tokens=100,
            confidence_score=0.6,
            fallback_triggered=False,
        )

        # Mock the expansion
        with patch.object(itr.tool_corpus, "get_expanded_set") as mock_expand:
            mock_expand.return_value = [
                Tool("tool1", "tool1", "Tool 1", {}, token_count=30),
                Tool("tool2", "tool2", "Tool 2", {}, token_count=35),
            ]

            fallback_result = itr.handle_fallback(original_result, "Test query")

            assert fallback_result.fallback_triggered is True
            assert fallback_result.confidence_score > original_result.confidence_score
            assert len(fallback_result.tools) > len(original_result.tools)

    def test_empty_corpora_handling(self):
        """Test behavior with empty corpora."""
        itr = ITR()

        # Should work even with empty corpora
        result = itr.step("Test query")

        assert isinstance(result, RetrievalResult)
        assert result.instructions == []
        assert result.tools == []

    def test_configuration_propagation(self):
        """Test that configuration changes propagate correctly."""
        config = ITRConfig(
            top_m_instructions=10,
            top_m_tools=5,
            token_budget=2000,
        )
        itr = ITR(config)

        # Verify all components have the same config
        assert itr.retriever.config.top_m_instructions == 10
        assert itr.selector.config.token_budget == 2000
        assert itr.prompt_builder.config.top_m_tools == 5


class TestITRExports:
    """Test module exports."""

    def test_module_exports(self):
        """Test that all expected classes are exported."""
        from itr import ITR, InstructionFragment, ITRConfig, RetrievalResult, Tool

        assert ITR is not None
        assert ITRConfig is not None
        assert RetrievalResult is not None
        assert InstructionFragment is not None
        assert Tool is not None

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        import itr

        expected_exports = [
            "ITR",
            "ITRConfig",
            "RetrievalResult",
            "InstructionFragment",
            "Tool",
        ]
        assert hasattr(itr, "__all__")
        assert set(itr.__all__) == set(expected_exports)


class TestITRErrorHandling:
    """Test error handling in ITR methods."""

    @pytest.fixture
    def itr_instance(self):
        """Provide basic ITR instance."""
        return ITR()

    def test_step_with_invalid_query_type(self, itr_instance):
        """Test step with invalid query type."""
        # Should handle non-string queries gracefully
        with patch.object(itr_instance.retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = []
            result = itr_instance.step(None)
            assert isinstance(result, RetrievalResult)

    def test_step_with_retrieval_error(self, itr_instance):
        """Test step when retrieval fails."""
        with patch.object(itr_instance.retriever, "retrieve") as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval failed")

            with pytest.raises(Exception, match="Retrieval failed"):
                itr_instance.step("Test query")

    def test_fallback_with_expansion_error(self, itr_instance):
        """Test fallback when expansion fails."""
        result = RetrievalResult(
            instructions=[],
            tools=[],
            total_tokens=0,
            confidence_score=0.5,
        )

        with patch.object(itr_instance.tool_corpus, "get_expanded_set") as mock_expand:
            mock_expand.side_effect = Exception("Expansion failed")

            with pytest.raises(Exception, match="Expansion failed"):
                itr_instance.handle_fallback(result, "Test query")
