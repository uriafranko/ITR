"""Test cases for itr.assembly.prompt_builder module."""

import pytest

from itr.assembly.prompt_builder import PromptBuilder
from itr.core.config import ITRConfig
from itr.core.types import FragmentType, InstructionFragment, Tool


class TestPromptBuilder:
    """Test cases for PromptBuilder class."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return ITRConfig(
            k_a_instructions=3,
            k_b_tools=2,
            token_budget=1000,
            tool_exemplars=2,  # Limit tool examples
        )

    @pytest.fixture
    def prompt_builder(self, config):
        """Provide PromptBuilder instance."""
        return PromptBuilder(config)

    @pytest.fixture
    def sample_instructions(self):
        """Provide sample instruction fragments."""
        return [
            InstructionFragment(
                id="inst_1",
                content="You are a helpful assistant.",
                token_count=5,
                fragment_type=FragmentType.ROLE_GUIDANCE,
                priority=3,
            ),
            InstructionFragment(
                id="inst_2",
                content="Always be accurate and precise.",
                token_count=5,
                fragment_type=FragmentType.STYLE_RULE,
                priority=1,
            ),
            InstructionFragment(
                id="inst_3",
                content="Prioritize user safety above all else.",
                token_count=6,
                fragment_type=FragmentType.SAFETY_POLICY,
                priority=5,
            ),
        ]

    @pytest.fixture
    def sample_tools(self):
        """Provide sample tools."""
        return [
            Tool(
                id="tool_1",
                name="calculator",
                description="Perform mathematical calculations",
                schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                exemplars=[
                    "Calculate 2+2",
                    "Find square root of 16",
                    "Compute factorial of 5",
                ],
                token_count=30,
            ),
            Tool(
                id="tool_2",
                name="text_analyzer",
                description="Analyze text properties",
                schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "analysis_type": {
                            "type": "string",
                            "enum": ["sentiment", "keywords"],
                        },
                    },
                    "required": ["text"],
                },
                exemplars=["Analyze sentiment", "Extract keywords"],
                token_count=25,
            ),
        ]

    def test_init(self, config):
        """Test PromptBuilder initialization."""
        builder = PromptBuilder(config)
        assert builder.config is config

    def test_assemble_empty_inputs(self, prompt_builder):
        """Test assembling prompt with empty instructions and tools."""
        result = prompt_builder.assemble([], [])

        assert "No tools are available. Please provide a direct response." in result
        assert "## Instructions" not in result
        assert "## Available Tools" not in result

    def test_assemble_instructions_only(self, prompt_builder, sample_instructions):
        """Test assembling prompt with instructions but no tools."""
        result = prompt_builder.assemble(sample_instructions, [])

        # Should contain instructions section
        assert "## Instructions" in result

        # Instructions should be sorted by priority (highest first)
        lines = result.split("\n")
        instructions_start = lines.index("## Instructions") + 1

        # Find instruction content lines (skip empty lines)
        instruction_lines = []
        for i in range(instructions_start, len(lines)):
            if (
                lines[i].startswith("##")
                or lines[i]
                == "No tools are available. Please provide a direct response."
            ):
                break
            if lines[i].strip():
                instruction_lines.append(lines[i])

        # Verify priority order (5, 3, 1)
        assert "Prioritize user safety above all else." in instruction_lines[0]
        assert "You are a helpful assistant." in instruction_lines[1]
        assert "Always be accurate and precise." in instruction_lines[2]

        # Should indicate no tools
        assert "No tools are available. Please provide a direct response." in result

    def test_assemble_tools_only(self, prompt_builder, sample_tools):
        """Test assembling prompt with tools but no instructions."""
        result = prompt_builder.assemble([], sample_tools)

        # Should contain tools section
        assert "## Available Tools" in result
        assert "### calculator" in result
        assert "### text_analyzer" in result

        # Should indicate multiple tools
        assert (
            "You have access to 2 tools. Select the most appropriate one(s) for the task."
            in result
        )

        # Should not contain instructions section
        assert "## Instructions" not in result

    def test_assemble_full_prompt(
        self, prompt_builder, sample_instructions, sample_tools
    ):
        """Test assembling complete prompt with both instructions and tools."""
        result = prompt_builder.assemble(sample_instructions, sample_tools)

        # Should contain both sections
        assert "## Instructions" in result
        assert "## Available Tools" in result

        # Check section order
        instructions_pos = result.find("## Instructions")
        tools_pos = result.find("## Available Tools")
        routing_pos = result.find("You have access to 2 tools")

        assert instructions_pos < tools_pos < routing_pos

        # Verify content
        assert "calculator" in result
        assert "text_analyzer" in result
        assert "Prioritize user safety above all else." in result

    def test_format_tool_basic(self, prompt_builder):
        """Test basic tool formatting."""
        tool = Tool(
            id="test_tool",
            name="test_tool",
            description="A test tool for testing",
            schema={"type": "object"},
            token_count=10,
        )

        result = prompt_builder._format_tool(tool)

        assert "### test_tool" in result
        assert "A test tool for testing" in result

    def test_format_tool_with_exemplars(self, prompt_builder, sample_tools):
        """Test tool formatting with exemplars."""
        tool = sample_tools[0]  # calculator with exemplars

        result = prompt_builder._format_tool(tool)

        assert "### calculator" in result
        assert "Perform mathematical calculations" in result
        assert "Examples:" in result
        assert "Calculate 2+2" in result
        assert "Find square root of 16" in result
        # Should limit to config.tool_exemplars (2)
        assert "Compute factorial of 5" not in result

    def test_format_tool_no_exemplars(self, prompt_builder):
        """Test tool formatting without exemplars."""
        tool = Tool(
            id="simple_tool",
            name="simple_tool",
            description="Simple tool without examples",
            schema={"type": "object"},
        )

        result = prompt_builder._format_tool(tool)

        assert "### simple_tool" in result
        assert "Simple tool without examples" in result
        assert "Examples:" not in result

    def test_get_routing_note_no_tools(self, prompt_builder):
        """Test routing note with no tools."""
        result = prompt_builder._get_routing_note(0)
        assert result == "No tools are available. Please provide a direct response."

    def test_get_routing_note_one_tool(self, prompt_builder):
        """Test routing note with one tool."""
        result = prompt_builder._get_routing_note(1)
        assert (
            result == "You have access to 1 tool. Use it if appropriate for the task."
        )

    def test_get_routing_note_multiple_tools(self, prompt_builder):
        """Test routing note with multiple tools."""
        result = prompt_builder._get_routing_note(3)
        assert (
            result
            == "You have access to 3 tools. Select the most appropriate one(s) for the task."
        )

        result = prompt_builder._get_routing_note(10)
        assert (
            result
            == "You have access to 10 tools. Select the most appropriate one(s) for the task."
        )

    def test_tool_exemplars_limit_configuration(self, config):
        """Test that tool exemplars are limited by configuration."""
        # Create a config with different exemplar limit
        config.tool_exemplars = 1
        builder = PromptBuilder(config)

        tool = Tool(
            id="multi_example_tool",
            name="multi_example_tool",
            description="Tool with many examples",
            schema={"type": "object"},
            exemplars=["Example 1", "Example 2", "Example 3", "Example 4"],
        )

        result = builder._format_tool(tool)

        assert "Example 1" in result
        assert "Example 2" not in result
        assert "Example 3" not in result
        assert "Example 4" not in result

    def test_instruction_priority_sorting(self, prompt_builder):
        """Test that instructions are properly sorted by priority."""
        instructions = [
            InstructionFragment(
                id="low",
                content="Low priority",
                token_count=2,
                fragment_type=FragmentType.STYLE_RULE,
                priority=1,
            ),
            InstructionFragment(
                id="high",
                content="High priority",
                token_count=2,
                fragment_type=FragmentType.SAFETY_POLICY,
                priority=10,
            ),
            InstructionFragment(
                id="medium",
                content="Medium priority",
                token_count=2,
                fragment_type=FragmentType.ROLE_GUIDANCE,
                priority=5,
            ),
        ]

        result = prompt_builder.assemble(instructions, [])

        lines = result.split("\n")
        instructions_section = []
        in_instructions = False

        for line in lines:
            if line == "## Instructions":
                in_instructions = True
                continue
            elif line.startswith("##") or line.startswith("No tools"):
                in_instructions = False
            elif in_instructions and line.strip():
                instructions_section.append(line)

        # Should be sorted by priority: high (10), medium (5), low (1)
        assert "High priority" in instructions_section[0]
        assert "Medium priority" in instructions_section[1]
        assert "Low priority" in instructions_section[2]

    def test_empty_sections_formatting(self, prompt_builder):
        """Test proper formatting with empty sections."""
        result = prompt_builder.assemble([], [])

        # Should not have empty sections
        assert "## Instructions" not in result
        assert "## Available Tools" not in result
        assert result.strip().endswith(
            "No tools are available. Please provide a direct response."
        )

    def test_complex_tool_schema_formatting(self, prompt_builder):
        """Test formatting of tool with complex schema."""
        tool = Tool(
            id="complex_tool",
            name="complex_tool",
            description="A tool with complex schema",
            schema={
                "type": "object",
                "properties": {
                    "required_param": {
                        "type": "string",
                        "description": "A required parameter",
                    },
                    "optional_param": {
                        "type": "integer",
                        "description": "An optional parameter",
                        "default": 42,
                    },
                    "enum_param": {
                        "type": "string",
                        "enum": ["option1", "option2", "option3"],
                    },
                },
                "required": ["required_param"],
            },
            exemplars=["Use complex tool with param"],
            token_count=50,
        )

        result = prompt_builder._format_tool(tool)

        assert "### complex_tool" in result
        assert "A tool with complex schema" in result
        assert "Examples:" in result
        assert "Use complex tool with param" in result


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder."""

    def test_realistic_prompt_assembly(self):
        """Test assembling a realistic prompt with various components."""
        config = ITRConfig(tool_exemplars=3)
        builder = PromptBuilder(config)

        instructions = [
            InstructionFragment(
                id="role",
                content="You are a data analysis assistant.",
                token_count=6,
                fragment_type=FragmentType.ROLE_GUIDANCE,
                priority=3,
            ),
            InstructionFragment(
                id="safety",
                content="Never execute potentially harmful operations.",
                token_count=6,
                fragment_type=FragmentType.SAFETY_POLICY,
                priority=5,
            ),
            InstructionFragment(
                id="style",
                content="Provide clear explanations with examples.",
                token_count=6,
                fragment_type=FragmentType.STYLE_RULE,
                priority=2,
            ),
        ]

        tools = [
            Tool(
                id="data_processor",
                name="data_processor",
                description="Process and analyze datasets",
                schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array"},
                        "operation": {
                            "type": "string",
                            "enum": ["filter", "aggregate", "sort"],
                        },
                    },
                    "required": ["data", "operation"],
                },
                exemplars=[
                    "Filter sales data",
                    "Aggregate user metrics",
                    "Sort by date",
                ],
                token_count=40,
            )
        ]

        result = builder.assemble(instructions, tools)

        # Verify structure
        sections = result.split("## ")
        assert (
            len([s for s in sections if s.strip()]) >= 2
        )  # Instructions and Tools sections

        # Verify priority ordering
        assert result.find(
            "Never execute potentially harmful operations"
        ) < result.find("You are a data analysis assistant")

        # Verify tool formatting
        assert "### data_processor" in result
        assert "Process and analyze datasets" in result
        assert "Filter sales data" in result
        assert "You have access to 1 tool" in result

    def test_prompt_structure_consistency(self):
        """Test that prompt structure is consistent across different inputs."""
        config = ITRConfig(tool_exemplars=2)
        builder = PromptBuilder(config)

        # Test various combinations
        test_cases = [
            ([], []),  # Empty
            (
                [InstructionFragment("1", "Test", 1, FragmentType.ROLE_GUIDANCE)],
                [],
            ),  # Instructions only
            ([], [Tool("1", "test", "Test tool", {})]),  # Tools only
        ]

        for instructions, tools in test_cases:
            result = builder.assemble(instructions, tools)

            # Should always end with routing note
            lines = result.strip().split("\n")
            last_line = lines[-1]
            assert any(
                phrase in last_line
                for phrase in [
                    "No tools are available",
                    "You have access to",
                ]
            )

            # Should have consistent formatting
            # Look for both ## (main sections) and ### (tool headers)
            main_sections = [line for line in lines if line.startswith("## ")]
            tool_sections = [line for line in lines if line.startswith("### ")]

            # Main sections should have proper formatting
            for section in main_sections:
                assert section.startswith("## ")

            # Tool sections should have proper formatting
            for section in tool_sections:
                assert section.startswith("### ")
