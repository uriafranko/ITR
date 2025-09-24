"""Test cases for itr.selection.budget_selector module."""

import pytest

from itr.core.config import ITRConfig
from itr.core.types import FragmentType, InstructionFragment, Tool
from itr.selection.budget_selector import BudgetAwareSelector


class TestBudgetAwareSelector:
    """Test cases for BudgetAwareSelector class."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return ITRConfig(
            k_a_instructions=3,
            k_b_tools=2,
            safety_overlay_tokens=50,
            token_budget=1000,
        )

    @pytest.fixture
    def selector(self, config):
        """Provide BudgetAwareSelector instance."""
        return BudgetAwareSelector(config)

    @pytest.fixture
    def sample_instructions(self):
        """Provide sample instruction fragments."""
        return [
            InstructionFragment(
                id="inst_1",
                content="You are a helpful assistant.",
                token_count=20,
                fragment_type=FragmentType.ROLE_GUIDANCE,
                priority=3,
            ),
            InstructionFragment(
                id="inst_2",
                content="Always be accurate and precise.",
                token_count=15,
                fragment_type=FragmentType.STYLE_RULE,
                priority=1,
            ),
            InstructionFragment(
                id="inst_3",
                content="Prioritize user safety above all else.",
                token_count=25,
                fragment_type=FragmentType.SAFETY_POLICY,
                priority=5,
            ),
            InstructionFragment(
                id="inst_4",
                content="Provide detailed explanations.",
                token_count=10,
                fragment_type=FragmentType.DOMAIN_SPECIFIC,
                priority=2,
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
                schema={"type": "object"},
                token_count=30,
            ),
            Tool(
                id="tool_2",
                name="text_analyzer",
                description="Analyze text properties",
                schema={"type": "object"},
                token_count=25,
            ),
            Tool(
                id="tool_3",
                name="web_search",
                description="Search the web",
                schema={"type": "object"},
                token_count=40,
            ),
        ]

    def test_init(self, config):
        """Test BudgetAwareSelector initialization."""
        selector = BudgetAwareSelector(config)
        assert selector.config is config

    def test_select_empty_inputs(self, selector):
        """Test selection with empty instruction and tool lists."""
        instructions, tools = selector.select([], [], 1000)

        assert instructions == []
        assert tools == []

    def test_select_within_budget(self, selector, sample_instructions, sample_tools):
        """Test selection when all items fit within budget."""
        # Large budget that can fit all items
        budget = 500

        instructions, tools = selector.select(sample_instructions, sample_tools, budget)

        # Should select up to k_a_instructions and k_b_tools
        assert len(instructions) <= selector.config.k_a_instructions
        assert len(tools) <= selector.config.k_b_tools

        # Verify all selected items are from the input
        for inst in instructions:
            assert inst in sample_instructions
        for tool in tools:
            assert tool in sample_tools

    def test_select_tight_budget(self, selector, sample_instructions, sample_tools):
        """Test selection with very tight budget."""
        # Small budget that can only fit a few items
        budget = 100  # After safety overlay (50), only 50 tokens available

        instructions, tools = selector.select(sample_instructions, sample_tools, budget)

        # Should select fewer items due to budget constraints
        total_tokens = sum(inst.token_count for inst in instructions) + sum(
            tool.token_count for tool in tools
        )
        available_budget = budget - selector.config.safety_overlay_tokens
        assert total_tokens <= available_budget

    def test_select_safety_fragment_priority(self, selector):
        """Test that safety fragments get prioritized."""
        instructions = [
            InstructionFragment(
                id="safety",
                content="Safety instruction",
                token_count=10,
                fragment_type=FragmentType.SAFETY_POLICY,
            ),
            InstructionFragment(
                id="regular",
                content="Regular instruction",
                token_count=10,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]

        # Small budget that can only fit one instruction
        budget = 60  # After safety overlay (50), only 10 tokens available

        selected_instructions, _ = selector.select(instructions, [], budget)

        # Safety fragment should be selected due to 2x value boost
        # Safety gets: 1.0 * 2.0 / 10 = 0.2 value per token
        # Regular gets: 0.5 / 10 = 0.05 value per token
        assert len(selected_instructions) == 1
        assert selected_instructions[0].fragment_type == FragmentType.SAFETY_POLICY

    def test_select_value_per_token_sorting(self, selector):
        """Test that items are sorted by value per token."""
        instructions = [
            InstructionFragment(
                id="expensive",
                content="Expensive instruction with many tokens",
                token_count=50,  # Low value per token (first position, many tokens)
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="cheap",
                content="Cheap",
                token_count=5,  # High value per token (second position, few tokens)
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]

        budget = 70  # After safety overlay, 20 tokens available

        selected_instructions, _ = selector.select(instructions, [], budget)

        # Should select the cheap instruction first due to better value per token
        assert len(selected_instructions) == 1
        assert selected_instructions[0].id == "cheap"

    def test_select_respects_k_limits(self, selector):
        """Test that selection respects k_a_instructions and k_b_tools limits."""
        # Create more items than the limits allow
        instructions = [
            InstructionFragment(
                id=f"inst_{i}",
                content=f"Instruction {i}",
                token_count=5,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            )
            for i in range(10)  # More than k_a_instructions (3)
        ]

        tools = [
            Tool(
                id=f"tool_{i}",
                name=f"tool_{i}",
                description=f"Tool {i}",
                schema={"type": "object"},
                token_count=5,
            )
            for i in range(10)  # More than k_b_tools (2)
        ]

        budget = 1000  # Large budget

        selected_instructions, selected_tools = selector.select(
            instructions, tools, budget
        )

        # Should not exceed the limits
        assert len(selected_instructions) <= selector.config.k_a_instructions
        assert len(selected_tools) <= selector.config.k_b_tools

    def test_select_zero_token_items(self, selector):
        """Test handling of items with zero tokens."""
        instructions = [
            InstructionFragment(
                id="zero_tokens",
                content="",
                token_count=0,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="normal",
                content="Normal instruction",
                token_count=10,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]

        tools = [
            Tool(
                id="zero_tool",
                name="zero_tool",
                description="",
                schema={"type": "object"},
                token_count=0,
            )
        ]

        budget = 100

        selected_instructions, selected_tools = selector.select(
            instructions, tools, budget
        )

        # Should handle zero-token items without error
        assert isinstance(selected_instructions, list)
        assert isinstance(selected_tools, list)

    def test_select_budget_exactly_used(self, selector):
        """Test selection when budget is exactly consumed."""
        instructions = [
            InstructionFragment(
                id="exact",
                content="Exact fit",
                token_count=20,  # After safety overlay (50), budget is 20
                fragment_type=FragmentType.ROLE_GUIDANCE,
            )
        ]

        budget = 70  # 70 - 50 = 20 available

        selected_instructions, _ = selector.select(instructions, [], budget)

        assert len(selected_instructions) == 1
        assert selected_instructions[0].token_count == 20

    def test_select_budget_exceeded_by_single_item(self, selector):
        """Test when a single item exceeds available budget."""
        instructions = [
            InstructionFragment(
                id="too_large",
                content="This instruction is too large for the budget",
                token_count=100,  # Larger than available budget after safety overlay
                fragment_type=FragmentType.ROLE_GUIDANCE,
            )
        ]

        budget = 70  # 70 - 50 = 20 available, but item needs 100

        selected_instructions, _ = selector.select(instructions, [], budget)

        # Should not select the item that exceeds budget
        assert len(selected_instructions) == 0

    def test_select_position_based_priority(self, selector):
        """Test that earlier positions get higher priority."""
        instructions = [
            InstructionFragment(
                id="first",
                content="First instruction",
                token_count=10,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="second",
                content="Second instruction",
                token_count=10,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="third",
                content="Third instruction",
                token_count=10,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
        ]

        budget = 70  # Can fit 2 instructions after safety overlay

        selected_instructions, _ = selector.select(instructions, [], budget)

        # Should select first two instructions due to position priority
        assert len(selected_instructions) == 2
        selected_ids = {inst.id for inst in selected_instructions}
        assert "first" in selected_ids
        assert "second" in selected_ids

    def test_select_mixed_types_budget_allocation(
        self, selector, sample_instructions, sample_tools
    ):
        """Test budget allocation between instructions and tools."""
        budget = 150  # 150 - 50 = 100 available

        selected_instructions, selected_tools = selector.select(
            sample_instructions[:2], sample_tools[:2], budget
        )

        # Should select some of both types
        total_tokens = sum(inst.token_count for inst in selected_instructions) + sum(
            tool.token_count for tool in selected_tools
        )

        available_budget = budget - selector.config.safety_overlay_tokens
        assert total_tokens <= available_budget
        assert len(selected_instructions) > 0 or len(selected_tools) > 0

    def test_select_early_stopping_when_limits_reached(self, selector):
        """Test that selection stops early when both limits are reached."""
        # Create many high-value items
        instructions = [
            InstructionFragment(
                id=f"inst_{i}",
                content="Short",
                token_count=1,  # Very efficient
                fragment_type=FragmentType.ROLE_GUIDANCE,
            )
            for i in range(10)
        ]

        tools = [
            Tool(
                id=f"tool_{i}",
                name=f"tool_{i}",
                description="Short",
                schema={"type": "object"},
                token_count=1,  # Very efficient
            )
            for i in range(10)
        ]

        budget = 1000  # Large budget

        selected_instructions, selected_tools = selector.select(
            instructions, tools, budget
        )

        # Should stop at the limits even though budget allows more
        assert len(selected_instructions) == selector.config.k_a_instructions
        assert len(selected_tools) == selector.config.k_b_tools

    def test_config_safety_overlay_deduction(self, selector):
        """Test that safety overlay tokens are properly deducted from budget."""
        instructions = [
            InstructionFragment(
                id="test",
                content="Test instruction",
                token_count=1,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            )
        ]

        # Budget exactly equals safety overlay tokens
        budget = selector.config.safety_overlay_tokens

        selected_instructions, _ = selector.select(instructions, [], budget)

        # Should not select anything as available budget is 0
        assert len(selected_instructions) == 0


class TestBudgetAwareSelectorIntegration:
    """Integration tests for BudgetAwareSelector."""

    def test_realistic_selection_scenario(self):
        """Test a realistic selection scenario with mixed content."""
        config = ITRConfig(
            k_a_instructions=5,
            k_b_tools=3,
            safety_overlay_tokens=100,
            token_budget=1000,
        )
        selector = BudgetAwareSelector(config)

        instructions = [
            InstructionFragment(
                id="role",
                content="You are a helpful AI assistant specialized in data analysis.",
                token_count=50,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="safety",
                content="Never execute potentially harmful operations.",
                token_count=30,
                fragment_type=FragmentType.SAFETY_POLICY,
            ),
            InstructionFragment(
                id="style",
                content="Be concise and accurate.",
                token_count=20,
                fragment_type=FragmentType.STYLE_RULE,
            ),
            InstructionFragment(
                id="domain",
                content="When analyzing data, always validate inputs first.",
                token_count=40,
                fragment_type=FragmentType.DOMAIN_SPECIFIC,
            ),
        ]

        tools = [
            Tool(
                id="data_processor",
                name="data_processor",
                description="Process and analyze datasets with validation",
                schema={"type": "object"},
                token_count=60,
            ),
            Tool(
                id="visualizer",
                name="visualizer",
                description="Create charts and graphs",
                schema={"type": "object"},
                token_count=45,
            ),
            Tool(
                id="validator",
                name="validator",
                description="Validate data integrity",
                schema={"type": "object"},
                token_count=35,
            ),
        ]

        budget = 400  # 400 - 100 = 300 available

        selected_instructions, selected_tools = selector.select(
            instructions, tools, budget
        )

        # Verify selection makes sense
        assert len(selected_instructions) > 0
        assert len(selected_tools) > 0

        # Safety instruction should be prioritized due to 2x boost
        safety_selected = any(
            inst.fragment_type == FragmentType.SAFETY_POLICY
            for inst in selected_instructions
        )
        assert safety_selected

        # Total tokens should be within budget
        total_tokens = sum(inst.token_count for inst in selected_instructions) + sum(
            tool.token_count for tool in selected_tools
        )
        assert total_tokens <= 300  # Available budget after safety overlay

    def test_edge_case_all_items_too_expensive(self):
        """Test when all items exceed the available budget."""
        config = ITRConfig(
            k_a_instructions=2,
            k_b_tools=2,
            safety_overlay_tokens=90,
        )
        selector = BudgetAwareSelector(config)

        instructions = [
            InstructionFragment(
                id="expensive1",
                content="Very long instruction that exceeds budget",
                token_count=50,
                fragment_type=FragmentType.ROLE_GUIDANCE,
            ),
            InstructionFragment(
                id="expensive2",
                content="Another very long instruction that exceeds budget",
                token_count=60,
                fragment_type=FragmentType.STYLE_RULE,
            ),
        ]

        tools = [
            Tool(
                id="expensive_tool",
                name="expensive_tool",
                description="Tool with very long description that exceeds budget",
                schema={"type": "object"},
                token_count=70,
            )
        ]

        budget = 100  # 100 - 90 = 10 available, but all items need 50+

        selected_instructions, selected_tools = selector.select(
            instructions, tools, budget
        )

        # Should not select anything
        assert len(selected_instructions) == 0
        assert len(selected_tools) == 0
