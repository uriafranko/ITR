from typing import List, Tuple

from ..core.config import ITRConfig
from ..core.types import InstructionFragment, Tool


class BudgetAwareSelector:
    """Select items within token budget using greedy algorithm."""

    def __init__(self, config: ITRConfig):
        self.config = config

    def select(
        self, instructions: List[InstructionFragment], tools: List[Tool], budget: int
    ) -> Tuple[List[InstructionFragment], List[Tool]]:
        """Select items within token budget."""
        # Reserve tokens for safety overlay
        available_budget = budget - self.config.safety_overlay_tokens

        # Create combined list with estimated value
        items = []

        for i, inst in enumerate(instructions):
            # Prioritize by position (earlier = higher priority) and fragment type
            value = 1.0 / (i + 1)
            if inst.fragment_type.value == "safety":
                value *= 2.0  # Boost safety fragments

            items.append(
                {
                    "item": inst,
                    "type": "instruction",
                    "tokens": inst.token_count,
                    "value": value,
                    "value_per_token": (
                        value / inst.token_count if inst.token_count > 0 else 0
                    ),
                }
            )

        for i, tool in enumerate(tools):
            # Prioritize by position (earlier = higher priority)
            value = 1.0 / (i + 1)

            items.append(
                {
                    "item": tool,
                    "type": "tool",
                    "tokens": tool.token_count,
                    "value": value,
                    "value_per_token": (
                        value / tool.token_count if tool.token_count > 0 else 0
                    ),
                }
            )

        # Sort by value per token (greedy approach)
        items.sort(key=lambda x: x["value_per_token"], reverse=True)

        selected_instructions = []
        selected_tools = []
        used_tokens = 0

        for item in items:
            if used_tokens + item["tokens"] <= available_budget:
                if item["type"] == "instruction":
                    if len(selected_instructions) < self.config.k_a_instructions:
                        selected_instructions.append(item["item"])
                        used_tokens += item["tokens"]
                else:  # tool
                    if len(selected_tools) < self.config.k_b_tools:
                        selected_tools.append(item["item"])
                        used_tokens += item["tokens"]

            # Stop if we have enough of both
            if (
                len(selected_instructions) >= self.config.k_a_instructions
                and len(selected_tools) >= self.config.k_b_tools
            ):
                break

        return selected_instructions, selected_tools
