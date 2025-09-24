from typing import List

from ..core.config import ITRConfig
from ..core.types import InstructionFragment, Tool


class PromptBuilder:
    """Assemble dynamic prompts from selected components."""

    def __init__(self, config: ITRConfig):
        self.config = config

    def assemble(
        self, instructions: List[InstructionFragment], tools: List[Tool]
    ) -> str:
        """Assemble dynamic prompt."""
        sections = []

        # 1. Selected instructions (ordered by priority)
        if instructions:
            sections.append("## Instructions")
            instructions_sorted = sorted(
                instructions, key=lambda x: x.priority, reverse=True
            )
            for inst in instructions_sorted:
                sections.append(inst.content)
            sections.append("")

        # 2. Tool schemas
        if tools:
            sections.append("## Available Tools")
            for tool in tools:
                tool_text = self._format_tool(tool)
                sections.append(tool_text)
            sections.append("")

        # 3. Routing note
        sections.append(self._get_routing_note(len(tools)))
        sections.append("")

        return "\n".join(sections)

    def _format_tool(self, tool: Tool) -> str:
        """Format a tool for the prompt."""
        lines = [f"### {tool.name}", tool.description]
        if tool.exemplars:
            lines.append("Examples:")
            for ex in tool.exemplars[: self.config.tool_exemplars]:  # Limit examples
                lines.append(f"- {ex}")

        return "\n".join(lines)

    def _get_routing_note(self, num_tools: int) -> str:
        """Generate routing instruction."""
        if num_tools == 0:
            return "No tools are available. Please provide a direct response."
        elif num_tools == 1:
            return "You have access to 1 tool. Use it if appropriate for the task."
        else:
            return f"You have access to {num_tools} tools. Select the most appropriate one(s) for the task."
