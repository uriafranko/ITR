from typing import List, Optional

from .__version__ import __version__
from .assembly.prompt_builder import PromptBuilder
from .core.config import ITRConfig
from .core.types import InstructionFragment, RetrievalResult, Tool
from .indexing.corpus import InstructionCorpus, ToolCorpus
from .retrieval.hybrid_retriever import HybridRetriever
from .selection.budget_selector import BudgetAwareSelector


class ITR:
    """Main ITR system for instruction and tool retrieval."""

    def __init__(self, config: Optional[ITRConfig] = None):
        self.config = config or ITRConfig()
        self.instruction_corpus = InstructionCorpus()
        self.tool_corpus = ToolCorpus()
        self.retriever = HybridRetriever(self.config)
        self.selector = BudgetAwareSelector(self.config)
        self.prompt_builder = PromptBuilder(self.config)

    def step(self, query: str, history: Optional[List[str]] = None) -> RetrievalResult:
        """Execute ITR step for query."""
        history = history or []

        # Retrieve candidates
        inst_candidates = self.retriever.retrieve(
            query, self.instruction_corpus, self.config.top_m_instructions
        )
        tool_candidates = self.retriever.retrieve(
            query, self.tool_corpus, self.config.top_m_tools
        )

        # Budget-aware selection
        selected_inst, selected_tools = self.selector.select(
            inst_candidates, tool_candidates, self.config.token_budget
        )

        # Calculate total tokens
        total_tokens = sum(i.token_count for i in selected_inst)
        total_tokens += sum(t.token_count for t in selected_tools)
        total_tokens += self.config.safety_overlay_tokens

        # Create result
        result = RetrievalResult(
            instructions=selected_inst,
            tools=selected_tools,
            total_tokens=total_tokens,
            confidence_score=0.8,  # Placeholder confidence
            fallback_triggered=False,
        )

        return result

    def get_prompt(self, query: str, history: Optional[List[str]] = None) -> str:
        """Get assembled prompt for query."""
        result = self.step(query, history)
        prompt = self.prompt_builder.assemble(
            result.instructions,
            result.tools,
        )
        return prompt

    def handle_fallback(
        self, original_result: RetrievalResult, query: str
    ) -> RetrievalResult:
        """Handle discovery fallback by expanding tool set."""
        # Expand tool selection
        expanded_tools = self.tool_corpus.get_expanded_set(
            original_result.tools,
            expansion_factor=self.config.discovery_expansion_factor,
        )

        # Recalculate tokens
        total_tokens = sum(i.token_count for i in original_result.instructions)
        total_tokens += sum(t.token_count for t in expanded_tools)
        total_tokens += self.config.safety_overlay_tokens

        # Create new result
        new_result = RetrievalResult(
            instructions=original_result.instructions,
            tools=expanded_tools,
            total_tokens=total_tokens,
            confidence_score=0.9,  # Higher confidence with more tools
            fallback_triggered=True,
        )

        return new_result

    def load_instructions(self, filepath: str):
        """Load instruction corpus from file."""
        from pathlib import Path

        self.instruction_corpus.load_from_file(Path(filepath))

    def load_tools(self, filepath: str):
        """Load tool corpus from file."""
        from pathlib import Path

        self.tool_corpus.load_from_file(Path(filepath))

    def add_instruction(self, content: str, metadata: Optional[dict] = None):
        """Add an instruction to the corpus."""
        self.instruction_corpus.add_system_prompt(content, metadata)

    def add_instruction_fragments(self, fragments: List[InstructionFragment]):
        """Add pre-created instruction fragments to the corpus."""
        self.instruction_corpus.add_fragments(fragments)

    def add_tool(self, tool_spec: dict):
        """Add a tool to the corpus."""
        self.tool_corpus.add_tool(tool_spec)


# Export main classes
__all__ = ["ITR", "ITRConfig", "RetrievalResult", "InstructionFragment", "Tool"]
