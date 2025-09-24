import json
from pathlib import Path
from typing import Dict, List, Optional

from ..core.exceptions import CorpusException
from ..core.types import FragmentType, InstructionFragment, Tool
from .chunker import TextChunker


class InstructionCorpus:
    """Manages instruction fragments."""

    def __init__(self, chunk_size_range=(200, 600)):
        self.fragments: Dict[str, InstructionFragment] = {}
        self.chunk_size_range = chunk_size_range
        self.chunker = TextChunker(chunk_size_range)

    def add_system_prompt(self, prompt: str, metadata: Optional[Dict] = None):
        """Chunk and index system prompt."""
        metadata = metadata or {}
        chunks = self.chunker.chunk(prompt)

        for i, chunk in enumerate(chunks):
            fragment = InstructionFragment(
                id=f"{metadata.get('source', 'unknown')}_{i}",
                content=chunk.text,
                token_count=chunk.token_count,
                fragment_type=self._infer_type(chunk.text),
                metadata=metadata,
            )
            self.fragments[fragment.id] = fragment

    def add_fragments(self, fragments: List[InstructionFragment]):
        """Add pre-created instruction fragments."""
        for fragment in fragments:
            self.fragments[fragment.id] = fragment

    def _infer_type(self, text: str) -> FragmentType:
        """Infer fragment type from content."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["safety", "harm", "ethical"]):
            return FragmentType.SAFETY_POLICY
        elif any(word in text_lower for word in ["style", "format", "tone"]):
            return FragmentType.STYLE_RULE
        elif any(word in text_lower for word in ["role", "persona", "act as"]):
            return FragmentType.ROLE_GUIDANCE
        elif any(word in text_lower for word in ["example", "e.g.", "for instance"]):
            return FragmentType.EXEMPLAR
        else:
            return FragmentType.DOMAIN_SPECIFIC

    def get(self, fragment_id: str) -> Optional[InstructionFragment]:
        """Get fragment by ID."""
        return self.fragments.get(fragment_id)

    def get_all(self) -> List[InstructionFragment]:
        """Get all fragments."""
        return list(self.fragments.values())

    def load_from_file(self, filepath: Path):
        """Load instructions from a text file."""
        if not filepath.exists():
            raise CorpusException(f"File not found: {filepath}")

        with open(filepath) as f:
            content = f.read()

        metadata = {"source": filepath.stem}
        self.add_system_prompt(content, metadata)

    def save(self, filepath: Path):
        """Save the entire instruction corpus to a JSON file.

        Serializes all fragments with their metadata to JSON format
        for persistence and later loading.

        Args:
            filepath: Path where to save the JSON file

        Example:
            >>> corpus.save(Path("instruction_corpus.json"))
        """
        data = [
            {
                "id": f.id,
                "content": f.content,
                "token_count": f.token_count,
                "fragment_type": f.fragment_type.value,
                "metadata": f.metadata,
                "priority": f.priority,
            }
            for f in self.fragments.values()
        ]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class ToolCorpus:
    """Manages a corpus of tools available to AI agents.

    The ToolCorpus handles storage, indexing, and retrieval of tool definitions.
    It manages tool metadata, token counting, and provides methods for expanding
    tool sets during fallback scenarios.

    Attributes:
        tools: Dictionary mapping tool IDs to Tool objects
        chunker: TextChunker instance for token counting

    Example:
        >>> tool_corpus = ToolCorpus()
        >>> tool_spec = {
        ...     "name": "calculator",
        ...     "description": "Performs calculations",
        ...     "schema": {"type": "object"}
        ... }
        >>> tool_corpus.add_tool(tool_spec)
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.chunker = TextChunker()

    def add_tool(self, tool_spec: Dict):
        """Add a tool to the corpus from a specification dictionary.

        Creates a Tool object from the specification and adds it to the corpus
        with automatic token counting for the tool representation.

        Args:
            tool_spec: Dictionary containing tool specification with keys:
                - name: Tool name (required)
                - description: Tool description (optional)
                - schema: JSON schema for parameters (optional)
                - exemplars: Usage examples (optional)
                - id: Unique identifier (optional, defaults to name)

        Example:
            >>> tool_spec = {
            ...     "name": "web_search",
            ...     "description": "Search the web for information",
            ...     "schema": {
            ...         "type": "object",
            ...         "properties": {"query": {"type": "string"}}
            ...     }
            ... }
            >>> corpus.add_tool(tool_spec)
        """
        tool = Tool(
            id=tool_spec.get("id", tool_spec["name"]),
            name=tool_spec["name"],
            description=tool_spec.get("description", ""),
            schema=tool_spec.get("schema", {}),
            exemplars=tool_spec.get("exemplars", []),
            token_count=self._count_tokens(tool_spec),
        )
        self.tools[tool.id] = tool

    def _count_tokens(self, tool_spec: Dict) -> int:
        """Count tokens for tool representation."""
        text = f"{tool_spec.get('name', '')}\n{tool_spec.get('description', '')}"
        if tool_spec.get("schema"):
            text += f"\n{json.dumps(tool_spec['schema'])}"
        # Use word count approximation if tokenizer not available
        try:
            return self.chunker.count_tokens(text)
        except (AttributeError, ValueError, RuntimeError):
            # Fallback to word count approximation
            return int(len(text.split()) * 1.3)

    def get(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID."""
        return self.tools.get(tool_id)

    def get_all(self) -> List[Tool]:
        """Get all tools."""
        return list(self.tools.values())

    def get_expanded_set(
        self, base_tools: List[Tool], expansion_factor: float = 2.0
    ) -> List[Tool]:
        """Get expanded tool set for fallback."""
        base_ids = {t.id for t in base_tools}
        all_tools = self.get_all()

        # Include base tools
        result = base_tools.copy()

        # Add more tools up to expansion factor
        target_count = int(len(base_tools) * expansion_factor)
        for tool in all_tools:
            if len(result) >= target_count:
                break
            if tool.id not in base_ids:
                result.append(tool)

        return result

    def load_from_file(self, filepath: Path):
        """Load tools from JSON file."""
        if not filepath.exists():
            raise CorpusException(f"File not found: {filepath}")

        with open(filepath) as f:
            tools = json.load(f)

        for tool_spec in tools:
            self.add_tool(tool_spec)

    def save(self, filepath: Path):
        """Save corpus to JSON file."""
        data = [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "schema": t.schema,
                "exemplars": t.exemplars,
                "token_count": t.token_count,
            }
            for t in self.tools.values()
        ]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
