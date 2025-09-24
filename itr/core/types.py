from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class FragmentType(Enum):
    """Types of instruction fragments for categorization and prioritization.

    Used to classify instruction fragments based on their content and purpose,
    allowing for better selection and prioritization during retrieval.
    """

    ROLE_GUIDANCE = "role"
    STYLE_RULE = "style"
    SAFETY_POLICY = "safety"
    DOMAIN_SPECIFIC = "domain"
    EXEMPLAR = "exemplar"


@dataclass
class InstructionFragment:
    """Represents a chunk of system instructions with metadata and embeddings.

    An instruction fragment is a discrete piece of guidance for an AI agent,
    created through chunking of larger instruction texts. Each fragment contains
    the instruction content, token count, type classification, and optional
    embedding for semantic search.

    Attributes:
        id: Unique identifier for the fragment
        content: The actual instruction text
        token_count: Number of tokens in the content
        fragment_type: Classification of the fragment type
        metadata: Additional metadata (source, priority, etc.)
        embedding: Optional dense embedding for semantic search
        priority: Priority score for selection (higher = more important)

    Example:
        >>> fragment = InstructionFragment(
        ...     id="safety_1",
        ...     content="Always prioritize user safety",
        ...     token_count=5,
        ...     fragment_type=FragmentType.SAFETY_POLICY,
        ...     metadata={"source": "safety_guidelines"}
        ... )
    """

    id: str
    content: str
    token_count: int
    fragment_type: FragmentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    priority: int = 0


@dataclass
class Tool:
    """Represents a tool or function available to an AI agent.

    Tools define capabilities that an agent can use to perform actions or
    retrieve information. Each tool includes a schema defining its parameters,
    usage examples, and operational metadata.

    Attributes:
        id: Unique identifier for the tool
        name: Human-readable tool name
        description: Description of what the tool does
        schema: JSON schema defining the tool's parameters
        exemplars: Example usage patterns
        token_count: Estimated tokens when serialized
        embedding: Optional embedding for semantic search
        preconditions: Conditions required before tool use
        postconditions: Expected outcomes after tool use
        failure_modes: Common failure scenarios

    Example:
        >>> tool = Tool(
        ...     id="calc",
        ...     name="calculator",
        ...     description="Perform mathematical calculations",
        ...     schema={"type": "object", "properties": {"expr": {"type": "string"}}}
        ... )
    """

    id: str
    name: str
    description: str
    schema: Dict[str, Any]
    exemplars: List[str] = field(default_factory=list)
    token_count: int = 0
    embedding: Optional[np.ndarray] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Results from an ITR retrieval operation.

    Contains the selected instructions and tools along with metadata about
    the retrieval process, including token usage and confidence metrics.

    Attributes:
        instructions: List of selected instruction fragments
        tools: List of selected tools
        total_tokens: Total token count of selected items
        confidence_score: Confidence in the retrieval quality (0-1)
        fallback_triggered: Whether fallback expansion was used

    Example:
        >>> result = RetrievalResult(
        ...     instructions=[fragment1, fragment2],
        ...     tools=[tool1],
        ...     total_tokens=150,
        ...     confidence_score=0.85
        ... )
    """

    instructions: List[InstructionFragment]
    tools: List[Tool]
    total_tokens: int
    confidence_score: float
    fallback_triggered: bool = False


@dataclass
class SearchResult:
    """Result from a search operation with relevance scoring.

    Represents a single item returned from a search query, including
    the item itself and its relevance score.

    Attributes:
        id: Unique identifier of the found item
        score: Relevance score (higher = more relevant)
        item: The actual item (InstructionFragment or Tool)

    Example:
        >>> result = SearchResult(
        ...     id="instruction_1",
        ...     score=0.92,
        ...     item=instruction_fragment
        ... )
    """

    id: str
    score: float
    item: Any
