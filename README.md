# ITR: Instruction-Tool Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/uriafranko/ITR)](https://github.com/uriafranko/ITR/stargazers)
[![GitHub contributors](https://img.shields.io/github/contributors/uriafranko/ITR)](https://github.com/uriafranko/ITR/graphs/contributors)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**ITR (Instruction-Tool Retrieval)** is a sophisticated system for efficiently retrieving and assembling the most relevant instructions and tools for agentic Large Language Models (LLMs). It enables intelligent context management for AI agents by dynamically selecting the optimal subset of instructions and tools based on user queries and token budget constraints.

> 💡 **Key Benefits**: Reduce context bloat, improve tool selection and latency, and optimize token usage through intelligent hybrid retrieval and budget-aware selection algorithms.

## 📑 Table of Contents

- [ITR: Instruction-Tool Retrieval](#itr-instruction-tool-retrieval)
  - [📑 Table of Contents](#-table-of-contents)
  - [🚀 Features](#-features)
  - [📦 Installation](#-installation)
    - [Using uv (Recommended)](#using-uv-recommended)
    - [Using pip](#using-pip)
    - [Development Installation](#development-installation)
  - [⚡ Quick Start](#-quick-start)
    - [Basic Usage](#basic-usage)
    - [Loading from Files](#loading-from-files)
    - [Using Pre-created Fragments](#using-pre-created-fragments)
  - [🖥️ CLI Usage](#️-cli-usage)
    - [Basic Retrieval](#basic-retrieval)
    - [Interactive Mode](#interactive-mode)
    - [Generate Configuration File](#generate-configuration-file)
    - [Using Custom Configuration](#using-custom-configuration)
  - [⚙️ Configuration](#️-configuration)
    - [Configuration Files](#configuration-files)
  - [🔍 How It Works](#-how-it-works)
    - [Fragment Types](#fragment-types)
  - [🎯 Performance](#-performance)
    - [Token Optimization Results](#token-optimization-results)
  - [🤝 Contributing](#-contributing)
    - [Development Setup](#development-setup)
  - [📄 License](#-license)
  - [🔗 Links](#-links)
  - [📚 Examples](#-examples)
  - [🔧 Troubleshooting](#-troubleshooting)
    - [Common Issues](#common-issues)

## 🚀 Features

- **Hybrid Retrieval**: Combines dense (embedding-based) and sparse (BM25) retrieval methods
- **Budget-Aware Selection**: Intelligent token budget management with greedy optimization
- **Dynamic Chunking**: Smart text chunking with configurable size ranges
- **Fallback Mechanisms**: Automatic expansion of tool sets for better coverage
- **CLI Interface**: Easy-to-use command-line interface for testing and integration
- **Flexible Configuration**: Configurable parameters for different use cases
- **Type Safety**: Full type hints for better development experience

## 📦 Installation

### Using uv (Recommended)

```bash
# Install ITR
uv add instruction-tool-retrieval

# Or install from source
git clone https://github.com/uriafranko/ITR.git
cd ITR
uv sync
```

### Using pip

```bash
pip install instruction-tool-retrieval
```

### Development Installation

```bash
git clone https://github.com/uriafranko/ITR.git
cd ITR
uv sync --dev
```

## ⚡ Quick Start

### Basic Usage

```python
from itr import ITR, ITRConfig

# Initialize ITR with custom configuration
config = ITRConfig(
    k_a_instructions=3,  # Max instructions to select
    k_b_tools=2,         # Max tools to select
    token_budget=1500    # Total token budget
)
itr = ITR(config)

# Add instructions
itr.add_instruction(
    "You are a helpful AI assistant. Be concise and accurate.",
    metadata={"source": "base_personality", "priority": 1}
)
itr.add_instruction(
    "Always prioritize safety and ethical considerations in your responses.",
    metadata={"source": "safety_guidelines", "priority": 2}
)

# Add tools
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations and arithmetic operations",
    "schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        }
    }
}
itr.add_tool(calculator_tool)

# Perform retrieval
query = "What is 15% of 240?"
result = itr.step(query)

print(f"Selected {len(result.instructions)} instructions and {len(result.tools)} tools")
print(f"Total tokens: {result.total_tokens}")
print(f"Confidence: {result.confidence_score:.2f}")

# Get assembled prompt
prompt = itr.get_prompt(query)
print("\\nAssembled Prompt:")
print(prompt)
```

### Loading from Files

```python
from itr import ITR

itr = ITR()

# Load instructions from text file
itr.load_instructions("path/to/instructions.txt")

# Load tools from JSON file
itr.load_tools("path/to/tools.json")

# Perform retrieval
result = itr.step("How do I process a CSV file?")
```

### Using Pre-created Fragments

```python
from itr import ITR, InstructionFragment, FragmentType

# Create fragments manually
fragments = [
    InstructionFragment(
        id="custom_1",
        content="Use clear and concise language",
        token_count=6,
        fragment_type=FragmentType.STYLE_RULE,
        metadata={"custom": True}
    ),
    InstructionFragment(
        id="custom_2",
        content="Provide step-by-step explanations for complex topics",
        token_count=9,
        fragment_type=FragmentType.DOMAIN_SPECIFIC
    )
]

itr = ITR()
itr.add_instruction_fragments(fragments)

result = itr.step("Explain machine learning")
```

## 🖥️ CLI Usage

ITR provides a command-line interface for easy testing and integration:

### Basic Retrieval

```bash
itr retrieve --query "How do I calculate compound interest?" \\
             --instructions instructions.txt \\
             --tools tools.json \\
             --show-prompt
```

### Interactive Mode

```bash
itr interactive
```

### Generate Configuration File

```bash
itr init-config --output my-config.json
```

### Using Custom Configuration

```bash
itr retrieve --config my-config.json \\
             --query "Analyze this dataset" \\
             --instructions data_instructions.txt \\
             --tools analysis_tools.json
```

## ⚙️ Configuration

ITR uses a flexible configuration system. You can customize behavior through the `ITRConfig` class:

```python
from itr import ITRConfig

config = ITRConfig(
    # Retrieval parameters
    top_m_instructions=20,      # Candidates to retrieve
    top_m_tools=15,            # Tool candidates to retrieve
    k_a_instructions=4,        # Max instructions to select
    k_b_tools=2,              # Max tools to select

    # Scoring weights for hybrid retrieval
    dense_weight=0.4,         # Embedding similarity weight
    sparse_weight=0.3,        # BM25 score weight
    rerank_weight=0.3,        # Reranking weight

    # Budget management
    token_budget=2000,        # Total token limit
    safety_overlay_tokens=200, # Reserved tokens

    # Fallback settings
    confidence_threshold=0.7,         # Trigger fallback below this
    discovery_expansion_factor=2.0,   # Tool expansion multiplier

    # Model settings
    embedding_model="all-MiniLM-L6-v2",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### Configuration Files

ITR supports JSON, YAML and .env configuration files:

```json
{
  "k_a_instructions": 3,
  "k_b_tools": 2,
  "token_budget": 1500,
  "confidence_threshold": 0.8,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

## 🔍 How It Works

ITR uses a multi-stage pipeline for instruction and tool retrieval:

1. **Indexing**: Instructions are chunked and indexed with both dense embeddings and sparse representations
2. **Retrieval**: User queries are processed through hybrid retrieval (dense + sparse)
3. **Selection**: Budget-aware selection algorithm picks optimal subset within token limits
4. **Assembly**: Selected instructions and tools are assembled into final prompt
5. **Fallback**: If confidence is low, expanded tool sets are used

### Fragment Types

ITR automatically categorizes instruction fragments:

- `ROLE_GUIDANCE`: Defines AI agent roles and personas
- `STYLE_RULE`: Formatting and communication style guidelines
- `SAFETY_POLICY`: Safety and ethical constraints
- `DOMAIN_SPECIFIC`: Task-specific instructions
- `EXEMPLAR`: Examples and demonstrations

## 🎯 Performance

ITR is designed for efficiency:

- **Fast Retrieval**: Optimized hybrid search with caching
- **Memory Efficient**: Lazy loading and smart chunking
- **Scalable**: Handles large instruction/tool corpora
- **Token Aware**: Precise token counting and budget management

### Token Optimization Results

![ITR Performance Results](examples/results.png)

*ITR achieves significant token reduction while maintaining high confidence and functionality across diverse analysis types.*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/uria-franko/ITR.git
cd ITR
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy itr/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI Package**: *Coming Soon*
- **Issue Tracker**: [https://github.com/uriafranko/ITR/issues](https://github.com/uriafranko/ITR/issues)

## 📚 Examples

Check out the [examples](examples/) directory for more detailed usage examples:

- Basic retrieval workflows
- Custom instruction creation
- Tool integration patterns
- Configuration examples
- Performance optimization tips

## 🔧 Troubleshooting

### Common Issues

**"Module not found" errors**: Make sure you've installed all dependencies:

```bash
uv sync
```

**Memory issues with large corpora**: Use chunking and increase available memory:

```python
config = ITRConfig(chunk_size_range=(100, 400))  # Smaller chunks
```

**Poor retrieval quality**: Try adjusting scoring weights:

```python
config = ITRConfig(
    dense_weight=0.6,    # Increase embedding weight
    sparse_weight=0.4    # Decrease keyword weight
)
```

For more help, please [open an issue](https://github.com/uriafranko/ITR/issues) on GitHub.

---

<div align="center">
  <p><strong>Built with ❤️ for the AI Agent community</strong></p>
  <p>⭐ Star this repo if you find ITR useful!</p>
</div>
