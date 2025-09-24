#!/usr/bin/env python3
"""Quick start example for ITR."""

from itr import ITR, ITRConfig


def main():
    # Initialize ITR with custom config
    config = ITRConfig(k_a_instructions=3, k_b_tools=2, token_budget=1500)
    itr = ITR(config)

    # Add some instructions to the corpus
    instructions = [
        "You are a helpful AI assistant focused on providing accurate and concise answers.",
        "Always consider safety and ethical implications in your responses.",
        "When answering technical questions, provide code examples when appropriate.",
        "Be polite and professional in all interactions.",
        "If you're unsure about something, acknowledge the uncertainty.",
    ]

    for instruction in instructions:
        itr.add_instruction(instruction, {"source": "system"})

    # Add some tools
    tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for current information",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
            },
        },
        {
            "name": "file_reader",
            "description": "Read contents of a file",
            "schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
            },
        },
        {
            "name": "code_executor",
            "description": "Execute Python code",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
            },
        },
    ]

    for tool in tools:
        itr.add_tool(tool)

    # Example queries
    queries = [
        "Calculate the square root of 144",
        "What's the current weather in San Francisco?",
        "Write a Python function to reverse a string",
        "Explain quantum computing in simple terms",
    ]

    print("ITR Quick Start Demo")
    print("=" * 50)

    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)

        # Perform retrieval
        result = itr.step(query)

        print("✅ Results:")
        print(f"  • Instructions: {len(result.instructions)} selected")
        print(f"  • Tools: {len(result.tools)} selected")
        print(f"  • Tokens: {result.total_tokens}/{config.token_budget}")
        print(f"  • Confidence: {result.confidence_score:.2f}")

        # Show selected tools
        if result.tools:
            print("\n🔧 Selected Tools:")
            for tool in result.tools:
                print(f"  - {tool.name}: {tool.description}")

        # Check if fallback needed
        if result.confidence_score < config.confidence_threshold:
            print("\n⚠️  Low confidence, triggering fallback...")
            result = itr.handle_fallback(result, query)
            print(f"  • Expanded to {len(result.tools)} tools")

    # Show how to get the assembled prompt
    print("\n" + "=" * 50)
    print("Example Assembled Prompt:")
    print("=" * 50)
    prompt = itr.get_prompt(queries[0])
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)


if __name__ == "__main__":
    main()
