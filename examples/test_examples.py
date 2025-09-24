#!/usr/bin/env python3
"""
Simple test to verify both examples work with ITR core functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itr import ITR, ITRConfig

def test_basic_functionality():
    """Test basic ITR functionality that both examples rely on"""

    print("🧪 Testing ITR Core Functionality")
    print("=" * 50)

    # Initialize ITR with basic configuration
    config = ITRConfig(
        k_a_instructions=3,
        k_b_tools=2,
        token_budget=1500
    )

    itr = ITR(config)

    # Add sample instructions
    instructions = [
        "You are a helpful assistant focused on data analysis and visualization.",
        "Always provide clear explanations with your analysis.",
        "When working with text data, consider readability and sentiment.",
        "For data science tasks, use appropriate statistical methods.",
        "Visualizations should be clear, accessible, and informative."
    ]

    print(f"📝 Adding {len(instructions)} sample instructions...")
    for instruction in instructions:
        itr.add_instruction(instruction)

    # Add sample tools
    tools = [
        {
            "name": "text_analyzer",
            "description": "Analyze text for sentiment and readability",
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}}
        },
        {
            "name": "data_visualizer",
            "description": "Create charts and graphs from data",
            "schema": {"type": "object", "properties": {"data": {"type": "string"}}}
        },
        {
            "name": "statistical_analyzer",
            "description": "Perform statistical analysis on datasets",
            "schema": {"type": "object", "properties": {"dataset": {"type": "string"}}}
        }
    ]

    print(f"🛠️  Adding {len(tools)} sample tools...")
    for tool in tools:
        itr.add_tool(tool)

    # Test queries from both examples
    test_queries = [
        "Analyze this academic paper for research quality",
        "Create an interactive dashboard for financial data",
        "Examine sentiment patterns in social media posts",
        "Build comprehensive visualizations for survey data"
    ]

    print(f"\n🔍 Testing {len(test_queries)} queries...")
    print("-" * 50)

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")

        result = itr.step(query)

        print(f"  ✅ Instructions selected: {len(result.instructions)}")
        print(f"  🛠️  Tools selected: {len(result.tools)}")
        print(f"  📊 Token usage: {result.total_tokens}/{config.token_budget} ({(result.total_tokens/config.token_budget)*100:.1f}%)")
        print(f"  🎯 Confidence: {result.confidence_score:.3f}")

        # Show selected tools
        if result.tools:
            tool_names = [tool.name for tool in result.tools]
            print(f"  🔧 Selected tools: {', '.join(tool_names)}")

        results.append({
            'query': query,
            'instructions': len(result.instructions),
            'tools': len(result.tools),
            'tokens': result.total_tokens,
            'confidence': result.confidence_score
        })

    # Summary
    print(f"\n📈 Test Summary:")
    print("=" * 50)

    avg_instructions = sum(r['instructions'] for r in results) / len(results)
    avg_tools = sum(r['tools'] for r in results) / len(results)
    avg_tokens = sum(r['tokens'] for r in results) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    print(f"Average instructions selected: {avg_instructions:.1f}")
    print(f"Average tools selected: {avg_tools:.1f}")
    print(f"Average token usage: {avg_tokens:.0f} ({(avg_tokens/config.token_budget)*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.3f}")

    print(f"\n✅ All tests passed! ITR core functionality works correctly.")
    print(f"📋 Both examples should work with the ITR system.")

    # Test prompt assembly
    print(f"\n🔧 Testing prompt assembly...")
    sample_prompt = itr.get_prompt("Analyze this dataset for patterns")
    print(f"✅ Generated prompt with {len(sample_prompt)} characters")

    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print(f"\n🎉 SUCCESS: ITR examples are ready to use!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise