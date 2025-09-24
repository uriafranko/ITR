import json
from pathlib import Path
from typing import Optional

import click

from .. import ITR
from ..core.config import ITRConfig


def load_config(config_path: Optional[str]) -> ITRConfig:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        return ITRConfig(**config_dict)
    return ITRConfig()


@click.group()
def cli():
    """ITR - Instruction-Tool Retrieval CLI."""
    pass


@cli.command()
@click.option("--config", type=click.Path(), help="Configuration file path")
@click.option("--instructions", type=click.Path(), help="Instructions file path")
@click.option("--tools", type=click.Path(), help="Tools JSON file path")
@click.option("--query", type=str, required=True, help="Query to process")
@click.option("--show-prompt", is_flag=True, help="Show assembled prompt")
def retrieve(config, instructions, tools, query, show_prompt):
    """Perform ITR retrieval for a query."""
    # Load configuration
    cfg = load_config(config)

    # Initialize ITR
    itr = ITR(cfg)

    # Load corpus if provided
    if instructions:
        click.echo(f"Loading instructions from {instructions}")
        itr.load_instructions(instructions)

    if tools:
        click.echo(f"Loading tools from {tools}")
        itr.load_tools(tools)

    # Perform retrieval
    result = itr.step(query)

    # Display results
    click.echo("\n" + "=" * 50)
    click.echo(f"Query: {query}")
    click.echo("=" * 50)

    click.echo("\n📊 Retrieval Results:")
    click.echo(f"  • Instructions selected: {len(result.instructions)}")
    click.echo(f"  • Tools selected: {len(result.tools)}")
    click.echo(f"  • Total tokens: {result.total_tokens}")
    click.echo(f"  • Confidence: {result.confidence_score:.2f}")

    if result.instructions:
        click.echo("\n📝 Selected Instructions:")
        for i, inst in enumerate(result.instructions[:3], 1):  # Show first 3
            preview = (
                inst.content[:100] + "..." if len(inst.content) > 100 else inst.content
            )
            click.echo(f"  {i}. [{inst.fragment_type.value}] {preview}")

    if result.tools:
        click.echo("\n🔧 Selected Tools:")
        for i, tool in enumerate(result.tools, 1):
            click.echo(f"  {i}. {tool.name}: {tool.description[:80]}...")

    if show_prompt:
        click.echo("\n" + "=" * 50)
        click.echo("Assembled Prompt:")
        click.echo("=" * 50)
        prompt = itr.get_prompt(query)
        click.echo(prompt)


@cli.command()
@click.option(
    "--output", type=click.Path(), default="config.json", help="Output config file"
)
def init_config(output):
    """Generate a default configuration file."""
    config = ITRConfig()
    config_dict = {
        "top_m_instructions": config.top_m_instructions,
        "top_m_tools": config.top_m_tools,
        "k_a_instructions": config.k_a_instructions,
        "k_b_tools": config.k_b_tools,
        "dense_weight": config.dense_weight,
        "sparse_weight": config.sparse_weight,
        "rerank_weight": config.rerank_weight,
        "token_budget": config.token_budget,
        "safety_overlay_tokens": config.safety_overlay_tokens,
        "confidence_threshold": config.confidence_threshold,
        "discovery_expansion_factor": config.discovery_expansion_factor,
        "cache_ttl_seconds": config.cache_ttl_seconds,
        "max_cache_size_mb": config.max_cache_size_mb,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
    }

    with open(output, "w") as f:
        json.dump(config_dict, f, indent=2)

    click.echo(f"Configuration file created: {output}")


@cli.command()
def interactive():
    """Run ITR in interactive mode."""
    click.echo("ITR Interactive Mode")
    click.echo("Type 'quit' to exit\n")

    # Initialize ITR
    itr = ITR()

    # Simple demo data
    itr.add_instruction(
        "You are a helpful AI assistant. Be concise and accurate.",
        {"source": "default"},
    )
    itr.add_instruction(
        "Always prioritize safety and ethical considerations.", {"source": "safety"}
    )
    itr.add_instruction(
        "When dealing with technical questions, provide detailed explanations.",
        {"source": "technical"},
    )

    # Add some demo tools
    demo_tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "schema": {"expression": "string"},
        },
        {
            "name": "web_search",
            "description": "Search the web for information",
            "schema": {"query": "string", "max_results": "integer"},
        },
        {
            "name": "code_executor",
            "description": "Execute Python code snippets",
            "schema": {"code": "string"},
        },
    ]

    for tool in demo_tools:
        itr.add_tool(tool)

    while True:
        query = click.prompt("\n💭 Enter your query", type=str)

        if query.lower() == "quit":
            click.echo("Goodbye!")
            break

        # Perform retrieval
        result = itr.step(query)

        # Display results
        click.echo(
            f"\n✅ Retrieved {len(result.instructions)} instructions and {len(result.tools)} tools"
        )
        click.echo(f"📊 Token usage: {result.total_tokens}/{itr.config.token_budget}")

        if result.confidence_score < itr.config.confidence_threshold:
            click.echo(
                f"⚠️  Low confidence ({result.confidence_score:.2f}), triggering fallback..."
            )
            result = itr.handle_fallback(result, query)
            click.echo(f"✅ Fallback complete: now {len(result.tools)} tools available")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
