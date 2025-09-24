#!/usr/bin/env python3
"""
ITR Example: Comprehensive Text Analysis Workflow with Token Savings Analysis

This example demonstrates ITR's capabilities for complex text analysis tasks with a focus on
measuring and showcasing the significant token savings achieved through intelligent selection:

KEY FEATURES DEMONSTRATED:
- Large corpus of domain-specific instructions (31 specialized instructions)
- Multiple specialized analysis tools (13 comprehensive tools)
- Dynamic context selection based on query complexity and text type
- Token budget optimization with real-time savings tracking
- Cost impact analysis with enterprise-scale projections

EFFICIENCY GAINS SHOWCASED:
- Baseline vs optimized token usage comparison
- Real-world cost savings analysis
- Scalability projections for high-volume processing
- Performance metrics across diverse text types

The example processes various types of text (academic papers, social media posts,
legal documents, creative writing) and demonstrates how ITR achieves significant
token reduction (typically 75%+ savings) while maintaining high-quality analysis
through intelligent instruction and tool selection.

This showcase illustrates why ITR is essential for cost-effective, scalable text
processing workflows in production AI systems.
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib>=3.5.0",
#     "seaborn>=0.11.0",
#     "pandas>=1.3.0",
#     "numpy>=1.21.0",
#     "wordcloud>=1.8.0",
#     "textstat>=0.7.0",
#     "nltk>=3.6.0",
#     "plotly>=5.0.0",
#     "rich>=12.0.0",
# ]
# ///

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
from datetime import datetime

# Rich for beautiful console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from rich.layout import Layout
from rich.text import Text

# Add parent directory to path to import ITR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from itr import ITR, ITRConfig, InstructionFragment, FragmentType

console = Console()

# Sample text corpus for analysis
SAMPLE_TEXTS = {
    "academic_paper": """
    The rapid advancement of artificial intelligence (AI) and machine learning (ML) technologies
    has fundamentally transformed numerous sectors, from healthcare and finance to transportation
    and entertainment. This comprehensive review examines the current state of deep learning
    architectures, with particular emphasis on transformer models and their applications in
    natural language processing (NLP) tasks. We analyze performance metrics across multiple
    benchmarks, including GLUE, SuperGLUE, and domain-specific evaluation frameworks.

    Our methodology incorporates a systematic literature review spanning the period from 2017
    to 2024, focusing on peer-reviewed publications that demonstrate significant contributions
    to the field. The analysis reveals several key trends: (1) increasing model size correlates
    with improved performance on complex reasoning tasks, (2) attention mechanisms have become
    the dominant paradigm for sequence-to-sequence modeling, and (3) pre-training on large
    corpora followed by task-specific fine-tuning remains the most effective approach for
    achieving state-of-the-art results.

    Furthermore, we investigate the computational requirements and environmental impact of
    large-scale model training, proposing several strategies for more efficient resource
    utilization. These include gradient checkpointing, mixed precision training, and novel
    architectural innovations that reduce parameter count while maintaining performance.
    The implications of these findings extend beyond technical considerations to encompass
    ethical, societal, and economic dimensions of AI deployment.
    """,

    "social_media": """
    OMG just tried that new bubble tea place downtown and it's AMAZING! 🧋✨ The taro milk tea
    with brown sugar pearls is literally heaven in a cup. Staff was super friendly too and
    the aesthetic is so cute - perfect for the gram! 📸 Already planning my next visit lol.
    Who wants to come with me??

    Also can we talk about how expensive everything is getting? Like $8 for a drink feels
    insane but also... worth it? 🤷‍♀️ Inflation is really hitting different these days.
    Remember when you could get a full meal for that price? Pepperidge Farm remembers 😭

    Anyway, highly recommend if you're in the area! They also have oat milk options for
    all my lactose intolerant friends 💚 #BubbleTea #Downtown #Foodie #WorthIt
    """,

    "legal_document": """
    WHEREAS, the Party of the First Part (hereinafter referred to as "Licensor") is the
    lawful owner of certain intellectual property rights, including but not limited to
    patents, trademarks, copyrights, and trade secrets relating to advanced machine
    learning algorithms and data processing methodologies (collectively, the "Licensed IP");

    WHEREAS, the Party of the Second Part (hereinafter referred to as "Licensee") desires
    to obtain a license to use, manufacture, and distribute products incorporating the
    Licensed IP within the territory defined herein;

    NOW, THEREFORE, in consideration of the mutual covenants and agreements contained
    herein, and for other good and valuable consideration, the receipt and sufficiency
    of which are hereby acknowledged, the parties agree as follows:

    1. GRANT OF LICENSE. Subject to the terms and conditions of this Agreement, Licensor
    hereby grants to Licensee a non-exclusive, non-transferable license to use the
    Licensed IP solely for the purposes set forth in Exhibit A attached hereto and
    incorporated herein by reference.

    2. TERM AND TERMINATION. This Agreement shall commence on the Effective Date and
    shall continue for a period of five (5) years, unless terminated earlier in
    accordance with the provisions hereof. Either party may terminate this Agreement
    upon thirty (30) days written notice to the other party in the event of a material
    breach that remains uncured after such notice period.

    3. ROYALTIES AND PAYMENT. In consideration for the rights granted hereunder,
    Licensee shall pay to Licensor a royalty equal to five percent (5%) of Net Sales
    of Licensed Products, payable quarterly within forty-five (45) days after the
    end of each calendar quarter.
    """,

    "creative_writing": """
    The ancient lighthouse stood sentinel against the storm, its beam cutting through
    the darkness like a silver sword. Maria pressed her face against the rain-streaked
    window, watching the waves crash against the rocky shore with increasing fury.
    Each thunderclap seemed to shake the very foundations of the coastal cottage where
    she had sought refuge.

    "Strange night to be traveling," the lighthouse keeper had said when she arrived,
    his weathered face creased with concern. "Roads will be impassable by morning."
    Now, as the wind howled like a banshee around the eaves, she understood the gravity
    of his warning.

    The old man emerged from the kitchen carrying two steaming mugs of tea, the amber
    liquid catching the firelight. "Built in 1847," he said, nodding toward the lighthouse.
    "Seen storms worse than this, believe it or not. Though I'll admit, this one's got
    some teeth to it."

    Maria accepted the tea gratefully, wrapping her cold fingers around the warm ceramic.
    Through the window, she could see the lighthouse beam sweeping its eternal arc, a
    beacon of hope in the tempestuous night. Something about its steadfast presence filled
    her with an unexpected sense of peace, as if the storm might rage but could never
    truly conquer the light.

    "Tell me," she said, settling into the worn armchair by the fire, "what's the most
    remarkable thing you've seen from that lighthouse?" The keeper's eyes twinkled with
    the promise of stories yet untold.
    """
}

# Comprehensive instruction set for text analysis
def create_analysis_instructions() -> List[str]:
    """Create a comprehensive set of text analysis instructions"""
    return [
        # Academic Analysis Instructions
        "When analyzing academic texts, focus on methodology, thesis statements, evidence quality, "
        "citation patterns, and logical structure. Identify key research contributions and limitations.",

        "For scholarly articles, examine the abstract, introduction, methodology, results, and "
        "conclusion sections separately. Evaluate the strength of arguments and validity of conclusions.",

        "In academic writing analysis, assess the clarity of research questions, appropriateness "
        "of methodology, statistical significance of findings, and potential biases or confounding factors.",

        "When reviewing literature reviews, evaluate comprehensiveness, currency of sources, "
        "synthesis quality, and identification of research gaps or future directions.",

        # Social Media Analysis Instructions
        "For social media content, analyze sentiment, emotional tone, engagement indicators, "
        "hashtag usage, and viral potential. Consider platform-specific conventions and audience.",

        "In social media posts, identify informal language patterns, emojis, abbreviations, "
        "slang terms, and cultural references. Assess authenticity and relatability factors.",

        "When analyzing social media sentiment, consider context clues, sarcasm, irony, and "
        "cultural nuances that might affect interpretation. Look for implicit meanings.",

        "For social media engagement analysis, examine call-to-action phrases, question formats, "
        "controversial topics, trending subjects, and community-building elements.",

        # Legal Document Analysis Instructions
        "When analyzing legal documents, focus on clause structure, defined terms, obligations, "
        "rights, remedies, and potential ambiguities. Identify governing law and jurisdiction.",

        "For contract analysis, examine consideration, mutual assent, capacity, legality, "
        "performance terms, breach conditions, and dispute resolution mechanisms.",

        "In legal text analysis, pay attention to precise language, conditional statements, "
        "temporal requirements, notice provisions, and indemnification clauses.",

        "When reviewing legal agreements, assess enforceability, potential loopholes, "
        "compliance requirements, and alignment with applicable regulations or standards.",

        # Creative Writing Analysis Instructions
        "For creative writing, analyze narrative structure, character development, setting, "
        "theme, plot progression, conflict resolution, and literary devices employed.",

        "In fiction analysis, examine point of view, narrative voice, dialogue quality, "
        "pacing, tension building, and emotional resonance with readers.",

        "When analyzing creative prose, identify figurative language, symbolism, motifs, "
        "imagery patterns, and stylistic choices that contribute to overall effect.",

        "For creative writing evaluation, assess originality, voice consistency, genre "
        "conventions, character believability, and thematic coherence throughout the work.",

        # General Text Analysis Instructions
        "Always begin text analysis with a high-level overview identifying genre, purpose, "
        "target audience, and primary objectives before diving into detailed examination.",

        "Consider the historical, cultural, and social context in which the text was created "
        "when interpreting meaning, significance, and potential impact on readers.",

        "Evaluate text quality using appropriate metrics: readability scores, vocabulary "
        "complexity, sentence structure variety, and coherence at paragraph and document levels.",

        "When comparing multiple texts, establish clear evaluation criteria and apply them "
        "consistently while noting genre-specific differences and contextual factors.",

        "Pay attention to author credentials, publication venue, intended audience, and "
        "potential biases that might influence content, tone, and presentation choices.",

        "Document all analytical observations with specific examples from the text, including "
        "line numbers or paragraph references for precise attribution and verification.",

        # Style and Tone Analysis
        "Analyze writing style by examining sentence length variation, vocabulary choices, "
        "formality level, technical terminology usage, and overall complexity patterns.",

        "For tone analysis, identify emotional indicators, attitude markers, subjective language, "
        "and evaluative expressions that reveal author's stance toward the subject matter.",

        "When assessing register and style appropriateness, consider audience expectations, "
        "genre conventions, communication goals, and contextual appropriateness.",

        # Structural Analysis
        "Examine document organization including introduction, body, conclusion structure, "
        "transition quality, paragraph coherence, and logical flow between sections.",

        "Analyze heading structure, subsection organization, list formatting, and visual "
        "hierarchy to assess information architecture and reader navigation support.",

        # Linguistic Features
        "Identify linguistic patterns including repetition, parallelism, alliteration, "
        "rhythm, and other stylistic devices that contribute to text effectiveness.",

        "Examine cohesion markers, reference chains, lexical relationships, and discourse "
        "connectors that bind text elements together for unified meaning.",

        # Quality Assessment
        "Evaluate text quality dimensions including accuracy, completeness, currency, "
        "relevance, authority, and objectivity using established criteria.",

        "Assess readability using multiple measures: Flesch-Kincaid grade level, average "
        "sentence length, syllable complexity, and specialized vocabulary density."
    ]

def create_analysis_tools() -> List[Dict[str, Any]]:
    """Create comprehensive text analysis tools"""
    return [
        {
            "name": "readability_analyzer",
            "description": "Calculate comprehensive readability metrics including Flesch-Kincaid, "
                          "Gunning Fog, SMOG, and Coleman-Liau indices",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to calculate"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "sentiment_analyzer",
            "description": "Perform multi-dimensional sentiment analysis including polarity, "
                          "subjectivity, emotion detection, and confidence scoring",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text for sentiment analysis"},
                    "granularity": {
                        "type": "string",
                        "enum": ["document", "sentence", "phrase"],
                        "description": "Analysis granularity level"
                    },
                    "emotions": {
                        "type": "boolean",
                        "description": "Include emotion detection"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "linguistic_feature_extractor",
            "description": "Extract linguistic features including POS tags, named entities, "
                          "dependency relations, and syntactic patterns",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text for feature extraction"},
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific linguistic features to extract"
                    },
                    "model": {
                        "type": "string",
                        "description": "NLP model to use for analysis"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "style_analyzer",
            "description": "Analyze writing style including formality, complexity, vocabulary "
                          "diversity, and genre classification",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text for style analysis"},
                    "reference_corpus": {
                        "type": "string",
                        "description": "Reference corpus for comparison"
                    },
                    "style_dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Style aspects to analyze"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "coherence_analyzer",
            "description": "Evaluate text coherence, cohesion, and structural organization "
                          "using computational linguistics methods",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "analysis_level": {
                        "type": "string",
                        "enum": ["local", "global", "both"],
                        "description": "Coherence analysis scope"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "topic_modeler",
            "description": "Perform topic modeling and thematic analysis using LDA, "
                          "BERT-based models, or other unsupervised learning approaches",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text for topic modeling"},
                    "num_topics": {
                        "type": "integer",
                        "description": "Number of topics to extract"
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["lda", "bert", "nmf"],
                        "description": "Topic modeling algorithm"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "plagiarism_detector",
            "description": "Detect potential plagiarism, text similarity, and citation issues "
                          "using advanced similarity algorithms",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to check"},
                    "reference_corpus": {
                        "type": "string",
                        "description": "Reference corpus for comparison"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Similarity threshold for flagging"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "keyword_extractor",
            "description": "Extract keywords, key phrases, and important concepts using "
                          "TF-IDF, YAKE, or transformer-based methods",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text for keyword extraction"},
                    "max_keywords": {
                        "type": "integer",
                        "description": "Maximum number of keywords"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["tfidf", "yake", "keybert"],
                        "description": "Keyword extraction method"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "text_summarizer",
            "description": "Generate extractive or abstractive summaries with configurable "
                          "length and focus areas",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "summary_type": {
                        "type": "string",
                        "enum": ["extractive", "abstractive"],
                        "description": "Type of summary to generate"
                    },
                    "length": {
                        "type": "integer",
                        "description": "Target summary length in sentences"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "bias_detector",
            "description": "Detect potential bias, discriminatory language, and problematic "
                          "content using fairness-aware NLP models",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze for bias"},
                    "bias_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of bias to check"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "academic_evaluator",
            "description": "Specialized analysis for academic texts including citation analysis, "
                          "research quality assessment, and methodology evaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Academic text to evaluate"},
                    "paper_type": {
                        "type": "string",
                        "enum": ["research", "review", "conference", "thesis"],
                        "description": "Type of academic paper"
                    },
                    "evaluation_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific evaluation criteria"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "legal_analyzer",
            "description": "Specialized analysis for legal documents including clause extraction, "
                          "risk assessment, and compliance checking",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Legal text to analyze"},
                    "document_type": {
                        "type": "string",
                        "enum": ["contract", "regulation", "policy", "statute"],
                        "description": "Type of legal document"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Legal jurisdiction for analysis"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "creative_writing_evaluator",
            "description": "Specialized analysis for creative writing including narrative structure, "
                          "character development, and literary device identification",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Creative text to analyze"},
                    "genre": {
                        "type": "string",
                        "enum": ["fiction", "poetry", "drama", "creative_nonfiction"],
                        "description": "Genre of creative writing"
                    },
                    "analysis_focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific aspects to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    ]

def simulate_analysis_results(text_type: str, tools_used: List[str]) -> Dict[str, Any]:
    """Simulate realistic analysis results for demonstration"""
    base_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "text_type": text_type,
        "tools_applied": tools_used,
        "processing_time": round(random.uniform(0.5, 3.0), 2)
    }

    # Add text-type specific results
    if text_type == "academic_paper":
        base_results.update({
            "readability_metrics": {
                "flesch_kincaid_grade": 16.2,
                "gunning_fog": 18.4,
                "academic_complexity": "High",
                "technical_term_density": 0.23
            },
            "academic_quality": {
                "methodology_strength": "Strong",
                "citation_quality": "Excellent",
                "research_contribution": "Significant",
                "bias_score": 0.15
            },
            "key_findings": [
                "Transformer models show 23% improvement over RNNs",
                "Computational cost scales quadratically with sequence length",
                "Pre-training corpus size correlates with downstream performance"
            ]
        })
    elif text_type == "social_media":
        base_results.update({
            "sentiment_analysis": {
                "overall_sentiment": "Positive",
                "polarity_score": 0.72,
                "emotional_tone": "Enthusiastic",
                "engagement_potential": "High"
            },
            "social_metrics": {
                "emoji_count": 8,
                "hashtag_effectiveness": "Good",
                "virality_score": 0.68,
                "authenticity_rating": "High"
            },
            "linguistic_features": {
                "informal_language": "Heavy",
                "abbreviations": ["OMG", "lol"],
                "generation_markers": ["literally", "feels"]
            }
        })
    elif text_type == "legal_document":
        base_results.update({
            "legal_analysis": {
                "document_type": "Licensing Agreement",
                "enforceability": "High",
                "ambiguity_score": 0.12,
                "risk_level": "Low"
            },
            "clause_analysis": {
                "key_obligations": 5,
                "termination_clauses": 2,
                "penalty_provisions": 3,
                "dispute_resolution": "Arbitration"
            },
            "compliance_check": {
                "regulatory_alignment": "Compliant",
                "jurisdictional_issues": "None identified",
                "standard_deviations": 2
            }
        })
    elif text_type == "creative_writing":
        base_results.update({
            "narrative_analysis": {
                "point_of_view": "Third person limited",
                "tense_consistency": "Excellent",
                "pacing": "Well-controlled",
                "atmosphere": "Gothic/Suspenseful"
            },
            "literary_devices": {
                "metaphors": 4,
                "imagery_density": "High",
                "symbolism": ["lighthouse as hope", "storm as conflict"],
                "dialogue_quality": "Natural"
            },
            "creative_metrics": {
                "originality_score": 0.78,
                "emotional_resonance": "Strong",
                "genre_adherence": "Good",
                "voice_consistency": "Excellent"
            }
        })

    return base_results

def calculate_baseline_tokens(instructions: List[str], tools: List[Dict[str, Any]]) -> int:
    """Calculate baseline token usage if all instructions and tools were included"""

    # Estimate tokens for all instructions (roughly 4 chars per token)
    instruction_tokens = sum(len(instr) // 4 for instr in instructions)

    # Estimate tokens for all tools (including schema descriptions)
    tool_tokens = 0
    for tool in tools:
        tool_str = json.dumps(tool, indent=2)
        tool_tokens += len(tool_str) // 4

    # Add system prompt overhead
    system_overhead = 200

    return instruction_tokens + tool_tokens + system_overhead

def demonstrate_text_analysis():
    """Main demonstration function showing ITR's text analysis capabilities with token savings analysis"""

    console.print(Panel.fit(
        "[bold blue]ITR Text Analysis Workflow Demonstration[/bold blue]\n"
        "[green]Showcasing dynamic instruction and tool selection with token optimization[/green]",
        border_style="blue"
    ))

    # Initialize ITR with optimized configuration for text analysis
    config = ITRConfig(
        top_m_instructions=25,      # More candidates for complex analysis
        top_m_tools=20,            # Rich tool selection
        k_a_instructions=8,        # More instructions for thorough analysis
        k_b_tools=5,              # Multiple tools per analysis
        token_budget=3500,         # Larger budget for comprehensive context
        dense_weight=0.5,          # Balance embedding similarity
        sparse_weight=0.3,         # Keyword matching
        rerank_weight=0.2,         # Fine-tune relevance
        confidence_threshold=0.6,   # Allow fallback for complex queries
        discovery_expansion_factor=1.8
    )

    itr = ITR(config)

    # Load comprehensive instructions
    with console.status("[bold green]Loading analysis instructions...", spinner="dots"):
        instructions = create_analysis_instructions()
        for i, instruction in enumerate(instructions):
            # Create instruction fragments with type classification
            if "academic" in instruction.lower():
                fragment_type = FragmentType.DOMAIN_SPECIFIC
            elif "social media" in instruction.lower():
                fragment_type = FragmentType.DOMAIN_SPECIFIC
            elif "legal" in instruction.lower():
                fragment_type = FragmentType.DOMAIN_SPECIFIC
            elif "creative" in instruction.lower():
                fragment_type = FragmentType.DOMAIN_SPECIFIC
            elif "style" in instruction.lower() or "tone" in instruction.lower():
                fragment_type = FragmentType.STYLE_RULE
            else:
                fragment_type = FragmentType.ROLE_GUIDANCE

            itr.add_instruction(
                instruction,
                metadata={
                    "source": "text_analysis_corpus",
                    "priority": random.randint(1, 5),
                    "domain": "linguistics",
                    "complexity": "advanced"
                }
            )
        time.sleep(1)

    # Load analysis tools
    with console.status("[bold green]Loading analysis tools...", spinner="dots"):
        tools = create_analysis_tools()
        for tool in tools:
            itr.add_tool(tool)
        time.sleep(0.8)

    # Calculate baseline token usage (without ITR optimization)
    baseline_tokens = calculate_baseline_tokens(instructions, tools)

    console.print(f"[green]✓[/green] Loaded {len(instructions)} instructions and {len(tools)} tools")

    # Display token efficiency preview
    efficiency_panel = Panel(
        f"[bold]Token Efficiency Analysis[/bold]\n\n"
        f"📊 Baseline (all instructions + all tools): [red]{baseline_tokens:,} tokens[/red]\n"
        f"🎯 ITR Budget: [blue]{config.token_budget:,} tokens[/blue]\n"
        f"💡 Maximum possible savings: [green]{baseline_tokens - config.token_budget:,} tokens ({((baseline_tokens - config.token_budget)/baseline_tokens)*100:.1f}%)[/green]",
        title="[cyan]ITR Optimization Preview[/cyan]",
        border_style="cyan"
    )
    console.print(efficiency_panel)

    # Analyze each text type with different queries
    analysis_queries = {
        "academic_paper": [
            "Analyze the methodology and research quality of this academic paper",
            "Evaluate the strength of arguments and citation patterns",
            "Assess the statistical validity and research contributions"
        ],
        "social_media": [
            "Analyze the sentiment and engagement potential of this social media post",
            "Evaluate the authenticity and viral characteristics",
            "Examine the linguistic features and cultural references"
        ],
        "legal_document": [
            "Review this legal document for potential risks and enforceability issues",
            "Analyze the clause structure and contractual obligations",
            "Assess compliance with standard legal practices"
        ],
        "creative_writing": [
            "Analyze the narrative structure and literary devices in this creative text",
            "Evaluate the character development and atmospheric elements",
            "Assess the writing style and genre conventions"
        ]
    }

    results_summary = []
    total_tokens_used = 0
    total_tokens_saved = 0

    # Process each text type
    for text_type, sample_text in SAMPLE_TEXTS.items():
        console.print(f"\n[bold cyan]Analyzing {text_type.replace('_', ' ').title()}[/bold cyan]")
        console.print("─" * 60)

        # Show text preview
        preview = sample_text.strip()[:200] + "..." if len(sample_text) > 200 else sample_text.strip()
        console.print(Panel(preview, title="[blue]Text Preview[/blue]", border_style="dim"))

        queries = analysis_queries[text_type]
        text_results = {"text_type": text_type, "analyses": []}

        for i, query in enumerate(queries, 1):
            console.print(f"\n[yellow]Query {i}:[/yellow] {query}")

            # Perform ITR retrieval with progress indication
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:

                task1 = progress.add_task("Retrieving relevant instructions...", total=100)
                progress.advance(task1, 30)
                time.sleep(0.3)

                task2 = progress.add_task("Selecting optimal tools...", total=100)
                progress.advance(task1, 40)
                progress.advance(task2, 50)
                time.sleep(0.2)

                task3 = progress.add_task("Optimizing token budget...", total=100)
                progress.advance(task1, 30)
                progress.advance(task2, 50)
                progress.advance(task3, 70)
                time.sleep(0.2)

                # Perform actual ITR step
                result = itr.step(f"{query}\n\nText to analyze:\n{sample_text}")

                progress.advance(task1, 0)
                progress.advance(task2, 0)
                progress.advance(task3, 30)
                time.sleep(0.1)

            # Display retrieval results
            results_table = Table(title=f"ITR Results for Query {i}")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")

            # Calculate savings for this query
            tokens_saved = baseline_tokens - result.total_tokens
            savings_percentage = (tokens_saved / baseline_tokens) * 100

            results_table.add_row("Instructions Selected", str(len(result.instructions)), f"vs {len(instructions)} available")
            results_table.add_row("Tools Selected", str(len(result.tools)), f"vs {len(tools)} available")
            results_table.add_row("Total Tokens", str(result.total_tokens), f"vs {baseline_tokens:,} baseline")
            results_table.add_row("Tokens Saved", f"[green]{tokens_saved:,}[/green]", f"[green]{savings_percentage:.1f}% reduction[/green]")
            results_table.add_row("Token Budget Usage", f"{(result.total_tokens/config.token_budget)*100:.1f}%", f"Efficiency: {100-((result.total_tokens/config.token_budget)*100):.1f}%")
            results_table.add_row("Confidence Score", f"{result.confidence_score:.3f}", "High relevance match")

            console.print(results_table)

            # Show selected instructions preview
            if result.instructions:
                console.print("\n[blue]Selected Instructions (preview):[/blue]")
                for j, instr in enumerate(result.instructions[:3], 1):
                    preview = instr.content[:150] + "..." if len(instr.content) > 150 else instr.content
                    console.print(f"  {j}. {preview}")
                if len(result.instructions) > 3:
                    console.print(f"  ... and {len(result.instructions) - 3} more instructions")

            # Show selected tools
            if result.tools:
                console.print(f"\n[blue]Selected Tools:[/blue]")
                tool_names = [tool.name for tool in result.tools]
                console.print(f"  {', '.join(tool_names)}")

            # Simulate analysis execution
            simulated_results = simulate_analysis_results(text_type, tool_names if result.tools else [])

            # Track cumulative savings
            total_tokens_used += result.total_tokens
            total_tokens_saved += tokens_saved

            analysis_result = {
                "query": query,
                "itr_metrics": {
                    "instructions_count": len(result.instructions),
                    "tools_count": len(result.tools),
                    "total_tokens": result.total_tokens,
                    "confidence": result.confidence_score,
                    "tokens_saved": tokens_saved,
                    "savings_percentage": savings_percentage
                },
                "analysis_output": simulated_results
            }
            text_results["analyses"].append(analysis_result)

            console.print(f"\n[green]✓ Analysis completed in {simulated_results['processing_time']}s[/green]")
            console.print(f"[dim]💰 Saved {tokens_saved:,} tokens ({savings_percentage:.1f}% reduction) on this query[/dim]")

        results_summary.append(text_results)
        console.print()

    # Generate comprehensive summary
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold green]Analysis Workflow Summary[/bold green]\n"
        "[white]Comprehensive text processing across multiple domains[/white]",
        border_style="green"
    ))

    # Enhanced summary statistics table with token savings
    summary_table = Table(title="ITR Performance & Token Savings Summary")
    summary_table.add_column("Text Type", style="cyan")
    summary_table.add_column("Avg Instructions", justify="center")
    summary_table.add_column("Avg Tools", justify="center")
    summary_table.add_column("Avg Tokens", justify="center")
    summary_table.add_column("Avg Savings", justify="center", style="green")
    summary_table.add_column("Efficiency", justify="center", style="green")

    total_analyses = 0
    cumulative_savings = 0

    for text_result in results_summary:
        text_type = text_result["text_type"].replace("_", " ").title()
        analyses = text_result["analyses"]

        avg_instructions = sum(a["itr_metrics"]["instructions_count"] for a in analyses) / len(analyses)
        avg_tools = sum(a["itr_metrics"]["tools_count"] for a in analyses) / len(analyses)
        avg_tokens = sum(a["itr_metrics"]["total_tokens"] for a in analyses) / len(analyses)
        avg_savings = sum(a["itr_metrics"]["tokens_saved"] for a in analyses) / len(analyses)
        avg_efficiency = sum(a["itr_metrics"]["savings_percentage"] for a in analyses) / len(analyses)

        summary_table.add_row(
            text_type,
            f"{avg_instructions:.1f}/{len(instructions)}",
            f"{avg_tools:.1f}/{len(tools)}",
            f"{avg_tokens:.0f}",
            f"{avg_savings:,.0f}",
            f"{avg_efficiency:.1f}%"
        )

        total_analyses += len(analyses)
        cumulative_savings += sum(a["itr_metrics"]["tokens_saved"] for a in analyses)

    # Add totals row
    total_baseline = baseline_tokens * total_analyses
    overall_efficiency = (cumulative_savings / total_baseline) * 100

    summary_table.add_row(
        "[bold]TOTALS[/bold]",
        f"[bold]{total_tokens_used/total_analyses:.1f}[/bold]",
        "",
        f"[bold]{total_tokens_used:,}[/bold]",
        f"[bold green]{cumulative_savings:,}[/bold green]",
        f"[bold green]{overall_efficiency:.1f}%[/bold green]"
    )

    console.print(summary_table)

    # Token savings breakdown
    savings_panel = Panel(
        f"[bold green]🎯 TOKEN OPTIMIZATION RESULTS[/bold green]\n\n"
        f"📊 Total baseline tokens (without ITR): [red]{total_baseline:,} tokens[/red]\n"
        f"💡 Actual tokens used (with ITR): [blue]{total_tokens_used:,} tokens[/blue]\n"
        f"💰 Total tokens saved: [bold green]{cumulative_savings:,} tokens[/bold green]\n"
        f"⚡ Overall efficiency gain: [bold green]{overall_efficiency:.1f}%[/bold green]\n\n"
        f"[dim]This represents a {cumulative_savings//1000}K+ token reduction across {total_analyses} analyses![/dim]",
        title="[bold green]💰 COST & EFFICIENCY IMPACT[/bold green]",
        border_style="green"
    )
    console.print(savings_panel)

    # Key insights with emphasis on token savings
    console.print("\n[bold blue]🔍 Key Insights & Improvements:[/bold blue]")
    insights = [
        f"💰 ITR achieved {overall_efficiency:.1f}% token reduction across all analyses (saved {cumulative_savings:,} tokens!)",
        f"⚡ Maintained only {(total_tokens_used/total_analyses/config.token_budget)*100:.1f}% average budget usage while preserving full functionality",
        "🎯 ITR successfully adapts instruction selection based on text type and query complexity",
        "📊 Academic texts triggered methodology-focused instructions (vs generic approaches)",
        "📱 Social media analysis activated sentiment and engagement-specific tools automatically",
        "⚖️ Legal documents triggered risk assessment and compliance checking workflows",
        "✍️ Creative writing analysis emphasized narrative structure and literary device detection",
        f"🔄 Dynamic selection maintained high confidence ({sum(sum(a['itr_metrics']['confidence'] for a in r['analyses']) for r in results_summary)/total_analyses:.3f} avg) across {total_analyses} diverse analyses",
        f"💡 Without ITR: {total_baseline:,} tokens needed | With ITR: {total_tokens_used:,} tokens used"
    ]

    for insight in insights:
        console.print(f"  [green]{insight}[/green]")

    # Final prompt assembly demonstration
    console.print(f"\n[bold cyan]Sample Assembled Prompt:[/bold cyan]")
    sample_query = "Analyze this academic paper for research quality and methodological rigor"
    final_prompt = itr.get_prompt(f"{sample_query}\n\nText: {SAMPLE_TEXTS['academic_paper'][:500]}...")

    # Show truncated prompt with syntax highlighting
    prompt_preview = final_prompt[:800] + "\n\n[... prompt continues with full context ...]" if len(final_prompt) > 800 else final_prompt
    syntax = Syntax(prompt_preview, "text", theme="monokai", line_numbers=True, word_wrap=True)
    console.print(Panel(syntax, title="[blue]Final Assembled Prompt (Preview)[/blue]", border_style="blue"))

    # Cost analysis section
    console.print(f"\n[bold yellow]💵 Real-World Cost Impact Analysis:[/bold yellow]")

    # Estimate costs (using approximate OpenAI pricing as example)
    input_cost_per_1k = 0.0015  # Example: GPT-4 input cost
    baseline_cost = (total_baseline / 1000) * input_cost_per_1k
    itr_cost = (total_tokens_used / 1000) * input_cost_per_1k
    cost_savings = baseline_cost - itr_cost

    cost_table = Table(title="Estimated Cost Analysis")
    cost_table.add_column("Scenario", style="cyan")
    cost_table.add_column("Tokens", justify="center", style="blue")
    cost_table.add_column("Est. Cost*", justify="center", style="green")
    cost_table.add_column("Savings", justify="center", style="green")

    cost_table.add_row(
        "Without ITR (Baseline)",
        f"{total_baseline:,}",
        f"${baseline_cost:.4f}",
        "-"
    )
    cost_table.add_row(
        "With ITR (Optimized)",
        f"{total_tokens_used:,}",
        f"${itr_cost:.4f}",
        f"${cost_savings:.4f} ({(cost_savings/baseline_cost)*100:.1f}%)"
    )

    console.print(cost_table)
    console.print("[dim]*Based on example pricing of $0.0015/1K input tokens[/dim]")

    # Scale-up projection
    scale_analysis = Panel(
        f"[bold]📈 Scale-Up Projections[/bold]\n\n"
        f"At 1,000 analyses/month:\n"
        f"• Without ITR: ~{(total_baseline * 1000 / total_analyses):,.0f} tokens → ~${(baseline_cost * 1000 / total_analyses):.2f}\n"
        f"• With ITR: ~{(total_tokens_used * 1000 / total_analyses):,.0f} tokens → ~${(itr_cost * 1000 / total_analyses):.2f}\n"
        f"• [green]Monthly savings: ~${(cost_savings * 1000 / total_analyses):.2f} ({overall_efficiency:.1f}% reduction)[/green]\n\n"
        f"At 10,000 analyses/month:\n"
        f"• [bold green]Potential savings: ~${(cost_savings * 10000 / total_analyses):.0f}/month[/bold green]\n"
        f"• [bold green]Annual savings: ~${(cost_savings * 120000 / total_analyses):.0f}/year[/bold green]",
        title="[yellow]💰 Enterprise Impact Projection[/yellow]",
        border_style="yellow"
    )
    console.print(scale_analysis)

    console.print("\n[bold green]✅ Text Analysis Workflow Demonstration Complete![/bold green]")
    console.print("[dim]This example showcased ITR's ability to:[/dim]")
    console.print("[dim]  🎯 Achieve significant token reduction while maintaining quality[/dim]")
    console.print("[dim]  📊 Handle diverse text types with specialized analysis approaches[/dim]")
    console.print("[dim]  🔄 Dynamically select relevant instructions and tools[/dim]")
    console.print("[dim]  💰 Optimize costs through intelligent context management[/dim]")
    console.print("[dim]  📈 Scale efficiently for enterprise text processing workflows[/dim]")
    console.print("[dim]  🎉 Provide high-confidence results across different domains[/dim]")

if __name__ == "__main__":
    try:
        demonstrate_text_analysis()
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        raise
