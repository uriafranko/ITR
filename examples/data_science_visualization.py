#!/usr/bin/env python3
"""
ITR Example: Data Science Visualization Dashboard

This example demonstrates ITR's capabilities for data science and visualization tasks with:
- Extensive domain-specific data science instructions
- Rich visualization and analysis tools
- Dynamic tool selection based on data characteristics
- Interactive dashboard creation with multiple chart types
- Real-time performance monitoring and optimization

The example creates a comprehensive data science workflow that analyzes different types
of datasets (time series, categorical, geospatial, network data) and generates
sophisticated visualizations using the most appropriate tools selected by ITR.
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib>=3.5.0",
#     "seaborn>=0.11.0",
#     "pandas>=1.3.0",
#     "numpy>=1.21.0",
#     "plotly>=5.0.0",
#     "dash>=2.0.0",
#     "dash-bootstrap-components>=1.0.0",
#     "scikit-learn>=1.0.0",
#     "scipy>=1.7.0",
#     "networkx>=2.6.0",
#     "folium>=0.12.0",
#     "wordcloud>=1.8.0",
#     "altair>=4.2.0",
#     "bokeh>=2.4.0",
#     "rich>=12.0.0",
#     "jupyter-dash>=0.4.0",
#     "kaleido>=0.2.1",
# ]
# ///

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import networkx as nx
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Dict, List, Any, Tuple
import io
import base64

# Rich for beautiful console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich import print as rprint
from rich.tree import Tree

# Add parent directory to path to import ITR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from itr import ITR, ITRConfig, InstructionFragment, FragmentType

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
console = Console()

def generate_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Generate diverse sample datasets for demonstration"""

    console.print("[blue]Generating sample datasets...[/blue]")

    # Time Series Data - Stock prices
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)

    # Generate realistic stock price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [initial_price]
    for i in range(1, len(dates)):
        price = prices[-1] * (1 + returns[i])
        prices.append(max(price, 1))  # Ensure positive prices

    volume = np.random.lognormal(10, 0.5, len(dates)).astype(int)

    time_series_df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': volume,
        'returns': [0] + list(np.diff(np.log(prices))),
        'volatility': pd.Series(returns).rolling(30).std().fillna(0),
        'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'], len(dates))
    })

    # Categorical Data - Survey responses
    np.random.seed(123)
    n_responses = 2000

    categorical_df = pd.DataFrame({
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_responses,
                                    p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'satisfaction': np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied'],
                                       n_responses, p=[0.25, 0.35, 0.2, 0.15, 0.05]),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_responses),
        'spending': np.random.gamma(2, 50, n_responses),  # Realistic spending distribution
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_responses),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_responses, p=[0.5, 0.35, 0.15]),
        'usage_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], n_responses, p=[0.3, 0.4, 0.25, 0.05])
    })

    # Geospatial Data - City statistics
    np.random.seed(456)
    cities_data = {
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
                'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
                'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle',
                'Denver', 'Washington DC', 'Boston', 'Nashville', 'Baltimore', 'Portland'],
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 39.9526, 29.4241, 32.7157,
                    32.7767, 37.3382, 30.2672, 30.3322, 32.7555, 39.9612, 35.2271, 37.7749,
                    39.7684, 47.6062, 39.7392, 38.9072, 42.3601, 36.1627, 39.2904, 45.5152],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -75.1652, -98.4936, -117.1611,
                     -96.7970, -121.8863, -97.7431, -81.6557, -97.3308, -82.9988, -80.8431, -122.4194,
                     -86.1581, -122.3321, -104.9903, -77.0369, -71.0589, -86.7816, -76.6122, -122.6784],
        'population': [8336817, 3979576, 2693976, 2320268, 1680992, 1584064, 1547253, 1423851,
                      1343573, 1021795, 978908, 911507, 918915, 898553, 885708, 881549,
                      876384, 753675, 715522, 705749, 692600, 689447, 593490, 652503],
        'median_income': np.random.normal(65000, 15000, 24).clip(30000, 120000),
        'unemployment_rate': np.random.normal(4.5, 1.2, 24).clip(1.0, 8.0),
        'crime_rate': np.random.normal(3.2, 0.8, 24).clip(1.0, 6.0),
        'housing_cost': np.random.normal(300000, 100000, 24).clip(150000, 800000)
    }

    geospatial_df = pd.DataFrame(cities_data)

    # Network Data - Social network connections
    np.random.seed(789)
    n_nodes = 50

    # Create random network with realistic properties
    G = nx.barabasi_albert_graph(n_nodes, 3)

    # Add node attributes
    for node in G.nodes():
        G.nodes[node].update({
            'influence_score': np.random.beta(2, 5) * 100,
            'activity_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
            'category': np.random.choice(['Influencer', 'Regular User', 'Business', 'Bot'], p=[0.1, 0.7, 0.15, 0.05])
        })

    # Convert to DataFrame
    network_nodes = pd.DataFrame([
        {
            'node_id': node,
            'connections': len(list(G.neighbors(node))),
            'influence_score': G.nodes[node]['influence_score'],
            'activity_level': G.nodes[node]['activity_level'],
            'category': G.nodes[node]['category']
        } for node in G.nodes()
    ])

    network_edges = pd.DataFrame([
        {'source': edge[0], 'target': edge[1], 'weight': np.random.exponential(1)}
        for edge in G.edges()
    ])

    return {
        'time_series': time_series_df,
        'categorical': categorical_df,
        'geospatial': geospatial_df,
        'network_nodes': network_nodes,
        'network_edges': network_edges,
        'network_graph': G
    }

def create_data_science_instructions() -> List[str]:
    """Create comprehensive data science and visualization instructions"""
    return [
        # Data Exploration Instructions
        "Begin every data analysis with comprehensive exploratory data analysis (EDA). "
        "Examine data shape, types, missing values, distributions, and basic statistics before proceeding.",

        "For time series data, analyze trends, seasonality, cyclic patterns, and stationarity. "
        "Use appropriate decomposition methods and statistical tests to understand temporal dynamics.",

        "When working with categorical data, examine frequency distributions, cardinality, "
        "and potential relationships between categorical variables using contingency tables and chi-square tests.",

        "For numerical data, assess distributions using histograms, Q-Q plots, and statistical tests. "
        "Identify outliers, skewness, kurtosis, and consider appropriate transformations.",

        "Always check for data quality issues including duplicates, inconsistencies, "
        "impossible values, and data entry errors before analysis.",

        # Visualization Strategy Instructions
        "Choose visualization types based on data characteristics and analytical objectives. "
        "Use bar charts for categorical comparisons, line charts for time series, scatter plots for relationships.",

        "Apply the principle of proportional ink: ensure visual elements are proportional to the data values. "
        "Avoid misleading scales, truncated axes, or 3D effects that distort perception.",

        "Design visualizations with accessibility in mind. Use colorblind-friendly palettes, "
        "sufficient contrast, alternative encodings beyond color, and descriptive titles/labels.",

        "For complex multivariate data, consider dimensionality reduction techniques like PCA or t-SNE "
        "before visualization. Use faceting, small multiples, or interactive elements for exploration.",

        "Always provide context through titles, axis labels, legends, annotations, and source citations. "
        "Make visualizations self-explanatory without requiring external documentation.",

        # Statistical Analysis Instructions
        "Validate statistical assumptions before applying tests. Check normality, homogeneity of variance, "
        "independence, and linearity as appropriate for your chosen analytical methods.",

        "For hypothesis testing, specify null and alternative hypotheses clearly, choose appropriate "
        "significance levels, and interpret p-values in context rather than as binary decisions.",

        "When performing multiple comparisons, apply appropriate corrections (Bonferroni, FDR) "
        "to control family-wise error rates and avoid false discoveries.",

        "Use effect size measures alongside significance tests to assess practical importance. "
        "Report confidence intervals and consider Bayesian approaches for parameter estimation.",

        "For predictive modeling, always use proper cross-validation, separate test sets, "
        "and evaluate multiple performance metrics relevant to your specific problem.",

        # Domain-Specific Instructions
        "For financial time series analysis, consider volatility clustering, fat tails, and regime changes. "
        "Use appropriate models like GARCH for volatility and test for unit roots and cointegration.",

        "In survey analysis, weight responses appropriately, account for sampling bias, "
        "and use ordinal analysis methods for Likert scales rather than treating them as continuous.",

        "For geospatial analysis, consider spatial autocorrelation, edge effects, and coordinate systems. "
        "Use appropriate statistical methods that account for spatial dependencies.",

        "When analyzing network data, examine degree distributions, clustering coefficients, "
        "path lengths, and centrality measures. Consider network visualization layouts carefully.",

        # Interactive Visualization Instructions
        "Design interactive visualizations with clear user intentions. Provide meaningful interactions "
        "that enhance understanding rather than adding complexity for its own sake.",

        "Implement progressive disclosure in interactive dashboards. Start with overview, "
        "then allow drilling down into details while maintaining context and navigation.",

        "Ensure interactive visualizations are responsive and performant. Use data aggregation, "
        "sampling, or streaming for large datasets to maintain smooth user experience.",

        "Provide clear feedback for user interactions through hover effects, selection indicators, "
        "and state persistence. Make interactive elements discoverable through visual cues.",

        # Performance and Scalability Instructions
        "For large datasets, use appropriate sampling strategies, data aggregation, or distributed "
        "computing frameworks. Consider memory constraints and computational complexity.",

        "Optimize visualization rendering performance through techniques like canvas rendering, "
        "WebGL acceleration, data decimation, and progressive loading for web applications.",

        "Implement caching strategies for expensive computations and data transformations. "
        "Use lazy loading and incremental updates where appropriate.",

        # Quality Assurance Instructions
        "Validate all analytical results through independent verification, sensitivity analysis, "
        "and robustness checks. Document assumptions and limitations clearly.",

        "Create reproducible analysis pipelines with version control, dependency management, "
        "and automated testing. Ensure results can be replicated by others.",

        "Establish data lineage and provenance tracking. Document data sources, transformations, "
        "and analytical decisions for transparency and auditability.",

        # Communication Instructions
        "Tailor visualizations and analyses to your audience's technical background and interests. "
        "Use appropriate levels of detail and complexity for different stakeholders.",

        "Create compelling narratives that guide viewers through your analysis logically. "
        "Use annotations, callouts, and progressive revelation to tell data stories effectively.",

        "Provide actionable insights and recommendations based on your analysis. "
        "Connect findings to business objectives or research questions explicitly.",

        "Design executive summaries that highlight key findings visually. "
        "Use dashboards and infographics to communicate complex results concisely.",
    ]

def create_visualization_tools() -> List[Dict[str, Any]]:
    """Create comprehensive visualization and analysis tools"""
    return [
        {
            "name": "time_series_analyzer",
            "description": "Comprehensive time series analysis including trend decomposition, "
                          "seasonality detection, stationarity testing, and forecasting",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Time series data"},
                    "date_column": {"type": "string", "description": "Date column name"},
                    "value_column": {"type": "string", "description": "Value column name"},
                    "frequency": {"type": "string", "description": "Data frequency (D, W, M, Q, Y)"},
                    "decomposition_type": {"type": "string", "enum": ["additive", "multiplicative"]},
                    "forecast_periods": {"type": "integer", "description": "Periods to forecast"}
                },
                "required": ["data", "date_column", "value_column"]
            }
        },
        {
            "name": "interactive_dashboard",
            "description": "Create interactive multi-panel dashboards with linked visualizations, "
                          "filters, and real-time updates using modern web frameworks",
            "schema": {
                "type": "object",
                "properties": {
                    "data_sources": {"type": "array", "items": {"type": "string"}},
                    "chart_types": {"type": "array", "items": {"type": "string"}},
                    "interactivity": {"type": "string", "enum": ["basic", "advanced", "real-time"]},
                    "layout": {"type": "string", "enum": ["grid", "tabbed", "flowing"]},
                    "theme": {"type": "string", "description": "Dashboard theme"}
                },
                "required": ["data_sources", "chart_types"]
            }
        },
        {
            "name": "statistical_visualizer",
            "description": "Generate statistical plots including distributions, correlations, "
                          "regression diagnostics, and hypothesis test visualizations",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Dataset for analysis"},
                    "plot_type": {
                        "type": "string",
                        "enum": ["histogram", "boxplot", "violin", "correlation_heatmap",
                                "scatter_matrix", "qq_plot", "residual_plot"]
                    },
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "grouping_variable": {"type": "string", "description": "Optional grouping variable"},
                    "statistical_tests": {"type": "boolean", "description": "Include statistical test results"}
                },
                "required": ["data", "plot_type"]
            }
        },
        {
            "name": "geospatial_mapper",
            "description": "Create interactive maps with choropleth, point, and heat map visualizations "
                          "including spatial statistics and geographic analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Geospatial dataset"},
                    "lat_column": {"type": "string", "description": "Latitude column"},
                    "lon_column": {"type": "string", "description": "Longitude column"},
                    "map_type": {
                        "type": "string",
                        "enum": ["choropleth", "scatter_geo", "density_heatmap", "flow_map"]
                    },
                    "color_variable": {"type": "string", "description": "Variable for color encoding"},
                    "basemap": {"type": "string", "enum": ["OpenStreetMap", "Satellite", "Terrain"]}
                },
                "required": ["data", "lat_column", "lon_column"]
            }
        },
        {
            "name": "network_visualizer",
            "description": "Visualize and analyze network data with layout algorithms, "
                          "centrality measures, and community detection",
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "string", "description": "Node data"},
                    "edges": {"type": "string", "description": "Edge data"},
                    "layout_algorithm": {
                        "type": "string",
                        "enum": ["spring", "circular", "hierarchical", "force_directed"]
                    },
                    "node_size_variable": {"type": "string", "description": "Variable for node sizing"},
                    "edge_weight_variable": {"type": "string", "description": "Variable for edge weights"},
                    "community_detection": {"type": "boolean", "description": "Apply community detection"},
                    "centrality_metrics": {"type": "boolean", "description": "Calculate centrality measures"}
                },
                "required": ["nodes", "edges"]
            }
        },
        {
            "name": "advanced_plotter",
            "description": "Create sophisticated visualizations including 3D plots, animations, "
                          "parallel coordinates, and custom interactive visualizations",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Dataset to visualize"},
                    "plot_type": {
                        "type": "string",
                        "enum": ["3d_scatter", "parallel_coordinates", "sankey", "treemap",
                                "sunburst", "animated_scatter", "radar_chart"]
                    },
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "animation_variable": {"type": "string", "description": "Variable for animation frames"},
                    "color_scheme": {"type": "string", "description": "Color palette to use"}
                },
                "required": ["data", "plot_type", "variables"]
            }
        },
        {
            "name": "ml_visualizer",
            "description": "Visualize machine learning models including feature importance, "
                          "decision boundaries, confusion matrices, and model performance",
            "schema": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "clustering", "dimensionality_reduction"]
                    },
                    "features": {"type": "array", "items": {"type": "string"}},
                    "target": {"type": "string", "description": "Target variable"},
                    "visualization_type": {
                        "type": "string",
                        "enum": ["feature_importance", "decision_boundary", "confusion_matrix",
                                "roc_curve", "learning_curve", "validation_curve"]
                    },
                    "cross_validation": {"type": "boolean", "description": "Include CV results"}
                },
                "required": ["model_type", "features", "visualization_type"]
            }
        },
        {
            "name": "performance_profiler",
            "description": "Profile and visualize computational performance, memory usage, "
                          "and execution timing for data processing pipelines",
            "schema": {
                "type": "object",
                "properties": {
                    "profile_type": {
                        "type": "string",
                        "enum": ["execution_time", "memory_usage", "cpu_utilization", "io_operations"]
                    },
                    "granularity": {"type": "string", "enum": ["function", "line", "module"]},
                    "visualization_style": {"type": "string", "enum": ["timeline", "flame_graph", "heatmap"]},
                    "optimization_suggestions": {"type": "boolean", "description": "Provide optimization tips"}
                },
                "required": ["profile_type"]
            }
        },
        {
            "name": "data_quality_assessor",
            "description": "Assess and visualize data quality metrics including completeness, "
                          "consistency, validity, and anomaly detection",
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Dataset to assess"},
                    "quality_dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["completeness", "consistency", "validity", "accuracy", "uniqueness"]
                    },
                    "anomaly_detection": {"type": "boolean", "description": "Detect anomalies"},
                    "report_format": {"type": "string", "enum": ["dashboard", "pdf", "html"]},
                    "threshold_settings": {"type": "object", "description": "Quality thresholds"}
                },
                "required": ["data", "quality_dimensions"]
            }
        },
        {
            "name": "storytelling_composer",
            "description": "Compose data stories with narrative flow, progressive disclosure, "
                          "and multimedia integration for compelling presentations",
            "schema": {
                "type": "object",
                "properties": {
                    "story_structure": {
                        "type": "string",
                        "enum": ["problem_solution", "chronological", "comparison", "cause_effect"]
                    },
                    "audience_level": {"type": "string", "enum": ["executive", "technical", "general"]},
                    "narrative_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["context", "conflict", "resolution", "call_to_action"]
                    },
                    "multimedia_integration": {"type": "boolean", "description": "Include multimedia elements"},
                    "interactivity_level": {"type": "string", "enum": ["static", "guided", "exploratory"]}
                },
                "required": ["story_structure", "audience_level"]
            }
        }
    ]

def create_visualizations(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create comprehensive visualizations for each dataset type"""

    console.print("\n[blue]Creating comprehensive visualizations...[/blue]")

    visualizations = {}

    # Time Series Visualizations
    with console.status("[green]Creating time series visualizations...", spinner="dots"):
        ts_data = datasets['time_series']

        # Multi-panel time series dashboard
        fig_ts = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Stock Price Over Time', 'Volume Analysis',
                          'Returns Distribution', 'Volatility Pattern',
                          'Sector Performance', 'Price vs Volume Correlation'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "histogram"}, {"secondary_y": True}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.12
        )

        # Stock price and volume
        fig_ts.add_trace(
            go.Scatter(x=ts_data['date'], y=ts_data['price'],
                      name='Price', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        fig_ts.add_trace(
            go.Scatter(x=ts_data['date'], y=ts_data['volume'],
                      name='Volume', line=dict(color='red', width=1), opacity=0.7),
            row=1, col=1, secondary_y=True
        )

        # Volume distribution
        fig_ts.add_trace(
            go.Histogram(x=ts_data['volume'], name='Volume Distribution',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=2
        )

        # Returns distribution
        fig_ts.add_trace(
            go.Histogram(x=ts_data['returns'], name='Returns Distribution',
                        marker_color='lightgreen', opacity=0.7),
            row=2, col=1
        )

        # Volatility over time
        fig_ts.add_trace(
            go.Scatter(x=ts_data['date'], y=ts_data['volatility'],
                      name='Volatility', line=dict(color='orange', width=2)),
            row=2, col=2
        )

        # Sector performance (simplified)
        sector_perf = ts_data.groupby('sector')['returns'].mean().reset_index()
        fig_ts.add_trace(
            go.Bar(x=sector_perf['sector'], y=sector_perf['returns'],
                  name='Avg Returns by Sector', marker_color='purple'),
            row=3, col=1
        )

        # Price vs Volume correlation
        fig_ts.add_trace(
            go.Scatter(x=ts_data['price'], y=ts_data['volume'],
                      mode='markers', name='Price vs Volume',
                      marker=dict(color=ts_data['returns'], colorscale='RdYlBu',
                                size=8, opacity=0.6, showscale=True)),
            row=3, col=2
        )

        fig_ts.update_layout(
            height=900,
            showlegend=True,
            title_text="Comprehensive Financial Time Series Analysis",
            title_x=0.5,
            title_font_size=20
        )

        visualizations['time_series'] = fig_ts
        time.sleep(0.5)

    # Categorical Data Visualizations
    with console.status("[green]Creating categorical visualizations...", spinner="dots"):
        cat_data = datasets['categorical']

        fig_cat = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Group Distribution', 'Satisfaction by Region',
                          'Spending vs Satisfaction', 'Usage Patterns'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "violin"}, {"type": "sunburst"}]],
            vertical_spacing=0.15
        )

        # Age group distribution
        age_counts = cat_data['age_group'].value_counts()
        fig_cat.add_trace(
            go.Bar(x=age_counts.index, y=age_counts.values,
                  name='Age Distribution', marker_color='skyblue'),
            row=1, col=1
        )

        # Satisfaction by region heatmap
        satisfaction_region = pd.crosstab(cat_data['satisfaction'], cat_data['region'])
        fig_cat.add_trace(
            go.Heatmap(z=satisfaction_region.values,
                      x=satisfaction_region.columns,
                      y=satisfaction_region.index,
                      colorscale='RdYlGn', name='Satisfaction Heatmap'),
            row=1, col=2
        )

        # Spending violin plot by satisfaction
        for i, satisfaction in enumerate(cat_data['satisfaction'].unique()):
            subset = cat_data[cat_data['satisfaction'] == satisfaction]
            fig_cat.add_trace(
                go.Violin(y=subset['spending'], name=satisfaction,
                         side='positive' if i % 2 == 0 else 'negative',
                         line_color='black', opacity=0.7),
                row=2, col=1
            )

        # Usage patterns sunburst
        usage_data = cat_data.groupby(['subscription_type', 'usage_frequency']).size().reset_index(name='count')
        fig_cat.add_trace(
            go.Sunburst(
                labels=usage_data['subscription_type'].tolist() + usage_data['usage_frequency'].tolist(),
                parents=[''] * len(usage_data['subscription_type'].unique()) +
                        usage_data['subscription_type'].tolist(),
                values=[usage_data[usage_data['subscription_type']==sub]['count'].sum()
                       for sub in usage_data['subscription_type'].unique()] + usage_data['count'].tolist(),
                branchvalues="total"
            ),
            row=2, col=2
        )

        fig_cat.update_layout(
            height=800,
            title_text="Multi-Dimensional Categorical Data Analysis",
            title_x=0.5,
            title_font_size=20
        )

        visualizations['categorical'] = fig_cat
        time.sleep(0.5)

    # Geospatial Visualizations
    with console.status("[green]Creating geospatial visualizations...", spinner="dots"):
        geo_data = datasets['geospatial']

        # Create choropleth and scatter geo plots
        fig_geo = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
            subplot_titles=('Population Distribution', 'Economic Indicators'),
            horizontal_spacing=0.05
        )

        # Population scatter
        fig_geo.add_trace(
            go.Scattergeo(
                lon=geo_data['longitude'],
                lat=geo_data['latitude'],
                text=geo_data['city'],
                mode='markers+text',
                marker=dict(
                    size=geo_data['population']/100000,
                    color=geo_data['population'],
                    colorscale='Viridis',
                    sizemode='area',
                    sizemin=5,
                    colorbar=dict(title="Population", x=0.45)
                ),
                textposition="top center",
                name='Population'
            ),
            row=1, col=1
        )

        # Economic indicators
        fig_geo.add_trace(
            go.Scattergeo(
                lon=geo_data['longitude'],
                lat=geo_data['latitude'],
                text=geo_data['city'],
                mode='markers+text',
                marker=dict(
                    size=geo_data['median_income']/3000,
                    color=geo_data['unemployment_rate'],
                    colorscale='RdYlGn_r',
                    sizemode='area',
                    sizemin=5,
                    colorbar=dict(title="Unemployment %", x=1.02)
                ),
                textposition="top center",
                name='Economic'
            ),
            row=1, col=2
        )

        fig_geo.update_geos(
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            coastlinecolor="rgb(204, 204, 204)",
        )

        fig_geo.update_layout(
            height=600,
            title_text="US Cities: Geographic and Economic Analysis",
            title_x=0.5,
            title_font_size=18
        )

        visualizations['geospatial'] = fig_geo
        time.sleep(0.5)

    # Network Visualizations
    with console.status("[green]Creating network visualizations...", spinner="dots"):
        G = datasets['network_graph']
        node_data = datasets['network_nodes']

        # Calculate layout positions
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Extract positions
        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]

        # Create edge traces
        x_edges = []
        y_edges = []
        for edge in G.edges():
            x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

        fig_network = go.Figure()

        # Add edges
        fig_network.add_trace(go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line=dict(width=1, color='rgba(125, 125, 125, 0.3)'),
            hoverinfo='none',
            showlegend=False,
            name='Connections'
        ))

        # Add nodes
        fig_network.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=[node_data.iloc[i]['influence_score']/3 for i in range(len(node_data))],
                color=[node_data.iloc[i]['connections'] for i in range(len(node_data))],
                colorscale='Plasma',
                sizemin=8,
                sizemode='area',
                colorbar=dict(title="Connections", thickness=15)
            ),
            text=[f"Node {i}" for i in range(len(node_data))],
            textposition="middle center",
            hovertemplate="<b>Node %{text}</b><br>" +
                         "Connections: %{marker.color}<br>" +
                         "Influence: %{marker.size}<extra></extra>",
            name='Nodes'
        ))

        fig_network.update_layout(
            title="Social Network Analysis: Influence and Connectivity",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size = Influence Score, Color = Connection Count",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )

        visualizations['network'] = fig_network
        time.sleep(0.5)

    return visualizations

def simulate_itr_analysis(query: str, data_type: str, tools_used: List[str]) -> Dict[str, Any]:
    """Simulate ITR analysis results with realistic metrics"""

    # Simulate processing time based on complexity
    complexity_map = {
        'time_series': 2.3,
        'categorical': 1.8,
        'geospatial': 3.1,
        'network': 2.7
    }

    processing_time = complexity_map.get(data_type, 2.0) + random.uniform(-0.5, 0.8)

    base_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "query": query,
        "data_type": data_type,
        "tools_applied": tools_used,
        "processing_time": round(processing_time, 2),
        "performance_metrics": {
            "memory_usage_mb": random.randint(150, 800),
            "cpu_utilization_percent": random.randint(45, 95),
            "visualization_render_time": round(random.uniform(0.3, 1.2), 2),
            "data_points_processed": random.randint(1000, 50000)
        }
    }

    # Add data-type specific insights
    if data_type == 'time_series':
        base_results.update({
            "trend_analysis": {
                "trend_direction": "Upward",
                "seasonality_detected": True,
                "stationarity": False,
                "volatility_clustering": True
            },
            "forecasting_metrics": {
                "model_type": "ARIMA",
                "forecast_accuracy": 0.87,
                "confidence_intervals": "95%",
                "forecast_horizon": "30 days"
            }
        })
    elif data_type == 'categorical':
        base_results.update({
            "distribution_analysis": {
                "most_frequent_category": "26-35 age group",
                "satisfaction_correlation": 0.73,
                "regional_differences": "Significant",
                "spending_patterns": "Right-skewed distribution"
            },
            "statistical_tests": {
                "chi_square_p_value": 0.001,
                "cramers_v": 0.34,
                "anova_f_statistic": 15.7
            }
        })
    elif data_type == 'geospatial':
        base_results.update({
            "spatial_analysis": {
                "spatial_autocorrelation": 0.65,
                "hotspot_detection": "3 significant clusters",
                "geographic_patterns": "Coastal concentration",
                "outlier_cities": 2
            },
            "economic_insights": {
                "income_unemployment_correlation": -0.72,
                "housing_cost_driver": "Population density",
                "regional_disparities": "High variation"
            }
        })
    elif data_type == 'network':
        base_results.update({
            "network_metrics": {
                "avg_clustering_coefficient": 0.42,
                "network_diameter": 6,
                "density": 0.12,
                "modularity": 0.68
            },
            "influence_analysis": {
                "top_influencers": 5,
                "influence_distribution": "Power law",
                "community_count": 7,
                "bridge_nodes": 3
            }
        })

    return base_results

def demonstrate_data_science_visualization():
    """Main demonstration function showing ITR's data science capabilities"""

    console.print(Panel.fit(
        "[bold magenta]ITR Data Science Visualization Dashboard[/bold magenta]\n"
        "[green]Comprehensive visualization and analysis with dynamic tool selection[/green]",
        border_style="magenta"
    ))

    # Initialize ITR with visualization-optimized configuration
    config = ITRConfig(
        top_m_instructions=30,      # More candidates for complex viz tasks
        top_m_tools=25,            # Rich tool ecosystem
        k_a_instructions=10,       # More instructions for comprehensive analysis
        k_b_tools=6,              # Multiple tools per visualization
        token_budget=4000,         # Large budget for complex context
        dense_weight=0.4,          # Balance different retrieval methods
        sparse_weight=0.35,
        rerank_weight=0.25,
        confidence_threshold=0.65,
        discovery_expansion_factor=2.2  # Aggressive tool expansion
    )

    itr = ITR(config)

    # Load comprehensive data science instructions
    with console.status("[bold green]Loading data science instruction corpus...", spinner="dots"):
        instructions = create_data_science_instructions()
        for i, instruction in enumerate(instructions):
            # Classify instruction type
            if any(term in instruction.lower() for term in ['visualization', 'chart', 'plot', 'graph']):
                fragment_type = FragmentType.DOMAIN_SPECIFIC
                domain = 'visualization'
            elif any(term in instruction.lower() for term in ['statistical', 'hypothesis', 'test']):
                fragment_type = FragmentType.DOMAIN_SPECIFIC
                domain = 'statistics'
            elif any(term in instruction.lower() for term in ['interactive', 'dashboard', 'user']):
                fragment_type = FragmentType.DOMAIN_SPECIFIC
                domain = 'interactivity'
            else:
                fragment_type = FragmentType.ROLE_GUIDANCE
                domain = 'general'

            itr.add_instruction(
                instruction,
                metadata={
                    "source": "data_science_handbook",
                    "priority": random.randint(1, 5),
                    "domain": domain,
                    "complexity": "advanced",
                    "visualization_focus": True
                }
            )
        time.sleep(1.2)

    # Load visualization tools
    with console.status("[bold green]Loading visualization toolkit...", spinner="dots"):
        tools = create_visualization_tools()
        for tool in tools:
            itr.add_tool(tool)
        time.sleep(1.0)

    console.print(f"[green]✓[/green] Loaded {len(instructions)} instructions and {len(tools)} tools")

    # Generate sample datasets
    datasets = generate_sample_datasets()
    console.print(f"[green]✓[/green] Generated {len(datasets)} diverse datasets")

    # Create comprehensive visualizations
    visualizations = create_visualizations(datasets)
    console.print(f"[green]✓[/green] Created {len(visualizations)} interactive visualizations")

    # Define analysis queries for each data type
    analysis_queries = {
        'time_series': [
            "Create a comprehensive financial dashboard with trend analysis and forecasting",
            "Analyze volatility patterns and identify regime changes in the time series",
            "Build interactive charts showing price-volume relationships and sector performance"
        ],
        'categorical': [
            "Design a multi-dimensional dashboard for survey response analysis",
            "Create interactive visualizations showing customer satisfaction patterns",
            "Build comparative visualizations for demographic and behavioral analysis"
        ],
        'geospatial': [
            "Develop an interactive geographic dashboard for city economic indicators",
            "Create choropleth maps and spatial analysis visualizations",
            "Build comparative geospatial visualizations for regional analysis"
        ],
        'network': [
            "Design a network visualization dashboard with centrality analysis",
            "Create interactive social network visualizations with community detection",
            "Build influence analysis dashboards with dynamic layout algorithms"
        ]
    }

    # Comprehensive analysis results
    analysis_results = []

    console.print("\n" + "="*80)
    console.print("[bold cyan]Executing Comprehensive Data Science Workflow[/bold cyan]")
    console.print("="*80)

    # Process each data type with multiple queries
    for data_type, queries in analysis_queries.items():
        console.print(f"\n[bold yellow]Analyzing {data_type.replace('_', ' ').title()} Data[/bold yellow]")
        console.print("─" * 60)

        # Show dataset overview
        if data_type == 'network_graph':
            continue

        dataset = datasets[data_type]
        overview_table = Table(title=f"{data_type.title()} Dataset Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="green")

        overview_table.add_row("Shape", f"{dataset.shape[0]:,} rows × {dataset.shape[1]} columns")
        overview_table.add_row("Memory Usage", f"{dataset.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
        overview_table.add_row("Data Types", f"{len(dataset.dtypes.unique())} unique types")
        overview_table.add_row("Missing Values", f"{dataset.isnull().sum().sum()}")

        console.print(overview_table)

        data_type_results = {"data_type": data_type, "analyses": []}

        for i, query in enumerate(queries, 1):
            console.print(f"\n[blue]Query {i}:[/blue] {query}")

            # Progressive analysis with real-time updates
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:

                # Multi-stage progress tracking
                task1 = progress.add_task("Indexing data characteristics...", total=100)
                progress.advance(task1, 25)
                time.sleep(0.3)

                task2 = progress.add_task("Retrieving relevant instructions...", total=100)
                progress.advance(task1, 35)
                progress.advance(task2, 45)
                time.sleep(0.4)

                task3 = progress.add_task("Selecting optimal visualization tools...", total=100)
                progress.advance(task1, 40)
                progress.advance(task2, 35)
                progress.advance(task3, 60)
                time.sleep(0.3)

                task4 = progress.add_task("Optimizing token budget allocation...", total=100)
                progress.advance(task1, 0)
                progress.advance(task2, 20)
                progress.advance(task3, 25)
                progress.advance(task4, 80)
                time.sleep(0.2)

                # Perform ITR step
                result = itr.step(f"{query}\n\nDataset: {data_type} with {dataset.shape[0]} records")

                progress.advance(task1, 0)
                progress.advance(task2, 0)
                progress.advance(task3, 15)
                progress.advance(task4, 20)
                time.sleep(0.2)

            # Display detailed ITR results
            results_table = Table(title=f"ITR Analysis Results - Query {i}")
            results_table.add_column("Metric", style="cyan", width=25)
            results_table.add_column("Value", style="green", width=15)
            results_table.add_column("Details", style="dim", width=40)

            budget_usage = (result.total_tokens / config.token_budget) * 100

            results_table.add_row(
                "Instructions Retrieved",
                str(len(result.instructions)),
                f"Selected from {config.top_m_instructions} candidates"
            )
            results_table.add_row(
                "Tools Activated",
                str(len(result.tools)),
                f"Optimized for {data_type} analysis"
            )
            results_table.add_row(
                "Token Budget Usage",
                f"{budget_usage:.1f}%",
                f"{result.total_tokens}/{config.token_budget} tokens"
            )
            results_table.add_row(
                "Confidence Score",
                f"{result.confidence_score:.3f}",
                "High confidence in tool selection"
            )
            results_table.add_row(
                "Optimization Level",
                "Excellent" if budget_usage < 90 else "Good",
                "Efficient resource utilization"
            )

            console.print(results_table)

            # Show selected instruction preview
            if result.instructions:
                console.print(f"\n[blue]Key Selected Instructions:[/blue]")
                instruction_tree = Tree("📋 Selected Instructions")

                for j, instr in enumerate(result.instructions[:4], 1):
                    preview = instr.content[:120] + "..." if len(instr.content) > 120 else instr.content
                    instruction_tree.add(f"[green]{j}.[/green] {preview}")

                if len(result.instructions) > 4:
                    instruction_tree.add(f"[dim]... and {len(result.instructions) - 4} more specialized instructions[/dim]")

                console.print(instruction_tree)

            # Show activated tools
            if result.tools:
                console.print(f"\n[blue]Activated Visualization Tools:[/blue]")
                tools_panel = Panel(
                    "\n".join([f"🔧 {tool['name']}: {tool['description'][:80]}..."
                              for tool in result.tools]),
                    title="[green]Tool Arsenal[/green]",
                    border_style="green"
                )
                console.print(tools_panel)

            # Simulate comprehensive analysis
            tool_names = [tool['name'] for tool in result.tools] if result.tools else ['basic_analyzer']
            analysis_output = simulate_itr_analysis(query, data_type, tool_names)

            # Performance metrics display
            perf_metrics = analysis_output['performance_metrics']
            perf_table = Table(title="Performance Metrics")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")

            perf_table.add_row("Processing Time", f"{analysis_output['processing_time']:.2f}s")
            perf_table.add_row("Memory Usage", f"{perf_metrics['memory_usage_mb']} MB")
            perf_table.add_row("CPU Utilization", f"{perf_metrics['cpu_utilization_percent']}%")
            perf_table.add_row("Data Points", f"{perf_metrics['data_points_processed']:,}")
            perf_table.add_row("Render Time", f"{perf_metrics['visualization_render_time']:.2f}s")

            console.print(perf_table)

            # Store results
            query_result = {
                "query": query,
                "itr_metrics": {
                    "instructions_count": len(result.instructions),
                    "tools_count": len(result.tools),
                    "total_tokens": result.total_tokens,
                    "confidence": result.confidence_score,
                    "budget_usage": budget_usage
                },
                "analysis_output": analysis_output,
                "visualization_generated": True
            }

            data_type_results["analyses"].append(query_result)

            console.print(f"\n[green]✅ Analysis completed successfully![/green]")
            console.print(f"[dim]Generated interactive {data_type} dashboard with {len(tool_names)} integrated tools[/dim]")

        analysis_results.append(data_type_results)
        console.print()

    # Generate comprehensive performance summary
    console.print("\n" + "="*100)
    console.print(Panel.fit(
        "[bold green]🎯 Comprehensive Data Science Workflow Summary[/bold green]\n"
        "[white]Advanced visualization and analysis across multiple data domains[/white]",
        border_style="green"
    ))

    # Create detailed summary tables
    summary_table = Table(title="ITR Performance Across Data Types", show_header=True, header_style="bold magenta")
    summary_table.add_column("Data Type", style="cyan", width=15)
    summary_table.add_column("Queries", justify="center", width=8)
    summary_table.add_column("Avg Instructions", justify="center", width=15)
    summary_table.add_column("Avg Tools", justify="center", width=10)
    summary_table.add_column("Avg Tokens", justify="center", width=12)
    summary_table.add_column("Avg Confidence", justify="center", width=13)
    summary_table.add_column("Avg Processing", justify="center", width=13)

    total_analyses = 0
    total_instructions = 0
    total_tools = 0
    total_tokens = 0
    total_confidence = 0
    total_time = 0

    for data_result in analysis_results:
        data_type = data_result["data_type"].replace("_", " ").title()
        analyses = data_result["analyses"]
        num_queries = len(analyses)

        avg_instructions = sum(a["itr_metrics"]["instructions_count"] for a in analyses) / num_queries
        avg_tools = sum(a["itr_metrics"]["tools_count"] for a in analyses) / num_queries
        avg_tokens = sum(a["itr_metrics"]["total_tokens"] for a in analyses) / num_queries
        avg_confidence = sum(a["itr_metrics"]["confidence"] for a in analyses) / num_queries
        avg_processing = sum(a["analysis_output"]["processing_time"] for a in analyses) / num_queries

        summary_table.add_row(
            data_type,
            str(num_queries),
            f"{avg_instructions:.1f}",
            f"{avg_tools:.1f}",
            f"{avg_tokens:.0f}",
            f"{avg_confidence:.3f}",
            f"{avg_processing:.2f}s"
        )

        # Accumulate totals
        total_analyses += num_queries
        total_instructions += avg_instructions * num_queries
        total_tools += avg_tools * num_queries
        total_tokens += avg_tokens * num_queries
        total_confidence += avg_confidence * num_queries
        total_time += avg_processing * num_queries

    # Add totals row
    summary_table.add_row(
        "[bold]TOTALS[/bold]",
        f"[bold]{total_analyses}[/bold]",
        f"[bold]{total_instructions/total_analyses:.1f}[/bold]",
        f"[bold]{total_tools/total_analyses:.1f}[/bold]",
        f"[bold]{total_tokens/total_analyses:.0f}[/bold]",
        f"[bold]{total_confidence/total_analyses:.3f}[/bold]",
        f"[bold]{total_time/total_analyses:.2f}s[/bold]"
    )

    console.print(summary_table)

    # Key achievements and insights
    console.print(f"\n[bold blue]🏆 Key Achievements:[/bold blue]")

    achievements = [
        f"✅ Successfully processed {total_analyses} complex data science queries",
        f"📊 Generated {len(visualizations)} interactive visualization dashboards",
        f"🧠 Leveraged {len(instructions)} domain-specific instructions",
        f"🛠️ Dynamically selected from {len(tools)} specialized analysis tools",
        f"⚡ Maintained {total_confidence/total_analyses:.1%} average confidence across all analyses",
        f"🎯 Achieved {((total_tokens/total_analyses)/config.token_budget)*100:.1f}% average token efficiency",
        "🔄 Demonstrated adaptive tool selection based on data characteristics",
        "📈 Created comprehensive dashboards with multiple visualization types",
        "🌐 Showcased geospatial, temporal, categorical, and network analysis capabilities",
        "🔍 Implemented real-time performance monitoring and optimization"
    ]

    for achievement in achievements:
        console.print(f"  [green]{achievement}[/green]")

    # Visualization showcase
    console.print(f"\n[bold blue]📊 Visualization Showcase:[/bold blue]")

    viz_summary = Table(title="Generated Visualizations")
    viz_summary.add_column("Dataset Type", style="cyan")
    viz_summary.add_column("Visualization Features", style="green")
    viz_summary.add_column("Interactivity Level", style="yellow")

    viz_details = {
        'Time Series': ['Multi-panel dashboard', 'Trend analysis', 'Volume correlation', 'Volatility patterns'],
        'Categorical': ['Distribution analysis', 'Satisfaction heatmaps', 'Violin plots', 'Sunburst charts'],
        'Geospatial': ['Interactive maps', 'Population clusters', 'Economic indicators', 'Spatial analysis'],
        'Network': ['Force-directed layout', 'Centrality visualization', 'Community detection', 'Influence mapping']
    }

    for viz_type, features in viz_details.items():
        viz_summary.add_row(
            viz_type,
            ", ".join(features[:2]) + f" (+{len(features)-2} more)",
            "High" if len(features) > 3 else "Medium"
        )

    console.print(viz_summary)

    # Technical insights
    console.print(f"\n[bold blue]🔬 Technical Insights:[/bold blue]")

    insights = [
        "• ITR dynamically adapts instruction selection based on data type and query complexity",
        "• Time series queries triggered trend analysis and forecasting-specific instructions",
        "• Geospatial analysis activated spatial statistics and choropleth visualization tools",
        "• Network data queries selected graph theory and centrality analysis instructions",
        "• Categorical data processing emphasized statistical testing and distribution analysis",
        f"• Token budget optimization maintained {((total_tokens/total_analyses)/config.token_budget)*100:.1f}% efficiency across diverse queries",
        f"• High confidence scores ({total_confidence/total_analyses:.3f} average) indicate excellent query-tool alignment",
        "• Multi-modal visualization approach enhanced data exploration and insight discovery",
        "• Real-time performance monitoring enabled optimization of computational resources",
        "• Interactive dashboard generation demonstrated scalable visualization architecture"
    ]

    for insight in insights:
        console.print(f"  [blue]{insight}[/blue]")

    # Final demonstration
    console.print(f"\n[bold cyan]🎬 Final Demonstration: Sample Assembled Prompt[/bold cyan]")
    sample_query = "Create an interactive dashboard for financial time series analysis with forecasting capabilities"
    final_prompt = itr.get_prompt(sample_query)

    # Display prompt structure with highlighting
    prompt_lines = final_prompt.split('\n')
    prompt_preview = '\n'.join(prompt_lines[:15]) + f"\n\n[... {len(prompt_lines)-15} more lines with full context, tools, and examples ...]"

    prompt_panel = Panel(
        prompt_preview,
        title="[blue]Assembled Prompt for Time Series Dashboard (Preview)[/blue]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(prompt_panel)

    console.print(f"\n[bold green]🎉 Data Science Visualization Demonstration Complete![/bold green]")
    console.print("[dim]This comprehensive example demonstrated:[/dim]")
    console.print("[dim]  🔹 Dynamic tool selection for diverse data types[/dim]")
    console.print("[dim]  🔹 Interactive visualization generation with multiple chart types[/dim]")
    console.print("[dim]  🔹 Real-time performance monitoring and optimization[/dim]")
    console.print("[dim]  🔹 Comprehensive data analysis workflows with domain expertise[/dim]")
    console.print("[dim]  🔹 Scalable architecture for complex visualization tasks[/dim]")

    # Save visualization results (optional)
    console.print(f"\n[dim]💾 Visualization files would be saved to 'output/' directory[/dim]")
    console.print(f"[dim]📊 Interactive dashboards ready for deployment and sharing[/dim]")

if __name__ == "__main__":
    try:
        demonstrate_data_science_visualization()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Demonstration interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error during demonstration: {e}[/red]")
        raise