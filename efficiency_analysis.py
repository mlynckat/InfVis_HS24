import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from datasets import load_dataset


def load_data():
    """Load and preprocess the data."""
    dataset = load_dataset("open-llm-leaderboard/contents")
    df = pd.DataFrame(dataset["train"])
    # Rename columns
    df = df.rename(
        columns={
            "model_family": "Model Family",
            "eval_name": "Eval Name",
            "fullname": "Full Name",
        }
    )
    # Create Model Family from Base Model if it doesn't exist
    if "Model Family" not in df.columns:
        df["Model Family"] = df["Base Model"].apply(lambda x: x.split("/")[0])
    return df


def create_filters():
    """Create and return filter values from sidebar."""
    st.sidebar.header("Filters")

    # Load data first to get filter options
    df = load_data()

    # Architecture filter
    architectures = ["All"] + sorted(df["Architecture"].unique().tolist())
    selected_architecture = st.sidebar.selectbox("Select Architecture", architectures)

    # Model size filter - dynamically updated based on architecture
    if selected_architecture == "All":
        available_sizes = df["#Params (B)"].dropna().unique()
    else:
        available_sizes = (
            df[df["Architecture"] == selected_architecture]["#Params (B)"]
            .dropna()
            .unique()
        )

    # Create size range slider
    min_size = min(size for size in available_sizes if size is not None and size != -1)
    max_size = max(size for size in available_sizes if size is not None and size != -1)

    # Check if min and max are the same
    if min_size == max_size:
        st.sidebar.info(f"Only models with {min_size}B parameters available")
        size_range = (min_size, min_size)
    else:
        size_range = st.sidebar.slider(
            "Select Model Size Range (in Billions)",
            min_value=int(min_size),
            max_value=int(max_size),
            value=(int(min_size), int(max_size)),
            step=1,
        )

    # Type filter using pills
    st.sidebar.subheader("Model Types")
    available_types = sorted(df["Type"].unique().tolist())
    selected_types = st.sidebar.pills(
        "Select Types",
        options=available_types,
        selection_mode="multi",
        default=["🟢 pretrained"],
        help="Filter models by their type classification",
    )

    return selected_architecture, size_range, selected_types


def filter_data(df, selected_architecture, size_range, selected_types):
    """Apply filters to the dataframe."""
    filtered_df = df.copy()

    # Apply architecture filter
    if selected_architecture != "All":
        filtered_df = filtered_df[filtered_df["Architecture"] == selected_architecture]

    # Apply size filter
    filtered_df = filtered_df[
        (filtered_df["#Params (B)"].between(size_range[0], size_range[1]))
        & (filtered_df["#Params (B)"] != -1)
    ]

    # Apply type filter
    if selected_types:
        selected_types_list = (
            [selected_types] if isinstance(selected_types, str) else selected_types
        )
        filtered_df = filtered_df[filtered_df["Type"].isin(selected_types_list)]

    return filtered_df


def create_efficiency_plot(filtered_df):
    """Create the efficiency plot."""
    # Calculate efficiency metrics
    efficiency_df = filtered_df.copy()
    efficiency_df["Performance per Param"] = (
        efficiency_df["Average ⬆️"] / efficiency_df["#Params (B)"]
    )

    # Calculate percentile ranks
    efficiency_df["Size_Percentile"] = efficiency_df["#Params (B)"].rank(pct=True) * 100
    efficiency_df["Performance_Percentile"] = (
        efficiency_df["Average ⬆️"].rank(pct=True) * 100
    )
    # Calculate LOWESS trendline values
    x_sorted = np.sort(efficiency_df["#Params (B)"].values)
    lowess_result = lowess(
        efficiency_df["Average ⬆️"].values,
        efficiency_df["#Params (B)"].values,
        return_sorted=False,
    )

    # Add trendline performance difference to dataframe
    efficiency_df["Trend_Difference"] = efficiency_df["Average ⬆️"] - lowess_result
    efficiency_df["Above_Trend"] = efficiency_df["Trend_Difference"] > 0

    # Define models to label with their positions and descriptions
    models_to_label = {
        "suayptalha_HomerCreativeAnvita-Mix-Qw7B_bfloat16": {
            "x": -1,
            "y": 55,
            "desc": "Best model under 13B parameters<br>(16GB VRAM):<br>"
            "Lightweight language models<br>ideal for prototyping,<br>"
            "chatbots, and basic NLP tasks.",
        },
        "upstage_solar-pro-preview-instruct_bfloat16": {
            "x": 13.5,
            "y": 55,
            "desc": "Best model under 26B parameters<br>(32GB VRAM):<br>"
            "Intermediate-scale models<br>balancing performance<br>"
            "and VRAM efficiency.",
        },
        "rombodawg_Rombos-LLM-V2.5-Qwen-32b_bfloat16": {
            "x": 27,
            "y": 55,
            "desc": "Best model under 40B parameters<br>(48GB VRAM):<br>"
            "Larger models with enhanced<br>reasoning and contexual<br>"
            "depth.",
        },
        "MaziyarPanahi_calme-3.2-instruct-78b_bfloat16": {
            "x": 60,
            "y": 55,
            "desc": "Best model overall (64GB+ VRAM):<br>"
            "State-of-the-art models requiring<br>extensive computational power.",
        },
    }

    # Remove these models from the main scatter plot
    main_df = efficiency_df[~efficiency_df["Eval Name"].isin(models_to_label.keys())]

    # Create scatter plot with updated colors and hover data
    fig = px.scatter(
        main_df,  # Use filtered dataframe here instead of efficiency_df
        x="#Params (B)",
        y="Average ⬆️",
        color="Above_Trend",
        color_discrete_map={
            True: "rgba(70, 150, 70, 0.6)",
            False: "rgba(150, 70, 70, 0.6)",
        },
        hover_data={
            "Eval Name": True,
            "Above_Trend": False,
        },
        title="How Model Size Impacts Performance: Spotlight on 'David' Models",
        template="plotly_white",
        labels={
            "#Params (B)": "Number of Parameters (Billions)",
            "Average ⬆️": "Average Model Performance",
            "Precision": "Model Precision",
            "Type": "Type",
            "Eval Name": "Model Name",
        },
    )

    # Remove legend
    fig.update_layout(showlegend=False)

    # Update title and subtitle
    fig.update_layout(
        title={
            "text": "LLM Performance vs. Model Size and VRAM<br>"
            "<sup style='font-size:14px'>Optimal Models for 16GB, 32GB, 48GB, and 64GB+ Configurations</sup>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        }
    )

    # Update layout to remove horizontal gridlines
    fig.update_layout(yaxis=dict(showgrid=False))  # This removes horizontal gridlines

    # Replace the single threshold line with four hardcoded lines
    for param_size in [13, 26, 40, 53]:
        fig.add_vline(
            x=param_size,
            line_dash="dash",
            line_color="gray",
            line_width=0.75,
            annotation_text=f"{param_size}B",
            annotation_position="top",
        )

    # Add trend line with modified appearance and label
    trendline = px.scatter(
        efficiency_df, x="#Params (B)", y="Average ⬆️", trendline="lowess"
    ).data[1]
    trendline.line.color = "#636EFA"  # Plotly's default blue color
    trendline.line.dash = "solid"
    trendline.line.width = 1.3
    trendline.showlegend = False  # Hide from legend
    fig.add_trace(trendline)

    # Add trendline description text box with updated explanation
    trendline_text = (
        f"<b><span style='color:#636EFA;'>Performance Trend</span></b><br>"
        f"The blue line shows the general relationship between<br>"
        f"model size and performance using LOWESS smoothing.<br><br>"
        f"Models in <span style='color:rgba(70,150,70,0.9)'>green</span> perform above the trend line.<br>"
        f"Models in <span style='color:rgba(150,70,70,0.9)'>red</span> perform below the trend line."
    )

    fig.add_annotation(
        x=0.85,
        y=0.39,
        xref="paper",
        yref="paper",
        text=trendline_text,
        showarrow=False,
        bgcolor="white",
        # bgcolor="rgba(99, 110, 250, 0.1)",  # Light blue background matching trendline
        bordercolor="#636EFA",  # Border matching trendline
        borderwidth=2,
        align="left",
        font=dict(size=14),
        xanchor="right",
        yanchor="top",
        width=355,
        borderpad=7,
    )

    for model, position in models_to_label.items():
        if model in efficiency_df["Eval Name"].values:
            model_data = efficiency_df[efficiency_df["Eval Name"] == model].iloc[0]

            # Add highlighted point
            fig.add_trace(
                go.Scatter(
                    x=[model_data["#Params (B)"]],
                    y=[model_data["Average ⬆️"]],
                    mode="markers",
                    marker=dict(
                        size=13,  # Increased size
                        color="rgba(255, 215, 0, 0.8)",  # Gold color with some transparency
                        line=dict(color="black", width=2),  # Thicker black border
                        symbol="star",  # Changed to star shape
                    ),
                    showlegend=False,
                )
            )

            # Add annotation with width constraint
            fig.add_annotation(
                x=position["x"],
                y=position["y"],
                text=(
                    f"<b>{''.join(model.split('_')[1:]).replace('bfloat16', '').replace('-', ' ')}</b><br>"
                    f"{position['desc']}"
                ),
                xref="x",
                yref="y",
                xanchor="left",
                yanchor="middle",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=0,
                ay=0,
                width=200,  # Control width of text box
                align="left",  # Align text to the left
                bordercolor="black",  # Optional: adds border to make width more visible
                borderwidth=1,
            )

    fig.update_layout(height=700, width=1500)
    return fig


def create_model_metrics_radar(df, model_name):
    """Create a radar chart showing individual metrics for a selected model."""
    # Define the metrics we want to show
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]

    # Get the model's data
    model_data = df[df["Eval Name"] == model_name].iloc[0]

    # Extract values for each metric
    values = [model_data[metric] for metric in metrics]

    # Add the first value again to close the polygon
    values.append(values[0])
    metrics.append(metrics[0])

    # Create the radar chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics,
            fill="toself",
            fillcolor="rgba(70, 150, 70, 0.3)",
            line=dict(color="rgb(70, 150, 70)", width=2),
            name=model_name,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100]  # Assuming scores are percentages
            )
        ),
        showlegend=False,
        title=dict(
            text=f"Performance Metrics: {model_name.split('_')[1]}",
            x=0.5,
            y=0.95,
            xanchor="center",  # Ensures title is centered and fully visible
        ),
        height=400,
        width=500,  # Increased width to ensure title fits
        margin=dict(t=50),  # Added top margin to prevent title cutoff
    )

    return fig


def create_model_metrics_radar_minimal_1(df, model_name):
    """Create a minimal radar chart showing individual metrics for a selected model."""
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]

    # Get the model's data
    model_data = df[df["Eval Name"] == model_name].iloc[0]

    # Extract values for each metric
    values = [model_data[metric] for metric in metrics]

    # Add the first value again to close the polygon
    values.append(values[0])
    metrics.append(metrics[0])

    # Create the radar chart
    fig = go.Figure()

    # Add lines from center to each point
    for i in range(len(values) - 1):
        fig.add_trace(
            go.Scatterpolar(
                r=[0, values[i]],
                theta=[metrics[i], metrics[i]],
                mode="lines",
                line=dict(color="rgb(70, 150, 70)", width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),  # Hide the radial axis
            angularaxis=dict(
                visible=False,  # Hide the angular axis
                showline=False,
                showticklabels=False,
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        margin=dict(l=0, r=0, t=30, b=0),  # Minimal margins
        height=150,
        width=200,
    )

    return fig


def create_model_metrics_radar_minimal_2(df, model_name):
    """Create a minimal radar chart showing individual metrics for a selected model."""
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]

    # Get the model's data
    model_data = df[df["Eval Name"] == model_name].iloc[0]

    # Extract values for each metric
    values = [model_data[metric] for metric in metrics]

    # Add the first value again to close the polygon
    values.append(values[0])
    metrics.append(metrics[0])

    # Create the radar chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=metrics,
            fill="toself",
            fillcolor="rgba(70, 150, 70, 0.3)",
            line=dict(color="rgb(70, 150, 70)", width=2),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),  # Hide the radial axis
            angularaxis=dict(
                visible=False,  # Hide the angular axis
                showline=False,
                showticklabels=False,
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        margin=dict(l=0, r=0, t=30, b=0),  # Minimal margins
        height=150,
        width=200,
    )

    return fig


def create_model_metrics_comparison(df, model_name):
    """Create a horizontal bar chart comparing model metrics to average."""
    # metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    metrics = ["BBH", "IFEval", "MATH Lvl 5", "MUSR", "GPQA", "MMLU-PRO"]

    # Get the model's data
    model_data = df[df["Eval Name"] == model_name].iloc[0]

    # Calculate averages for each metric
    averages = {metric: df[metric].mean() for metric in metrics}

    # Calculate differences from average
    differences = {metric: model_data[metric] - averages[metric] for metric in metrics}

    # Create the bar chart
    fig = go.Figure()

    # Add bars
    colors = ["#FFD700" if x >= 0 else "#EF5350" for x in differences.values()]

    fig.add_trace(
        go.Bar(
            x=list(differences.values()),
            y=list(metrics),
            orientation="h",
            marker_color=colors,
            text=[f"{diff:+.1f}" for diff in differences.values()],
            textposition="inside",
        )
    )

    # Add vertical line at x=0
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="gray")

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Performance vs. Average: {model_name.split('_')[1]}",
            x=0.5,
            y=0.95,
            xanchor="center",
        ),
        xaxis_title="Difference from Average (%)",
        yaxis_title=None,
        height=400,
        width=400,
        showlegend=False,
        margin=dict(l=20, r=20),  # Increase left margin for metric names
    )

    return fig


def create_model_metrics_comparison_minimal(df, model_name):
    """Create a minimal horizontal bar chart comparing model metrics to average."""
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]

    # Get the model's data
    model_data = df[df["Eval Name"] == model_name].iloc[0]

    # Calculate averages for each metric
    averages = {metric: df[metric].mean() for metric in metrics}

    # Calculate differences from average
    differences = {metric: model_data[metric] - averages[metric] for metric in metrics}

    # Create the bar chart
    fig = go.Figure()

    # Add bars with consistent color
    fig.add_trace(
        go.Bar(
            x=list(differences.values()),
            y=list(metrics),
            orientation="h",
            marker_color="rgba(70, 150, 70, 0.6)",
        )
    )

    # Add vertical line at x=0
    fig.add_vline(
        x=0, line_width=1, line_dash="solid", line_color="rgba(128, 128, 128, 0.3)"
    )

    # Update layout to remove all text and labels
    fig.update_layout(
        showlegend=False,
        title=dict(
            text=f"Performance vs. Average: {model_name.split('_')[1]}",
            x=0.5,
            y=0.95,
            xanchor="center",
        ),
        xaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, showline=False
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, showline=False
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=150,
        width=200,
    )

    return fig


def create_qwen_comparison_plot(df):
    """Create a horizontal bar plot comparing metrics across Qwen models."""
    # Define the metrics and models to compare
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    model_names = [
        "Qwen_Qwen2.5-7B_bfloat16",
        "Qwen_Qwen2.5-14B_bfloat16",
        "Qwen_Qwen2.5-32B_bfloat16",
        "Qwen_Qwen2.5-72B_bfloat16",
    ]

    # Filter for just these models
    qwen_df = df[df["Eval Name"].isin(model_names)]

    # Create figure
    fig = go.Figure()

    # Define colors for different model sizes
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#2196F3"]

    # Add bars for each metric
    for i, model in enumerate(model_names):
        if model in qwen_df["Eval Name"].values:
            model_data = qwen_df[qwen_df["Eval Name"] == model].iloc[0]
            values = [model_data[metric] for metric in metrics]

            fig.add_trace(
                go.Bar(
                    name=f"Qwen {int(model_data['#Params (B)'])}B",
                    y=metrics,
                    x=values,
                    orientation="h",
                    marker_color=colors[i],
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Qwen Model Family Performance Comparison",
            x=0.5,
            y=0.95,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis_title="Score (%)",
        yaxis_title=None,
        barmode="group",
        height=500,
        width=800,
        yaxis=dict(
            categoryorder="array",
            categoryarray=metrics[::-1],  # Reverse order to match radar chart
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def create_qwen_comparison_plot_vertical(df):
    """Create a vertical bar plot comparing metrics across Qwen models."""
    # Define the metrics and models to compare
    metrics = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    metrics = ["Average ⬆️", "MUSR", "GPQA", "MATH Lvl 5", "IFEval", "BBH", "MMLU-PRO"]
    model_names = [
        "Qwen_Qwen2.5-7B_bfloat16",
        "Qwen_Qwen2.5-14B_bfloat16",
        "Qwen_Qwen2.5-32B_bfloat16",
        "Qwen_Qwen2.5-72B_bfloat16",
    ]

    # Filter for just these models
    qwen_df = df[df["Eval Name"].isin(model_names)]

    # Create figure
    fig = go.Figure()

    # Define colors for different model sizes
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#2196F3"]

    # Add bars for each metric
    for i, model in enumerate(model_names):
        if model in qwen_df["Eval Name"].values:
            model_data = qwen_df[qwen_df["Eval Name"] == model].iloc[0]
            values = [model_data[metric] for metric in metrics]

            fig.add_trace(
                go.Bar(
                    name=f"Qwen {int(model_data['#Params (B)'])}B",
                    x=metrics,
                    y=values,
                    marker_color=colors[i],
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Qwen Model Family Performance Comparison",
            x=0.5,
            y=0.95,
            xanchor="center",
            font=dict(size=20),
        ),
        yaxis_title="Score (%)",
        xaxis_title=None,
        barmode="group",
        height=600,
        width=1000,
        xaxis=dict(
            tickangle=45,  # Angle the metric labels for better readability
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=20, r=20, t=80, b=100
        ),  # Increased bottom margin for angled labels
    )

    return fig


def main():
    st.set_page_config(page_title="LLM Efficiency Analysis", layout="wide")

    # Create filters
    selected_architecture, size_range, selected_types = create_filters()

    # Load and filter data
    df = load_data()
    filtered_df = filter_data(df, selected_architecture, size_range, selected_types)

    # Create and display the plot
    fig = create_efficiency_plot(filtered_df)
    st.plotly_chart(fig, use_container_width=False)

    # After creating the efficiency plot, add model selection and radar chart
    if len(filtered_df) > 0:
        st.subheader("Individual Model Analysis")

        # Model selector
        selected_model = st.selectbox(
            "Select a model to view detailed metrics",
            options=filtered_df["Eval Name"].tolist(),
            format_func=lambda x: x.split("_")[1],  # Show cleaner model names
        )

        # Create layout for the two charts
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            radar_fig = create_model_metrics_radar(filtered_df, selected_model)
            st.plotly_chart(radar_fig, use_container_width=True)

        with col2:
            radar_fig_minimal = create_model_metrics_radar_minimal_1(
                filtered_df, selected_model
            )
            st.plotly_chart(radar_fig_minimal, use_container_width=True)

        with col3:
            comparison_fig = create_model_metrics_comparison(
                filtered_df, selected_model
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

        with col4:
            comparison_fig_minimal = create_model_metrics_comparison_minimal(
                filtered_df, selected_model
            )
            st.plotly_chart(comparison_fig_minimal, use_container_width=True)

    # Add download buttons for the plot
    col1, col2 = st.columns(2)

    with col1:
        buffer = fig.to_html(include_plotlyjs="cdn", full_html=False)
        st.download_button(
            label="Download Plot as HTML",
            data=buffer,
            file_name="llm_efficiency_plot.html",
            mime="text/html",
            help="Download the plot as an interactive HTML file",
        )

    with col2:
        svg_buffer = fig.to_image(format="svg")
        st.download_button(
            label="Download Plot as SVG",
            data=svg_buffer,
            file_name="llm_efficiency_plot.svg",
            mime="image/svg+xml",
            help="Download the plot as a vector SVG file",
        )

    # Create and display the Qwen comparison plot
    col1, col2 = st.columns(2)

    with col1:
        qwen_comparison_fig = create_qwen_comparison_plot(df)
        st.plotly_chart(qwen_comparison_fig, use_container_width=True)

    with col2:
        qwen_comparison_fig_vertical = create_qwen_comparison_plot_vertical(df)
        st.plotly_chart(qwen_comparison_fig_vertical, use_container_width=True)


if __name__ == "__main__":
    main()
