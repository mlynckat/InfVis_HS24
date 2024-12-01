import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def load_data(file_path="data/open_llm_leaderboard.csv"):
    """Load and preprocess the data."""
    df = pd.read_csv(file_path)
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


def create_efficiency_plot(filtered_df, size_threshold, performance_threshold):
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

    # Define "Davids" based on thresholds
    davids = efficiency_df[
        (efficiency_df["Size_Percentile"] <= size_threshold)  # Smaller models
        & (
            efficiency_df["Performance_Percentile"] >= performance_threshold
        )  # Better performers
    ].sort_values("Performance per Param", ascending=False)

    # Create scatter plot
    fig = px.scatter(
        efficiency_df,
        x="#Params (B)",
        y="Average ⬆️",
        color="Type",
        title="Model Performance vs Size (Highlighting 'David' Models)",
        template="plotly_white",
    )

    # Add reference lines for thresholds
    size_threshold_value = np.percentile(efficiency_df["#Params (B)"], size_threshold)
    performance_threshold_value = np.percentile(
        efficiency_df["Average ⬆️"], performance_threshold
    )

    fig.add_vline(
        x=size_threshold_value,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{int(np.ceil(size_threshold_value))}B",
        annotation_position="top",
    )
    fig.add_hline(
        y=performance_threshold_value,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{100-performance_threshold}th Performance Percentile",
        annotation_position="right",
    )

    # Add trend line with modified appearance
    trendline = px.scatter(
        efficiency_df, x="#Params (B)", y="Average ⬆️", trendline="lowess"
    ).data[1]
    trendline.line.color = "grey"  # Change color to grey
    trendline.line.dash = "solid"  # Make line solid instead of dashed
    trendline.line.width = 1  # Make the line thinner
    fig.add_trace(trendline)

    fig.update_traces(showlegend=False)

    # Highlight top David model if any exist
    if len(davids) > 0:
        top_david = davids.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[top_david["#Params (B)"]],
                y=[top_david["Average ⬆️"]],
                mode="markers",
                marker=dict(
                    size=15,
                    color="#FFD700",  # Gold
                    line=dict(color="black", width=2),
                    symbol="circle",
                ),
                name=top_david["Eval Name"],
                showlegend=False,
            )
        )

        # Update annotation text to include all metrics
        annotation_text = (
            f"Top Model: "
            f"<b>{top_david['Eval Name']}</b><br>"
            f"• Size: {int(top_david['#Params (B)'])}B params<br>"
            f"• Architecture: {top_david['Architecture']}<br>"
            # f"• Efficiency: {top_david['Performance per Param']:.2f}<br>"
            f"<br>Individual Metrics:<br>"
            f"• IFEval: {top_david.get('IFEval', 'N/A'):.2f}<br>"
            f"• BBH: {top_david.get('BBH', 'N/A'):.2f}<br>"
            f"• MATH Lvl 5: {top_david.get('MATH Lvl 5', 'N/A'):.2f}<br>"
            f"• GPQA: {top_david.get('GPQA', 'N/A'):.2f}<br>"
            f"• MUSR: {top_david.get('MUSR', 'N/A'):.2f}<br>"
            f"• MMLU-PRO: {top_david.get('MMLU-PRO', 'N/A'):.2f}<br>"
            f"• Average Score: {top_david['Average ⬆️']:.2f}<br>"
        )

        # Update annotation box with matching gold theme
        fig.add_annotation(
            x=0.92,
            y=0.08,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255, 215, 0, 0.1)",  # Light gold background
            bordercolor="#FFD700",  # Gold border matching the highlight
            borderwidth=2,
            align="left",
            font=dict(size=11),
            xanchor="right",
            yanchor="bottom",
            width=300,
        )

    fig.update_layout(height=600)
    return fig, davids


def main():
    st.set_page_config(page_title="LLM Efficiency Analysis", layout="wide")

    # Create filters
    selected_architecture, size_range, selected_types = create_filters()

    # Load and filter data
    df = load_data()
    filtered_df = filter_data(df, selected_architecture, size_range, selected_types)

    # Threshold controls
    st.subheader("Threshold Controls")
    col1, col2 = st.columns(2)

    with col1:
        size_threshold = st.slider(
            "Size Threshold (percentile)",
            min_value=5,
            max_value=95,
            value=77,
            step=1,
            help="Models below this size percentile will be considered. Lower value = smaller models.",
        )

    with col2:
        performance_threshold = st.slider(
            "Performance Threshold (percentile)",
            min_value=50,
            max_value=95,
            value=90,
            step=5,
            help="Models above this performance percentile will be considered. Higher value = better performers.",
        )

    # Create and display the plot
    fig, davids = create_efficiency_plot(
        filtered_df, size_threshold, performance_threshold
    )
    st.plotly_chart(fig, use_container_width=True)

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

    # Show number of models found
    st.metric(
        "Models Found",
        len(davids),
        help="Number of models meeting the selected criteria",
    )


if __name__ == "__main__":
    main()
