import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(
    page_title="Open LLM Leaderboard Explorer", page_icon="🤖", layout="wide"
)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/open_llm_leaderboard.csv")
    df["model_family"] = df["Base Model"].apply(lambda x: x.split("/")[0])
    return df


# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Architecture filter
architectures = ["All"] + sorted(df["Architecture"].unique().tolist())
selected_architecture = st.sidebar.selectbox("Select Architecture", architectures)

# Model size filter - dynamically updated based on architecture
if selected_architecture == "All":
    available_sizes = df["#Params (B)"].dropna().unique()
else:
    available_sizes = (
        df[df["Architecture"] == selected_architecture]["#Params (B)"].dropna().unique()
    )

# Convert to integers and sort numerically, then format as strings with "B"
size_ranges = ["All"] + [
    f"{int(size)}B" for size in sorted(available_sizes) if size is not None
]
selected_size = st.sidebar.selectbox("Select Model Size", size_ranges)

# Model family filter - dynamically updated based on architecture and size
filtered_df = df.copy()
if selected_architecture != "All":
    filtered_df = filtered_df[filtered_df["Architecture"] == selected_architecture]
if selected_size != "All":
    size_value = int(selected_size.replace("B", ""))
    filtered_df = filtered_df[filtered_df["#Params (B)"] == size_value]

available_families = filtered_df["model_family"].unique()
model_families = ["All"] + sorted(available_families.tolist())
selected_family = st.sidebar.selectbox("Select Model Family", model_families)

# Apply final family filter
if selected_family != "All":
    filtered_df = filtered_df[filtered_df["model_family"] == selected_family]

# Main content
st.title("Open LLM Leaderboard Explorer")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Scatter Plot", "📋 Data Table", "🏆 Top Performers"])

with tab1:
    st.header("Performance Comparison")

    # Create groups of metrics for better organization
    performance_metrics = [
        "Average ⬆️",
        "IFEval",
        "BBH",
        "MATH Lvl 5",
        "GPQA",
        "MUSR",
        "MMLU-PRO",
    ]

    # Model characteristics (numerical)
    model_metrics = [
        "#Params (B)",
        "Generation",
        "Precision",
        "Hub ❤️",
        "Submission Date",
    ]

    # Model properties (categorical)
    model_properties = ["T", "Type", "Weight Type"]  # model type (emoji categories)

    # Combine all metrics with group labels
    all_metrics = [
        ("Performance Metrics", metric) for metric in performance_metrics
    ] + [
        ("Model Properties", metric)
        for metric in model_properties + model_metrics
        if metric in df.columns
    ]

    # Create formatted labels for the selectbox
    metric_labels = {f"{group}: {metric}": metric for group, metric in all_metrics}

    # Create layout columns for the selectors
    col1, col2, col3 = st.columns(3)

    with col1:
        # X-axis metric selector
        x_metric_label = st.selectbox(
            "X-axis metric",
            options=list(metric_labels.keys()),
            index=list(metric_labels.keys()).index(f"Model Properties: #Params (B)"),
        )

    with col2:
        # Y-axis metric selector
        y_metric_label = st.selectbox(
            "Y-axis metric",
            options=list(metric_labels.keys()),
            index=list(metric_labels.keys()).index("Performance Metrics: Average ⬆️"),
        )

    with col3:
        # Group by selector
        group_by = st.selectbox(
            "Group by", options=["Architecture", "Type", "model_family", "T"], index=0
        )

    # Get actual metric names from labels
    x_metric = metric_labels[x_metric_label]
    y_metric = metric_labels[y_metric_label]

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x=x_metric,
        y=y_metric,
        color=group_by,  # Use the selected grouping variable
        hover_data=[
            "eval_name",
            "model_family",
            "Base Model",
            "T",
        ],
        title=f"{y_metric} vs {x_metric}",
        template="plotly_white",
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        legend_title=group_by,  # Update legend title to match grouping
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Explorer")

    # Column selector
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display",
        all_columns,
        default=[
            "eval_name",
            "Architecture",
            "#Params (B)",
            "Average ⬆️",
            "IFEval",
            "BBH",
            "MATH Lvl 5",
        ],
    )

    # Show filtered dataframe
    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns], use_container_width=True, hide_index=True
        )
    else:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Top Performers Analysis")

    # Define metrics for ranking
    metrics = [
        "Average ⬆️",
        "IFEval",
        "BBH",
        "MATH Lvl 5",
        "GPQA",
        "MUSR",
        "MMLU-PRO",
        "#Params (B)",
        "Generation",
    ]

    # Select metric for top performers
    top_metric = st.selectbox("Select metric for ranking", metrics, index=0)

    # Number of top performers to show
    top_n = st.slider("Number of top performers to show", 5, 20, 10)

    # Get top performers
    top_performers = filtered_df.nlargest(top_n, top_metric)

    # Create bar chart
    fig = px.bar(
        top_performers,
        x="eval_name",
        y=top_metric,
        color="Architecture",
        title=f"Top {top_n} Models by {top_metric}",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_tickangle=-45, xaxis_title="Model Name", yaxis_title=top_metric
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed stats
    st.subheader("Detailed Statistics")
    # Only include top_metric if it's not already in the display columns
    if top_metric in ["eval_name", "Architecture", "#Params (B)"]:
        display_columns = ["eval_name", "Architecture", "#Params (B)"]
    else:
        display_columns = ["eval_name", "Architecture", "#Params (B)", top_metric]

    st.dataframe(
        top_performers[display_columns], use_container_width=True, hide_index=True
    )

# Footer
st.markdown("---")
st.markdown(
    "Data source: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)"
)
