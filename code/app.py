import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

# Set page config
st.set_page_config(
    page_title="Open LLM Leaderboard Explorer", page_icon="🤖", layout="wide"
)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/open_llm_leaderboard.csv")
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

selected_size = size_range  # Tuple containing (min, max) values

# Model family filter - dynamically updated based on architecture and size
filtered_df = df.copy()
if selected_architecture != "All":
    filtered_df = filtered_df[filtered_df["Architecture"] == selected_architecture]

# Apply size filter using the range from the slider
filtered_df = filtered_df[
    (filtered_df["#Params (B)"].between(selected_size[0], selected_size[1]))
    & (filtered_df["#Params (B)"] != -1)
]

available_families = filtered_df["Model Family"].unique()
model_families = ["All"] + sorted(available_families.tolist(), key=str.lower)
selected_family = st.sidebar.selectbox("Select Model Family", model_families)

# Apply final family filter
if selected_family != "All":
    filtered_df = filtered_df[filtered_df["Model Family"] == selected_family]

# Main content
st.title("Open LLM Leaderboard Explorer")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Scatter Plot", "📋 Data Table", "🏆 Top Performers", "📈 Model Analysis"]
)

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

    # Filter out string columns for size metric
    numeric_metrics = {
        label: metric
        for label, metric in metric_labels.items()
        if df[metric].dtype in ["int64", "float64"]
    }

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
        # Size metric selector - only numeric columns
        size_metric_label = st.selectbox(
            "Point size metric",
            options=["None"] + list(numeric_metrics.keys()),
            index=list(numeric_metrics.keys()).index("Model Properties: Hub ❤️") + 1,
        )

    # Group by and highlight selectors in new row
    col4, col5 = st.columns(2)

    with col4:
        # Group by selector
        group_by = st.selectbox(
            "Group by",
            options=sorted(
                ["Architecture", "Type", "Model Family", "Precision", "MoE"],
                key=str.lower,
            ),
            index=0,
        )

    with col5:
        # Highlight model selector
        highlight_model = st.selectbox(
            "Highlight specific model",
            options=["None"] + sorted(filtered_df["Eval Name"].tolist(), key=str.lower),
            index=0,
        )

    # Get actual metric names from labels
    x_metric = metric_labels[x_metric_label]
    y_metric = metric_labels[y_metric_label]
    size_metric = (
        numeric_metrics[size_metric_label] if size_metric_label != "None" else None
    )

    # Create base scatter plot
    scatter_plot_args = {
        "data_frame": filtered_df,
        "x": x_metric,
        "y": y_metric,
        "color": group_by,
        "hover_data": [
            "Eval Name",
            "Model Family",
            "Base Model",
            "T",
        ],
        "title": f"{y_metric} vs {x_metric}",
        "template": "plotly_white",
    }

    # Add size parameter only if a size metric is selected
    if size_metric is not None:
        scatter_plot_args.update(
            {
                "size": size_metric,
                "size_max": 30,
            }
        )
        scatter_plot_args["hover_data"].append(size_metric)
        scatter_plot_args["title"] += f" (size: {size_metric})"

    fig = px.scatter(**scatter_plot_args)

    # If a model is selected for highlighting, add it as a separate trace
    if highlight_model != "None":
        highlight_df = filtered_df[filtered_df["Eval Name"] == highlight_model]

        # Add highlighted point
        fig.add_trace(
            go.Scatter(
                x=highlight_df[x_metric],
                y=highlight_df[y_metric],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=15,
                    color="yellow",
                    line=dict(color="black", width=2),
                ),
                name=highlight_model,
                hovertext=[highlight_model],
                showlegend=True,
            )
        )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        legend_title=group_by,
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
            "Eval Name",
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

    # Add statistical overview section
    st.subheader("Statistical Overview")

    # Add radio button for data selection
    stats_data_selection = st.radio(
        "Select data for statistics",
        ["Filtered Data", "Complete Dataset"],
        horizontal=True,
    )

    # Get only numeric columns from selected columns that are present in the full dataset
    numeric_cols = [
        col
        for col in selected_columns
        if col in df.columns and df[col].dtype in ["int64", "float64"]
    ]

    if numeric_cols:
        # Choose dataset based on user selection
        data_for_stats = filtered_df if stats_data_selection == "Filtered Data" else df

        # Create description DataFrame
        desc_df = data_for_stats[numeric_cols].describe()

        # Round all values to 2 decimal places
        desc_df = desc_df.round(2)

        # Transpose for better readability
        desc_df_t = desc_df.transpose()

        # Add column names as a new column
        desc_df_t.insert(0, "Metric", desc_df_t.index)

        # Add info about number of models included
        st.info(f"Statistics based on {len(data_for_stats)} models")

        # Display the statistics
        st.dataframe(
            desc_df_t,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": "Metric",
                "count": "Count",
                "mean": "Mean",
                "std": "Std Dev",
                "min": "Min",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Max",
            },
        )
    else:
        st.info("No numeric columns selected for statistical analysis.")


with tab3:
    st.header("Top Performers Analysis")

    # Add radio button for analysis type
    analysis_type = st.radio(
        "Analysis Level", ["Individual Models", "Architectures"], horizontal=True
    )

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

    # Create two columns for selectors
    col1, col2 = st.columns(2)

    with col1:
        # Select metric for top performers
        top_metric = st.selectbox("Select metric for ranking", metrics, index=0)

    with col2:
        # Only show color selector for individual models
        if analysis_type == "Individual Models":
            color_by = st.selectbox(
                "Color by",
                options=sorted(
                    ["Architecture", "Type", "Model Family", "Precision", "MoE"],
                    key=str.lower,
                ),
                index=0,
            )

    # Number of top performers to show
    top_n = st.slider(f"Number of top {analysis_type.lower()} to show", 5, 20, 10)

    if analysis_type == "Individual Models":
        # Original model-level analysis
        top_performers = filtered_df.nlargest(top_n, top_metric).sort_values(
            top_metric, ascending=False
        )
        x_axis = "Eval Name"
    else:
        # Architecture-level analysis
        arch_stats = (
            filtered_df.groupby("Architecture")
            .agg({top_metric: ["mean", "max", "count"], "#Params (B)": "mean"})
            .round(2)
        )

        arch_stats.columns = [
            "Average Score",
            "Best Score",
            "Number of Models",
            "Avg Model Size",
        ]
        arch_stats = arch_stats.sort_values("Average Score", ascending=False).head(
            top_n
        )

        top_performers = arch_stats.reset_index()
        x_axis = "Architecture"

    # Add info about the number of models included
    st.info(
        f"Statistics based on {len(filtered_df)} models with {selected_size[0]}B to {selected_size[1]}B parameters"
    )

    # Create bar chart
    fig = px.bar(
        top_performers,
        x=x_axis,
        y="Average Score" if analysis_type == "Architectures" else top_metric,
        color=color_by if analysis_type == "Individual Models" else None,
        title=f"Top {top_n} {analysis_type} by {top_metric}",
        template="plotly_white",
        category_orders={
            x_axis: top_performers[x_axis].tolist()
        },  # Force the x-axis order
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title=(
            "Architecture" if analysis_type == "Architectures" else "Model Name"
        ),
        yaxis_title=top_metric,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed stats
    st.subheader("Detailed Statistics")

    if analysis_type == "Individual Models":
        # Original model-level display
        if top_metric in ["Eval Name", "Architecture", "#Params (B)"]:
            display_columns = ["Eval Name", "Architecture", "#Params (B)"]
        else:
            display_columns = ["Eval Name", "Architecture", "#Params (B)", top_metric]
    else:
        # Show all architecture statistics
        display_columns = [
            "Architecture",
            "Average Score",
            "Best Score",
            "Number of Models",
            "Avg Model Size",
        ]

    st.dataframe(
        (
            top_performers[display_columns]
            if analysis_type == "Individual Models"
            else top_performers
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.header("Architecture Performance Analysis")

    # Add option to analyze all architectures or a specific one
    analysis_type = st.radio(
        "Analysis Type", ["Single Architecture", "All Architectures"]
    )

    if analysis_type == "Single Architecture":
        # Existing single architecture analysis code
        if selected_architecture == "All":
            analysis_architecture = st.selectbox(
                "Select Architecture",
                options=sorted(df["Architecture"].unique().tolist()),
                index=0,
            )
            # Apply size filter if selected
            architecture_df = df[df["Architecture"] == analysis_architecture]
            architecture_df = architecture_df[
                (
                    architecture_df["#Params (B)"].between(
                        selected_size[0], selected_size[1]
                    )
                )
                & (architecture_df["#Params (B)"] != -1)
            ]
        else:
            analysis_architecture = selected_architecture
            architecture_df = filtered_df

        # Show size filter status
        st.info(
            f"Showing analysis for {len(filtered_df)} models between {selected_size[0]}B and {selected_size[1]}B parameters"
        )

    else:  # All Architectures analysis
        architecture_df = (
            filtered_df
            if selected_architecture == "All"
            else df[df["Architecture"].isin([selected_architecture])]
        )

        # Show comparative metrics across architectures
        st.subheader("Architecture Comparison")

        # Calculate average performance for each architecture
        arch_comparison = []
        for arch in architecture_df["Architecture"].unique():
            arch_data = architecture_df[architecture_df["Architecture"] == arch]
            metrics_avg = {
                "Architecture": arch,
                "Number of Models": len(arch_data),
                "Avg Model Size": arch_data["#Params (B)"].mean(),
                "Average Score": arch_data["Average ⬆️"].mean(),
                "Max Score": arch_data["Average ⬆️"].max(),
            }
            arch_comparison.append(metrics_avg)

        comparison_df = pd.DataFrame(arch_comparison)

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Architectures", len(comparison_df))
        with col2:
            st.metric("Total Models", comparison_df["Number of Models"].sum())
        with col3:
            st.metric(
                "Best Architecture",
                comparison_df.loc[
                    comparison_df["Average Score"].idxmax(), "Architecture"
                ],
            )

        # Create comparison visualizations
        fig1 = px.bar(
            comparison_df,
            x="Architecture",
            y="Average Score",
            title="Average Performance by Architecture",
            template="plotly_white",
        )
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

        # Model size distribution by architecture
        fig2 = px.box(
            architecture_df,
            x="Architecture",
            y="#Params (B)",
            title="Model Size Distribution by Architecture",
            template="plotly_white",
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed metrics table
        st.subheader("Architecture Metrics")
        comparison_df = comparison_df.round(2)
        st.dataframe(
            comparison_df.sort_values("Average Score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # Continue with existing performance metrics and specialization analysis...
    performance_metrics = [
        "Average ⬆️",
        "IFEval",
        "BBH",
        "MATH Lvl 5",
        "GPQA",
        "MUSR",
        "MMLU-PRO",
    ]

    # Only proceed if there are models to analyze
    if len(architecture_df) > 0:
        # Calculate differences from average for each metric
        model_data = []
        for metric in performance_metrics:
            overall_avg = df[metric].mean()
            arch_avg = architecture_df[metric].mean()
            diff_from_avg = arch_avg - overall_avg
            model_data.append(
                {
                    "Metric": metric,
                    "Difference": diff_from_avg,
                    "Architecture Average": arch_avg,
                    "Overall Average": overall_avg,
                }
            )

        # Create DataFrame for plotting
        comparison_df = pd.DataFrame(model_data)

        # Create bar chart
        fig = go.Figure()

        # Add bars for difference from average
        fig.add_trace(
            go.Bar(
                x=comparison_df["Metric"],
                y=comparison_df["Difference"],
                name="Difference from Average",
                marker_color=comparison_df["Difference"].apply(
                    lambda x: "green" if x > 0 else "red"
                ),
            )
        )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Update layout with conditional title
        if analysis_type == "Single Architecture":
            title = (
                f"{analysis_architecture} Architecture Performance vs. Overall Average"
            )
            if selected_size != "All":
                title += f" ({selected_size} models)"
        else:
            title = "Selected Architectures Performance vs. Overall Average"
            if selected_size != "All":
                title += f" ({selected_size} models)"

        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Difference from Average",
            height=500,
            showlegend=False,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create two columns for metrics and details
        col1, col2 = st.columns(2)

        with col1:
            # Show detailed metrics table
            st.subheader("Detailed Metrics")

            # Format the numbers to 2 decimal places
            comparison_df_display = comparison_df.copy()
            for col in ["Difference", "Architecture Average", "Overall Average"]:
                comparison_df_display[col] = comparison_df_display[col].round(2)

            st.dataframe(
                comparison_df_display, use_container_width=True, hide_index=True
            )

        with col2:
            # Show architecture details
            st.subheader("Architecture Details")

            # Calculate architecture statistics
            arch_stats = [
                {"Metric": "📊 Total Models", "Value": str(len(architecture_df))},
                {
                    "Metric": "📏 Average Size",
                    "Value": f"{architecture_df['#Params (B)'].mean():.1f}B",
                },
                {
                    "Metric": "📐 Size Range",
                    "Value": f"{architecture_df['#Params (B)'].min():.1f}B - {architecture_df['#Params (B)'].max():.1f}B",
                },
                {
                    "Metric": "👥 Most Common Family",
                    "Value": str(architecture_df["Model Family"].mode().iloc[0]),
                },
                {
                    "Metric": "🏷️ Most Common Type",
                    "Value": str(
                        architecture_df["Type"].mode().iloc[0]
                        if "Type" in architecture_df.columns
                        else "N/A"
                    ),
                },
            ]

            # Create DataFrame explicitly as strings
            stats_df = pd.DataFrame(arch_stats)
            stats_df = stats_df.astype(str)  # Ensure all values are strings

            # Display with explicit column configuration
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                },
            )

        # Show all models in this architecture
        st.subheader("Models in this Architecture")
        st.dataframe(
            architecture_df[["Eval Name", "#Params (B)", "Average ⬆️"]].sort_values(
                "Average ⬆️", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Add a new section for specialization analysis
        st.subheader("Model Specialization Analysis")

        # Calculate specialization scores for each model
        performance_metrics = [
            "IFEval",
            "BBH",
            "MATH Lvl 5",
            "GPQA",
            "MUSR",
            "MMLU-PRO",
        ]

        def calculate_specialization(row):
            # Calculate normalized differences from mean for each metric
            diffs = []
            for metric in performance_metrics:
                metric_mean = df[metric].mean()
                metric_std = df[metric].std()
                normalized_diff = (
                    (row[metric] - metric_mean) / metric_std if metric_std != 0 else 0
                )
                diffs.append(normalized_diff)

            # Calculate specialization score and identify specialized areas
            specialization_score = np.std(diffs)
            specialized_metrics = []
            for metric, diff in zip(performance_metrics, diffs):
                if abs(diff) > 1:  # More than 1 standard deviation from mean
                    specialized_metrics.append(f"{metric} ({diff:+.2f}σ)")

            return pd.Series(
                {
                    "specialization_score": specialization_score,
                    "specialized_metrics": (
                        ", ".join(specialized_metrics)
                        if specialized_metrics
                        else "None"
                    ),
                }
            )

        # Calculate specialization for filtered models
        specialization_results = architecture_df.apply(calculate_specialization, axis=1)
        specialization_df = pd.concat(
            [
                architecture_df[
                    [
                        "Eval Name",
                        "Architecture",
                        "#Params (B)",
                        "Average ⬆️",
                        "Model Family",
                    ]
                ],
                specialization_results,
            ],
            axis=1,
        ).sort_values("specialization_score", ascending=False)

        # Display results in a clean layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("📈 Most Specialized Models")
            st.dataframe(
                specialization_df[
                    ["Eval Name", "#Params (B)", "specialization_score"]
                ].head(),
                use_container_width=True,
                hide_index=True,
            )

            st.write("📉 Most Generalist Models")
            st.dataframe(
                specialization_df[
                    ["Eval Name", "#Params (B)", "specialization_score"]
                ].tail(),
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.write("🔍 Detailed Analysis")
            st.dataframe(
                specialization_df[
                    [
                        "Eval Name",
                        "#Params (B)",
                        "Model Family",
                        "specialization_score",
                        "specialized_metrics",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # Visualize specialization distribution
        fig = px.histogram(
            specialization_df,
            x="specialization_score",
            title="Distribution of Model Specialization",
            template="plotly_white",
            nbins=20,
        )
        fig.update_layout(
            xaxis_title="Specialization Score",
            yaxis_title="Number of Models",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"No models found for {analysis_architecture} with {selected_size} parameters"
        )

# Footer
st.markdown("---")
st.markdown(
    "Data source: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)"
)
