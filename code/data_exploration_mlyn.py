import pandas as pd
import altair as alt
import os

# Create a directory to store individual charts
os.makedirs('individual_charts', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/open_llm_leaderboard.csv')

# Columns to exclude
exclude_columns = ["Maintainer's Highlight", "Model", "Model sha"]


# Add dataset overview
print("\nDataset Overview:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn descriptions:")
for column in df.columns:
    dtype = df[column].dtype
    n_unique = df[column].nunique()
    n_missing = df[column].isna().sum()
    print(f"\n{column}:")
    print(f"  - Type: {dtype}")
    print(f"  - Unique values: {n_unique}")
    print(f"  - Missing values: {n_missing}")
    
    # Add value range or unique values based on dtype and number of unique values
    if df[column].dtype in ['int64', 'float64']:
        print(f"  - Range: {df[column].min()} to {df[column].max()}")
    elif df[column].dtype == 'object' and n_unique <= 10:  # Only show if 10 or fewer unique values
        unique_vals = df[column].unique()
        print(f"  - Values: {', '.join(str(val) for val in unique_vals if pd.notna(val))}")
    elif df[column].dtype == 'object':
        # For columns with many unique string values, show a few examples
        sample_vals = df[column].dropna().sample(min(3, n_unique))
        print(f"  - Sample values: {', '.join(str(val) for val in sample_vals)}")

# 1. Distribution of Precision Types
precision_dist = alt.Chart(df).mark_bar().encode(
    x=alt.X('Precision:N', title='Precision Type'),
    y=alt.Y('count():Q', title='Count'),
    color='Precision:N'
).properties(
    title='Distribution of Precision Types',
    width=400,
    height=300
)
precision_dist.save('individual_charts/1_precision_distribution.html')

# 2. Average Score vs. Model Precision
precision_perf = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('Precision:N', title='Precision Type'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color='Precision:N'
).properties(
    title='Performance Distribution by Precision Type',
    width=400,
    height=300
)
precision_perf.save('individual_charts/2_precision_performance.html')

# 3. Model Performance Across Different Types
type_perf = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('Type:N', title='Model Type'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color='Type:N'
).properties(
    title='Performance Distribution by Model Type',
    width=400,
    height=300
)
type_perf.save('individual_charts/3_type_performance.html')

# 4. Evolution of Model Performance Over Time
# Convert date to datetime
df['Upload To Hub Date'] = pd.to_datetime(df['Upload To Hub Date'])
time_perf = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X('Upload To Hub Date:T', title='Upload Date'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    tooltip=['Model', 'Average ⬆️', 'Upload To Hub Date']
).properties(
    title='Model Performance Over Time',
    width=600,
    height=300
)
time_perf.save('individual_charts/4_time_performance.html')

# 5. Hub Likes vs. Model Performance
likes_perf = alt.Chart(df).mark_circle().encode(
    x=alt.X('Hub ❤️:Q', title='Hub Likes'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    tooltip=['Model', 'Hub ❤️', 'Average ⬆️']
).properties(
    title='Hub Likes vs Performance',
    width=400,
    height=300
)
likes_perf.save('individual_charts/5_likes_performance.html')

# 6. Model Performance by Architecture and Precision
arch_precision = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('Architecture:N', title='Architecture'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color='Precision:N',
    column=alt.Column('Precision:N', title='Precision Type')
).properties(
    title='Performance by Architecture and Precision',
    width=200,
    height=300
)
arch_precision.save('individual_charts/6_arch_precision_performance.html')

# 7. MoE vs Non-MoE Performance
moe_perf = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('MoE:N', title='MoE Type'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color='MoE:N'
).properties(
    title='MoE vs Non-MoE Performance',
    width=400,
    height=300
)
moe_perf.save('individual_charts/7_moe_performance.html')

# 8. Comparison of Performance Metrics
metrics = ['Average ⬆️', 'IFEval', 'BBH', 'MMLU-PRO', 'MATH Lvl 5']
metrics_df = df[metrics].melt(var_name='Metric', value_name='Score')
metrics_comparison = alt.Chart(metrics_df).mark_boxplot().encode(
    x=alt.X('Metric:N', title='Metric'),
    y=alt.Y('Score:Q', title='Score'),
    color='Metric:N'
).properties(
    title='Comparison of Performance Metrics',
    width=500,
    height=300
)
metrics_comparison.save('individual_charts/8_metrics_comparison.html')

# 9. Architecture Entries vs Performance (Bubble Plot)
arch_summary = df.groupby('Architecture').agg({
    'Average ⬆️': 'mean',
    'Model': 'count'
}).reset_index()

bubble_plot = alt.Chart(arch_summary).mark_circle().encode(
    x=alt.X('Architecture:N', title='Architecture'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    size=alt.Size('Model:Q', title='Number of Models'),
    tooltip=['Architecture', 'Average ⬆️', 'Model']
).properties(
    title='Architecture Performance and Popularity',
    width=600,
    height=400
)
bubble_plot.save('individual_charts/9_architecture_bubble.html')

# 10. Generalization vs Specialization
performance_metrics = ['Average ⬆️', 'IFEval', 'BBH', 'MMLU-PRO', 'MATH Lvl 5']
correlation_matrix = df[performance_metrics].corr()

# Create the base chart with both points and diagonal line
scatter_matrix = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    tooltip=['Model'] + performance_metrics
).properties(
    width=150,
    height=150
)

# Create diagonal line data
min_val = min(df[performance_metrics].min())
max_val = max(df[performance_metrics].max())
diagonal_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})

# Base diagonal line
diag_line = alt.Chart(diagonal_df).mark_line(
    color='red',
    strokeDash=[10, 10]
).encode(
    x=alt.X('x:Q'),
    y=alt.Y('y:Q')
)

# 10a. Color by Architecture
scatter_matrix_arch = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color=alt.Color('Architecture:N', legend=alt.Legend(orient='right')),
    tooltip=['Model', 'Architecture'] + performance_metrics
).properties(
    width=150,
    height=150
)

scatter_matrix_arch_final = (scatter_matrix_arch + diag_line).repeat(
    row=performance_metrics,
    column=performance_metrics
).properties(
    title='Performance Metrics Comparison by Architecture'
)
scatter_matrix_arch_final.save('individual_charts/10a_performance_matrix_architecture.html')

# 10b. Color by Model Family
scatter_matrix_family = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color=alt.Color('Base Model:N', legend=alt.Legend(orient='right')),
    tooltip=['Model', 'Base Model'] + performance_metrics
).properties(
    width=150,
    height=150
)

scatter_matrix_family_final = (scatter_matrix_family + diag_line).repeat(
    row=performance_metrics,
    column=performance_metrics
).properties(
    title='Performance Metrics Comparison by Model Family'
)
scatter_matrix_family_final.save('individual_charts/10b_performance_matrix_family.html')

# 10c. Color by Type
scatter_matrix_type = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color=alt.Color('Type:N', legend=alt.Legend(orient='right')),
    tooltip=['Model', 'Type'] + performance_metrics
).properties(
    width=150,
    height=150
)

scatter_matrix_type_final = (scatter_matrix_type + diag_line).repeat(
    row=performance_metrics,
    column=performance_metrics
).properties(
    title='Performance Metrics Comparison by Type'
)
scatter_matrix_type_final.save('individual_charts/10c_performance_matrix_type.html')

# Print correlations
print("\nCorrelation Matrix between Performance Metrics:")
print(correlation_matrix.round(3))

# 1. Reshape the data for faceting
metrics_to_plot = performance_metrics[1:]  # Skip 'Average ⬆️' as it's used for y-axis
plot_data = df.melt(
    id_vars=['Model', 'Type', 'Average ⬆️'], 
    value_vars=metrics_to_plot,
    var_name='Metric', 
    value_name='Score'
)

# 2. Create a combined dataset for both the scatter plot and diagonal line
min_val = min(plot_data['Score'].min(), plot_data['Average ⬆️'].min())
max_val = max(plot_data['Score'].max(), plot_data['Average ⬆️'].max())

# 3. Add diagonal line points to each metric group
diagonal_data = []
for metric in metrics_to_plot:
    for type_val in df['Type'].unique():  # Add this loop to create lines for each Type
        diagonal_data.append({
            'Model': 'reference',
            'Type': type_val,  # Use the specific type value
            'Average ⬆️': min_val,
            'Metric': metric,
            'Score': min_val,
            'is_diagonal': True
        })
        diagonal_data.append({
            'Model': 'reference',
            'Type': type_val,  # Use the specific type value
            'Average ⬆️': max_val,
            'Metric': metric,
            'Score': max_val,
            'is_diagonal': True
        })

# Combine the original data with diagonal line data
plot_data['is_diagonal'] = False
combined_data = pd.concat([plot_data, pd.DataFrame(diagonal_data)])

# 4. Create the layered chart first

# Points layer for the scatter plot
points = alt.Chart(combined_data).mark_circle().encode(
    x=alt.X('Score:Q', title='Metric Score'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color=alt.Color('Type:N', legend=alt.Legend(orient='right')),
    tooltip=['Model', 'Type', 'Average ⬆️', 'Metric', 'Score']
).transform_filter(
    alt.datum.is_diagonal == False
)

# Diagonal line layer
lines = alt.Chart(combined_data).mark_line(
    color='red',
    strokeDash=[10, 10]
).encode(
    x='Score:Q',
    y='Average ⬆️:Q'
).transform_filter(
    alt.datum.is_diagonal == True
)

# 5. Combine points and lines, then facet the result
final_chart = alt.vconcat(
    (points + lines).properties(
        width=150,
        height=150
    ).facet(
        row='Type:N',
        column='Metric:N'
    ).properties(
        title='Performance Metrics Comparison Faceted by Type'
    )
)

# Save the chart
final_chart.save('individual_charts/11_faceted_performance_by_type.html')

# 12. Create faceted performance comparison by MoE
# 1. Reshape the data for faceting
metrics_to_plot = performance_metrics[1:]  # Skip 'Average ⬆️' as it's used for y-axis
plot_data_moe = df.melt(
    id_vars=['Model', 'MoE', 'Average ⬆️'], 
    value_vars=metrics_to_plot,
    var_name='Metric', 
    value_name='Score'
)

# 2. Create diagonal line data for MoE facets
diagonal_data_moe = []
for metric in metrics_to_plot:
    for moe_val in df['MoE'].unique():
        diagonal_data_moe.append({
            'Model': 'reference',
            'MoE': moe_val,
            'Average ⬆️': min_val,
            'Metric': metric,
            'Score': min_val,
            'is_diagonal': True
        })
        diagonal_data_moe.append({
            'Model': 'reference',
            'MoE': moe_val,
            'Average ⬆️': max_val,
            'Metric': metric,
            'Score': max_val,
            'is_diagonal': True
        })

# Combine the original data with diagonal line data
plot_data_moe['is_diagonal'] = False
combined_data_moe = pd.concat([plot_data_moe, pd.DataFrame(diagonal_data_moe)])

# Points layer for the scatter plot
points_moe = alt.Chart(combined_data_moe).mark_circle().encode(
    x=alt.X('Score:Q', title='Metric Score'),
    y=alt.Y('Average ⬆️:Q', title='Average Performance'),
    color=alt.Color('MoE:N', legend=alt.Legend(orient='right')),
    tooltip=['Model', 'MoE', 'Average ⬆️', 'Metric', 'Score']
).transform_filter(
    alt.datum.is_diagonal == False
)

# Diagonal line layer
lines_moe = alt.Chart(combined_data_moe).mark_line(
    color='red',
    strokeDash=[10, 10]
).encode(
    x='Score:Q',
    y='Average ⬆️:Q'
).transform_filter(
    alt.datum.is_diagonal == True
)

# Combine points and lines, then facet the result
final_chart_moe = alt.vconcat(
    (points_moe + lines_moe).properties(
        width=150,
        height=150
    ).facet(
        row='MoE:N',
        column='Metric:N'
    ).properties(
        title='Performance Metrics Comparison Faceted by MoE'
    )
)

# Save the chart
final_chart_moe.save('individual_charts/12_faceted_performance_by_moe.html')

# 13. Distribution of models across different categories
# Create separate charts for each category

# Type distribution
type_dist = alt.Chart(df).mark_bar().encode(
    x=alt.X('Type:N', title='Model Type'),
    y=alt.Y('count():Q', title='Number of Models'),
    color=alt.Color('Type:N', legend=None),
    tooltip=['Type', alt.Tooltip('count():Q', title='Count')]
).properties(
    width=300,
    height=200,
    title='Distribution of Models by Type'
)

# Base Model distribution
# Get top 10 base models by frequency to avoid overcrowding
top_base_models = (df['Base Model']
                  .value_counts()
                  .nlargest(10)
                  .index
                  .tolist())

base_model_dist = alt.Chart(
    df[df['Base Model'].isin(top_base_models)]
).mark_bar().encode(
    x=alt.X('Base Model:N', 
            title='Base Model (Top 10)',
            sort='-y'),
    y=alt.Y('count():Q', title='Number of Models'),
    color=alt.Color('Base Model:N', legend=None),
    tooltip=['Base Model', alt.Tooltip('count():Q', title='Count')]
).properties(
    width=300,
    height=200,
    title='Distribution of Models by Base Model (Top 10)'
)

# MoE distribution
moe_dist = alt.Chart(df).mark_bar().encode(
    x=alt.X('MoE:N', title='MoE Status'),
    y=alt.Y('count():Q', title='Number of Models'),
    color=alt.Color('MoE:N', legend=None),
    tooltip=['MoE', alt.Tooltip('count():Q', title='Count')]
).properties(
    width=300,
    height=200,
    title='Distribution of Models by MoE Status'
)

# Combine all charts vertically
combined_dist = alt.vconcat(
    type_dist,
    base_model_dist,
    moe_dist
).properties(
    title='Distribution of Models Across Different Categories'
)

# Save the combined chart
combined_dist.save('individual_charts/13_model_distributions.html')
