import pandas as pd
import altair as alt
import os

# Create a directory to store individual charts
os.makedirs('individual_charts', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/open_llm_leaderboard.csv')

# Columns to exclude
exclude_columns = ["Maintainer's Highlight", "Model", "Model sha"]

# Function to create a chart based on data type
def create_chart(df, column):
    field = alt.FieldName(column)
    
    if 'date' in column.lower():
        # Treat as time series
        df[column] = pd.to_datetime(df[column])
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X(field, type='temporal'),
            y='count()',
        ).properties(
            title=f'Time Series of {column}',
            width=400,
            height=200
        )
    elif df[column].dtype == 'object':
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(field, type='nominal', sort='-y'),
            y='count()',
            color=alt.Color(field, type='nominal')
        ).properties(
            title=f'Distribution of {column}',
            width=300,
            height=200
        )
    elif df[column].dtype in ['int64', 'float64']:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(field, type='quantitative', bin=alt.Bin(maxbins=30)),
            y='count()',
        ).properties(
            title=f'Distribution of {column}',
            width=300,
            height=200
        )
    elif df[column].dtype == 'bool':
        chart = alt.Chart(df).mark_arc().encode(
            theta='count()',
            color=alt.Color(field, type='nominal')
        ).properties(
            title=f'Distribution of {column}',
            width=200,
            height=200
        )
    else:
        print(f"Skipping column {column} due to unsupported dtype: {df[column].dtype}")
        return None
    
    # Save the chart to an individual HTML file
    safe_column_name = column.replace(" ", "_").replace("'", "").replace("#", "num")
    chart_path = f'individual_charts/{safe_column_name}.html'
    chart.save(chart_path)
    print(f"Chart for {column} saved to {chart_path}")
    return chart

# Create charts for each column
charts = []
for column in df.columns:
    if column not in exclude_columns:
        chart = create_chart(df, column)
        if chart:
            charts.append(chart)

# Special charts
print("Creating scatter plot")
chart_scatter = alt.Chart(df).mark_circle().encode(
    x=alt.X(alt.FieldName('#Params (B)'), type='quantitative', scale=alt.Scale(type='log')),
    y=alt.Y(alt.FieldName('Average ⬆️'), type='quantitative'),
    color=alt.Color('Architecture', type='nominal'),
    tooltip=[alt.Tooltip('Architecture', type='nominal'), 
             alt.Tooltip(alt.FieldName('#Params (B)'), type='quantitative'), 
             alt.Tooltip(alt.FieldName('Average ⬆️'), type='quantitative')]
).properties(
    title='Number of Parameters vs Average Score',
    width=400,
    height=300
)
chart_scatter.save('individual_charts/scatter_plot.html')
charts.append(chart_scatter)

print("Creating box plot")
tasks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
melted_df = df.melt(id_vars=['Architecture'], value_vars=tasks, var_name='Task', value_name='Score')
chart_tasks = alt.Chart(melted_df).mark_boxplot().encode(
    x=alt.X('Task', type='nominal'),
    y=alt.Y('Score', type='quantitative'),
    color=alt.Color('Task', type='nominal')
).properties(
    title='Distribution of Scores Across Different Tasks',
    width=400,
    height=300
)
chart_tasks.save('individual_charts/box_plot.html')
charts.append(chart_tasks)

# Combine all charts
print("Combining all charts")
combined_chart = alt.vconcat(*charts)

# Save the combined chart
combined_chart.save('open_llm_leaderboard_analysis.html')

print("\nAll charts have been saved individually in the 'individual_charts' directory.")
print("Combined chart has been saved to 'open_llm_leaderboard_analysis.html'")
