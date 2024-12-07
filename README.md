# Open LLM Leaderboard Explorer

An interactive dashboard for exploring and analyzing models from the Hugging Face Open LLM Leaderboard.

## Features

### 📊 Scatter Plot

- Compare any two metrics with customizable axes
- Size points by a third metric
- Group models by architecture, type, or model family
- Highlight specific models
- Add jitter to reduce overlapping points

### 📋 Data Table

- Search across all columns
- Customize visible columns
- View statistical summaries
- Export filtered data

### 🏆 Top Performers

- Analyze top models by any metric
- Compare individual models or architectures
- View detailed statistics and rankings
- Visualize performance distributions

### 📈 Model Analysis

- Deep dive into architecture performance
- Compare against overall averages
- Analyze model specialization
- View size distributions and trends

### 🔍 Model Deep Dive

- Compare up to three models
- Radar charts for performance visualization
- Detailed metric comparisons
- Performance relative to average

### 🔀 Parallel Plot

- Visualize relationships across multiple metrics
- Interactive filtering on any dimension
- Color-coded by average performance
- Compare architectures and model sizes

## Getting Started

1. Install dependencies:

   ```bash
   pip install streamlit pandas plotly scipy datasets
   ```

2. Run the app:

   ```bash
   streamlit run app.py
   ```

## Using the Filters

The sidebar contains several filters to help you focus on specific models:

- **Architecture**: Filter by model architecture
- **Model Size**: Select a range of model sizes (in billions of parameters)
- **Model Types**: Multi-select pill buttons for model types
- **Model Family**: Filter by model family/organization

## Tips

- Use the tabs at the top to switch between different views
- Hover over data points for detailed information
- Click and drag in plots to zoom
- Double-click to reset the view
- Use the search box in the Data Table to find specific models
- Click column headers to sort the data
- Use the parallel plot to find models that excel in specific metrics

## Data Source

Data is loaded directly from the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) using the Hugging Face datasets library.
