# Visualizing Performance Patterns of Open-Source Large Language Models

Authors:
Lars Schmid (schmila7@students.zhaw.ch)
Katsiaryna Mlynchyk (mlynckat@students.zhaw.ch)
Supervisor: Prof. Dr. Susanne Bleisch (susanne.bleisch@fhnw.ch)
Date: 8 December 2024

## Abstract

We developed an interactive Streamlit app to explore the Open LLM Leaderboard dataset, enabling users to filter models, create visualizations, and analyze performance metrics. Key findings revealed a strong correlation between model size and performance, with diminishing returns as size increases. The app guided the creation of infographics on selecting models for specific GPU configurations and understanding the environmental trade-offs of scaling up. This poster showcases the app’s methodology, tools, and insights.

## Exploration

Using Python, Pandas, and Plotly, we developed an interactive Streamlit app to analyze the Open LLM Leaderboard dataset, enabling dynamic data exploration. The app included several key features, such as interactive filters to sort models by architecture, size, type, or family. Users could also create customizable visualizations, including scatter plots, radar charts, heatmaps, and parallel coordinate plots, with adjustable axes and grouping options. Metric analysis tools allowed for the identification of top-performing models and an evaluation of efficiency through performance-to-parameter ratios. Additionally, the app provided insights through statistical summaries, z-score normalization, and clustering techniques, offering a deeper understanding of the data. This tool played a central role in guiding our exploration and uncovering patterns while validating hypotheses.

## Journey

The Open LLM Leaderboard dataset contains performance metrics for large language models. Through an initial exploration of the data, trends and patterns were identified, particularly in relation to model size, performance, and efficiency. The Streamlit app served as a primary tool for this analysis, allowing for interactive hypothesis testing and a more detailed understanding of the dataset. This exploration led to two key findings. First, there was a strong correlation between the number of parameters in a model and its performance across various metrics, confirming the hypothesis that larger models generally perform better. Second, it became evident that as model sizes increased, the performance gains began to plateau, indicating a threshold beyond which further increases in size resulted in diminishing returns. These findings led us to create the following infographics:

## Learning

Reflecting on this task, we learned the importance of dedicating time to thorough data exploration and the surprising complexity of creating impactful infographics, which demand far more attention to detail than typical plots. The development of an interactive app has made it much easier for us to identify patterns and test hypotheses and demonstrates the value of interactive tools for data analysis.

## Sources

“Open LLM Leaderboard”, available at https://huggingface.co/datasets/open-llm-leaderboard/contents
“Calculating GPU memory for serving LLMs”, available at https://www.substratus.ai/blog/calculating-gpu-memory-for-llm
