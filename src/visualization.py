import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    #Plots a correlation heatmap using Plotly.
    
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")
    return fig

def plot_distribution(df, column, mood='hist'):
    #Plots distribution of a column.
    if mood == 'hist':
        fig = px.histogram(df, x=column, title=f"Distribution of {column}", nbins=30)
    else:
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig

def plot_scatter(df, x_col, y_col, color_col=None):
    #Plots a scatter plot.
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
    return fig

def plot_target_distribution(df, target_col):

    #Plots the count of target classes.
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    fig = px.bar(counts, x=target_col, y='Count', title="Outcome Class Distribution", color=target_col)
    return fig

def plot_model_comparison(results_df):
    #Plots a bar chart comparing model metrics.

    melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig = px.bar(melted, x='Model', y='Score', color='Metric', barmode='group', title="Model Performance Comparison")
    return fig

def plot_confusion_matrix(cm, labels):
    #Plots a confusion matrix using Plotly.

    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    return fig
