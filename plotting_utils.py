import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_top_n(df, target_column, n=10):
    top_n = df.nlargest(n, target_column)

    # Initialize the bar plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Set width for each bar and their positions
    width = 0.28
    ind = np.arange(len(top_n))

    # Plot target_column and MMLU_average on the primary y-axis with adjusted positions
    ax1.bar(ind - width, top_n[target_column], width=width, color='blue', label=target_column)
    ax1.bar(ind, top_n['MMLU_average'], width=width, color='orange', label='MMLU_average')

    # Set the primary y-axis labels and title
    ax1.set_title(f'Top {n} performing models on {target_column}')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')

    # Create a secondary y-axis for Parameters
    ax2 = ax1.twinx()

    # Plot Parameters as bars on the secondary y-axis with adjusted position
    ax2.bar(ind + width, top_n['Parameters'], width=width, color='red', label='Parameters')

    # Set the secondary y-axis labels
    ax2.set_ylabel('Parameters', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set the x-ticks and their labels
    ax1.set_xticks(ind)
    ax1.set_xticklabels(top_n.index, rotation=45, ha="right")

    # Adjust the legend
    fig.tight_layout()
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the plot
    st.pyplot(fig)

# Function to create an unfilled radar chart
def create_radar_chart_unfilled(df, model_names, metrics):
    fig = go.Figure()
    min_value = df.loc[model_names, metrics].min().min()
    max_value = df.loc[model_names, metrics].max().max()
    for model_name in model_names:
        values_model = df.loc[model_name, metrics]
        fig.add_trace(go.Scatterpolar(
            r=values_model,
            theta=metrics,
            name=model_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min_value, max_value]
            )),
        showlegend=True,
        width=800,  # Change the width as needed
        height=600   # Change the height as needed
    )
    return fig



# Function to create a line chart
def create_line_chart(df, model_names, metrics):
    line_data = []
    for model_name in model_names:
        values_model = df.loc[model_name, metrics]
        for metric, value in zip(metrics, values_model):
            line_data.append({'Model': model_name, 'Metric': metric, 'Value': value})

    line_df = pd.DataFrame(line_data)

    fig = px.line(line_df, x='Metric', y='Value', color='Model', title='Comparison of Models', line_dash_sequence=['solid'])
    fig.update_layout(showlegend=True)
    return fig

def create_plot(df, x_values, y_values, models=None, title=None):
    if models is not None:
        df = df[df.index.isin(models)]

    # remove rows with NaN values
    df = df.dropna(subset=[x_values, y_values])

    plot_data = pd.DataFrame({
        'Model': df.index,
        x_values: df[x_values],
        y_values: df[y_values],
    })

    plot_data['color'] = 'purple'
    fig = px.scatter(plot_data, x=x_values, y=y_values, color='color', hover_data=['Model'], trendline="ols")
    
    # If title is not provided, use x_values vs. y_values as the default title
    if title is None:
        title = x_values + " vs. " + y_values
    
    layout_args = dict(
        showlegend=False, 
        xaxis_title=x_values,
        yaxis_title=y_values,
        xaxis=dict(),
        yaxis=dict(),
        title=title,
        height=500,
        width=1000,
    )
    fig.update_layout(**layout_args)
    
    # Add a dashed line at 0.25 for the y_values
    x_min = df[x_values].min()
    x_max = df[x_values].max()

    y_min = df[y_values].min()
    y_max = df[y_values].max()

    if x_values.startswith('MMLU'): 
        fig.add_shape(
        type='line',
        x0=0.25, x1=0.25,
        y0=y_min, y1=y_max,
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
        )

    if y_values.startswith('MMLU'):
        fig.add_shape(
        type='line',
        x0=x_min, x1=x_max,
        y0=0.25, y1=0.25,
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
        )

    return fig